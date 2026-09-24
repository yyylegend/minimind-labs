"""Torch 同进程 GRPO rollout，以及 completion token 概率对齐。"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from trainer.trainer_utils import autocast_context


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_logps: torch.Tensor
    responses: list[str]


def trim_completion(completion_ids: torch.Tensor, eos_token_id: int | None) -> torch.Tensor:
    """保留首个 EOS（含 EOS）；EOS 之后是 generate 补出的填充 token。"""
    if completion_ids.ndim != 1:
        raise ValueError("completion_ids 必须是一维 token 序列")
    if eos_token_id is None:
        return completion_ids
    eos_positions = torch.where(completion_ids == eos_token_id)[0]
    if eos_positions.numel() == 0:
        return completion_ids
    return completion_ids[: int(eos_positions[0].item()) + 1]


def compute_completion_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """取模型对每条 completion token 的 log-prob；prompt/padding 不返回。"""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids 和 attention_mask 必须形状相同且为二维")
    if prompt_lengths.shape != input_ids.shape[:1]:
        raise ValueError("prompt_lengths 必须每条序列一个长度")
    if completion_mask.ndim != 2 or completion_mask.shape[0] != input_ids.shape[0]:
        raise ValueError("completion_mask 必须为 [序列数, completion长度]")
    if input_ids.shape[1] < 2:
        raise ValueError("输入至少需要一个 prompt token 和一个 completion token")

    logits = model(input_ids, attention_mask=attention_mask).logits
    target_logps = F.log_softmax(logits[:, :-1, :].float(), dim=-1).gather(
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)

    offsets = torch.arange(completion_mask.shape[1], device=input_ids.device)
    positions = prompt_lengths.to(input_ids.device).unsqueeze(1) - 1 + offsets.unsqueeze(0)
    # 被 mask 掉的 padding 位置不会参与 loss；clamp 使 gather 对这些位置也始终安全。
    positions = positions.clamp(min=0, max=target_logps.shape[1] - 1)
    completion_logps = target_logps.gather(dim=1, index=positions)
    return completion_logps * completion_mask.to(dtype=completion_logps.dtype)


class TorchRolloutEngine:
    """每个 prompt 单独生成一组回答，避免左 padding 改变 RoPE 位置。"""

    def __init__(
        self,
        policy_model,
        tokenizer,
        device: torch.device,
        dtype: torch.dtype,
        max_seq_len: int,
        max_gen_len: int,
        temperature: float = 0.8,
    ) -> None:
        if max_seq_len < 1 or max_gen_len < 1:
            raise ValueError("prompt 和 generation 长度上限必须大于 0")
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.temperature = temperature

    def rollout(self, rows: list[dict], num_generations: int) -> RolloutBatch:
        if not rows:
            raise ValueError("rollout 至少需要一个 prompt")
        if num_generations < 2:
            raise ValueError("GRPO 每个 prompt 至少需要生成 2 个回答")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer 必须提供 pad_token_id 或 eos_token_id")

        was_training = self.policy_model.training
        self.policy_model.eval()
        sequences = []
        completions = []
        prompt_lengths = []
        responses = []
        try:
            for row in rows:
                encoded = self.tokenizer(
                    row["prompt"],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                )
                prompt_ids = encoded["input_ids"].to(self.device)
                if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
                    raise ValueError("每条 GRPO prompt 必须编码成非空的一维 token 序列")
                prompt_mask = torch.ones_like(prompt_ids)
                with torch.no_grad(), autocast_context(self.device, self.dtype):
                    generated = self.policy_model.generate(
                        input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                        attention_mask=prompt_mask.repeat_interleave(num_generations, dim=0),
                        max_new_tokens=self.max_gen_len,
                        temperature=self.temperature,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        use_cache=True,
                    ).clone()

                for output in generated:
                    completion = trim_completion(output[prompt_ids.shape[1] :], self.tokenizer.eos_token_id)
                    if completion.numel() == 0:
                        raise ValueError("模型生成了空 completion")
                    completion = completion.detach().to(device=self.device, dtype=torch.long)
                    sequences.append(torch.cat((prompt_ids[0], completion)))
                    completions.append(completion)
                    prompt_lengths.append(prompt_ids.shape[1])
                    responses.append(self.tokenizer.decode(completion, skip_special_tokens=True))
        finally:
            if was_training:
                self.policy_model.train()

        max_input_length = max(sequence.numel() for sequence in sequences)
        max_completion_length = max(completion.numel() for completion in completions)
        input_ids = torch.full(
            (len(sequences), max_input_length),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        completion_ids = torch.full(
            (len(completions), max_completion_length),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        completion_mask = torch.zeros_like(completion_ids)
        for index, (sequence, completion) in enumerate(zip(sequences, completions)):
            input_ids[index, : sequence.numel()] = sequence
            attention_mask[index, : sequence.numel()] = 1
            completion_ids[index, : completion.numel()] = completion
            completion_mask[index, : completion.numel()] = 1

        prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.long, device=self.device)
        was_training = self.policy_model.training
        self.policy_model.eval()
        try:
            with torch.no_grad(), autocast_context(self.device, self.dtype):
                old_logps = compute_completion_logps(
                    self.policy_model,
                    input_ids,
                    attention_mask,
                    prompt_lengths_tensor,
                    completion_mask,
                )
        finally:
            if was_training:
                self.policy_model.train()

        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths_tensor,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            old_logps=old_logps.detach(),
            responses=responses,
        )

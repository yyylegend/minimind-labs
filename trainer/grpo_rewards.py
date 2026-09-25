"""MiniMind CISPO 使用的 reward-model 与规则奖励。"""

from dataclasses import asdict, dataclass
import re

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class RewardBreakdown:
    reward_model: float
    length: float
    thinking: float
    repetition_penalty: float

    @property
    def total(self) -> float:
        return self.reward_model + self.length + self.thinking - self.repetition_penalty

    def to_dict(self) -> dict[str, float]:
        values = asdict(self)
        values["total"] = self.total
        return values


class MiniMindRewardModel:
    """包装 MiniMind 官方 InternLM reward model 的 get_score 接口。"""

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

    @torch.inference_mode()
    def get_score(self, messages: list[dict], response: str) -> float:
        history_text = "\n".join(
            f"{message['role']}: {message.get('content', '')}" for message in messages[:-1]
        )
        last_query = messages[-1].get("content", "") if messages else ""
        message_context = (
            f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}"
            if history_text
            else last_query
        )
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response},
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(float(score), 3.0), -3.0)


def repetition_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    repeated = len(ngrams) - len(set(ngrams))
    return min(cap, repeated * cap * 2 / len(ngrams))


def score_completions(
    messages_per_prompt: list[list[dict]],
    responses: list[str],
    reward_model,
    num_generations: int,
) -> list[RewardBreakdown]:
    """按 prompt 分组给回答打分，并保留官方 reward 的各组成项。"""
    if num_generations < 1:
        raise ValueError("num_generations 必须至少为 1")
    if len(responses) != len(messages_per_prompt) * num_generations:
        raise ValueError("responses 数量必须等于 prompt 数量乘以 num_generations")

    results = []
    for prompt_index, messages in enumerate(messages_per_prompt):
        for generation_index in range(num_generations):
            response = responses[prompt_index * num_generations + generation_index]
            length_reward = 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            thinking_reward = 0.0
            answer = response.strip()
            if "</think>" in response:
                thinking, answer = response.split("</think>", 1)
                thinking_reward += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
                thinking_reward += 0.25 if response.count("</think>") == 1 else -0.25
                answer = answer.strip()

            repetition = repetition_penalty(answer)
            reward_score = max(min(float(reward_model.get_score(messages, answer)), 3.0), -3.0)
            results.append(
                RewardBreakdown(
                    reward_model=reward_score,
                    length=length_reward,
                    thinking=thinking_reward,
                    repetition_penalty=repetition,
                )
            )
    return results

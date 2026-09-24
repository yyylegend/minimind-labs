"""MiniMind 数据集适配：复用主仓库的 JSONL 格式。"""

import json
import random
from typing import Any

import torch
from torch.utils.data import Dataset


def _load_json_dataset(path: str, features=None):
    """用 HuggingFace datasets 读取 JSONL，避免一次性把大文件读进内存。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("读取 MiniMind 数据集需要 datasets，请先安装主仓库 requirements.txt 中的依赖。") from exc
    kwargs = {"data_files": path, "split": "train"}
    if features is not None:
        kwargs["features"] = features
    return load_dataset("json", **kwargs)


def _conversation_features():
    """显式声明 MiniMind SFT 对话字段，兼容 tool-call 样本的可选字段。"""
    from datasets import Features, Value

    return Features(
        {
            "conversations": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                    "reasoning_content": Value("string"),
                    "tools": Value("string"),
                    "tool_calls": Value("string"),
                }
            ]
        }
    )


def _input_ids(encoded: Any) -> list[int]:
    """兼容 tokenizer 返回 BatchEncoding 或普通字典。"""
    return list(encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids)


def _pad_lm_example(
    token_ids: list[int],
    max_length: int,
    pad_token_id: int,
    bos_token_id: int | None,
    eos_token_id: int | None,
) -> dict[str, torch.Tensor]:
    """把一段 Token 变成训练需要的 input_ids、labels 和 attention_mask。"""
    prefix = [] if bos_token_id is None else [bos_token_id]
    suffix = [] if eos_token_id is None else [eos_token_id]
    tokens = (prefix + token_ids + suffix)[:max_length]
    valid_length = len(tokens)
    padding = max_length - valid_length

    input_ids = torch.tensor(tokens + [pad_token_id] * padding, dtype=torch.long)
    attention_mask = torch.tensor([1] * valid_length + [0] * padding, dtype=torch.long)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class PretrainDataset(Dataset):
    """读取 ``{"text": "..."}`` 格式的 next-token 预训练数据。"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = _load_json_dataset(data_path)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        encoded = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        )
        return _pad_lm_example(
            _input_ids(encoded),
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )


def _preprocess_chat(conversations: list[dict[str, Any]], add_system_ratio: float = 0.2):
    """沿用 MiniMind 的轻量 system prompt 增强。"""
    if any(message.get("tools") for message in conversations):
        return conversations
    if conversations and conversations[0].get("role") != "system" and random.random() < add_system_ratio:
        system_prompts = [
            "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
            "你是minimind，一个小巧但有用的语言模型。",
            "你是一个专业的AI助手，请提供有价值的回答。",
            "You are a helpful AI assistant.",
        ]
        return [{"role": "system", "content": random.choice(system_prompts)}] + conversations
    return conversations


def _postprocess_chat(prompt: str, empty_think_ratio: float = 0.2) -> str:
    empty_think = "<think>\n\n</think>\n\n"
    if empty_think in prompt and random.random() > empty_think_ratio:
        prompt = prompt.replace(empty_think, "")
    return prompt


class SFTDataset(Dataset):
    """读取 MiniMind 对话 JSONL，只让 assistant 部分参与 loss。"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024, augment: bool = True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.samples = _load_json_dataset(data_path, features=_conversation_features())
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = _input_ids(tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False))
        self.eos_id = _input_ids(tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False))

    def __len__(self) -> int:
        return len(self.samples)

    def create_chat_prompt(self, conversations: list[dict[str, Any]]) -> str:
        messages = []
        tools = None
        for raw_message in conversations:
            message = dict(raw_message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
        )

    def generate_labels(self, input_ids: list[int]) -> list[int]:
        """只标记 assistant 段；其他位置用 -100 忽略。"""
        labels = [-100] * len(input_ids)
        index = 0
        while index < len(input_ids):
            if input_ids[index : index + len(self.bos_id)] == self.bos_id:
                start = index + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for label_index in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[label_index] = input_ids[label_index]
                index = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                index += 1
        return labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        conversations = sample["conversations"]
        if self.augment:
            conversations = _preprocess_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        if self.augment:
            prompt = _postprocess_chat(prompt)
        input_ids = _input_ids(self.tokenizer(prompt, add_special_tokens=False))[: self.max_length]
        labels = self.generate_labels(input_ids)
        padding = self.max_length - len(input_ids)
        input_ids += [self.pad_token_id] * padding
        labels += [-100] * padding
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * (self.max_length - padding) + [0] * padding,
                dtype=torch.long,
            ),
        }


class RLAIFDataset(Dataset):
    """读取 MiniMind RLAIF 对话，只把待回答的上下文交给策略模型。"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 768,
        thinking_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if max_length < 1:
            raise ValueError("GRPO max_length 必须至少为 1")
        if not 0.0 <= thinking_ratio <= 1.0:
            raise ValueError("thinking_ratio 必须在 [0, 1] 范围内")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
        # 官方 RLAIF 文件可能带额外顶层字段；只读 conversations，避开 Arrow 对整行列名的强 schema cast。
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as data_file:
            for line_number, line in enumerate(data_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"RLAIF JSONL 第 {line_number} 行不是有效 JSON") from exc
                conversations = row.get("conversations") if isinstance(row, dict) else None
                if not isinstance(conversations, list):
                    raise ValueError(f"RLAIF JSONL 第 {line_number} 行缺少 conversations 列表")
                self.samples.append(conversations)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        conversations = self.samples[index]
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError(f"RLAIF 样本 {index} 至少需要一条上下文消息和一条末尾 assistant 消息")
        if conversations[-1].get("role") != "assistant":
            raise ValueError(f"RLAIF 样本 {index} 的最后一条消息必须是 assistant")

        # 末尾 assistant 是数据集携带的参考回复；GRPO 只使用它前面的对话作为问题。
        messages = [dict(message) for message in conversations[:-1]]
        messages = _preprocess_chat(messages)
        tools = None
        for message in messages:
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=random.random() < self.thinking_ratio,
            tools=tools,
        )
        return {"prompt": prompt, "messages": messages}


class DPODataset(Dataset):
    """读取 chosen/rejected 对话，只对最后一条 assistant 回复计算偏好。"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 768) -> None:
        super().__init__()
        if max_length < 2:
            raise ValueError("DPO max_length 必须至少为 2")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.samples = _load_json_dataset(data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_conversation(self, messages: list[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        conversation_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = _input_ids(self.tokenizer(prompt_text, add_special_tokens=False))
        input_ids = _input_ids(self.tokenizer(conversation_text, add_special_tokens=False))

        # 用 prompt 和完整对话的最长共同 token 前缀定位回复起点，避免把问题算进偏好分数。
        shared_length = 0
        for prompt_id, token_id in zip(prompt_ids, input_ids):
            if prompt_id != token_id:
                break
            shared_length += 1
        if shared_length == 0 or shared_length == len(input_ids):
            raise ValueError("DPO 样本无法定位 assistant 回复，请检查 tokenizer chat template")

        prompt_ids = input_ids[:shared_length]
        response_ids = input_ids[shared_length:]
        response_length = min(len(response_ids), self.max_length)
        prompt_length = min(len(prompt_ids), self.max_length - response_length)
        retained_prompt = prompt_ids[-prompt_length:] if prompt_length else []
        retained_response = response_ids[:response_length]
        retained_ids = retained_prompt + retained_response
        response_mask = [0] * len(retained_prompt) + [1] * len(retained_response)
        valid_length = len(retained_ids)
        padding_length = self.max_length - valid_length

        return (
            torch.tensor(retained_ids + [self.pad_token_id] * padding_length, dtype=torch.long),
            torch.tensor([1] * valid_length + [0] * padding_length, dtype=torch.long),
            torch.tensor(response_mask + [0] * padding_length, dtype=torch.long),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        chosen = sample.get("chosen")
        rejected = sample.get("rejected")
        if not isinstance(chosen, list) or not isinstance(rejected, list) or not chosen or not rejected:
            raise ValueError(f"DPO 样本 {index} 必须包含非空 chosen/rejected 对话")
        if chosen[:-1] != rejected[:-1]:
            raise ValueError(f"DPO 样本 {index} 的 chosen/rejected 必须共享相同对话上下文")
        if chosen[-1].get("role") != "assistant" or rejected[-1].get("role") != "assistant":
            raise ValueError(f"DPO 样本 {index} 的最后一条消息必须来自 assistant")

        chosen_ids, chosen_attention, chosen_response = self._encode_conversation(chosen)
        rejected_ids, rejected_attention, rejected_response = self._encode_conversation(rejected)
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_attention,
            "chosen_response_mask": chosen_response,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_attention,
            "rejected_response_mask": rejected_response,
        }

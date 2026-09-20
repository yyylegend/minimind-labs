"""MiniMind 数据集适配：复用主仓库的 JSONL 格式。"""

import json
import random
from typing import Any

import torch
from torch.utils.data import Dataset


def _load_json_dataset(path: str):
    """用 HuggingFace datasets 读取 JSONL，避免一次性把大文件读进内存。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("读取 MiniMind 数据集需要 datasets，请先安装主仓库 requirements.txt 中的依赖。") from exc
    return load_dataset("json", data_files=path, split="train")


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

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = _load_json_dataset(data_path)
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
        conversations = _preprocess_chat(sample["conversations"])
        prompt = _postprocess_chat(self.create_chat_prompt(conversations))
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

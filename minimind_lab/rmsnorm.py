"""RMSNorm：Transformer 中的均方根归一化。"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """沿最后一个维度归一化，并保留一个可学习缩放参数。"""

    # dim：输入向量的长度，比如 MiniMind 中可能是 768；eps：防止除以 0 的极小值
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 float32 做统计更稳，再恢复输入 dtype，和 MiniMind 保持一致。
        normalized = self._normalize(x.float())
        return (self.weight * normalized).to(dtype=x.dtype)

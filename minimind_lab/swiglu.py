"""SwiGLU / FFN：Transformer 中负责逐 token 加工的前馈网络。"""

import torch
from torch import nn


class SwiGLU(nn.Module):
    """用门控机制加工每个 token 的隐藏向量。

    输入和输出形状都是 [..., hidden_size]，中间会暂时扩展到
    [..., intermediate_size]。
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # gate：决定哪些特征应该通过多少。
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # up：生成待加工的候选特征。
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # down：把扩展后的特征压回 hidden_size，才能接残差。
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MiniMind 的核心公式：
        # down_proj(SiLU(gate_proj(x)) * up_proj(x))
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FeedForward(SwiGLU):
    """和 MiniMind 主仓库保持一致的命名。"""

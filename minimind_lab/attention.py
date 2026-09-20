"""Attention：把 RMSNorm、RoPE、GQA 和 KV Cache 串起来。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rmsnorm import RMSNorm
from .rope import apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把较少的 KV 头复制成 Q 头数量，用于 GQA。

    输入 x 的形状是 [batch, sequence, kv_heads, head_dim]。
    例如 4 个 Q 头、2 个 KV 头时，n_rep=2。
    """
    if n_rep == 1:
        return x

    batch_size, sequence_length, kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch_size, sequence_length, kv_heads, n_rep, head_dim)
        .reshape(batch_size, sequence_length, kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """一个可独立测试的因果自注意力模块。

    输入 x 的形状为 [batch, sequence, hidden_size]。
    position_embeddings 是 (cos, sin)，形状均为 [sequence, head_dim]。
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int | None = None,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-5,
        flash_attn: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")

        num_key_value_heads = num_key_value_heads or num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_rep = num_attention_heads // num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        # Q/K/V 投影：把每个 token 的隐藏向量变成注意力使用的三种视角。
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # 这是当前 MiniMind Attention 中的 Q/K 归一化，和输入层的 RMSNorm 不同，
        # 它归一化的是每个 head 内部的 head_dim 维向量。
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.is_causal = True
        self.flash = flash_attn and hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """执行一次因果注意力，并可返回当前累积的 K/V。

        past_key_value 中的 K/V 形状为 [batch, past_sequence, kv_heads, head_dim]。
        使用缓存时，调用方需要给当前位置对应的 cos、sin。
        attention_mask 若提供，形状为 [batch, total_key_sequence]，1 表示有效。
        """
        batch_size, sequence_length, _ = x.shape
        cos, sin = position_embeddings

        # 1. 线性投影并拆头：
        #    [B, S, hidden] -> Q:[B, S, q_heads, D]
        #                         K/V:[B, S, kv_heads, D]
        q = self.q_proj(x).view(batch_size, sequence_length, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        # 2. 先稳定 Q/K 的数值，再注入位置信息；V 不做这两步。
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. 把历史 K/V 接到当前 K/V 前面。
        #    历史 K 已经在它原来的位置做过 RoPE，不需要重复旋转。
        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_length = past_k.shape[1]
            k = torch.cat((past_k, k), dim=1)
            v = torch.cat((past_v, v), dim=1)

        present_key_value = (k, v) if use_cache else None
        total_key_length = k.shape[1]

        # 4. GQA：KV 头少于 Q 头时，让多个 Q 头共享同一组 K/V。
        q = q.transpose(1, 2)  # [B, q_heads, S, D]
        k = repeat_kv(k, self.n_rep).transpose(1, 2)  # [B, q_heads, total_S, D]
        v = repeat_kv(v, self.n_rep).transpose(1, 2)  # [B, q_heads, total_S, D]

        # 5. 长序列训练时优先使用 PyTorch 的融合实现；它和下面的手写公式等价，
        #    但通常更快、更省显存。KV Cache 或 Padding 掩码场景先走手写路径，便于
        #    处理当前 query 与历史 key 的不等长情况。
        can_use_flash = (
            self.flash
            and sequence_length > 1
            and past_key_value is None
            and (attention_mask is None or torch.all(attention_mask == 1))
        )
        if can_use_flash:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            # 6. 手写注意力：QK^T 得到“当前 token 要关注哪些历史 token”的分数。
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 因果掩码：第 i 个当前位置不能看到它后面的 token。
            # 有 KV Cache 时，当前 query 的绝对位置从 past_length 开始。
            query_positions = past_length + torch.arange(sequence_length, device=x.device)
            key_positions = torch.arange(total_key_length, device=x.device)
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            scores = scores.masked_fill(~causal_mask, float("-inf"))

            # Padding 掩码：补齐位置也不能被关注。
            if attention_mask is not None:
                if attention_mask.shape != (batch_size, total_key_length):
                    raise ValueError("attention_mask 必须是 [batch, total_key_sequence]")
                scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))

            attention_weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
            attention_weights = self.attn_dropout(attention_weights)
            output = attention_weights @ v

        # 8. 合并多个头，并投影回 hidden_size，交给外部残差连接。
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, present_key_value

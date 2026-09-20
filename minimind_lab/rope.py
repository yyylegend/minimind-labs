"""RoPE：用旋转给 Attention 的 Q、K 注入位置信息。"""

import torch


def precompute_freqs_cis(
    dim: int,
    end: int,
    rope_base: float = 1e6,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预先计算每个位置、每个维度对应的 cos 和 sin。

    参数：
        dim: 每个 Attention head 的维度，必须是偶数，因为旋转需要两两配对。
        end: 最多计算多少个位置，例如序列长度 128 就传 128。
        rope_base: 控制不同维度旋转速度的基数。
        device: 计算表放在哪个设备上；要和后续的 Q、K 保持一致。

    返回：
        freqs_cos、freqs_sin，形状都是 [end, dim]。
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("RoPE 的 dim 必须是正偶数")
    if end < 0:
        raise ValueError("RoPE 的 end 不能是负数")

    # 每隔两个维度取一个频率。前面的频率转得快，后面的频率转得慢。
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # positions: [end], inv_freq: [dim // 2]
    # 外积后得到每个“位置 × 频率”的旋转角度。
    positions = torch.arange(end, device=device).float()
    angles = torch.outer(positions, inv_freq).float()  # [end, dim // 2]

    # 把一半角度复制到另一半，配合 rotate_half 完成二维旋转。
    freqs_cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    freqs_sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    return freqs_cos, freqs_sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把向量后半段搬到前面并取负，作为旋转公式中的另一项。"""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q、K 做 RoPE 旋转。

    约定 q、k 的形状为 [batch_size, sequence_length, num_heads, head_dim]，
    cos、sin 的形状为 [sequence_length, head_dim]。
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q 和 k 必须是 [batch, sequence, heads, head_dim] 四维张量")
    if q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q 和 k 的 sequence_length、head_dim 必须一致")
    if cos.shape != sin.shape or cos.shape != (q.shape[1], q.shape[-1]):
        raise ValueError("cos、sin 必须是 [sequence_length, head_dim]")

    # [sequence, head_dim] -> [1, sequence, 1, head_dim]
    # 这样一个位置的 cos/sin 会广播给该位置的所有 batch 和 head。
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # 旋转公式：x' = x * cos(theta) + rotate_half(x) * sin(theta)
    q_embed = (q * cos + _rotate_half(q) * sin).to(dtype=q.dtype)
    k_embed = (k * cos + _rotate_half(k) * sin).to(dtype=k.dtype)
    return q_embed, k_embed

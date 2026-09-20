import torch

from minimind_lab import apply_rotary_pos_emb, precompute_freqs_cis


def main() -> None:
    torch.manual_seed(0)
    sequence_length, head_dim = 4, 4
    cos, sin = precompute_freqs_cis(dim=head_dim, end=sequence_length)

    # Attention 在投影 Q、K 后，常见形状是 [batch, sequence, heads, head_dim]。
    q = torch.randn(1, sequence_length, 1, head_dim)
    k = torch.randn(1, sequence_length, 1, head_dim)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    print(f"cos 表形状: {cos.shape}")
    print(f"Q 形状: {q.shape} -> 旋转后: {q_rot.shape}")
    print("\n第 0 个位置的 Q（位置 0 的旋转角度为 0）：")
    print(q[0, 0, 0])
    print(q_rot[0, 0, 0])
    print("\n第 1 个位置的 Q（已经发生旋转）：")
    print(q[0, 1, 0])
    print(q_rot[0, 1, 0])


if __name__ == "__main__":
    main()

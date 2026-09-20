import torch

from minimind_lab import Attention, precompute_freqs_cis


def main() -> None:
    torch.manual_seed(0)
    batch_size, sequence_length = 1, 4
    hidden_size, num_heads, num_kv_heads = 8, 4, 2
    head_dim = hidden_size // num_heads

    attention = Attention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
    )
    x = torch.randn(batch_size, sequence_length, hidden_size)
    cos, sin = precompute_freqs_cis(dim=head_dim, end=sequence_length)

    output, cache = attention(x, (cos, sin), use_cache=True)

    print(f"输入 x: {x.shape}")
    print(f"Q 头数: {num_heads}, KV 头数: {num_kv_heads}")
    print(f"Attention 输出: {output.shape}")
    print(f"缓存 K: {cache[0].shape}, 缓存 V: {cache[1].shape}")
    print("\n含义：4 个 Q 头共享 2 个 KV 头，每个 KV 头被 2 个 Q 头复用。")


if __name__ == "__main__":
    main()

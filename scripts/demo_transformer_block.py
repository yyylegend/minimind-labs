import torch

from minimind_lab import TransformerBlock, precompute_freqs_cis


def main() -> None:
    torch.manual_seed(0)
    hidden_size, sequence_length = 8, 4
    block = TransformerBlock(
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=16,
    )
    x = torch.randn(1, sequence_length, hidden_size)
    cos, sin = precompute_freqs_cis(hidden_size // 4, sequence_length)
    y, _ = block(x, (cos, sin))

    print(f"输入 x: {x.shape}")
    print(f"输出 y: {y.shape}")
    print("\n流程：RMSNorm → Attention → 残差 → RMSNorm → FFN → 残差")


if __name__ == "__main__":
    main()

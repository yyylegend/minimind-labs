import torch

from minimind_lab import SwiGLU


def main() -> None:
    torch.manual_seed(0)
    ffn = SwiGLU(hidden_size=8, intermediate_size=16)
    x = torch.randn(2, 3, 8)
    y = ffn(x)

    print(f"输入 x: {x.shape}")
    print(f"gate/up 中间特征: {ffn.gate_proj.out_features}")
    print(f"输出 y: {y.shape}")
    print("\n流程：hidden_size → 扩展 → SiLU(gate) × up → 压回 hidden_size")


if __name__ == "__main__":
    main()

import torch

from minimind_lab import RMSNorm


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    y = RMSNorm(dim=4)(x)

    print("输入 x：")
    print(x)
    print("\nRMSNorm 输出：")
    print(y)
    print("\n每行输出平方均值：")
    print(y.square().mean(dim=-1))


if __name__ == "__main__":
    main()

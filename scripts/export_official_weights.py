"""从可恢复训练 checkpoint 中导出官方 MiniMind 风格的纯权重。"""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 MiniMind 官方风格 .pth 权重")
    parser.add_argument("--checkpoint", required=True, help="例如 out/pretrain_last.pt")
    parser.add_argument("--output", required=True, help="例如 out/pretrain_512.pth")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    weights = {name: value.half().cpu() for name, value in state_dict.items()}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, output_path)
    print(f"已导出：{output_path}")


if __name__ == "__main__":
    main()

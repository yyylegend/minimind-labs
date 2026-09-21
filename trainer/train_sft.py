"""使用 MiniMind 对话 JSONL 数据进行 SFT。"""

import argparse

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from dataset.lm_dataset import SFTDataset
from minimind_lab import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import (
    EpochRandomSampler,
    load_model_weights,
    resolve_device,
    resolve_dtype,
    setup_seed,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMind 学习版 SFT")
    parser.add_argument("--data_path", default="./data/sft_t2t_mini.jsonl")
    parser.add_argument("--tokenizer_path", default="../minimind/model")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--resume_checkpoint", default="", help="从 SFT last checkpoint 继续训练")
    parser.add_argument("--tensorboard_dir", default="./out/runs/sft")
    parser.add_argument("--output_dir", default="./out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=0, help="只跑指定 optimizer steps，0 表示完整训练")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.02, help="从 SFT 数据中固定划分的验证比例")
    parser.add_argument("--eval_interval", type=int, default=500, help="每隔多少 optimizer step 计算一次验证 loss")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    config = MiniMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_seq_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    model = MiniMindForCausalLM(config).to(device)
    if args.init_checkpoint:
        load_model_weights(model, args.init_checkpoint, device)

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len, augment=True)
    eval_dataloader = None
    train_dataset = dataset
    if not 0 <= args.eval_ratio < 1:
        raise ValueError("eval_ratio 必须在 [0, 1) 范围内")
    if args.eval_ratio > 0:
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator).tolist()
        eval_size = max(1, int(len(dataset) * args.eval_ratio))
        train_indices = indices[eval_size:]
        eval_indices = indices[:eval_size]
        train_dataset = Subset(dataset, train_indices)
        eval_base = SFTDataset(
            args.data_path,
            tokenizer,
            max_length=args.max_seq_len,
            augment=False,
        )
        eval_dataset = Subset(eval_base, eval_indices)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        print(f"SFT 数据划分：train={len(train_dataset)}, eval={len(eval_dataset)}")

    sampler = EpochRandomSampler(train_dataset, seed=args.seed)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_model(
        model=model,
        dataloader=dataloader,
        config=config,
        device=device,
        dtype=dtype,
        output_dir=args.output_dir,
        stage="sft",
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        accumulation_steps=args.accumulation_steps,
        grad_clip=args.grad_clip,
        save_interval=args.save_interval,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        resume_checkpoint=args.resume_checkpoint,
        tensorboard_dir=args.tensorboard_dir,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        eval_dataloader=eval_dataloader,
        eval_interval=args.eval_interval,
        require_targets=True,
    )


if __name__ == "__main__":
    main()

"""使用 chosen/rejected 偏好对继续训练 MiniMind 学习版模型。"""

import argparse
import math

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import DPODataset
from minimind_lab import MiniMindConfig, MiniMindForCausalLM
from trainer.dpo_utils import dpo_loss
from trainer.metrics import TrainingMetrics
from trainer.trainer_utils import (
    EpochRandomSampler,
    _build_resumed_dataloader,
    _recover_from_fp16_overflow,
    autocast_context,
    cosine_learning_rate,
    format_duration,
    load_model_weights,
    load_training_checkpoint,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    setup_seed,
)


def _sequence_logps(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """汇总每条 completion 的 token log-prob；prompt 和 padding 不计分。"""
    targets = input_ids[:, 1:]
    valid_targets = response_mask[:, 1:].bool() & attention_mask[:, 1:].bool()
    if not valid_targets.any(dim=1).all().item():
        raise ValueError("DPO batch 中存在没有有效 assistant target 的样本")
    token_logps = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    token_logps = token_logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logps * valid_targets).sum(dim=-1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMind 学习版 DPO")
    parser.add_argument("--data_path", default="./data/dpo.jsonl")
    parser.add_argument("--tokenizer_path", default="../minimind/model")
    parser.add_argument("--init_checkpoint", required=True, help="作为策略初始权重和冻结 reference 的 SFT 权重")
    parser.add_argument("--resume_checkpoint", default="", help="从 dpo_last.pt 恢复训练")
    parser.add_argument("--output_dir", default="./out/checkpoints/dpo")
    parser.add_argument("--tensorboard_dir", default="./out/runs/dpo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=2, help="每批偏好对数量；前向时会同时计算 chosen 和 rejected")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-8)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=0, help="0 表示完整训练；用于 smoke test 时可设为 2")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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

    # 两个模型从同一份 SFT 权重开始：policy 会更新，reference 始终冻结。
    policy = MiniMindForCausalLM(config).to(device)
    load_model_weights(policy, args.init_checkpoint, device)
    reference = MiniMindForCausalLM(config).to(device)
    load_model_weights(reference, args.init_checkpoint, device)
    reference.eval()
    reference.requires_grad_(False)

    dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = EpochRandomSampler(dataset, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    writer = None
    if args.tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(args.tensorboard_dir)
        except ImportError:
            print("未安装 tensorboard，跳过网页监控；训练本身继续运行。")

    start_epoch, global_step, resume_batch = 0, 0, 0
    if args.resume_checkpoint:
        start_epoch, global_step, resume_batch = load_training_checkpoint(
            policy, optimizer, scaler, args.resume_checkpoint, device
        )
        print(f"已恢复 DPO checkpoint：epoch={start_epoch}, step={global_step}, batch={resume_batch}")

    batches_per_epoch = len(dataloader)
    steps_per_epoch = math.ceil(batches_per_epoch / args.accumulation_steps)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    metrics = TrainingMetrics(start_step=global_step)
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    last_update_batch = resume_batch
    stop_training = False

    try:
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            skip_batches = resume_batch if epoch == start_epoch else 0
            resume_batch = 0
            epoch_loader = _build_resumed_dataloader(dataloader, skip_batches)
            for local_batch_index, batch in enumerate(epoch_loader, start=1):
                batch_index = skip_batches + local_batch_index
                # 把 chosen/rejected 合成一次前向；前半是 chosen，后半是 rejected。
                chosen_ids = batch["chosen_input_ids"]
                rejected_ids = batch["rejected_input_ids"]
                input_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
                attention_mask = torch.cat(
                    [batch["chosen_attention_mask"], batch["rejected_attention_mask"]], dim=0
                )
                response_mask = torch.cat(
                    [batch["chosen_response_mask"], batch["rejected_response_mask"]], dim=0
                )
                pair_count = chosen_ids.shape[0]
                target_labels = input_ids.masked_fill(response_mask == 0, -100)
                metrics.record_batch(input_ids, attention_mask, target_labels)

                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                response_mask = response_mask.to(device, non_blocking=True)
                with autocast_context(device, dtype):
                    with torch.no_grad():
                        reference_logps = _sequence_logps(
                            reference(input_ids, attention_mask=attention_mask).logits,
                            input_ids,
                            attention_mask,
                            response_mask,
                        )
                    policy_logits = policy(input_ids, attention_mask=attention_mask).logits
                    policy_logps = _sequence_logps(
                        policy_logits, input_ids, attention_mask, response_mask
                    )
                    # DPO 比较两模型对答案对的相对偏好，而不是单独拟合一个答案。
                    loss = dpo_loss(
                        policy_logps[:pair_count],
                        policy_logps[pair_count:],
                        reference_logps[:pair_count],
                        reference_logps[pair_count:],
                        beta=args.beta,
                    )
                    scaled_loss = loss / args.accumulation_steps

                if not torch.isfinite(loss.detach()).item():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f"DPO loss 变为非有限值：epoch={epoch + 1}, batch={batch_index}")
                scaler.scale(scaled_loss).backward()

                should_update = (
                    batch_index % args.accumulation_steps == 0 or batch_index == batches_per_epoch
                )
                if not should_update:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(policy.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm).item():
                    if use_scaler:
                        scale_before, scale_after = _recover_from_fp16_overflow(scaler, optimizer)
                        metrics.record_fp16_overflow()
                        print(f"DPO FP16 梯度溢出，跳过更新；GradScaler: {scale_before:.0f} -> {scale_after:.0f}")
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("DPO 梯度范数变为非有限值")

                next_step = global_step + 1
                current_lr = cosine_learning_rate(
                    next_step, total_steps, args.learning_rate,
                    warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                last_update_batch = batch_index
                snapshot = metrics.record_update(global_step, total_steps)

                if global_step % args.log_interval == 0 or batch_index == batches_per_epoch:
                    grad_norm_value = float(grad_norm.item())
                    print(
                        f"stage=dpo epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                        f"loss={loss.item():.4f} lr={current_lr:.2e} "
                        f"pairs/s={pair_count * snapshot.updates_per_second:.1f} "
                        f"target_tok/s={snapshot.target_tokens_per_second:.0f} "
                        f"grad={grad_norm_value:.3f} eta={format_duration(snapshot.eta_seconds)}"
                    )
                    if writer is not None:
                        writer.add_scalar("train/dpo_loss", loss.item(), global_step)
                        writer.add_scalar("train/learning_rate", current_lr, global_step)
                        writer.add_scalar("train/grad_norm", grad_norm_value, global_step)
                        writer.add_scalar("throughput/preference_pairs_per_second", pair_count * snapshot.updates_per_second, global_step)
                        writer.add_scalar("throughput/target_tokens_per_second", snapshot.target_tokens_per_second, global_step)

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    save_checkpoint(policy, optimizer, scaler, config, args.output_dir, "dpo", epoch, global_step, batch_index)
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break

            save_checkpoint(policy, optimizer, scaler, config, args.output_dir, "dpo", epoch, global_step, last_update_batch)
            if stop_training:
                break
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()

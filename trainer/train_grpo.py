"""MiniMind 学习版 GRPO/CISPO：Torch rollout、奖励打分和单卡训练。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from dataset.lm_dataset import RLAIFDataset
from minimind_lab import MiniMindConfig, MiniMindForCausalLM
from trainer.grpo_objective import cispo_loss, group_relative_advantages
from trainer.grpo_rewards import MiniMindRewardModel, score_completions
from trainer.grpo_rollout import TorchRolloutEngine, compute_completion_logps
from trainer.metrics import TrainingMetrics
from trainer.trainer_utils import (
    EpochRandomSampler,
    _build_resumed_dataloader,
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


def _identity_collate(rows: list[dict]) -> list[dict]:
    return rows


def _sha256_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"制品路径不存在：{path}")
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"制品目录为空：{path}")
    for file_path in files:
        if path.is_dir():
            digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    project_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return revision.stdout.strip() or "unknown", bool(status.stdout.strip())


def _load_data_manifest(path: str, data_path: str, data_sha256: str) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"必须提供数据准备脚本生成的 manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("source_url", "source_revision", "source_license", "source_sha256", "heldout_sha256")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError(f"数据 manifest 缺少来源信息：{missing}")
    if Path(manifest.get("training_path", "")).resolve() != Path(data_path).resolve():
        raise ValueError("--data_path 与 data manifest 中的 training_path 不一致")
    if manifest.get("training_sha256") != data_sha256:
        raise ValueError("训练数据 SHA-256 与 data manifest 不一致")
    for path_key, hash_key in (("source_path", "source_sha256"), ("heldout_path", "heldout_sha256")):
        artifact_path = Path(manifest[path_key])
        if not artifact_path.is_file():
            raise FileNotFoundError(f"原始/held-out 文件缺失：{artifact_path}")
        print(f"正在复核 {path_key} SHA-256：{artifact_path}")
        if _sha256_path(artifact_path) != manifest[hash_key]:
            raise ValueError(f"{path_key} SHA-256 与数据 manifest 不一致")
    return manifest


def _run_manifest(args, data_manifest: dict, data_sha256: str, mode: str) -> dict:
    artifact_paths = {
        "training_data": Path(args.data_path),
        "tokenizer": Path(args.tokenizer_path),
        "init_checkpoint": Path(args.init_checkpoint),
        "reward_model": Path(args.reward_model_path),
        "data_manifest": Path(args.data_manifest),
    }
    artifacts = {}
    for name, path in artifact_paths.items():
        print(f"正在计算 {name} SHA-256：{path}")
        artifact_hash = data_sha256 if name == "training_data" else _sha256_path(path)
        artifacts[name] = {"path": str(path.resolve()), "sha256": artifact_hash}
    artifacts["source_data"] = {
        "path": data_manifest["source_path"],
        "sha256": data_manifest["source_sha256"],
    }
    artifacts["heldout_prompts"] = {
        "path": data_manifest["heldout_path"],
        "sha256": data_manifest["heldout_sha256"],
    }
    revision, dirty = _git_state()
    run_config = {
        "mode": mode,
        "dtype": args.dtype,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "max_seq_len": args.max_seq_len,
        "max_gen_len": args.max_gen_len,
        "batch_size_prompts": args.batch_size,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "epsilon_high": args.epsilon_high,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "save_interval": args.save_interval,
        "seed": args.seed,
        "pilot_prompts": args.pilot_prompts if mode == "rollout_only" else None,
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "MiniMind CISPO",
        "code_revision": revision,
        "working_tree_dirty": dirty,
        "revisions": {
            "rlaif_dataset": data_manifest["source_revision"],
            "reward_model": args.reward_model_revision,
            "tokenizer": args.tokenizer_revision,
            "init_checkpoint": args.init_revision,
        },
        "licenses": {"rlaif_dataset": data_manifest["source_license"]},
        "artifacts": artifacts,
        "data_preparation": data_manifest,
        "run_config": run_config,
    }


def _save_or_validate_manifest(output_dir: str, manifest: dict) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "code_revision",
            "working_tree_dirty",
            "artifacts",
            "revisions",
            "licenses",
            "run_config",
        ):
            if previous.get(key) != manifest.get(key):
                raise ValueError(
                    f"已有 run_manifest.json 的 {key} 与本次运行不一致；请使用新的输出目录"
                )
        return
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _build_config(args, tokenizer) -> MiniMindConfig:
    if tokenizer.eos_token_id is None:
        raise ValueError("MiniMind tokenizer 必须设置 eos_token_id")
    return MiniMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        # completion 也要经过 RoPE，位置表需覆盖 prompt + generated response。
        max_position_embeddings=args.max_seq_len + args.max_gen_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )


def _load_models(args, tokenizer, config, device, include_reference: bool):
    policy = MiniMindForCausalLM(config)
    load_model_weights(policy, args.init_checkpoint, device)
    policy.to(device)
    reference = None
    if include_reference:
        reference = MiniMindForCausalLM(config)
        load_model_weights(reference, args.init_checkpoint, device)
        reference.to(device).eval()
        reference.requires_grad_(False)
    return policy, reference


def _make_dataset(args, tokenizer) -> RLAIFDataset:
    return RLAIFDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        thinking_ratio=args.thinking_ratio,
    )


def _make_rollout_engine(args, policy, tokenizer, device, dtype) -> TorchRolloutEngine:
    return TorchRolloutEngine(
        policy,
        tokenizer,
        device,
        dtype,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
    )


def _load_reward_model(args, device, dtype) -> MiniMindRewardModel:
    print(f"正在加载 reward model：{args.reward_model_path}")
    return MiniMindRewardModel(args.reward_model_path, device, dtype)


def _reward_tensor(reward_breakdowns, device) -> torch.Tensor:
    return torch.tensor([item.total for item in reward_breakdowns], dtype=torch.float32, device=device)


def _mean_reward_components(reward_breakdowns) -> dict[str, float]:
    if not reward_breakdowns:
        return {}
    return {
        "reward_model": sum(item.reward_model for item in reward_breakdowns) / len(reward_breakdowns),
        "length": sum(item.length for item in reward_breakdowns) / len(reward_breakdowns),
        "thinking": sum(item.thinking for item in reward_breakdowns) / len(reward_breakdowns),
        "repetition_penalty": sum(item.repetition_penalty for item in reward_breakdowns) / len(reward_breakdowns),
    }


def _pilot(args, dataset, rollout_engine, reward_model, device) -> None:
    prompt_count = min(args.pilot_prompts, len(dataset))
    if prompt_count < args.pilot_prompts:
        print(f"注意：数据只有 {prompt_count} 个 prompt，少于计划的 {args.pilot_prompts} 个。")
    if prompt_count == 0:
        raise ValueError("RLAIF 数据集中没有 prompt")

    subset = Subset(dataset, range(prompt_count))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_identity_collate,
    )
    output_dir = Path(args.pilot_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "rollouts.jsonl"
    groups = []
    all_breakdowns = []
    total_completions = 0
    empty_responses = 0
    response_token_lengths = []
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with samples_path.open("w", encoding="utf-8") as samples_file:
        for batch_index, rows in enumerate(loader, start=1):
            rollout = rollout_engine.rollout(rows, args.num_generations)
            breakdowns = score_completions(
                [row["messages"] for row in rows],
                rollout.responses,
                reward_model,
                args.num_generations,
            )
            rewards = _reward_tensor(breakdowns, device)
            grouped = rewards.view(len(rows), args.num_generations)
            group_stds = grouped.std(dim=1, unbiased=False)
            all_breakdowns.extend(breakdowns)
            total_completions += len(rollout.responses)
            empty_responses += sum(not response.strip() for response in rollout.responses)
            response_token_lengths.extend(rollout.completion_mask.sum(dim=1).tolist())

            for row_index, row in enumerate(rows):
                start = row_index * args.num_generations
                group = breakdowns[start : start + args.num_generations]
                groups.append(
                    {
                        "prompt": row["prompt"],
                        "group_reward_std": float(group_stds[row_index].item()),
                        "responses": [
                            {"text": rollout.responses[start + offset], **item.to_dict()}
                            for offset, item in enumerate(group)
                        ],
                    }
                )
            for row in groups[-len(rows) :]:
                samples_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"stage=grpo-pilot prompts={min(batch_index * args.batch_size, prompt_count)}/{prompt_count} "
                f"completions={total_completions} reward_mean={rewards.mean().item():.3f} "
                f"group_std_mean={group_stds.mean().item():.3f}"
            )

    elapsed = time.perf_counter() - started_at
    final_rewards = _reward_tensor(all_breakdowns, device)
    grouped_rewards = final_rewards.view(prompt_count, args.num_generations)
    group_stds = grouped_rewards.std(dim=1, unbiased=False)
    zero_variance_groups = int((group_stds <= 1e-8).sum().item())
    summary = {
        "mode": "rollout_only",
        "prompts": prompt_count,
        "generations_per_prompt": args.num_generations,
        "completions": total_completions,
        "empty_response_count": empty_responses,
        "mean_reward": float(final_rewards.mean().item()),
        "mean_group_reward_std": float(group_stds.mean().item()),
        "zero_variance_groups": zero_variance_groups,
        "mean_response_tokens": sum(response_token_lengths) / max(len(response_token_lengths), 1),
        "mean_reward_components": _mean_reward_components(all_breakdowns),
        "elapsed_seconds": elapsed,
        "completions_per_second": total_completions / elapsed if elapsed > 0 else 0.0,
        "peak_cuda_memory_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
        ),
        "samples_path": str(samples_path.resolve()),
        "decision": "review pilot outputs; thresholds are intentionally not guessed",
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"GRPO rollout 预检结果：{summary_path}")


def _training_labels(rollout) -> torch.Tensor:
    labels = torch.full_like(rollout.input_ids, -100)
    lengths = rollout.completion_mask.sum(dim=1).tolist()
    for index, (prompt_length, response_length) in enumerate(
        zip(rollout.prompt_lengths.tolist(), lengths)
    ):
        end = prompt_length + int(response_length)
        labels[index, prompt_length:end] = rollout.input_ids[index, prompt_length:end]
    return labels


def _write_scalars(
    writer,
    loss,
    current_lr,
    grad_norm,
    snapshot,
    rewards,
    breakdowns,
    num_generations,
    kl_mean,
    step,
):
    if writer is None:
        return
    components = _mean_reward_components(breakdowns)
    group_std = rewards.view(-1, num_generations).std(dim=1, unbiased=False).mean()
    writer.add_scalar("train/cispo_loss", float(loss.item()), step)
    writer.add_scalar("train/learning_rate", current_lr, step)
    writer.add_scalar("train/grad_norm", float(grad_norm.item()), step)
    writer.add_scalar("reward/total_mean", float(rewards.mean().item()), step)
    writer.add_scalar("reward/group_std_mean", float(group_std.item()), step)
    writer.add_scalar("reward/model_mean", components["reward_model"], step)
    writer.add_scalar("reward/length_mean", components["length"], step)
    writer.add_scalar("reward/thinking_mean", components["thinking"], step)
    writer.add_scalar("reward/repetition_penalty_mean", components["repetition_penalty"], step)
    writer.add_scalar("regularization/kl_mean", kl_mean, step)
    writer.add_scalar("throughput/raw_tokens_per_second", snapshot.raw_tokens_per_second, step)
    writer.add_scalar("throughput/valid_tokens_per_second", snapshot.valid_tokens_per_second, step)
    writer.add_scalar("throughput/target_tokens_per_second", snapshot.target_tokens_per_second, step)
    if math.isfinite(snapshot.eta_seconds):
        writer.add_scalar("progress/eta_seconds", snapshot.eta_seconds, step)


def _train(args, dataset, policy, reference, rollout_engine, reward_model, config, device, dtype) -> None:
    sampler = EpochRandomSampler(dataset, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_identity_collate,
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
        print(f"已恢复 GRPO checkpoint：epoch={start_epoch}, step={global_step}, batch={resume_batch}")

    steps_per_epoch = len(loader)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    if total_steps <= 0:
        raise ValueError("训练没有任何 optimizer step；请检查数据和 batch_size")
    if global_step >= total_steps:
        print(f"checkpoint 已达到目标 step={total_steps}，没有剩余训练步骤。")
        if writer is not None:
            writer.close()
        return

    metrics = TrainingMetrics(start_step=global_step)
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    current_epoch = start_epoch
    last_update_batch = resume_batch
    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            sampler.set_epoch(epoch)
            skip_batches = resume_batch if epoch == start_epoch else 0
            resume_batch = 0
            epoch_loader = _build_resumed_dataloader(loader, skip_batches)
            for local_index, rows in enumerate(epoch_loader, start=1):
                batch_index = skip_batches + local_index
                rollout = rollout_engine.rollout(rows, args.num_generations)
                reward_breakdowns = score_completions(
                    [row["messages"] for row in rows],
                    rollout.responses,
                    reward_model,
                    args.num_generations,
                )
                rewards = _reward_tensor(reward_breakdowns, device)
                advantages = group_relative_advantages(rewards, args.num_generations)
                labels = _training_labels(rollout)
                metrics.record_batch(rollout.input_ids, rollout.attention_mask, labels)

                with autocast_context(device, dtype):
                    policy_logps = compute_completion_logps(
                        policy,
                        rollout.input_ids,
                        rollout.attention_mask,
                        rollout.prompt_lengths,
                        rollout.completion_mask,
                    )
                    with torch.no_grad():
                        reference_logps = compute_completion_logps(
                            reference,
                            rollout.input_ids,
                            rollout.attention_mask,
                            rollout.prompt_lengths,
                            rollout.completion_mask,
                        )
                    loss = cispo_loss(
                        policy_logps,
                        rollout.old_logps,
                        reference_logps,
                        advantages,
                        rollout.completion_mask,
                        beta=args.beta,
                        epsilon_high=args.epsilon_high,
                    )

                if not torch.isfinite(loss.detach()).item():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"GRPO CISPO loss 非有限：epoch={epoch + 1}, batch={batch_index}, step={global_step}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(policy.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm).item():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"GRPO 梯度范数非有限：epoch={epoch + 1}, batch={batch_index}, step={global_step}"
                    )

                next_step = global_step + 1
                current_lr = cosine_learning_rate(
                    next_step,
                    total_steps,
                    args.learning_rate,
                    min_lr_ratio=args.min_lr_ratio,
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step = next_step
                last_update_batch = batch_index
                snapshot = metrics.record_update(global_step, total_steps)

                kl_div = reference_logps.detach().float() - policy_logps.detach().float()
                kl_values = torch.exp(kl_div) - kl_div - 1.0
                kl_mean = float(
                    (kl_values * rollout.completion_mask).sum().item()
                    / rollout.completion_mask.sum().clamp(min=1).item()
                )
                grouped_rewards = rewards.view(-1, args.num_generations)
                group_stds = grouped_rewards.std(dim=1, unbiased=False)
                components = _mean_reward_components(reward_breakdowns)

                if global_step % args.log_interval == 0 or batch_index == len(loader):
                    grad_value = float(grad_norm.item())
                    zero_variance = int((group_stds <= 1e-8).sum().item())
                    print(
                        f"stage=grpo-cispo epoch={epoch + 1}/{args.epochs} "
                        f"step={global_step}/{total_steps} loss={loss.item():.4f} lr={current_lr:.2e} "
                        f"reward={rewards.mean().item():.3f} group_std={group_stds.mean().item():.3f} "
                        f"zero_var={zero_variance}/{len(group_stds)} kl={kl_mean:.4f} "
                        f"rm/len/think/rep={components['reward_model']:.3f}/"
                        f"{components['length']:.3f}/{components['thinking']:.3f}/"
                        f"{components['repetition_penalty']:.3f} "
                        f"tok/s(raw/valid/target)="
                        f"{snapshot.raw_tokens_per_second:.0f}/"
                        f"{snapshot.valid_tokens_per_second:.0f}/"
                        f"{snapshot.target_tokens_per_second:.0f} "
                        f"grad={grad_value:.3f} eta={format_duration(snapshot.eta_seconds)}"
                    )
                    _write_scalars(
                        writer, loss, current_lr, grad_norm, snapshot, rewards,
                        reward_breakdowns, args.num_generations, kl_mean, global_step,
                    )

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    checkpoint = save_checkpoint(
                        policy, optimizer, scaler, config, args.output_dir,
                        "grpo", epoch, global_step, batch_index,
                    )
                    print(f"已保存 GRPO checkpoint：{checkpoint}")
                if args.max_steps > 0 and global_step >= args.max_steps:
                    save_checkpoint(
                        policy, optimizer, scaler, config, args.output_dir,
                        "grpo", epoch, global_step, batch_index,
                    )
                    return

            save_checkpoint(
                policy, optimizer, scaler, config, args.output_dir,
                "grpo", epoch + 1, global_step, 0,
            )
            print(f"epoch {epoch + 1} 完成；checkpoint 已更新。")
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        checkpoint = save_checkpoint(
            policy, optimizer, scaler, config, args.output_dir,
            "grpo", current_epoch, global_step, last_update_batch,
        )
        print(f"\n收到 Ctrl+C，已保存最近完成的 optimizer step：{checkpoint}")
    finally:
        if writer is not None:
            writer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMind 单卡 GRPO/CISPO 训练与 rollout 预检")
    parser.add_argument("--data_path", required=True, help="去重后的 RLAIF 训练 JSONL")
    parser.add_argument("--data_manifest", required=True, help="prepare_grpo_data.py 输出的 manifest")
    parser.add_argument("--tokenizer_path", required=True, help="本地 MiniMind tokenizer 目录")
    parser.add_argument("--tokenizer_revision", required=True, help="tokenizer 仓库 commit 或本地版本标识")
    parser.add_argument("--init_checkpoint", required=True, help="本地通用 full_sft_768 权重，只读")
    parser.add_argument("--init_revision", required=True, help="起始 checkpoint 版本或来源标识")
    parser.add_argument("--reward_model_path", required=True, help="本地 reward model 目录")
    parser.add_argument("--reward_model_revision", required=True, help="reward model 仓库 commit 或本地版本标识")
    parser.add_argument("--resume_checkpoint", default="", help="从 grpo_last.pt 恢复完整训练状态")
    parser.add_argument("--output_dir", default="./out/checkpoints/grpo_cispo")
    parser.add_argument("--tensorboard_dir", default="./out/runs/grpo_cispo")
    parser.add_argument("--pilot_output_dir", default="./out/evaluations/grpo_pilot")
    parser.add_argument("--rollout_only", action="store_true", help="只生成/打分并输出预检报告，不训练")
    parser.add_argument("--pilot_prompts", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=768, help="prompt token 上限")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="每条 completion 的最大 token 数")
    parser.add_argument("--batch_size", type=int, default=2, help="每批 prompt 数；实际 completion 数=batch_size × generations")
    parser.add_argument("--num_generations", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--thinking_ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=10, help="每多少 optimizer steps 保存一次")
    parser.add_argument("--max_steps", type=int, default=0, help="0 表示完整 epoch；正数表示全局目标 step")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.num_generations < 2:
        raise ValueError("batch_size 必须大于 0，num_generations 至少为 2")
    if args.pilot_prompts < 1 or args.save_interval < 1 or args.log_interval < 1:
        raise ValueError("pilot_prompts、save_interval 和 log_interval 必须大于 0")

    setup_seed(args.seed)
    device = resolve_device(args.device)
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("GRPO 请求了 CUDA 但当前环境没有 CUDA；为避免误在 CPU 长时间运行，已停止。")
    dtype = resolve_dtype(args.dtype, device)

    print("正在计算训练制品 SHA-256 并验证 data manifest...")
    data_sha256 = _sha256_path(Path(args.data_path))
    data_manifest = _load_data_manifest(args.data_manifest, args.data_path, data_sha256)
    manifest = _run_manifest(args, data_manifest, data_sha256, "rollout_only" if args.rollout_only else "train")

    print(f"正在加载 tokenizer：{args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    config = _build_config(args, tokenizer)
    policy, reference = _load_models(
        args, tokenizer, config, device, include_reference=not args.rollout_only
    )
    dataset = _make_dataset(args, tokenizer)
    print(
        f"GRPO 数据样本={len(dataset)}，每批 prompt={args.batch_size}，"
        f"每题回答={args.num_generations}，batch completions={args.batch_size * args.num_generations}"
    )

    if args.rollout_only:
        _save_or_validate_manifest(args.pilot_output_dir, manifest)
        reward_model = _load_reward_model(args, device, dtype)
        rollout_engine = _make_rollout_engine(args, policy, tokenizer, device, dtype)
        _pilot(args, dataset, rollout_engine, reward_model, device)
        return

    _save_or_validate_manifest(args.output_dir, manifest)
    reward_model = _load_reward_model(args, device, dtype)
    rollout_engine = _make_rollout_engine(args, policy, tokenizer, device, dtype)
    _train(args, dataset, policy, reference, rollout_engine, reward_model, config, device, dtype)


if __name__ == "__main__":
    main()

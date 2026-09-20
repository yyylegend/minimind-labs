"""训练公共逻辑：混合精度、梯度累积、裁剪和 checkpoint。"""

from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torch.nn.utils import clip_grad_norm_

from trainer.metrics import TrainingMetrics


class EpochRandomSampler(Sampler[int]):
    """按 epoch 生成可复现的随机顺序，便于 checkpoint 后跳过已处理 batch。"""

    def __init__(self, data_source, seed: int = 42) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


def _build_resumed_dataloader(dataloader, skip_batches: int):
    """直接从未处理的索引构造 DataLoader，避免恢复时重新读取旧 batch。"""
    if skip_batches <= 0:
        return dataloader
    if dataloader.batch_size is None:
        raise ValueError("恢复训练要求 DataLoader 使用 batch_size，而不是自定义 batch_sampler")

    # sampler 的顺序由 EpochRandomSampler.set_epoch() 决定，先生成索引不会触发 Dataset.__getitem__。
    indices = list(iter(dataloader.sampler))
    start_index = skip_batches * dataloader.batch_size
    remaining_dataset = Subset(dataloader.dataset, indices[start_index:])
    return DataLoader(
        remaining_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        persistent_workers=dataloader.persistent_workers if dataloader.num_workers else False,
    )


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--"
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s"


def cosine_learning_rate(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    """返回带线性 warmup 和 cosine decay 的当前学习率。"""
    if total_steps <= 0:
        return base_lr
    if base_lr < 0:
        raise ValueError("base_lr 不能小于 0")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio 必须在 [0, 1] 范围内")

    step = max(0, min(step, total_steps))
    warmup_steps = max(0, min(warmup_steps, total_steps))
    if warmup_steps and step < warmup_steps:
        return base_lr * step / warmup_steps

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _nonfinite_names(tensors) -> list[str]:
    return [
        name
        for name, value in tensors
        if torch.is_tensor(value)
        and value.is_floating_point()
        and not torch.isfinite(value).all().item()
    ]


def _assert_finite_model(model) -> None:
    bad_names = _nonfinite_names(model.state_dict().items())
    if bad_names:
        raise FloatingPointError(f"模型包含非有限权重，拒绝继续训练或保存：{bad_names[:5]}")


def _assert_finite_optimizer(optimizer) -> None:
    bad_names = []
    for parameter_id, state in optimizer.state.items():
        bad_names.extend(
            (f"{parameter_id}.{name}" for name in _nonfinite_names(state.items()))
        )
    if bad_names:
        raise FloatingPointError(f"优化器状态包含非有限值：{bad_names[:5]}")


def _atomic_torch_save(value, path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(value, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _recover_from_fp16_overflow(scaler, optimizer) -> tuple[float, float]:
    """让 GradScaler 跳过溢出的更新并降低 scale，而不是让训练直接退出。"""
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return scale_before, float(scaler.get_scale())


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("未检测到 CUDA，自动切换到 CPU。")
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            raise ValueError("当前 GPU 不适合 BF16，请改用 --dtype float16")
        return torch.bfloat16
    raise ValueError("dtype 只支持 float16 或 bfloat16")


def autocast_context(device: torch.device, dtype: torch.dtype):
    """GPU 使用自动混合精度，CPU 直接使用普通 float32。"""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def save_checkpoint(
    model,
    optimizer,
    scaler,
    config,
    output_dir: str,
    stage: str,
    epoch: int,
    step: int,
    batch_index: int = 0,
) -> Path:
    """同时保存模型权重和恢复训练所需的优化器状态。"""
    _assert_finite_model(model)
    _assert_finite_optimizer(optimizer)
    scaler_state = scaler.state_dict()
    scaler_scale = scaler_state.get("scale")
    if scaler_scale is not None and (not math.isfinite(scaler_scale) or scaler_scale <= 0):
        raise FloatingPointError(f"GradScaler 状态异常：scale={scaler_scale}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / f"{stage}_last.pt"
    state = {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler_state,
        "config": asdict(config),
        "epoch": epoch,
        "step": step,
        "batch_index": batch_index,
    }
    _atomic_torch_save(state, checkpoint_path)

    # 同时保存官方 MiniMind 风格的纯权重文件：官方训练器只需要这个 state_dict。
    official_stage = "full_sft" if stage == "sft" else stage
    official_path = output_path / f"{official_stage}_{config.hidden_size}.pth"
    official_state = {
        name: value.detach().half().cpu()
        for name, value in model.state_dict().items()
    }
    _atomic_torch_save(official_state, official_path)
    return checkpoint_path


def load_model_weights(model, checkpoint_path: str, device: torch.device) -> None:
    """加载已有模型权重，可用于从 Pretrain 开始 SFT。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    _assert_finite_model(model)


def load_training_checkpoint(model, optimizer, scaler, checkpoint_path: str, device: torch.device):
    """恢复完整训练状态，而不只是恢复模型权重。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    _assert_finite_model(model)
    _assert_finite_optimizer(optimizer)
    return (
        int(checkpoint.get("epoch", 0)),
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("batch_index", 0)),
    )


def train_model(
    model,
    dataloader,
    config,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: str,
    stage: str,
    epochs: int,
    learning_rate: float,
    accumulation_steps: int,
    grad_clip: float,
    save_interval: int,
    max_steps: int = 0,
    log_interval: int = 10,
    resume_checkpoint: str = "",
    tensorboard_dir: str = "",
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> None:
    """MiniMind 风格的单卡训练循环。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    writer = None
    if tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(tensorboard_dir)
        except ImportError:
            print("未安装 tensorboard，跳过网页监控；训练本身继续运行。")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_epoch, global_step, resume_batch = 0, 0, 0
    if resume_checkpoint:
        start_epoch, global_step, resume_batch = load_training_checkpoint(
            model, optimizer, scaler, resume_checkpoint, device
        )
        print(f"已恢复 checkpoint：epoch={start_epoch}, step={global_step}, batch={resume_batch}")

    batches_per_epoch = len(dataloader)
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * epochs
    metrics = TrainingMetrics(start_step=global_step)
    stop_training = False
    current_epoch = start_epoch
    last_update_batch = resume_batch
    consecutive_fp16_overflows = 0

    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            skip_batches = resume_batch if epoch == start_epoch else 0
            resume_batch = 0
            epoch_dataloader = dataloader
            if skip_batches:
                print(f"恢复训练：直接跳过已完成的 {skip_batches} 个 batch。")
                epoch_dataloader = _build_resumed_dataloader(dataloader, skip_batches)

            for local_batch_index, batch in enumerate(epoch_dataloader, start=1):
                batch_index = skip_batches + local_batch_index

                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]
                metrics.record_batch(input_ids, attention_mask, labels)
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)

                with autocast_context(device, dtype):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss + outputs.aux_loss
                    scaled_loss = loss / accumulation_steps

                if not torch.isfinite(loss.detach()).item():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"loss 变为非有限值：stage={stage}, epoch={epoch + 1}, "
                        f"batch={batch_index}, step={global_step}。请降低学习率或检查数据。"
                    )

                scaler.scale(scaled_loss).backward()
                should_update = batch_index % accumulation_steps == 0 or batch_index == batches_per_epoch
                if not should_update:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(grad_norm).item():
                    if use_scaler:
                        scale_before, scale_after = _recover_from_fp16_overflow(scaler, optimizer)
                        metrics.record_fp16_overflow()
                        consecutive_fp16_overflows += 1
                        print(
                            f"检测到 FP16 梯度溢出，已跳过 batch={batch_index} 的更新，"
                            f"GradScaler scale: {scale_before:.0f} -> {scale_after:.0f}。"
                        )
                        if consecutive_fp16_overflows >= 8:
                            raise FloatingPointError(
                                "连续 8 次 FP16 梯度溢出，训练未能自行恢复。"
                            )
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"梯度范数变为非有限值：stage={stage}, epoch={epoch + 1}, "
                        f"batch={batch_index}, step={global_step}。请降低学习率或检查精度。"
                    )

                next_step = global_step + 1
                current_lr = cosine_learning_rate(
                    next_step,
                    total_steps,
                    learning_rate,
                    warmup_steps=warmup_steps,
                    min_lr_ratio=min_lr_ratio,
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = current_lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                consecutive_fp16_overflows = 0
                global_step += 1
                last_update_batch = batch_index
                metrics_snapshot = metrics.record_update(global_step, total_steps)
                if global_step % log_interval == 0 or batch_index == batches_per_epoch:
                    grad_norm_value = float(grad_norm.item())
                    scaler_scale = float(scaler.get_scale())
                    if writer is not None:
                        writer.add_scalar("train/loss", loss.item(), global_step)
                        writer.add_scalar("train/learning_rate", current_lr, global_step)
                        writer.add_scalar(
                            "train/tokens_per_second",
                            metrics_snapshot.raw_tokens_per_second,
                            global_step,
                        )
                        writer.add_scalar(
                            "throughput/raw_tokens_per_second",
                            metrics_snapshot.raw_tokens_per_second,
                            global_step,
                        )
                        writer.add_scalar(
                            "throughput/valid_tokens_per_second",
                            metrics_snapshot.valid_tokens_per_second,
                            global_step,
                        )
                        writer.add_scalar(
                            "throughput/target_tokens_per_second",
                            metrics_snapshot.target_tokens_per_second,
                            global_step,
                        )
                        writer.add_scalar("data/padding_ratio", metrics_snapshot.padding_ratio, global_step)
                        writer.add_scalar("data/target_token_ratio", metrics_snapshot.target_token_ratio, global_step)
                        writer.add_scalar("train/grad_norm", grad_norm_value, global_step)
                        writer.add_scalar("stability/grad_scaler_scale", scaler_scale, global_step)
                        writer.add_scalar(
                            "stability/fp16_overflow_total",
                            metrics_snapshot.fp16_overflow_total,
                            global_step,
                        )
                        if math.isfinite(metrics_snapshot.eta_seconds):
                            writer.add_scalar("progress/eta_seconds", metrics_snapshot.eta_seconds, global_step)
                    if device.type == "cuda":
                            writer.add_scalar(
                                "system/gpu_memory_allocated_gb",
                                torch.cuda.memory_allocated(device) / 1024**3,
                                global_step,
                            )
                    print(
                        f"stage={stage} epoch={epoch + 1}/{epochs} "
                        f"step={global_step}/{total_steps} loss={loss.item():.4f} "
                        f"tok/s(raw/valid/target)="
                        f"{metrics_snapshot.raw_tokens_per_second:.0f}/"
                        f"{metrics_snapshot.valid_tokens_per_second:.0f}/"
                        f"{metrics_snapshot.target_tokens_per_second:.0f} "
                        f"pad={metrics_snapshot.padding_ratio:.1%} grad={grad_norm_value:.3f} "
                        f"scale={scaler_scale:.0f} ovf={metrics_snapshot.fp16_overflow_total} "
                        f"eta={format_duration(metrics_snapshot.eta_seconds)}"
                    )
                if save_interval > 0 and global_step % save_interval == 0:
                    save_checkpoint(
                        model, optimizer, scaler, config, output_dir, stage,
                        epoch, global_step, batch_index,
                    )
                if max_steps > 0 and global_step >= max_steps:
                    stop_training = True
                    break

            if stop_training:
                save_checkpoint(
                    model, optimizer, scaler, config, output_dir, stage,
                    epoch, global_step, last_update_batch,
                )
                break

            # epoch 已经完整结束，恢复时从下一个 epoch 开始。
            save_checkpoint(model, optimizer, scaler, config, output_dir, stage, epoch + 1, global_step, 0)
    except KeyboardInterrupt:
        # Ctrl+C 只保存最近一个完成 optimizer.step 的安全位置。
        optimizer.zero_grad(set_to_none=True)
        checkpoint_path = save_checkpoint(
            model, optimizer, scaler, config, output_dir, stage,
            current_epoch, global_step, last_update_batch,
        )
        print(f"\n收到 Ctrl+C，已保存安全 checkpoint：{checkpoint_path}")
    finally:
        if writer is not None:
            writer.close()

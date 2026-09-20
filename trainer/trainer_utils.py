"""训练公共逻辑：混合精度、梯度累积、裁剪和 checkpoint。"""

from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import Sampler
from torch.nn.utils import clip_grad_norm_


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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / f"{stage}_last.pt"
    state = {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "step": step,
        "batch_index": batch_index,
    }
    torch.save(state, checkpoint_path)

    # 同时保存官方 MiniMind 风格的纯权重文件：官方训练器只需要这个 state_dict。
    official_stage = "full_sft" if stage == "sft" else stage
    official_path = output_path / f"{official_stage}_{config.hidden_size}.pth"
    official_state = {
        name: value.detach().half().cpu()
        for name, value in model.state_dict().items()
    }
    torch.save(official_state, official_path)
    return checkpoint_path


def load_model_weights(model, checkpoint_path: str, device: torch.device) -> None:
    """加载已有模型权重，可用于从 Pretrain 开始 SFT。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)


def load_training_checkpoint(model, optimizer, scaler, checkpoint_path: str, device: torch.device):
    """恢复完整训练状态，而不只是恢复模型权重。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
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
) -> None:
    """MiniMind 风格的单卡训练循环。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
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

    steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * epochs
    started_at = time.perf_counter()
    tokens_seen = 0
    stop_training = False
    current_epoch = start_epoch
    last_update_batch = resume_batch

    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            skip_batches = resume_batch if epoch == start_epoch else 0
            resume_batch = 0

            for batch_index, batch in enumerate(dataloader, start=1):
                if batch_index <= skip_batches:
                    continue

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                tokens_seen += input_ids.numel()

                with autocast_context(device, dtype):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss + outputs.aux_loss
                    scaled_loss = loss / accumulation_steps

                scaler.scale(scaled_loss).backward()
                should_update = batch_index % accumulation_steps == 0 or batch_index == len(dataloader)
                if not should_update:
                    continue

                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                last_update_batch = batch_index

                elapsed = time.perf_counter() - started_at
                steps_per_second = global_step / elapsed if elapsed > 0 else 0.0
                remaining = (total_steps - global_step) / steps_per_second if steps_per_second else float("inf")
                tokens_per_second = tokens_seen / elapsed if elapsed > 0 else 0.0
                if global_step % log_interval == 0 or batch_index == len(dataloader):
                    if writer is not None:
                        writer.add_scalar("train/loss", loss.item(), global_step)
                        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                        writer.add_scalar("train/tokens_per_second", tokens_per_second, global_step)
                        if math.isfinite(remaining):
                            writer.add_scalar("train/eta_seconds", remaining, global_step)
                        if device.type == "cuda":
                            writer.add_scalar(
                                "system/gpu_memory_allocated_gb",
                                torch.cuda.memory_allocated(device) / 1024**3,
                                global_step,
                            )
                    print(
                        f"stage={stage} epoch={epoch + 1}/{epochs} "
                        f"step={global_step}/{total_steps} loss={loss.item():.4f} "
                        f"tok/s={tokens_per_second:.0f} eta={format_duration(remaining)}"
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

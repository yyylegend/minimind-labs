"""训练监控：区分计算负载、有效数据和真正参与 loss 的目标 token。"""

from collections import deque
from dataclasses import dataclass
import math
import time

import torch


@dataclass(frozen=True)
class TrainingMetricsSnapshot:
    raw_tokens_per_second: float
    valid_tokens_per_second: float
    target_tokens_per_second: float
    updates_per_second: float
    padding_ratio: float
    target_token_ratio: float
    eta_seconds: float
    fp16_overflow_total: int


@dataclass(frozen=True)
class _MetricPoint:
    at: float
    step: int
    raw_tokens: int
    valid_tokens: int
    target_tokens: int


class TrainingMetrics:
    """用最近 N 次成功更新生成可恢复训练也可信的吞吐与 ETA 指标。"""

    def __init__(
        self,
        start_step: int = 0,
        window_updates: int = 100,
        start_time: float | None = None,
    ) -> None:
        if window_updates < 1:
            raise ValueError("window_updates 必须至少为 1")
        started_at = time.perf_counter() if start_time is None else start_time
        self._window_updates = window_updates
        self._raw_tokens = 0
        self._valid_tokens = 0
        self._target_tokens = 0
        self._points = deque(
            [_MetricPoint(started_at, start_step, 0, 0, 0)],
            maxlen=window_updates + 1,
        )
        self.fp16_overflow_total = 0

    def record_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> None:
        """在 batch 仍位于 CPU 时记录 token 数，避免触发 GPU 同步。"""
        self._raw_tokens += input_ids.numel()
        self._valid_tokens += int(attention_mask.sum().item())
        if labels is not None and labels.size(-1) > 1:
            self._target_tokens += int(labels[..., 1:].ne(-100).sum().item())

    def record_fp16_overflow(self) -> None:
        self.fp16_overflow_total += 1

    def record_update(
        self,
        step: int,
        total_steps: int,
        now: float | None = None,
    ) -> TrainingMetricsSnapshot:
        """记录一次成功 optimizer 更新，并返回最近窗口的指标快照。"""
        current_time = time.perf_counter() if now is None else now
        point = _MetricPoint(
            current_time,
            step,
            self._raw_tokens,
            self._valid_tokens,
            self._target_tokens,
        )
        self._points.append(point)
        baseline = self._points[0]
        duration = max(point.at - baseline.at, 0.0)
        if duration == 0:
            raw_tokens_per_second = 0.0
            valid_tokens_per_second = 0.0
            target_tokens_per_second = 0.0
            updates_per_second = 0.0
        else:
            raw_tokens_per_second = (point.raw_tokens - baseline.raw_tokens) / duration
            valid_tokens_per_second = (point.valid_tokens - baseline.valid_tokens) / duration
            target_tokens_per_second = (point.target_tokens - baseline.target_tokens) / duration
            updates_per_second = (point.step - baseline.step) / duration

        raw_tokens = point.raw_tokens - baseline.raw_tokens
        valid_tokens = point.valid_tokens - baseline.valid_tokens
        target_tokens = point.target_tokens - baseline.target_tokens
        padding_ratio = 1.0 - valid_tokens / raw_tokens if raw_tokens else 0.0
        target_token_ratio = target_tokens / valid_tokens if valid_tokens else 0.0
        eta_seconds = (
            max(total_steps - step, 0) / updates_per_second
            if updates_per_second > 0
            else math.inf
        )
        return TrainingMetricsSnapshot(
            raw_tokens_per_second=raw_tokens_per_second,
            valid_tokens_per_second=valid_tokens_per_second,
            target_tokens_per_second=target_tokens_per_second,
            updates_per_second=updates_per_second,
            padding_ratio=padding_ratio,
            target_token_ratio=target_token_ratio,
            eta_seconds=eta_seconds,
            fp16_overflow_total=self.fp16_overflow_total,
        )

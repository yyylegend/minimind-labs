"""DPO 偏好损失。"""

import torch
from torch.nn import functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """提高策略模型相对 reference 对 chosen 回复的偏好。"""
    logps = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if beta <= 0:
        raise ValueError("DPO beta 必须大于 0")
    if any(value.shape != policy_chosen_logps.shape for value in logps[1:]):
        raise ValueError("四组 DPO log-prob 必须具有相同形状")

    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps
    preference_logits = beta * (policy_log_ratio - reference_log_ratio)
    return -F.logsigmoid(preference_logits).mean()

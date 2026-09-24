"""GRPO 组内优势和 MiniMind 风格 CISPO 目标函数。"""

import torch


def group_relative_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """对每个 prompt 生成的一组奖励单独标准化。"""
    if rewards.ndim != 1:
        raise ValueError("rewards 必须是一维张量，顺序为每个 prompt 的连续回答组")
    if num_generations < 2:
        raise ValueError("num_generations 必须至少为 2，才能进行组内比较")
    if rewards.numel() == 0 or rewards.numel() % num_generations:
        raise ValueError("rewards 数量必须是 num_generations 的正整数倍")
    if epsilon <= 0:
        raise ValueError("epsilon 必须大于 0")
    if not torch.isfinite(rewards).all().item():
        raise ValueError("rewards 包含非有限值")

    # FP32 统计量避免低精度 reward 在均值/方差计算中丢失精度。
    grouped_rewards = rewards.float().view(-1, num_generations)
    group_mean = grouped_rewards.mean(dim=1, keepdim=True)
    group_std = grouped_rewards.std(dim=1, unbiased=False, keepdim=True)
    advantages = (grouped_rewards - group_mean) / (group_std + epsilon)
    return advantages.reshape(-1)


def cispo_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    beta: float = 0.1,
    epsilon_high: float = 5.0,
) -> torch.Tensor:
    """计算 CISPO token loss，并忽略 EOS 后的 padding token。

    log-prob 张量形状为 ``[回答数, completion长度]``；advantages 每条回答一个值。
    old_logps 是生成时记录的概率，reference_logps 来自冻结的 SFT 参考模型。
    """
    if current_logps.ndim != 2:
        raise ValueError("log-prob 张量必须为 [回答数, completion长度]")
    if old_logps.shape != current_logps.shape or reference_logps.shape != current_logps.shape:
        raise ValueError("current/old/reference log-prob 必须形状相同")
    if completion_mask.shape != current_logps.shape:
        raise ValueError("completion_mask 必须和 log-prob 形状相同")
    if advantages.shape != current_logps.shape[:1]:
        raise ValueError("advantages 必须每条回答对应一个值")
    if beta < 0:
        raise ValueError("beta 不能小于 0")
    if epsilon_high <= 0:
        raise ValueError("epsilon_high 必须大于 0")

    # 概率比和 KL 在 FP32 计算，减少 autocast 下 exp 的数值风险。
    current = current_logps.float()
    old = old_logps.float()
    reference = reference_logps.float()
    ratio = torch.exp(current - old)
    clipped_ratio = ratio.clamp(max=epsilon_high).detach()

    log_ratio_to_reference = reference - current
    per_token_kl = torch.exp(log_ratio_to_reference) - log_ratio_to_reference - 1.0
    per_token_loss = -(
        clipped_ratio * advantages.float().unsqueeze(1) * current
        - beta * per_token_kl
    )

    mask = completion_mask.to(dtype=per_token_loss.dtype)
    valid_tokens = mask.sum(dim=1).clamp(min=1.0)
    per_response_loss = (per_token_loss * mask).sum(dim=1) / valid_tokens
    return per_response_loss.mean()

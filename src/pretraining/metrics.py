from typing import Any, Dict

import numpy as np
import torch
from torch.optim import AdamW

from src.model.pretrain_model import PretrainableGNN
from src.pretraining.losses import UncertaintyWeighter
from src.pretraining.schedulers import GRLLambdaScheduler


def compute_training_metrics(
    per_domain_per_task_raw_losses: Dict[str, Dict[str, float]],
    raw_losses: Dict[str, torch.Tensor],
    total_weighted_loss: torch.Tensor,
    weighter: UncertaintyWeighter,
    grl_scheduler: GRLLambdaScheduler,
    optimizer_model: AdamW,
    optimizer_uncertainty: AdamW,
    model: PretrainableGNN,
    epoch: int,
    step: int,
) -> Dict[str, Any]:
    metrics = {}

    per_task_raw_losses = {}
    for domain in per_domain_per_task_raw_losses:
        for task in per_domain_per_task_raw_losses[domain]:
            raw_loss = per_domain_per_task_raw_losses[domain][task]
            metrics[f'train/loss/{domain}/{task}_raw'] = raw_loss
            if task not in per_task_raw_losses:
                per_task_raw_losses[task] = []
            per_task_raw_losses[task].append(raw_loss)
    for task in per_task_raw_losses:
        metrics[f'train/loss/{task}_raw'] = float(np.mean(per_task_raw_losses[task]))
    if 'domain_adv' in raw_losses:
        metrics['train/loss/domain_adv_raw'] = float(raw_losses['domain_adv'].detach().cpu())

    metrics['train/loss/total_weighted'] = total_weighted_loss.detach().cpu().item()

    sigmas = weighter.get_task_sigmas()
    for task_name, sigma_value in sigmas.items():
        metrics[f'train/uncertainty/{task_name}_sigma'] = sigma_value

    if 'domain_adv' in raw_losses:
        metrics['train/domain_adv/lambda'] = grl_scheduler()

    metrics['train/lr/model'] = optimizer_model.param_groups[0]['lr']
    metrics['train/lr/uncertainty'] = optimizer_uncertainty.param_groups[0]['lr']

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    metrics['train/gradients/model_grad_norm'] = total_norm

    metrics['train/epoch'] = epoch
    metrics['train/step'] = step

    return metrics


def compute_validation_metrics(
    per_domain_per_task_raw_losses: Dict[str, Dict[str, float]],
    total_balanced_loss: float,
    weighter: UncertaintyWeighter,
) -> Dict[str, Any]:
    metrics = {}

    task_sigmas = weighter.get_task_sigmas()

    per_task_raw_losses = {}
    for domain in per_domain_per_task_raw_losses:
        for task in per_domain_per_task_raw_losses[domain]:
            raw_loss = per_domain_per_task_raw_losses[domain][task]
            metrics[f'val/loss/{domain}/{task}_raw'] = raw_loss
            if task not in per_task_raw_losses:
                per_task_raw_losses[task] = []
            per_task_raw_losses[task].append(raw_loss)
    for task in per_task_raw_losses:
        metrics[f'val/loss/{task}_raw'] = float(np.mean(per_task_raw_losses[task]))

    metrics['val/loss/total_balanced'] = total_balanced_loss

    for task_name, sigma_value in task_sigmas.items():
        metrics[f'val/uncertainty/{task_name}_sigma'] = sigma_value

    return metrics

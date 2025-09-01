import time
from typing import Any, Dict

import torch
from torch.optim import AdamW

from src.models.pretrain_model import PretrainableGNN
from src.pretrain.losses import UncertaintyWeighter
from src.pretrain.schedulers import GRLLambdaScheduler


def _compute_base_metrics(
    per_domain_per_task_raw_losses: Dict[str, Dict[str, float]],
    per_task_raw_losses: Dict[str, torch.Tensor],
    per_domain_weighted_losses: Dict[str, float],
    total_weighted_loss: torch.Tensor,
    weighter: UncertaintyWeighter,
    grl_scheduler: GRLLambdaScheduler,
    epoch: int,
    prefix: str
) -> Dict[str, Any]:
    metrics = {}

    for domain in per_domain_per_task_raw_losses:
        for task in per_domain_per_task_raw_losses[domain]:
            metrics[f'{prefix}/loss/{domain}/{task}_raw'] = per_domain_per_task_raw_losses[domain][task]

    for task_name, raw_loss_tensor in per_task_raw_losses.items():
        metrics[f'{prefix}/loss/{task_name}_raw'] = float(raw_loss_tensor.detach().cpu())

    for domain_name, weighted_loss in per_domain_weighted_losses.items():
        metrics[f'{prefix}/loss/{domain_name}_weighted'] = weighted_loss

    metrics[f'{prefix}/loss/total_weighted'] = float(total_weighted_loss.detach().cpu())

    task_sigmas = weighter.get_task_sigmas()
    for task_name, sigma_value in task_sigmas.items():
        metrics[f'{prefix}/uncertainty/{task_name}_sigma'] = sigma_value

    if 'domain_adv' in per_task_raw_losses:
        metrics[f'{prefix}/domain_adv/lambda'] = grl_scheduler()

    metrics[f'{prefix}/progress/epoch'] = epoch

    return metrics


def compute_training_metrics(
    per_domain_per_task_raw_losses: Dict[str, Dict[str, float]],
    per_task_raw_losses: Dict[str, torch.Tensor],
    per_domain_weighted_losses: Dict[str, float],
    total_weighted_loss: torch.Tensor,
    weighter: UncertaintyWeighter,
    grl_scheduler: GRLLambdaScheduler,
    optimizer_model: AdamW,
    optimizer_uncertainty: AdamW,
    model: PretrainableGNN,
    epoch: int,
    step: int,
    step_start_time: float
) -> Dict[str, Any]:
    metrics = _compute_base_metrics(
        per_domain_per_task_raw_losses,
        per_task_raw_losses,
        per_domain_weighted_losses,
        total_weighted_loss,
        weighter,
        grl_scheduler,
        epoch,
        'train'
    )

    metrics[f'train/progress/step'] = step

    metrics['train/lr/model'] = optimizer_model.param_groups[0]['lr']
    metrics['train/lr/uncertainty'] = optimizer_uncertainty.param_groups[0]['lr']

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    metrics['train/gradients/model_grad_norm'] = total_norm

    metrics['train/system/time_per_step'] = time.time() - step_start_time

    return metrics


def compute_validation_metrics(
    per_domain_per_task_raw_losses: Dict[str, Dict[str, float]],
    per_task_raw_losses: Dict[str, torch.Tensor],
    per_domain_weighted_losses: Dict[str, float],
    total_weighted_loss: float,
    weighter: UncertaintyWeighter,
    grl_scheduler: Any,
    epoch: int
) -> Dict[str, Any]:
    return _compute_base_metrics(
        per_domain_per_task_raw_losses,
        per_task_raw_losses,
        per_domain_weighted_losses,
        total_weighted_loss,
        weighter,
        grl_scheduler,
        epoch,
        'val'
    )

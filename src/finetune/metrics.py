import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from torch import Tensor

from src.data.data_setup import NUM_CLASSES

from src.models.finetune_model import FinetuneGNN


def _aggregate_batch_metrics(
    batch_metrics: List[Dict[str, float]],
    epoch: int,
    prefix: str
) -> Dict[str, float]:
    metrics = {}
    metric_names = set(batch_metrics[0].keys()) - {'num_samples'}

    total_samples = sum(batch['num_samples'] for batch in batch_metrics)
    for metric_name in metric_names:
        weighted_sum = sum(
            batch[metric_name] * batch['num_samples']
            for batch in batch_metrics
        )
        metrics[metric_name] = weighted_sum / total_samples

    if prefix != 'val':
        metrics[f'{prefix}/progress/epoch'] = epoch

    return metrics


def compute_batch_metrics(
    domain_name: str,
    targets: Tensor,
    predictions: Tensor,
    probabilities: Tensor,
    loss: torch.Tensor,
    prefix: str
) -> Dict[str, float]:
    is_binary_classification = NUM_CLASSES[domain_name] == 2

    y_true_np = targets.detach().cpu().numpy()
    y_pred_np = predictions.detach().cpu().numpy()
    y_prob_np = probabilities.detach().cpu().numpy()
    if is_binary_classification:
        y_prob_np = y_prob_np[:, 1]

    metrics = {}

    metrics[f'{prefix}/accuracy'] = float(accuracy_score(y_true_np, y_pred_np))

    unique_true = np.unique(y_true_np)
    unique_pred = np.unique(y_pred_np)

    average_type = 'binary' if is_binary_classification else 'macro'
    metrics[f'{prefix}/f1'] = float(f1_score(y_true_np, y_pred_np, average=average_type, zero_division=0))
    metrics[f'{prefix}/precision'] = float(precision_score(y_true_np, y_pred_np, average=average_type, zero_division=0))
    metrics[f'{prefix}/recall'] = float(recall_score(y_true_np, y_pred_np, average=average_type, zero_division=0))

    if len(unique_true) < 2:
        metrics[f'{prefix}/auc'] = 0.0
    else:
        try:
            if is_binary_classification:
                metrics[f'{prefix}/auc'] = float(roc_auc_score(y_true_np, y_prob_np))
            else:
                metrics[f'{prefix}/auc'] = float(roc_auc_score(y_true_np, y_prob_np, multi_class='ovr'))
        except (ValueError, RuntimeWarning):
            metrics[f'{prefix}/auc'] = 0.0

    metrics[f'{prefix}/loss'] = float(loss.item())
    metrics['num_samples'] = len(targets)

    return metrics


def compute_training_metrics(
    epoch: int,
    step: int,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    domain_name: str,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    step_start_time: float,
    model: FinetuneGNN
) -> Dict[str, float]:
    metrics = compute_batch_metrics(domain_name, targets, predictions, probabilities, loss, 'train')

    for pg in optimizer.param_groups:
        metrics[f'train/lr/{pg["name"]}'] = pg['lr']

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    metrics['train/gradients/model_grad_norm'] = total_norm

    metrics['train/progress/epoch'] = epoch
    metrics['train/progress/step'] = step

    step_time = (time.time() - step_start_time)
    metrics['train/system/time_per_step'] = step_time

    return metrics


def compute_validation_metrics(
    batch_metrics: List[Dict[str, float]],
    epoch: int
) -> Dict[str, float]:
    return _aggregate_batch_metrics(batch_metrics, epoch, 'val')


def compute_test_metrics(
    batch_metrics: List[Dict[str, float]],
    epoch: int,
    epochs_since_improvement: int,
    training_start_time: float,
    model: FinetuneGNN
) -> Dict[str, float]:
    test_metrics = _aggregate_batch_metrics(batch_metrics, epoch, 'test')

    test_metrics['test/convergence_epochs'] = epoch - epochs_since_improvement
    test_metrics['test/training_time'] = time.time() - training_start_time

    test_metrics['test/total_parameters'] = sum(p.numel() for p in model.parameters())
    test_metrics['test/trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return test_metrics

import time
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from torch import Tensor

from src.data.data_setup import NUM_CLASSES

from src.models.finetune_model import FinetuneGNN


def _aggregate_batch_metrics(batch_metrics: List[Dict[str, float]], epoch: int, prefix: str) -> Dict[str, float]:
    metrics = {}
    metric_names = set(batch_metrics[0].keys()) - {'num_samples'}

    total_samples = sum(batch['num_samples'] for batch in batch_metrics)
    for metric_name in metric_names:
        weighted_sum = sum(
            batch[metric_name] * batch['num_samples']
            for batch in batch_metrics
        )
        metrics[f'{prefix}/{metric_name}'] = weighted_sum / total_samples

    metrics[f'{prefix}/epoch'] = epoch

    return metrics


def compute_batch_metrics(
    domain_name: str,
    targets: Tensor,
    predictions: Tensor,
    probabilities: Tensor,
    loss: torch.Tensor
) -> Dict[str, float]:
    is_binary_classification = NUM_CLASSES[domain_name] == 2

    y_true_np = targets.detach().cpu().numpy()
    y_pred_np = predictions.detach().cpu().numpy()
    y_prob_np = probabilities.detach().cpu().numpy()

    metrics = {}

    metrics['accuracy'] = float(accuracy_score(y_true_np, y_pred_np))

    average_type = 'binary' if is_binary_classification else 'macro'
    metrics['f1'] = float(f1_score(y_true_np, y_pred_np, average=average_type))
    metrics['precision'] = float(precision_score(y_true_np, y_pred_np, average=average_type))
    metrics['recall'] = float(recall_score(y_true_np, y_pred_np, average=average_type))

    try:
        if is_binary_classification:
            y_prob_pos = y_prob_np[:, 1]
            metrics['auc'] = float(roc_auc_score(y_true_np, y_prob_pos))
        else:
            metrics['auc'] = float(roc_auc_score(y_true_np, y_prob_np, average='macro', multi_class='ovr'))
    except ValueError:
        metrics['auc'] = 0.0

    metrics['loss'] = float(loss.item())
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
    step_start_time: float
) -> Dict[str, float]:
    batch_metrics = compute_batch_metrics(domain_name, targets, predictions, probabilities, loss)

    metrics = {}
    metric_names = set(batch_metrics.keys()) - {'num_samples'}

    for metric_name in metric_names:
        metrics[f'finetune/train/{metric_name}'] = batch_metrics[metric_name]

    for pg in optimizer.param_groups:
        metrics[f'finetune/train/lr/{pg["name"]}'] = pg['lr']

    metrics['finetune/train/epoch'] = epoch
    metrics['finetune/train/step'] = step

    step_time = (time.time() - step_start_time)
    metrics['finetune/system/time_per_step'] = step_time

    return metrics


def compute_validation_metrics(
    batch_metrics: List[Dict[str, float]],
    epoch: int
) -> Dict[str, float]:
    return _aggregate_batch_metrics(batch_metrics, epoch, 'finetune/val')


def compute_test_metrics(
    batch_metrics: List[Dict[str, float]],
    epoch: int,
    epochs_since_improvement: int,
    training_start_time: float,
    model: FinetuneGNN
) -> Dict[str, float]:
    test_metrics = _aggregate_batch_metrics(batch_metrics, epoch, 'finetune/test')

    test_metrics['finetune/test/convergence_epochs'] = epoch - epochs_since_improvement
    test_metrics['finetune/test/training_time'] = time.time() - training_start_time

    test_metrics['finetune/test/total_parameters'] = sum(p.numel() for p in model.parameters())
    test_metrics['finetune/test/trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return test_metrics

from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor


def compute_evaluation_metrics(y_true: Tensor, y_pred: Tensor, y_prob: Tensor, task_type: str) -> Dict[str, float]:
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    metrics = {}

    metrics['accuracy'] = float(accuracy_score(y_true_np, y_pred_np))

    if task_type == 'binary':
        metrics['f1'] = float(f1_score(y_true_np, y_pred_np, average='binary'))
        metrics['f1_macro'] = float(f1_score(y_true_np, y_pred_np, average='macro'))
        metrics['f1_micro'] = float(f1_score(y_true_np, y_pred_np, average='micro'))
    else:
        metrics['f1_macro'] = float(f1_score(y_true_np, y_pred_np, average='macro'))
        metrics['f1_micro'] = float(f1_score(y_true_np, y_pred_np, average='micro'))
        metrics['f1_weighted'] = float(f1_score(y_true_np, y_pred_np, average='weighted'))

    y_prob_np = y_prob.detach().cpu().numpy()

    try:
        if task_type == 'binary':
            y_prob_pos = y_prob_np[:, 1]
            metrics['auc'] = float(roc_auc_score(y_true_np, y_prob_pos))
        else:
            metrics['auc_macro'] = float(roc_auc_score(y_true_np, y_prob_np, average='macro', multi_class='ovr'))
            metrics['auc_weighted'] = float(roc_auc_score(y_true_np, y_prob_np, average='weighted', multi_class='ovr'))
    except ValueError:
        if task_type == 'binary':
            metrics['auc'] = 0.0
        else:
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0

    return metrics


def aggregate_metrics(metric_batches: list, split_prefix: str) -> Dict[str, float]:
    total_samples = sum(batch['num_samples'] for batch in metric_batches)

    aggregated = {}

    metric_names = set(metric_batches[0].keys()) - {'num_samples'}

    for metric_name in metric_names:
        weighted_sum = sum(batch[metric_name] * batch['num_samples'] for batch in metric_batches)
        aggregated[f'{split_prefix}/{metric_name}'] = weighted_sum / total_samples

    aggregated[f'{split_prefix}/num_samples'] = total_samples

    return aggregated


def compute_training_metrics(epoch: int, step: int, train_loss: float, lr_backbone: float, lr_head: float, model: torch.nn.Module) -> Dict[str, float]:
    metrics = {
        'train/loss': train_loss,
        'train/lr/backbone': lr_backbone,
        'train/lr/head': lr_head,
        'train/epoch': epoch,
        'train/step': step,
    }

    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1

    total_norm = total_norm ** (1. / 2)
    metrics['train/grad_norm'] = total_norm

    return metrics

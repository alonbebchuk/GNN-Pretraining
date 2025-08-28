from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor


def compute_classification_metrics(
    y_true: Tensor,
    y_pred: Tensor,
    y_prob: Optional[Tensor] = None,
    task_type: str = 'multiclass'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted labels [N]  
        y_prob: Predicted probabilities [N, num_classes] (optional, for AUC)
        task_type: 'binary' or 'multiclass'
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy for sklearn compatibility
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = float(accuracy_score(y_true_np, y_pred_np))
    
    # F1 scores
    if task_type == 'binary':
        metrics['f1'] = float(f1_score(y_true_np, y_pred_np, average='binary'))
        metrics['f1_macro'] = float(f1_score(y_true_np, y_pred_np, average='macro'))
        metrics['f1_micro'] = float(f1_score(y_true_np, y_pred_np, average='micro'))
    else:  # multiclass
        metrics['f1_macro'] = float(f1_score(y_true_np, y_pred_np, average='macro'))
        metrics['f1_micro'] = float(f1_score(y_true_np, y_pred_np, average='micro'))
        metrics['f1_weighted'] = float(f1_score(y_true_np, y_pred_np, average='weighted'))
    
    # AUC scores (if probabilities provided)
    if y_prob is not None:
        y_prob_np = y_prob.detach().cpu().numpy()
        
        try:
            if task_type == 'binary':
                # For binary, use probability of positive class
                if y_prob_np.shape[1] == 2:
                    y_prob_pos = y_prob_np[:, 1]
                else:
                    y_prob_pos = y_prob_np.flatten()
                metrics['auc'] = float(roc_auc_score(y_true_np, y_prob_pos))
                
            else:  # multiclass
                metrics['auc_macro'] = float(roc_auc_score(
                    y_true_np, y_prob_np, average='macro', multi_class='ovr'
                ))
                metrics['auc_weighted'] = float(roc_auc_score(
                    y_true_np, y_prob_np, average='weighted', multi_class='ovr'
                ))
                
        except ValueError as e:
            # AUC computation can fail with single class or other issues
            print(f"AUC computation failed: {e}")
            if task_type == 'binary':
                metrics['auc'] = 0.0
            else:
                metrics['auc_macro'] = 0.0
                metrics['auc_weighted'] = 0.0
    
    return metrics


def compute_loss_and_metrics(
    model: torch.nn.Module,
    batch_or_data,
    device: torch.device,
    task_type: str,
    domain_name: str
) -> Dict[str, float]:
    """
    Compute loss and metrics for a single batch/data sample.
    Handles graph classification, node classification, and link prediction tasks.
    
    Args:
        model: Finetuning model
        batch_or_data: Different formats for different task types
        device: Torch device
        task_type: 'graph_classification', 'node_classification', or 'link_prediction'
        domain_name: Dataset name for determining binary vs multiclass
    
    Returns:
        Dictionary with loss and metrics
    """
    model.eval()
    
    with torch.no_grad():
        if task_type == 'graph_classification':
            # Standard graph classification
            batch = batch_or_data.to(device)
            logits = model(batch)
            targets = batch.y
            
            # Compute loss
            loss = F.cross_entropy(logits, targets, reduction='mean')
            
            # Compute predictions and probabilities
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
        elif task_type == 'node_classification':
            # Node classification with specific nodes
            data, node_indices, targets = batch_or_data
            data = data.to(device)
            node_indices = node_indices.to(device)
            targets = targets.to(device)
            
            # Forward pass on full graph, select specific nodes
            full_logits = model(data)
            logits = full_logits[node_indices]
            
            # Compute loss
            loss = F.cross_entropy(logits, targets, reduction='mean')
            
            # Compute predictions and probabilities
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
        elif task_type == 'link_prediction':
            # Link prediction
            data, edges, targets = batch_or_data
            data = data.to(device)
            edges = edges.to(device)
            targets = targets.to(device)
            
            # Forward pass for link prediction
            edge_probs = model(data, edge_index_for_prediction=edges)
            
            # Compute loss (binary cross entropy)
            loss = F.binary_cross_entropy(edge_probs, targets, reduction='mean')
            
            # For link prediction, predictions are binary (edge exists or not)
            predictions = (edge_probs > 0.5).float()
            
            # Convert to binary classification format for metrics
            probabilities = torch.stack([1 - edge_probs, edge_probs], dim=1)
            targets = targets.long()
            predictions = predictions.long()
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Determine if binary or multiclass
        if task_type == 'link_prediction':
            task_classification_type = 'binary'
        else:
            num_classes = logits.size(1)
            is_binary = (num_classes == 2) or (domain_name == 'PTC_MR')  # PTC_MR is binary
            task_classification_type = 'binary' if is_binary else 'multiclass'
        
        # Compute metrics
        metrics = compute_classification_metrics(
            targets, predictions, probabilities, task_classification_type
        )
        
        # Add loss to metrics
        metrics['loss'] = float(loss.item())
        metrics['num_samples'] = len(targets)
    
    return metrics


def aggregate_metrics(metric_batches: list) -> Dict[str, float]:
    """
    Aggregate metrics across batches using proper weighting.
    
    Args:
        metric_batches: List of metric dictionaries from individual batches
    
    Returns:
        Aggregated metrics dictionary
    """
    if not metric_batches:
        return {}
    
    # Calculate total samples for weighting
    total_samples = sum(batch['num_samples'] for batch in metric_batches)
    
    aggregated = {}
    
    # Get all metric names (excluding num_samples)
    metric_names = set(metric_batches[0].keys()) - {'num_samples'}
    
    for metric_name in metric_names:
        # Weighted average across batches
        weighted_sum = sum(
            batch[metric_name] * batch['num_samples'] 
            for batch in metric_batches
        )
        aggregated[metric_name] = weighted_sum / total_samples
    
    aggregated['total_samples'] = total_samples
    
    return aggregated


def compute_training_metrics(
    epoch: int,
    step: int,
    train_loss: float,
    lr_backbone: float,
    lr_head: float,
    model: torch.nn.Module
) -> Dict[str, float]:
    """
    Compute training metrics for logging.
    
    Args:
        epoch: Current epoch
        step: Current step
        train_loss: Training loss for current batch
        lr_backbone: Current backbone learning rate
        lr_head: Current head learning rate  
        model: Model for gradient monitoring
    
    Returns:
        Training metrics dictionary
    """
    metrics = {
        'train/loss': train_loss,
        'train/lr/backbone': lr_backbone,
        'train/lr/head': lr_head,
        'train/epoch': epoch,
        'train/step': step,
    }
    
    # Compute gradient norm for monitoring
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
        metrics['train/grad_norm'] = total_norm
    
    return metrics


def compute_validation_metrics(
    aggregated_metrics: Dict[str, float],
    split: str = 'val'
) -> Dict[str, float]:
    """
    Format validation/test metrics for logging.
    
    Args:
        aggregated_metrics: Aggregated metrics from aggregate_metrics()
        split: 'val' or 'test'
    
    Returns:
        Formatted metrics for logging
    """
    formatted_metrics = {}
    
    for metric_name, value in aggregated_metrics.items():
        if metric_name == 'total_samples':
            formatted_metrics[f'{split}/num_samples'] = value
        else:
            formatted_metrics[f'{split}/{metric_name}'] = value
    
    return formatted_metrics


def print_metrics_summary(metrics: Dict[str, float], split: str, domain: str) -> None:
    """Print a formatted summary of metrics."""
    print(f"\n{split.upper()} METRICS - {domain}")
    print("=" * 40)
    
    # Primary metrics
    if f'{split}/accuracy' in metrics:
        print(f"Accuracy:  {metrics[f'{split}/accuracy']:.4f}")
    if f'{split}/f1_macro' in metrics:
        print(f"F1 Macro:  {metrics[f'{split}/f1_macro']:.4f}")
    if f'{split}/auc_macro' in metrics:
        print(f"AUC Macro: {metrics[f'{split}/auc_macro']:.4f}")
    if f'{split}/loss' in metrics:
        print(f"Loss:      {metrics[f'{split}/loss']:.4f}")
    
    # Sample count
    if f'{split}/num_samples' in metrics:
        print(f"Samples:   {int(metrics[f'{split}/num_samples'])}")
    
    print("=" * 40)

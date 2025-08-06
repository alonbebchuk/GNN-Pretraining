#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics for Fine-tuning Pipeline.

This module provides detailed evaluation metrics for different task types
including graph classification, node classification, and link prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


class MetricsComputer:
    """
    Comprehensive metrics computation for different task types.
    """
    
    def __init__(self, task_type: str, num_classes: int):
        """
        Initialize metrics computer.
        
        Args:
            task_type: Type of task ('graph_classification', 'node_classification', 'link_prediction')
            num_classes: Number of classes
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probabilities: Optional[torch.Tensor] = None):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            probabilities: Prediction probabilities (optional)
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
        
        # Store predictions and targets
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
        
        if probabilities is not None:
            if probabilities.ndim > 1:
                # Multi-class: store max probability or full distribution
                self.probabilities.extend(probabilities.max(axis=1))
            else:
                # Binary: store probabilities directly
                self.probabilities.extend(probabilities.flatten())
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive metrics based on accumulated data.
        
        Returns:
            Dictionary with computed metrics
        """
        if not self.predictions or not self.targets:
            logger.warning("No data available for metrics computation")
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities) if self.probabilities else None
        
        metrics = {}
        
        # Basic classification metrics
        try:
            metrics['accuracy'] = accuracy_score(targets, predictions)
        except Exception as e:
            logger.warning(f"Failed to compute accuracy: {e}")
            metrics['accuracy'] = 0.0
        
        # Precision, Recall, F1
        try:
            if self.num_classes == 2:
                # Binary classification
                metrics['precision'] = precision_score(targets, predictions, zero_division=0)
                metrics['recall'] = recall_score(targets, predictions, zero_division=0)
                metrics['f1'] = f1_score(targets, predictions, zero_division=0)
            else:
                # Multi-class classification
                metrics['precision_macro'] = precision_score(targets, predictions, average='macro', zero_division=0)
                metrics['precision_micro'] = precision_score(targets, predictions, average='micro', zero_division=0)
                metrics['recall_macro'] = recall_score(targets, predictions, average='macro', zero_division=0)
                metrics['recall_micro'] = recall_score(targets, predictions, average='micro', zero_division=0)
                metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
                metrics['f1_micro'] = f1_score(targets, predictions, average='micro', zero_division=0)
        except Exception as e:
            logger.warning(f"Failed to compute precision/recall/f1: {e}")
        
        # AUC metrics (if probabilities available)
        if probabilities is not None and len(probabilities) > 0:
            try:
                if self.num_classes == 2:
                    metrics['auc_roc'] = roc_auc_score(targets, probabilities)
                    metrics['auc_pr'] = average_precision_score(targets, probabilities)
                else:
                    # Multi-class AUC (one-vs-rest)
                    if len(np.unique(targets)) > 1:
                        metrics['auc_roc_ovr'] = roc_auc_score(targets, probabilities, multi_class='ovr', average='macro')
            except Exception as e:
                logger.warning(f"Failed to compute AUC metrics: {e}")
        
        # Task-specific metrics
        if self.task_type == 'link_prediction':
            metrics.update(self._compute_link_prediction_metrics(predictions, targets, probabilities))
        elif self.task_type in ['graph_classification', 'node_classification']:
            metrics.update(self._compute_classification_metrics(predictions, targets))
        
        return metrics
    
    def _compute_link_prediction_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                                       probabilities: Optional[np.ndarray]) -> Dict[str, float]:
        """Compute link prediction specific metrics."""
        metrics = {}
        
        try:
            # Precision@K and Recall@K for different K values
            k_values = [10, 50, 100]
            
            if probabilities is not None:
                for k in k_values:
                    if len(probabilities) >= k:
                        # Get top-k predictions
                        top_k_indices = np.argsort(probabilities)[-k:]
                        top_k_targets = targets[top_k_indices]
                        
                        precision_at_k = np.sum(top_k_targets) / k
                        recall_at_k = np.sum(top_k_targets) / max(np.sum(targets), 1)
                        
                        metrics[f'precision_at_{k}'] = precision_at_k
                        metrics[f'recall_at_{k}'] = recall_at_k
            
            # Hit rate (for ranking evaluation)
            if probabilities is not None and len(probabilities) > 0:
                # Sort by probability
                sorted_indices = np.argsort(probabilities)[::-1]
                sorted_targets = targets[sorted_indices]
                
                # Hit rate at different positions
                for k in [1, 5, 10]:
                    if len(sorted_targets) >= k:
                        hit_rate = np.any(sorted_targets[:k])
                        metrics[f'hit_rate_at_{k}'] = float(hit_rate)
        
        except Exception as e:
            logger.warning(f"Failed to compute link prediction metrics: {e}")
        
        return metrics
    
    def _compute_classification_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute classification specific metrics."""
        metrics = {}
        
        try:
            # Confusion matrix based metrics
            cm = confusion_matrix(targets, predictions)
            
            if self.num_classes == 2:
                # Binary classification
                tn, fp, fn, tp = cm.ravel()
                
                # Specificity
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics['specificity'] = specificity
                
                # Sensitivity (same as recall)
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics['sensitivity'] = sensitivity
                
                # Balanced accuracy
                balanced_acc = (sensitivity + specificity) / 2
                metrics['balanced_accuracy'] = balanced_acc
            
            # Per-class metrics
            for i in range(min(self.num_classes, len(np.unique(targets)))):
                class_mask = (targets == i)
                if np.sum(class_mask) > 0:
                    class_predictions = predictions[class_mask]
                    class_accuracy = np.mean(class_predictions == i)
                    metrics[f'class_{i}_accuracy'] = class_accuracy
        
        except Exception as e:
            logger.warning(f"Failed to compute classification metrics: {e}")
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if not self.predictions or not self.targets:
            return "No data available for classification report"
        
        try:
            predictions = np.array(self.predictions)
            targets = np.array(self.targets)
            
            return classification_report(targets, predictions, zero_division=0)
        except Exception as e:
            logger.warning(f"Failed to generate classification report: {e}")
            return f"Failed to generate report: {e}"
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if not self.predictions or not self.targets:
            return np.array([])
        
        try:
            predictions = np.array(self.predictions)
            targets = np.array(self.targets)
            
            return confusion_matrix(targets, predictions)
        except Exception as e:
            logger.warning(f"Failed to compute confusion matrix: {e}")
            return np.array([])


class EvaluationTracker:
    """
    Tracks evaluation metrics across multiple epochs and provides analysis.
    """
    
    def __init__(self, task_type: str, num_classes: int):
        """
        Initialize evaluation tracker.
        
        Args:
            task_type: Type of task
            num_classes: Number of classes
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.epoch_metrics = []
        self.best_metrics = {}
        self.best_epoch = -1
    
    def update_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Update metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Computed metrics
        """
        epoch_data = {
            'epoch': epoch,
            'metrics': metrics.copy()
        }
        self.epoch_metrics.append(epoch_data)
        
        # Update best metrics (based on primary metric)
        primary_metric = self._get_primary_metric()
        if primary_metric in metrics:
            current_value = metrics[primary_metric]
            
            if not self.best_metrics or current_value > self.best_metrics.get(primary_metric, -float('inf')):
                self.best_metrics = metrics.copy()
                self.best_epoch = epoch
    
    def _get_primary_metric(self) -> str:
        """Get the primary metric for this task type."""
        if self.task_type == 'link_prediction':
            return 'auc_roc'
        else:
            return 'accuracy'
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics achieved."""
        return {
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'primary_metric': self._get_primary_metric()
        }
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get complete training history."""
        return self.epoch_metrics.copy()
    
    def analyze_training_progress(self) -> Dict[str, Any]:
        """Analyze training progress and provide insights."""
        if len(self.epoch_metrics) < 2:
            return {'status': 'insufficient_data'}
        
        primary_metric = self._get_primary_metric()
        values = [epoch_data['metrics'].get(primary_metric, 0) for epoch_data in self.epoch_metrics]
        
        analysis = {
            'total_epochs': len(self.epoch_metrics),
            'best_epoch': self.best_epoch,
            'final_value': values[-1] if values else 0,
            'best_value': max(values) if values else 0,
            'improvement_from_start': values[-1] - values[0] if len(values) > 1 else 0,
            'is_converged': self._check_convergence(values),
            'is_overfitting': self._check_overfitting(),
            'stability': self._compute_stability(values)
        }
        
        return analysis
    
    def _check_convergence(self, values: List[float], window: int = 10, threshold: float = 1e-4) -> bool:
        """Check if training has converged."""
        if len(values) < window:
            return False
        
        recent_values = values[-window:]
        variance = np.var(recent_values)
        
        return variance < threshold
    
    def _check_overfitting(self, patience: int = 10) -> bool:
        """Check for signs of overfitting."""
        if len(self.epoch_metrics) < patience * 2:
            return False
        
        # Simple heuristic: if best epoch is too early, might be overfitting
        total_epochs = len(self.epoch_metrics)
        early_stopping_point = total_epochs * 0.7
        
        return self.best_epoch < early_stopping_point
    
    def _compute_stability(self, values: List[float]) -> float:
        """Compute training stability (lower variance = more stable)."""
        if len(values) < 2:
            return 0.0
        
        # Compute coefficient of variation
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if mean_val == 0:
            return 0.0
        
        cv = std_val / abs(mean_val)
        
        # Convert to stability score (0-1, higher is better)
        stability = 1.0 / (1.0 + cv)
        
        return stability


def create_metrics_computer(task_type: str, num_classes: int) -> MetricsComputer:
    """
    Create metrics computer for given task type.
    
    Args:
        task_type: Type of task
        num_classes: Number of classes
        
    Returns:
        MetricsComputer instance
    """
    return MetricsComputer(task_type, num_classes)


def create_evaluation_tracker(task_type: str, num_classes: int) -> EvaluationTracker:
    """
    Create evaluation tracker for given task type.
    
    Args:
        task_type: Type of task
        num_classes: Number of classes
        
    Returns:
        EvaluationTracker instance
    """
    return EvaluationTracker(task_type, num_classes)


def compute_improvement_metrics(baseline_metrics: Dict[str, float], 
                              current_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compute improvement metrics compared to baseline.
    
    Args:
        baseline_metrics: Baseline performance metrics
        current_metrics: Current performance metrics
        
    Returns:
        Dictionary with improvement metrics
    """
    improvements = {}
    
    for metric_name, current_value in current_metrics.items():
        if metric_name in baseline_metrics:
            baseline_value = baseline_metrics[metric_name]
            
            if baseline_value != 0:
                # Relative improvement
                relative_improvement = (current_value - baseline_value) / abs(baseline_value)
                improvements[f'{metric_name}_relative_improvement'] = relative_improvement
            
            # Absolute improvement
            absolute_improvement = current_value - baseline_value
            improvements[f'{metric_name}_absolute_improvement'] = absolute_improvement
    
    return improvements 
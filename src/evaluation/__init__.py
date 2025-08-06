"""
Evaluation and fine-tuning pipeline for downstream tasks.

This module contains:
- Main fine-tuning script
- Enhanced fine-tuning trainers
- Model adaptation utilities
- Comprehensive evaluation metrics
- Link prediction enhancements
"""

from .main_finetune import main as finetune_main
from .enhanced_finetune_trainer import EnhancedFineTuningTrainer, AdaptiveFineTuningModel
from .model_adapter import ModelAdapter, DownstreamModelWrapper
from .evaluation_metrics import MetricsComputer, EvaluationTracker
from .link_prediction import LinkPredictionEvaluator, AdvancedNegativeSampler

__all__ = [
    'finetune_main',
    'EnhancedFineTuningTrainer',
    'AdaptiveFineTuningModel',
    'ModelAdapter',
    'DownstreamModelWrapper',
    'MetricsComputer',
    'EvaluationTracker',
    'LinkPredictionEvaluator',
    'AdvancedNegativeSampler'
]

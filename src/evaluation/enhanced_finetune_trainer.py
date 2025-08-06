#!/usr/bin/env python3
"""
Enhanced Fine-tuning Trainer with Comprehensive Evaluation.

This module provides an advanced trainer for downstream fine-tuning with:
- Comprehensive metrics computation
- Task-specific loss functions
- Advanced evaluation strategies
- Integration with enhanced components
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

try:
    from .evaluation_metrics import MetricsComputer, EvaluationTracker
    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    try:
        from evaluation_metrics import MetricsComputer, EvaluationTracker
        ENHANCED_METRICS_AVAILABLE = True
    except ImportError:
        logger.warning("Enhanced metrics not available, using basic metrics")
        ENHANCED_METRICS_AVAILABLE = False

try:
    from .link_prediction import LinkPredictionEvaluator, AdvancedNegativeSampler, LinkPredictionConfig
    ENHANCED_LINK_PRED_AVAILABLE = True
except ImportError:
    try:
        from link_prediction import LinkPredictionEvaluator, AdvancedNegativeSampler, LinkPredictionConfig
        ENHANCED_LINK_PRED_AVAILABLE = True
    except ImportError:
        logger.warning("Enhanced link prediction not available")
        ENHANCED_LINK_PRED_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class AdaptiveFineTuningModel(nn.Module):
    """
    Adaptive fine-tuning model that handles different task types and adaptation strategies.
    """
    
    def __init__(self,
                 encoder: nn.Module,
                 backbone: nn.Module,
                 task_head: nn.Module,
                 task_type: str,
                 adaptation_method: str = 'full',
                 freeze_encoder: bool = True):
        """
        Initialize adaptive fine-tuning model.
        
        Args:
            encoder: Input encoder (pre-trained or new)
            backbone: GNN backbone (pre-trained)
            task_head: Task-specific head (new)
            task_type: Type of downstream task
            adaptation_method: Adaptation strategy ('full', 'linear', 'adapter')
            freeze_encoder: Whether to freeze encoder initially
        """
        super().__init__()
        
        self.encoder = encoder
        self.backbone = backbone
        self.task_head = task_head
        self.task_type = task_type
        self.adaptation_method = adaptation_method
        
        # Initialize adapter layers if needed
        if adaptation_method == 'adapter':
            self.adapter_layers = self._create_adapter_layers()
        
        # Apply initial freezing
        if freeze_encoder:
            self.freeze_encoder()
    
    def _create_adapter_layers(self, bottleneck_dim: int = 64) -> nn.ModuleDict:
        """Create adapter layers for efficient fine-tuning."""
        adapters = nn.ModuleDict()
        
        # Add adapters to backbone layers if they exist
        if hasattr(self.backbone, 'layers'):
            for i, layer in enumerate(self.backbone.layers):
                if hasattr(layer, 'hidden_dim'):
                    hidden_dim = layer.hidden_dim
                    adapters[f'layer_{i}'] = nn.Sequential(
                        nn.Linear(hidden_dim, bottleneck_dim),
                        nn.ReLU(),
                        nn.Linear(bottleneck_dim, hidden_dim)
                    )
        
        return adapters
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        if self.adaptation_method == 'linear':
            # Also freeze backbone for linear probing
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        logger.info(f"Encoder frozen (adaptation: {self.adaptation_method})")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        if self.adaptation_method == 'full':
            # Also unfreeze backbone for full fine-tuning
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        logger.info(f"Encoder unfrozen (adaptation: {self.adaptation_method})")
    
    def forward(self, data):
        """Forward pass through the model."""
        # Encode input features
        h = self.encoder(data.x)
        
        # Pass through backbone
        node_embeddings = self.backbone(h, data.edge_index)
        
        # Apply adapters if using adapter method
        if self.adaptation_method == 'adapter' and hasattr(self, 'adapter_layers'):
            # Apply adapters (simplified - would need more sophisticated integration)
            for adapter_name, adapter in self.adapter_layers.items():
                if 'layer_0' in adapter_name:  # Example: apply to first layer output
                    node_embeddings = node_embeddings + adapter(node_embeddings)
        
        # Task-specific processing
        if self.task_type == 'graph_classification':
            # Global pooling for graph-level prediction
            try:
                from torch_geometric.nn import global_mean_pool
                batch = getattr(data, 'batch', torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device))
                graph_embedding = global_mean_pool(node_embeddings, batch)
                return self.task_head(graph_embedding)
            except ImportError:
                # Fallback: simple mean pooling
                graph_embedding = node_embeddings.mean(dim=0, keepdim=True)
                return self.task_head(graph_embedding)
        
        elif self.task_type == 'node_classification':
            # Direct node-level prediction
            return self.task_head(node_embeddings)
        
        elif self.task_type == 'link_prediction':
            # Return node embeddings for link prediction
            return node_embeddings
        
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")


class EnhancedDownstreamClassificationHead(nn.Module):
    """
    Enhanced classification head with flexible architecture.
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: List[int] = None,
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = False,
                 task_type: str = 'graph_classification'):
        """
        Initialize enhanced classification head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            task_type: Type of task (affects architecture)
        """
        super().__init__()
        
        self.task_type = task_type
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2] if input_dim > 64 else [32]
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Link prediction specific components
        if task_type == 'link_prediction':
            self.link_predictor = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim, num_classes)
            )
    
    def forward(self, x, edge_index=None):
        """Forward pass through classification head."""
        if self.task_type == 'link_prediction' and edge_index is not None:
            # Concatenate source and target node embeddings
            src_embeddings = x[edge_index[0]]
            dst_embeddings = x[edge_index[1]]
            edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
            return self.link_predictor(edge_embeddings)
        else:
            return self.classifier(x)


class EnhancedFineTuningTrainer:
    """
    Enhanced trainer for downstream fine-tuning with comprehensive evaluation.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: torch.device = None,
                 experiment_tracker: Optional[Any] = None,
                 evaluation_tracker: Optional[Any] = None):
        """
        Initialize enhanced fine-tuning trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
            device: Device to run on
            experiment_tracker: Experiment tracking (e.g., WandB)
            evaluation_tracker: Evaluation tracking
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_tracker = experiment_tracker
        self.evaluation_tracker = evaluation_tracker
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.scheduler = self._create_scheduler()
        
        # Initialize evaluation components
        self.metrics_computer = self._create_metrics_computer()
        self.link_pred_evaluator = self._create_link_pred_evaluator()
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = float('-inf')
        self.best_model_state = None
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_metrics': []
        }
        
        logger.info("Enhanced fine-tuning trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with component-specific learning rates."""
        # Get trainable parameters
        trainable_params = list(self.model.parameters())
        trainable_params = [p for p in trainable_params if p.requires_grad]
        
        if not trainable_params:
            logger.warning("No trainable parameters found!")
            # Make all parameters trainable as fallback
            for param in self.model.parameters():
                param.requires_grad = True
            trainable_params = list(self.model.parameters())
        
        # Create optimizer
        optimizer_name = self.config.get('training', {}).get('optimizer', 'AdamW')
        lr = self.config.get('training', {}).get('learning_rate', 1e-4)
        weight_decay = self.config.get('training', {}).get('weight_decay', 5e-4)
        
        if optimizer_name.lower() == 'adamw':
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on task type."""
        task_type = self.config.get('downstream_task', {}).get('task_type', 'graph_classification')
        num_classes = self.config.get('downstream_task', {}).get('num_classes', 2)
        
        if task_type == 'link_prediction' or num_classes == 2:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('training', {}).get('scheduler', {})
        
        if not scheduler_config or scheduler_config.get('type') == 'none':
            return None
        
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('training', {}).get('epochs', 200),
                eta_min=scheduler_config.get('min_lr_ratio', 0.01) * self.optimizer.param_groups[0]['lr']
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 50),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            return None
    
    def _create_metrics_computer(self):
        """Create metrics computer if available."""
        if not ENHANCED_METRICS_AVAILABLE:
            return None
        
        task_type = self.config.get('downstream_task', {}).get('task_type', 'graph_classification')
        num_classes = self.config.get('downstream_task', {}).get('num_classes', 2)
        
        return MetricsComputer(task_type, num_classes)
    
    def _create_link_pred_evaluator(self):
        """Create link prediction evaluator if available."""
        if not ENHANCED_LINK_PRED_AVAILABLE:
            return None
        
        task_type = self.config.get('downstream_task', {}).get('task_type', 'graph_classification')
        
        if task_type == 'link_prediction':
            return LinkPredictionEvaluator()
        
        return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Handle different data formats
                if hasattr(batch, 'x'):
                    # Graph data
                    batch = batch.to(self.device)
                    targets = batch.y
                else:
                    # Tensor data
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    batch = inputs
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                
                # Handle output shapes
                if outputs.dim() > 1 and outputs.size(0) == 1:
                    outputs = outputs.squeeze(0)
                if targets.dim() > 0 and len(targets) == 1:
                    targets = targets.squeeze()
                
                # Ensure compatible shapes
                if outputs.dim() == 1 and len(outputs) > 1:
                    outputs = outputs.unsqueeze(0)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping if specified
                grad_clip = self.config.get('training', {}).get('gradient_clipping', 0.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                
                # Get predictions for metrics
                if outputs.dim() > 1:
                    preds = outputs.argmax(dim=1)
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                
                all_predictions.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
                
            except Exception as e:
                logger.warning(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Compute epoch metrics
        avg_loss = total_loss / max(len(self.train_loader), 1)
        
        try:
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(all_targets, all_predictions) if all_targets else 0.0
        except:
            accuracy = 0.0
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }
    
    def evaluate(self, data_loader: DataLoader, split_name: str = 'val') -> Dict[str, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                try:
                    # Handle different data formats
                    if hasattr(batch, 'x'):
                        batch = batch.to(self.device)
                        targets = batch.y
                    else:
                        inputs, targets = batch
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        batch = inputs
                    
                    outputs = self.model(batch)
                    
                    # Handle output shapes
                    if outputs.dim() > 1 and outputs.size(0) == 1:
                        outputs = outputs.squeeze(0)
                    if targets.dim() > 0 and len(targets) == 1:
                        targets = targets.squeeze()
                    
                    if outputs.dim() == 1 and len(outputs) > 1:
                        outputs = outputs.unsqueeze(0)
                    
                    # Compute loss
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    # Get predictions and probabilities
                    if outputs.dim() > 1:
                        probs = torch.softmax(outputs, dim=1)
                        preds = outputs.argmax(dim=1)
                        all_probabilities.extend(probs.max(dim=1)[0].detach().cpu().numpy())
                    else:
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).long()
                        all_probabilities.extend(probs.detach().cpu().numpy())
                    
                    all_predictions.extend(preds.detach().cpu().numpy())
                    all_targets.extend(targets.detach().cpu().numpy())
                    
                except Exception as e:
                    logger.warning(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
        
        # Compute metrics
        avg_loss = total_loss / max(len(data_loader), 1)
        
        metrics = {f'{split_name}_loss': avg_loss}
        
        # Enhanced metrics if available
        if self.metrics_computer and all_targets:
            self.metrics_computer.reset()
            self.metrics_computer.update(
                torch.tensor(all_predictions),
                torch.tensor(all_targets),
                torch.tensor(all_probabilities) if all_probabilities else None
            )
            
            enhanced_metrics = self.metrics_computer.compute_metrics()
            for key, value in enhanced_metrics.items():
                metrics[f'{split_name}_{key}'] = value
        else:
            # Basic metrics
            try:
                from sklearn.metrics import accuracy_score, f1_score
                if all_targets:
                    metrics[f'{split_name}_accuracy'] = accuracy_score(all_targets, all_predictions)
                    metrics[f'{split_name}_f1'] = f1_score(all_targets, all_predictions, average='macro')
            except:
                metrics[f'{split_name}_accuracy'] = 0.0
                metrics[f'{split_name}_f1'] = 0.0
        
        return metrics
    
    def train(self, max_epochs: int = 200, patience: int = 10) -> Dict[str, Any]:
        """Run complete training loop."""
        logger.info(f"Starting enhanced fine-tuning for {max_epochs} epochs...")
        
        best_val_metric = float('-inf')
        patience_counter = 0
        unfreeze_epoch = self.config.get('fine_tuning_strategy', {}).get('unfreeze_epoch')
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Check if we should unfreeze the encoder
            if (unfreeze_epoch is not None and 
                epoch == unfreeze_epoch and 
                hasattr(self.model, 'unfreeze_encoder')):
                
                logger.info(f"Unfreezing encoder at epoch {epoch}")
                self.model.unfreeze_encoder()
                # Recreate optimizer to include encoder parameters
                self.optimizer = self._create_optimizer()
            
            # Training epoch
            train_metrics = self.train_epoch()
            
            # Validation epoch
            val_metrics = self.evaluate(self.val_loader, 'val')
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Track best model
            primary_metric = self.config.get('training', {}).get('validation_metric', 'val_accuracy')
            if primary_metric in val_metrics:
                current_val_metric = val_metrics[primary_metric]
                
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    logger.info(f"New best model! {primary_metric}: {current_val_metric:.4f}")
                else:
                    patience_counter += 1
            
            # Store metrics
            self.training_history['train_losses'].append(train_metrics['train_loss'])
            self.training_history['val_losses'].append(val_metrics['val_loss'])
            self.training_history['val_metrics'].append(val_metrics)
            
            # Log to experiment tracker
            if self.experiment_tracker and WANDB_AVAILABLE:
                try:
                    self.experiment_tracker.log(epoch_metrics, step=epoch)
                except:
                    pass
            
            # Update evaluation tracker
            if self.evaluation_tracker:
                try:
                    self.evaluation_tracker.update_epoch(epoch, val_metrics)
                except:
                    pass
            
            # Log progress
            logger.info(f"Epoch {epoch:3d}: Train Loss: {train_metrics['train_loss']:.4f}, "
                        f"Val Loss: {val_metrics['val_loss']:.4f}, "
                        f"Val Acc: {val_metrics.get('val_accuracy', 0):.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Load best model for final evaluation
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model for final evaluation")
        
        # Final test evaluation
        test_metrics = self.evaluate(self.test_loader, 'test')
        
        # Create final results
        final_results = {
            'best_val_metric': best_val_metric,
            'final_test_metrics': test_metrics,
            'training_history': self.training_history,
            'total_epochs': epoch + 1
        }
        
        # Log final results
        if self.experiment_tracker and WANDB_AVAILABLE:
            try:
                self.experiment_tracker.log({f"final_{k}": v for k, v in test_metrics.items()})
                self.experiment_tracker.log({"best_val_metric": best_val_metric})
            except:
                pass
        
        logger.info("Enhanced fine-tuning completed!")
        logger.info("Final test metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return final_results


def create_enhanced_downstream_model(pretrained_model: nn.Module,
                                   dataset_name: str,
                                   task_type: str,
                                   config: Dict[str, Any],
                                   device: torch.device) -> AdaptiveFineTuningModel:
    """
    Create enhanced downstream model with proper component extraction.
    
    Args:
        pretrained_model: Pre-trained model
        dataset_name: Target dataset name
        task_type: Type of downstream task
        config: Configuration dictionary
        device: Device to run on
        
    Returns:
        Enhanced downstream model
    """
    try:
        from .downstream_data_loading import get_dataset_info
        dataset_info = get_dataset_info(dataset_name)
    except ImportError:
        try:
            from downstream_data_loading import get_dataset_info
            dataset_info = get_dataset_info(dataset_name)
        except ImportError:
            # Fallback dataset info
            dataset_info = {
                'num_classes': config.get('downstream_task', {}).get('num_classes', 2),
                'input_dim': config.get('downstream_task', {}).get('input_dim', 256),
                'in_domain': False
            }
    
    # Extract components from pre-trained model
    if hasattr(pretrained_model, 'input_encoders') and hasattr(pretrained_model, 'gnn_backbone'):
        # Use pre-trained components
        domain_name = dataset_info.get('domain_name')
        
        if domain_name and domain_name in pretrained_model.input_encoders:
            # In-domain: use pre-trained encoder
            encoder = pretrained_model.input_encoders[domain_name]
        else:
            # Out-of-domain: create new encoder
            try:
                from models.gnn import InputEncoder
                encoder = InputEncoder(
                    dim_in=dataset_info['input_dim'],
                    hidden_dim=256,
                    dropout_rate=0.2
                )
            except ImportError:
                encoder = nn.Linear(dataset_info['input_dim'], 256)
        
        backbone = pretrained_model.gnn_backbone
    else:
        # Fallback: create simple components
        encoder = nn.Linear(dataset_info['input_dim'], 256)
        backbone = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
    
    # Create task head
    task_head = EnhancedDownstreamClassificationHead(
        input_dim=256,
        num_classes=dataset_info['num_classes'],
        hidden_dims=[128],
        dropout_rate=config.get('training', {}).get('dropout_rate', 0.2),
        task_type=task_type
    )
    
    # Create adaptive model
    model = AdaptiveFineTuningModel(
        encoder=encoder,
        backbone=backbone,
        task_head=task_head,
        task_type=task_type,
        adaptation_method=config.get('fine_tuning_strategy', {}).get('adaptation_method', 'full'),
        freeze_encoder=config.get('fine_tuning_strategy', {}).get('freeze_encoder', True)
    )
    
    return model.to(device)
#!/usr/bin/env python3
"""
Downstream fine-tuning script for evaluating pre-trained GNN models.

This script handles:
- Loading pre-trained models from checkpoints
- Fine-tuning on downstream tasks (graph/node/link classification)
- Both full fine-tuning and linear probing strategies
- Evaluation metrics computation and logging
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
import json
import time

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from infrastructure.config import load_config
    from data.data_loading import create_data_loaders
    from core.models.pretrain_model import PretrainableGNN
    from core.models.heads import MLPHead, DotProductDecoder
    from infrastructure.experiment_tracking import create_experiment_tracker
    from infrastructure.checkpointing import CheckpointManager
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class DownstreamModel(nn.Module):
    """
    Wrapper for downstream task evaluation.
    
    Handles both in-domain and out-of-domain scenarios with appropriate
    encoder and head configurations.
    """
    
    def __init__(self, 
                 pretrained_model: PretrainableGNN,
                 task_type: str,
                 num_classes: int,
                 input_dim: Optional[int] = None,
                 freeze_backbone: bool = False,
                 freeze_encoder: bool = False):
        """
        Initialize downstream model.
        
        Args:
            pretrained_model: Pre-trained GNN model
            task_type: 'graph_classification', 'node_classification', 'link_prediction'
            num_classes: Number of output classes
            input_dim: Input dimension for new encoder (out-of-domain only)
            freeze_backbone: Whether to freeze GNN backbone
            freeze_encoder: Whether to freeze input encoder
        """
        super().__init__()
        
        self.task_type = task_type
        self.freeze_backbone = freeze_backbone
        self.freeze_encoder = freeze_encoder
        
        # Use pre-trained components
        self.gnn_backbone = pretrained_model.gnn_backbone
        self.input_encoders = pretrained_model.input_encoders
        
        # Create new input encoder for out-of-domain tasks
        if input_dim is not None:
            self.new_input_encoder = nn.Sequential(
                nn.Linear(input_dim, pretrained_model.hidden_dim),
                nn.LayerNorm(pretrained_model.hidden_dim),
                nn.ReLU(),
                nn.Dropout(pretrained_model.dropout_rate)
            )
        else:
            self.new_input_encoder = None
        
        # Create task-specific head
        if task_type == 'graph_classification':
            self.task_head = MLPHead(
                dim_in=pretrained_model.hidden_dim,
                dim_hidden=pretrained_model.hidden_dim,
                dim_out=num_classes,
                dropout_rate=pretrained_model.dropout_rate
            )
        elif task_type == 'node_classification':
            self.task_head = MLPHead(
                dim_in=pretrained_model.hidden_dim,
                dim_hidden=pretrained_model.hidden_dim,
                dim_out=num_classes,
                dropout_rate=pretrained_model.dropout_rate
            )
        elif task_type == 'link_prediction':
            self.task_head = DotProductDecoder()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Freeze components if requested
        if freeze_backbone:
            for param in self.gnn_backbone.parameters():
                param.requires_grad = False
        
        if freeze_encoder:
            for encoder in self.input_encoders.values():
                for param in encoder.parameters():
                    param.requires_grad = False
    
    def forward(self, data, domain_name: Optional[str] = None):
        """Forward pass for downstream task."""
        # Encode input features
        if self.new_input_encoder is not None:
            # Out-of-domain: use new encoder
            h0 = self.new_input_encoder(data.x)
        else:
            # In-domain: use pre-trained encoder
            if domain_name and domain_name in self.input_encoders:
                h0 = self.input_encoders[domain_name](data.x)
            else:
                # Fallback to first available encoder
                encoder = next(iter(self.input_encoders.values()))
                h0 = encoder(data.x)
        
        # GNN forward pass
        node_embeddings = self.gnn_backbone(h0, data.edge_index)
        
        # Task-specific output
        if self.task_type == 'graph_classification':
            # Global pooling for graph-level prediction
            graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
            return self.task_head(graph_embedding)
        elif self.task_type == 'node_classification':
            return self.task_head(node_embeddings)
        elif self.task_type == 'link_prediction':
            # Enhanced link prediction handling
            if hasattr(data, 'edge_labels'):
                # Use pre-computed edges from enhanced dataset
                return self.task_head(node_embeddings, data.edge_index)
            else:
                # Fallback to basic link prediction
                return self.task_head(node_embeddings, data.edge_index)


class DownstreamTrainer:
    """Trainer for downstream task fine-tuning."""
    
    def __init__(self, 
                 model: DownstreamModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 experiment_tracker: Optional[Any] = None):
        """Initialize downstream trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.experiment_tracker = experiment_tracker
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss function
        if config['task_type'] == 'link_prediction':
            self.criterion = nn.BCEWithLogitsLoss()
        elif config['num_classes'] == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_val_score = float('-inf')
        self.patience_counter = 0
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for different components."""
        param_groups = []
        
        # New components (higher learning rate)
        new_params = []
        if hasattr(self.model, 'new_input_encoder') and self.model.new_input_encoder:
            new_params.extend(self.model.new_input_encoder.parameters())
        new_params.extend(self.model.task_head.parameters())
        
        if new_params:
            param_groups.append({
                'params': new_params,
                'lr': self.config.get('new_component_lr', 1e-3)
            })
        
        # Pre-trained components (lower learning rate)
        pretrained_params = []
        if not self.model.freeze_backbone:
            pretrained_params.extend(self.model.gnn_backbone.parameters())
        if not self.model.freeze_encoder:
            for encoder in self.model.input_encoders.values():
                pretrained_params.extend(encoder.parameters())
        
        if pretrained_params:
            param_groups.append({
                'params': pretrained_params,
                'lr': self.config.get('pretrained_component_lr', 1e-4)
            })
        
        return optim.AdamW(param_groups, weight_decay=self.config.get('weight_decay', 0.05))
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.config['task_type'] == 'graph_classification':
                # Handle batched graphs
                outputs = []
                labels = []
                for i in range(batch.batch.max().item() + 1):
                    mask = batch.batch == i
                    graph_data = batch.subgraph(mask)
                    output = self.model(graph_data)
                    outputs.append(output)
                    labels.append(batch.y[mask][0])  # Assuming graph-level labels
                
                outputs = torch.cat(outputs, dim=0)
                labels = torch.stack(labels)
            else:
                outputs = self.model(batch)
                labels = batch.y
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            if self.config['task_type'] == 'link_prediction' or self.config['num_classes'] == 2:
                preds = torch.sigmoid(outputs).detach().cpu().numpy()
            else:
                preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
        
        return total_loss / len(self.train_loader), all_preds, all_labels
    
    def evaluate(self, data_loader):
        """Evaluate on given data loader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Forward pass (similar to training)
                if self.config['task_type'] == 'graph_classification':
                    outputs = []
                    labels = []
                    for i in range(batch.batch.max().item() + 1):
                        mask = batch.batch == i
                        graph_data = batch.subgraph(mask)
                        output = self.model(graph_data)
                        outputs.append(output)
                        labels.append(batch.y[mask][0])
                    
                    outputs = torch.cat(outputs, dim=0)
                    labels = torch.stack(labels)
                else:
                    outputs = self.model(batch)
                    labels = batch.y
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Collect predictions
                if self.config['task_type'] == 'link_prediction' or self.config['num_classes'] == 2:
                    preds = torch.sigmoid(outputs).detach().cpu().numpy()
                else:
                    preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.detach().cpu().numpy())
        
        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(data_loader)
        
        return metrics
    
    def _compute_metrics(self, preds, labels):
        """Compute evaluation metrics."""
        import numpy as np
        
        preds = np.array(preds)
        labels = np.array(labels)
        
        metrics = {}
        
        if self.config['task_type'] == 'link_prediction':
            # Enhanced link prediction metrics
            try:
                from link_prediction import LinkPredictionEvaluator
                
                evaluator = LinkPredictionEvaluator()
                evaluator.add_batch(torch.tensor(preds), torch.tensor(labels))
                enhanced_metrics = evaluator.compute_metrics()
                
                # Use enhanced metrics if available
                if enhanced_metrics:
                    metrics.update(enhanced_metrics)
                    return metrics
                    
            except ImportError:
                logging.warning("Enhanced link prediction evaluation not available, using basic metrics")
            
            # Fallback to basic binary classification metrics
            binary_preds = (preds > 0.5).astype(int)
            metrics['accuracy'] = accuracy_score(labels, binary_preds)
            metrics['f1'] = f1_score(labels, binary_preds, average='binary')
            if len(np.unique(labels)) > 1:
                metrics['auc'] = roc_auc_score(labels, preds)
                
        elif self.config['num_classes'] == 2:
            # Binary classification
            binary_preds = (preds > 0.5).astype(int)
            metrics['accuracy'] = accuracy_score(labels, binary_preds)
            metrics['f1'] = f1_score(labels, binary_preds, average='binary')
            if len(np.unique(labels)) > 1:
                metrics['auc'] = roc_auc_score(labels, preds)
        else:
            # Multi-class classification
            class_preds = np.argmax(preds, axis=1)
            metrics['accuracy'] = accuracy_score(labels, class_preds)
            metrics['f1'] = f1_score(labels, class_preds, average='macro')
        
        return metrics
    
    def train(self, max_epochs: int = 200, patience: int = 10):
        """Full training loop."""
        logging.info(f"Starting downstream training for {max_epochs} epochs...")
        
        for epoch in range(max_epochs):
            # Training
            train_loss, train_preds, train_labels = self.train_epoch()
            train_metrics = self._compute_metrics(train_preds, train_labels)
            train_metrics['loss'] = train_loss
            
            # Validation
            val_metrics = self.evaluate(self.val_loader)
            
            # Logging
            logging.info(f"Epoch {epoch+1}/{max_epochs}")
            logging.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            logging.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            if self.experiment_tracker:
                metrics_to_log = {}
                for key, value in train_metrics.items():
                    metrics_to_log[f'train/{key}'] = value
                for key, value in val_metrics.items():
                    metrics_to_log[f'val/{key}'] = value
                
                self.experiment_tracker.log_metrics(metrics_to_log, epoch=epoch)
            
            # Early stopping
            val_score = val_metrics.get('accuracy', val_metrics.get('f1', 0))
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.patience_counter = 0
                # Save best model state
                best_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                self.model.load_state_dict(best_state)
                break
        
        # Final evaluation on test set
        test_metrics = self.evaluate(self.test_loader)
        logging.info("Final test results:")
        for key, value in test_metrics.items():
            logging.info(f"  {key}: {value:.4f}")
        
        return test_metrics


def main():
    """Main fine-tuning function."""
    parser = argparse.ArgumentParser(description='GNN Downstream Fine-tuning')
    parser.add_argument('--pretrain-checkpoint', type=str, default=None,
                       help='Path to pre-trained model checkpoint (if None, trains from scratch - B1 baseline)')
    parser.add_argument('--downstream-config', type=str, required=False,
                       help='Path to downstream task configuration (optional for B1 baseline)')
    parser.add_argument('--strategy', type=str, choices=['full', 'linear'], required=True,
                       help='Fine-tuning strategy: full or linear probing')
    parser.add_argument('--output-dir', type=str, default='results/finetune',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--offline', action='store_true',
                       help='Run without WandB logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    logger.info("Starting downstream fine-tuning...")
    if args.pretrain_checkpoint:
        logger.info(f"Pre-trained checkpoint: {args.pretrain_checkpoint}")
    else:
        logger.info("Mode: From-scratch training (B1 baseline)")
    logger.info(f"Downstream config: {args.downstream_config}")
    logger.info(f"Strategy: {args.strategy}")
    
    try:
        # Load downstream task configuration
        import yaml
        
        if args.downstream_config:
            with open(args.downstream_config, 'r') as f:
                downstream_config = yaml.safe_load(f)
        else:
            # Create B1 baseline configuration
            downstream_config = {
                'downstream_task': {
                    'dataset_name': 'MUTAG',
                    'task_type': 'graph_classification',
                    'batch_size': 32,
                    'num_classes': 2,
                    'in_domain': False
                },
                'training': {
                    'epochs': 200,
                    'patience': 10,
                    'optimizer': 'AdamW',
                    'learning_rate': 0.001,
                    'weight_decay': 0.0005,
                    'validation_metric': 'val_accuracy',
                    'metric_mode': 'max'
                },
                'fine_tuning_strategy': {
                    'freeze_encoder': False,
                    'adaptation_method': 'full'
                },
                'wandb': {
                    'enabled': not args.offline,
                    'project_name': 'gnn-b1-baseline',
                    'tags': ['b1', 'baseline', 'from-scratch']
                }
            }
            logger.info("Using B1 baseline configuration")
        
        # Enhanced model loading with adapter integration
        pretrained_model = None
        
        try:
            # Try to use the enhanced model adapter
            from model_adapter import create_model_adapter
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            adapter = create_model_adapter(device)
            
            if args.pretrain_checkpoint is not None:
                logger.info("Loading pre-trained model via adapter...")
                pretrained_model = adapter.load_pretrained_model(args.pretrain_checkpoint)
            else:
                logger.info("Creating fresh model for B1 baseline via adapter...")
                pretrained_model = adapter.load_pretrained_model('local-baseline')
            
            logger.info("Model loaded successfully via adapter")
            
        except ImportError as e:
            logger.warning(f"Model adapter not available: {e}")
            # Fallback to original implementation
            
            if args.pretrain_checkpoint is not None:
                logger.info("Loading pre-trained model (fallback method)...")
                try:
                    checkpoint = torch.load(args.pretrain_checkpoint, map_location='cpu', weights_only=False)
                    
                    # Try different import paths
                    try:
                        from models.pretrain_model import create_full_pretrain_model
                    except ImportError:
                        try:
                            from src.models.pretrain_model import create_full_pretrain_model
                        except ImportError:
                            from models import create_full_pretrain_model
                    
                    pretrained_model = create_full_pretrain_model()
                    
                    # Load state dict with error handling
                    if 'model_state_dict' in checkpoint:
                        pretrained_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    else:
                        pretrained_model.load_state_dict(checkpoint, strict=False)
                    
                    logger.info("Pre-trained model loaded successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to load pre-trained model: {e}")
                    logger.info("Creating fresh model as fallback...")
                    pretrained_model = None
            
            if pretrained_model is None:
                # B1: From-scratch training or fallback
                logger.info("Creating fresh model (B1 baseline or fallback)...")
                try:
                    from models.pretrain_model import create_full_pretrain_model
                except ImportError:
                    try:
                        from src.models.pretrain_model import create_full_pretrain_model
                    except ImportError:
                        from models import create_full_pretrain_model
                
                pretrained_model = create_full_pretrain_model()
        
        if pretrained_model is None:
            logger.error("Failed to create or load model")
            return 1
        
        # Create downstream model
        freeze_backbone = (args.strategy == 'linear')
        freeze_encoder = (args.strategy == 'linear' and downstream_config.get('in_domain', False))
        
        downstream_model = DownstreamModel(
            pretrained_model=pretrained_model,
            task_type=downstream_config['task_type'],
            num_classes=downstream_config['num_classes'],
            input_dim=downstream_config.get('input_dim'),  # For out-of-domain tasks
            freeze_backbone=freeze_backbone,
            freeze_encoder=freeze_encoder
        )
        
        # Load downstream data
        from downstream_data_loading import create_downstream_data_loaders
        data_loaders = create_downstream_data_loaders(downstream_config)
        
        if not data_loaders:
            logger.error("No data loaders created - check if data is processed")
            return 1
        
        # Create experiment tracker if not offline
        experiment_tracker = None
        if not args.offline:
            try:
                experiment_tracker = create_experiment_tracker(downstream_config, downstream_model)
            except Exception as e:
                logger.warning(f"Failed to create experiment tracker: {e}")
        
        # Create trainer with enhanced capabilities
        try:
            # Try to use enhanced trainer
            from enhanced_finetune_trainer import EnhancedFineTuningTrainer
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            trainer = EnhancedFineTuningTrainer(
                model=downstream_model,
                train_loader=data_loaders.get('train'),
                val_loader=data_loaders.get('val'),
                test_loader=data_loaders.get('test'),
                config=downstream_config,
                device=device,
                experiment_tracker=experiment_tracker
            )
            
            logger.info("Enhanced trainer created successfully")
            
        except ImportError as e:
            logger.warning(f"Enhanced trainer not available: {e}")
            # Fallback to original trainer
            trainer = DownstreamTrainer(
                model=downstream_model,
                train_loader=data_loaders.get('train'),
                val_loader=data_loaders.get('val'),
                test_loader=data_loaders.get('test'),
                config=downstream_config,
                experiment_tracker=experiment_tracker
            )
            
            logger.info("Original trainer created as fallback")
        
        # Train and evaluate
        results = trainer.train(
            max_epochs=downstream_config.get('max_epochs', 200),
            patience=downstream_config.get('patience', 10)
        )
        
        # Save results
        output_path = Path(args.output_dir) / f"{downstream_config['task_name']}_{args.strategy}_seed{args.seed}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 
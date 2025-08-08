#!/usr/bin/env python3
"""
Simplified Fine-tuning Script for GNN Evaluation.

This script provides a streamlined interface for fine-tuning pre-trained GNN models
on downstream tasks with minimal configuration required.
"""

import sys
import argparse
import logging
import yaml
import torch
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_default_config(dataset_name: str, task_type: str, strategy: str = 'full') -> Dict[str, Any]:
    """Create default configuration for a given dataset and task."""
    
    # Dataset-specific configurations
    dataset_configs = {
        'MUTAG': {'num_classes': 2, 'input_dim': 7, 'in_domain': True, 'batch_size': 32},
        'PROTEINS': {'num_classes': 2, 'input_dim': 4, 'in_domain': True, 'batch_size': 32},
        'NCI1': {'num_classes': 2, 'input_dim': 37, 'in_domain': True, 'batch_size': 32},
        'ENZYMES': {'num_classes': 6, 'input_dim': 21, 'in_domain': True, 'batch_size': 32},
        'FRANKENSTEIN': {'num_classes': 2, 'input_dim': 780, 'in_domain': False, 'batch_size': 16},
        'PTC_MR': {'num_classes': 2, 'input_dim': 18, 'in_domain': False, 'batch_size': 32},
        'Cora': {'num_classes': 7, 'input_dim': 1433, 'in_domain': False, 'batch_size': 1},
        'CiteSeer': {'num_classes': 6, 'input_dim': 3703, 'in_domain': False, 'batch_size': 1}
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_config = dataset_configs[dataset_name]
    
    # Task-specific adjustments
    if task_type == 'link_prediction':
        dataset_config['num_classes'] = 2
        if dataset_config['batch_size'] == 1:
            dataset_config['batch_size'] = 16
    
    # Strategy-specific learning rates
    if strategy == 'linear':
        base_lr = 0.001
        pretrained_lr = 0.0
        unfreeze_epoch = None
    else:  # full
        base_lr = 0.0001 if dataset_config['in_domain'] else 0.0002
        pretrained_lr = base_lr * 0.1
        unfreeze_epoch = 50 if dataset_config['in_domain'] else 75
    
    return {
        'model_artifact': {
            'path': 'checkpoints/s5_domain_invariant/best_model.pt'
        },
        'downstream_task': {
            'dataset_name': dataset_name,
            'task_type': task_type,
            'batch_size': dataset_config['batch_size'],
            'num_classes': dataset_config['num_classes'],
            'in_domain': dataset_config['in_domain'],
            'input_dim': dataset_config['input_dim'] if not dataset_config['in_domain'] else None
        },
        'training': {
            'epochs': 200,
            'patience': 15 if dataset_config['in_domain'] else 20,
            'optimizer': 'AdamW',
            'learning_rate': base_lr,
            'weight_decay': 0.0005,
            'validation_metric': 'val_accuracy',
            'metric_mode': 'max'
        },
        'fine_tuning_strategy': {
            'freeze_encoder': True,
            'unfreeze_epoch': unfreeze_epoch,
            'adaptation_method': strategy
        },
        'wandb': {
            'enabled': True,
            'project_name': f'gnn-simplified-{dataset_name.lower()}',
            'tags': [dataset_name.lower(), task_type, strategy, 'simplified']
        },
        'reproducibility': {
            'seed': 42
        }
    }


def main():
    """Main function for simplified fine-tuning."""
    parser = argparse.ArgumentParser(description='Simplified Fine-tuning for GNN Models')
    
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--task', type=str, help='Task type')
    parser.add_argument('--strategy', type=str, default='full', choices=['full', 'linear'])
    parser.add_argument('--offline', action='store_true', help='Run without WandB')
    parser.add_argument('--dry-run', action='store_true', help='Validate config only')
    
    args = parser.parse_args()
    
    try:
        # Load or create configuration
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        elif args.dataset and args.task:
            config = create_default_config(args.dataset, args.task, args.strategy)
        else:
            parser.error("Either --config or both --dataset and --task required")
        
        if args.dry_run:
            logger.info("Configuration validation:")
            print(yaml.dump(config, default_flow_style=False, indent=2))
            return 0
        
        # Import and run fine-tuning
        from evaluation.main_finetune import main as finetune_main
        
        # Create temporary config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config = f.name
        
        try:
            # Set up arguments (ensure seed is passed if present)
            original_argv = sys.argv
            sys.argv = ['main_finetune.py', '--downstream-config', temp_config, '--strategy', args.strategy]
            if args.offline:
                sys.argv.append('--offline')
            # Pass seed from config if present
            try:
                seed = config.get('reproducibility', {}).get('seed', None)
                if seed is not None:
                    sys.argv.extend(['--seed', str(seed)])
            except Exception:
                pass
            
            results = finetune_main()
            logger.info("✅ Fine-tuning completed successfully!")
            return 0
            
        finally:
            sys.argv = original_argv
            Path(temp_config).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
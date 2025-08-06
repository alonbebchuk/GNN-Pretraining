#!/usr/bin/env python3
"""
Main pre-training script for GNN multi-task learning.

This script orchestrates the complete pre-training pipeline including:
- Configuration loading and validation
- Data loading with domain-balanced sampling
- Model initialization
- Training loop with multi-task losses
- Experiment tracking and checkpointing
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.optim as optim
from typing import Dict, Any

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

try:
    from infrastructure.config import load_config, create_default_config_file
    from data.data_loading import create_data_loaders
    from core.models import create_full_pretrain_model
    from training.losses import MultiTaskLossComputer
    from infrastructure.scheduler import create_lr_scheduler, create_domain_adversarial_scheduler, SchedulerManager
    from infrastructure.experiment_tracking import create_experiment_tracker
    from infrastructure.checkpointing import create_checkpoint_manager
    from training.trainer import PretrainTrainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def create_optimizer(model, optimizer_config):
    """Create optimizer from configuration."""
    # Get all parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_config.name.lower() == 'adam':
        return optim.Adam(
            params,
            lr=optimizer_config.lr,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay
        )
    elif optimizer_config.name.lower() == 'adamw':
        return optim.AdamW(
            params,
            lr=optimizer_config.lr,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay
        )
    elif optimizer_config.name.lower() == 'sgd':
        return optim.SGD(
            params,
            lr=optimizer_config.lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")


def validate_config(config):
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check if data files exist
    data_dir = Path('data/processed')
    if not data_dir.exists():
        issues.append("Processed data directory not found. Run data_setup.py first.")
        return issues
    
    # Check for required dataset files
    for dataset_name in config.data.pretrain_datasets:
        data_file = data_dir / f"{dataset_name}_graphs.pt"
        splits_file = data_dir / f"{dataset_name}_splits.pt"
        
        if not data_file.exists():
            issues.append(f"Dataset file not found: {data_file}")
        if not splits_file.exists():
            issues.append(f"Splits file not found: {splits_file}")
    
    # Check task configuration
    enabled_tasks = [name for name, task in config.tasks.items() if task.enabled]
    if not enabled_tasks:
        issues.append("No tasks are enabled in configuration.")
    
    return issues


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='GNN Pre-training')
    parser.add_argument('--config', type=str,
                       help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--create-default-config', type=str,
                       help='Create default configuration file at specified path and exit')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--offline', action='store_true',
                       help='Run in offline mode (no WandB logging)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration and setup without training')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override random seed from configuration')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create default config if requested
    if args.create_default_config:
        create_default_config_file(args.create_default_config)
        logger.info(f"Default configuration created: {args.create_default_config}")
        return
    
    # Check if config is provided for normal operations
    if not args.config:
        parser.error("--config is required unless using --create-default-config")
    
    logger.info("Starting GNN pre-training...")
    logger.info(f"Configuration file: {args.config}")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # Override seed if provided
        if args.seed is not None:
            logger.info(f"Overriding config seed with: {args.seed}")
            # Set seed for reproducibility
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
            
            # Update run name to include seed
            if hasattr(config, 'run') and hasattr(config.run, 'run_name'):
                config.run.run_name = f"{config.run.run_name}_seed_{args.seed}"
        
        logger.info(f"Configuration loaded successfully")
        logger.info(f"Device: {config.device}")
        logger.info(f"Enabled tasks: {[name for name, task in config.tasks.items() if task.enabled]}")
        
        # Validate configuration
        logger.info("Validating configuration...")
        issues = validate_config(config)
        if issues:
            logger.warning("Configuration issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            
            if any("not found" in issue for issue in issues):
                logger.error("Critical issues found. Please fix them before proceeding.")
                return 1
        
        # Create data loaders
        logger.info("Creating data loaders...")
        data_loaders = create_data_loaders(
            processed_data_dir='data/processed',
            dataset_names=config.data.pretrain_datasets,
            batch_size=config.data.batch_size,
            domain_balanced_sampling=config.data.domain_balanced_sampling,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            shuffle=config.data.shuffle,
            drop_last=config.data.drop_last
        )
        
        if 'train' not in data_loaders:
            logger.error("No training data found!")
            return 1
        
        logger.info(f"Training batches: {len(data_loaders['train'])}")
        if 'val' in data_loaders:
            logger.info(f"Validation batches: {len(data_loaders['val'])}")
        
        # Create model
        logger.info("Creating model...")
        model = create_full_pretrain_model(
            device=config.device,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout_rate=config.model.dropout_rate,
            enable_augmentations=config.model.enable_augmentations
        )
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # Create optimizer
        logger.info("Creating optimizer...")
        optimizer = create_optimizer(model, config.optimizer)
        
        # Create loss computer
        logger.info("Creating loss computer...")
        loss_computer = MultiTaskLossComputer(
            task_configs=config.tasks,
            use_uncertainty_weighting=config.training.use_uncertainty_weighting,
            uncertainty_init=config.training.uncertainty_init
        )
        
        # Add uncertainty parameters to optimizer if using uncertainty weighting
        if config.training.use_uncertainty_weighting:
            uncertainty_params = loss_computer.get_parameters()
            if uncertainty_params:
                optimizer.param_groups.append({
                    'params': uncertainty_params,
                    'lr': config.optimizer.lr * 0.1  # Lower learning rate for uncertainty parameters
                })
                logger.info(f"Added {len(uncertainty_params)} uncertainty parameters to optimizer")
        
        # Create learning rate scheduler
        logger.info("Creating schedulers...")
        total_steps = config.training.max_steps if config.training.max_steps > 0 else config.training.max_epochs * len(data_loaders['train'])
        
        lr_scheduler = create_lr_scheduler(
            optimizer=optimizer,
            scheduler_config=config.scheduler,
            total_steps=total_steps,
            total_epochs=config.training.max_epochs

        )
        
        da_scheduler = None
        if config.domain_adversarial.enabled:
            da_scheduler = create_domain_adversarial_scheduler(
                da_config=config.domain_adversarial,
                total_epochs=config.training.max_epochs
            )
        
        scheduler_manager = SchedulerManager(
            lr_scheduler=lr_scheduler,
            da_scheduler=da_scheduler
        )
        
        # Create experiment tracker
        experiment_tracker = create_experiment_tracker(config, model, offline=args.offline)
        
        # Create checkpoint manager
        checkpoint_manager = create_checkpoint_manager(config)
        
        # Connect experiment tracker with checkpoint manager for artifact logging
        if experiment_tracker:
            experiment_tracker.connect_checkpoint_manager(checkpoint_manager)
        
        if args.dry_run:
            logger.info("Dry run completed successfully. All components initialized correctly.")
            return 0
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = PretrainTrainer(
            config=config,
            model=model,
            train_loader=data_loaders['train'],
            val_loader=data_loaders.get('val'),
            optimizer=optimizer,
            loss_computer=loss_computer,
            scheduler_manager=scheduler_manager,
            checkpoint_manager=checkpoint_manager,
            experiment_tracker=experiment_tracker
        )
        
        # Resume from checkpoint if requested
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.resume_from_checkpoint(args.resume)
        
        # Start training
        logger.info("Starting training...")
        training_results = trainer.train()
        
        # Log final results
        logger.info("Training completed!")
        logger.info(f"Total epochs: {training_results['total_epochs']}")
        logger.info(f"Total steps: {training_results['total_steps']}")
        logger.info(f"Training time: {training_results['training_time']:.2f} seconds")
        logger.info(f"Best validation score: {training_results['best_val_score']:.6f}")
        
        # Close experiment tracker
        if experiment_tracker:
            experiment_tracker.finish()
        
        logger.info("Pre-training pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory! Try reducing batch size or model size.")
            logger.error("Consider using gradient accumulation or mixed precision training.")
        else:
            logger.error(f"Runtime error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1




if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 
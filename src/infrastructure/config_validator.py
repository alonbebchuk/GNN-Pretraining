#!/usr/bin/env python3
"""
Configuration Validator for Fine-tuning Pipeline.

This module provides comprehensive validation of configuration files
to catch common errors early and provide helpful error messages.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import re

# Setup logging
logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigValidator:
    """
    Comprehensive validator for fine-tuning configuration files.
    """
    
    def __init__(self):
        """Initialize the configuration validator."""
        self.errors = []
        self.warnings = []
        
        # Define valid values for different configuration options
        self.valid_datasets = {
            'MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES', 'FRANKENSTEIN', 
            'PTC_MR', 'Cora', 'CiteSeer'
        }
        
        self.valid_task_types = {
            'graph_classification', 'node_classification', 'link_prediction'
        }
        
        self.valid_optimizers = {'Adam', 'AdamW', 'SGD'}
        
        self.valid_adaptation_strategies = {'full', 'linear_probe', 'adapter'}
        
        # Define dataset-specific information
        self.dataset_info = {
            'MUTAG': {'task_type': 'graph_classification', 'num_classes': 2, 'in_domain': True},
            'PROTEINS': {'task_type': 'graph_classification', 'num_classes': 2, 'in_domain': True},
            'NCI1': {'task_type': 'graph_classification', 'num_classes': 2, 'in_domain': True},
            'ENZYMES': {'task_type': 'graph_classification', 'num_classes': 6, 'in_domain': True},
            'FRANKENSTEIN': {'task_type': 'graph_classification', 'num_classes': 2, 'in_domain': False},
            'PTC_MR': {'task_type': 'graph_classification', 'num_classes': 2, 'in_domain': False},
            'Cora': {'task_type': 'node_classification', 'num_classes': 7, 'in_domain': False},
            'CiteSeer': {'task_type': 'node_classification', 'num_classes': 6, 'in_domain': False},
        }
    
    def validate_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a configuration file and return the loaded config.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigValidationError: If validation fails
        """
        self.errors.clear()
        self.warnings.clear()
        
        config_path = Path(config_path)
        
        # Check if file exists
        if not config_path.exists():
            raise ConfigValidationError(f"Configuration file not found: {config_path}")
        
        # Load YAML file
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to load configuration file: {e}")
        
        if config is None:
            raise ConfigValidationError("Configuration file is empty")
        
        # Validate configuration structure and content
        self._validate_structure(config)
        self._validate_model_artifact(config)
        self._validate_downstream_task(config)
        self._validate_training_config(config)
        self._validate_fine_tuning_strategy(config)
        self._validate_wandb_config(config)
        self._validate_cross_field_consistency(config)
        
        # Report errors and warnings
        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join([f"  - {error}" for error in self.errors])
            raise ConfigValidationError(error_msg)
        
        if self.warnings:
            logger.warning("Configuration validation warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        logger.info(f"Configuration validation passed: {config_path}")
        return config
    
    def _validate_structure(self, config: Dict[str, Any]):
        """Validate the overall structure of the configuration."""
        required_sections = ['model_artifact', 'downstream_task', 'training', 'fine_tuning_strategy', 'wandb']
        
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section: '{section}'")
            elif not isinstance(config[section], dict):
                self.errors.append(f"Section '{section}' must be a dictionary")
    
    def _validate_model_artifact(self, config: Dict[str, Any]):
        """Validate model artifact configuration."""
        if 'model_artifact' not in config:
            return
        
        artifact_config = config['model_artifact']
        
        # Check required fields
        if 'path' not in artifact_config:
            self.errors.append("model_artifact.path is required")
            return
        
        artifact_path = artifact_config['path']
        
        # Validate artifact path format
        if isinstance(artifact_path, str):
            # Check if it's a local path or WandB artifact path
            if artifact_path.startswith('local-'):
                self.warnings.append(f"Using local artifact path: {artifact_path}")
            elif '/' in artifact_path and ':' in artifact_path:
                # WandB artifact format: entity/project/artifact:version
                parts = artifact_path.split('/')
                if len(parts) < 3:
                    self.errors.append(f"Invalid WandB artifact path format: {artifact_path}")
                elif ':' not in parts[-1]:
                    self.errors.append(f"WandB artifact path missing version: {artifact_path}")
            else:
                self.warnings.append(f"Artifact path format may be invalid: {artifact_path}")
        else:
            self.errors.append("model_artifact.path must be a string")
    
    def _validate_downstream_task(self, config: Dict[str, Any]):
        """Validate downstream task configuration."""
        if 'downstream_task' not in config:
            return
        
        task_config = config['downstream_task']
        
        # Required fields
        required_fields = ['dataset_name', 'task_type', 'batch_size']
        for field in required_fields:
            if field not in task_config:
                self.errors.append(f"downstream_task.{field} is required")
        
        # Validate dataset name
        if 'dataset_name' in task_config:
            dataset_name = task_config['dataset_name']
            if dataset_name not in self.valid_datasets:
                self.errors.append(f"Unknown dataset: {dataset_name}. Valid options: {sorted(self.valid_datasets)}")
        
        # Validate task type
        if 'task_type' in task_config:
            task_type = task_config['task_type']
            if task_type not in self.valid_task_types:
                self.errors.append(f"Unknown task type: {task_type}. Valid options: {sorted(self.valid_task_types)}")
        
        # Validate batch size
        if 'batch_size' in task_config:
            batch_size = task_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                self.errors.append("downstream_task.batch_size must be a positive integer")
            elif batch_size > 128:
                self.warnings.append(f"Large batch size ({batch_size}) may cause memory issues")
        
        # Cross-validate dataset and task type
        if 'dataset_name' in task_config and 'task_type' in task_config:
            dataset_name = task_config['dataset_name']
            task_type = task_config['task_type']
            
            if dataset_name in self.dataset_info:
                expected_task_type = self.dataset_info[dataset_name]['task_type']
                if task_type != expected_task_type:
                    self.errors.append(f"Task type mismatch: {dataset_name} should use {expected_task_type}, not {task_type}")
    
    def _validate_training_config(self, config: Dict[str, Any]):
        """Validate training configuration."""
        if 'training' not in config:
            return
        
        training_config = config['training']
        
        # Required fields
        required_fields = ['epochs', 'optimizer', 'learning_rate']
        for field in required_fields:
            if field not in training_config:
                self.errors.append(f"training.{field} is required")
        
        # Validate epochs
        if 'epochs' in training_config:
            epochs = training_config['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                self.errors.append("training.epochs must be a positive integer")
            elif epochs > 1000:
                self.warnings.append(f"Very large number of epochs ({epochs}) may be unnecessary")
        
        # Validate optimizer
        if 'optimizer' in training_config:
            optimizer = training_config['optimizer']
            if optimizer not in self.valid_optimizers:
                self.errors.append(f"Unknown optimizer: {optimizer}. Valid options: {sorted(self.valid_optimizers)}")
        
        # Validate learning rate
        if 'learning_rate' in training_config:
            lr = training_config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append("training.learning_rate must be a positive number")
            elif lr > 0.1:
                self.warnings.append(f"Very large learning rate ({lr}) may cause training instability")
            elif lr < 1e-6:
                self.warnings.append(f"Very small learning rate ({lr}) may cause slow convergence")
        
        # Validate weight decay
        if 'weight_decay' in training_config:
            wd = training_config['weight_decay']
            if not isinstance(wd, (int, float)) or wd < 0:
                self.errors.append("training.weight_decay must be a non-negative number")
            elif wd > 0.1:
                self.warnings.append(f"Very large weight decay ({wd}) may hurt performance")
    
    def _validate_fine_tuning_strategy(self, config: Dict[str, Any]):
        """Validate fine-tuning strategy configuration."""
        if 'fine_tuning_strategy' not in config:
            return
        
        strategy_config = config['fine_tuning_strategy']
        
        # Validate freeze_encoder
        if 'freeze_encoder' in strategy_config:
            freeze_encoder = strategy_config['freeze_encoder']
            if not isinstance(freeze_encoder, bool):
                self.errors.append("fine_tuning_strategy.freeze_encoder must be a boolean")
        
        # Validate unfreeze_epoch
        if 'unfreeze_epoch' in strategy_config:
            unfreeze_epoch = strategy_config['unfreeze_epoch']
            if not isinstance(unfreeze_epoch, int) or unfreeze_epoch < 0:
                self.errors.append("fine_tuning_strategy.unfreeze_epoch must be a non-negative integer")
            
            # Cross-validate with training epochs
            if 'training' in config and 'epochs' in config['training']:
                total_epochs = config['training']['epochs']
                if unfreeze_epoch >= total_epochs:
                    self.warnings.append(f"unfreeze_epoch ({unfreeze_epoch}) >= total epochs ({total_epochs})")
        
        # Validate adaptation strategy
        if 'adaptation_strategy' in strategy_config:
            adaptation_strategy = strategy_config['adaptation_strategy']
            if adaptation_strategy not in self.valid_adaptation_strategies:
                self.errors.append(f"Unknown adaptation strategy: {adaptation_strategy}. Valid options: {sorted(self.valid_adaptation_strategies)}")
        
        # Logical consistency checks
        if ('freeze_encoder' in strategy_config and 
            'unfreeze_epoch' in strategy_config):
            
            freeze_encoder = strategy_config['freeze_encoder']
            unfreeze_epoch = strategy_config['unfreeze_epoch']
            
            if not freeze_encoder and unfreeze_epoch > 0:
                self.warnings.append("unfreeze_epoch specified but freeze_encoder is False")
    
    def _validate_wandb_config(self, config: Dict[str, Any]):
        """Validate WandB configuration."""
        if 'wandb' not in config:
            return
        
        wandb_config = config['wandb']
        
        # Required fields
        required_fields = ['project_name', 'run_name']
        for field in required_fields:
            if field not in wandb_config:
                self.errors.append(f"wandb.{field} is required")
        
        # Validate project name
        if 'project_name' in wandb_config:
            project_name = wandb_config['project_name']
            if not isinstance(project_name, str) or not project_name.strip():
                self.errors.append("wandb.project_name must be a non-empty string")
        
        # Validate run name
        if 'run_name' in wandb_config:
            run_name = wandb_config['run_name']
            if not isinstance(run_name, str) or not run_name.strip():
                self.errors.append("wandb.run_name must be a non-empty string")
            elif len(run_name) > 128:
                self.warnings.append(f"Very long run name ({len(run_name)} chars) may be truncated")
        
        # Validate entity (optional)
        if 'entity' in wandb_config:
            entity = wandb_config['entity']
            if entity is not None and (not isinstance(entity, str) or not entity.strip()):
                self.errors.append("wandb.entity must be a non-empty string or null")
    
    def _validate_cross_field_consistency(self, config: Dict[str, Any]):
        """Validate consistency across different configuration sections."""
        
        # Check dataset and task type consistency (already done in _validate_downstream_task)
        
        # Check learning rate and optimizer consistency
        if ('training' in config and 
            'optimizer' in config['training'] and 
            'learning_rate' in config['training']):
            
            optimizer = config['training']['optimizer']
            lr = config['training']['learning_rate']
            
            # Provide optimizer-specific learning rate recommendations
            if optimizer == 'SGD' and lr < 0.01:
                self.warnings.append(f"Learning rate {lr} may be too small for SGD optimizer")
            elif optimizer in ['Adam', 'AdamW'] and lr > 0.01:
                self.warnings.append(f"Learning rate {lr} may be too large for {optimizer} optimizer")
        
        # Check in-domain vs out-of-domain task configuration
        if 'downstream_task' in config and 'dataset_name' in config['downstream_task']:
            dataset_name = config['downstream_task']['dataset_name']
            
            if dataset_name in self.dataset_info:
                is_in_domain = self.dataset_info[dataset_name]['in_domain']
                
                if ('fine_tuning_strategy' in config and 
                    'freeze_encoder' in config['fine_tuning_strategy']):
                    
                    freeze_encoder = config['fine_tuning_strategy']['freeze_encoder']
                    
                    if is_in_domain and freeze_encoder:
                        self.warnings.append(f"In-domain dataset {dataset_name} may benefit from unfrozen encoder")
                    elif not is_in_domain and not freeze_encoder:
                        self.warnings.append(f"Out-of-domain dataset {dataset_name} may benefit from frozen encoder initially")
        
        # Check artifact path and run name consistency
        if ('model_artifact' in config and 'path' in config['model_artifact'] and
            'wandb' in config and 'run_name' in config['wandb']):
            
            artifact_path = config['model_artifact']['path']
            run_name = config['wandb']['run_name']
            
            # Extract model identifier from artifact path
            if ':' in artifact_path:
                model_version = artifact_path.split(':')[-1]
                if model_version not in run_name:
                    self.warnings.append(f"Run name may not reflect model version {model_version}")
    
    def validate_config_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ConfigValidationError: If validation fails
        """
        self.errors.clear()
        self.warnings.clear()
        
        if not isinstance(config, dict):
            raise ConfigValidationError("Configuration must be a dictionary")
        
        # Run all validation checks
        self._validate_structure(config)
        self._validate_model_artifact(config)
        self._validate_downstream_task(config)
        self._validate_training_config(config)
        self._validate_fine_tuning_strategy(config)
        self._validate_wandb_config(config)
        self._validate_cross_field_consistency(config)
        
        # Report errors and warnings
        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join([f"  - {error}" for error in self.errors])
            raise ConfigValidationError(error_msg)
        
        if self.warnings:
            logger.warning("Configuration validation warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        return config
    
    def create_template_config(self, output_path: Union[str, Path], 
                             dataset_name: str = 'MUTAG',
                             model_artifact: str = 'entity/project/model:v1') -> None:
        """
        Create a template configuration file.
        
        Args:
            output_path: Path to save the template configuration
            dataset_name: Dataset name to use in template
            model_artifact: Model artifact path to use in template
        """
        if dataset_name not in self.dataset_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.dataset_info[dataset_name]
        
        template_config = {
            'model_artifact': {
                'path': model_artifact
            },
            'downstream_task': {
                'dataset_name': dataset_name,
                'task_type': dataset_info['task_type'],
                'batch_size': 32 if dataset_info['task_type'] == 'graph_classification' else 1
            },
            'training': {
                'epochs': 100,
                'optimizer': 'Adam',
                'learning_rate': 0.0001,
                'weight_decay': 5.0e-4
            },
            'fine_tuning_strategy': {
                'freeze_encoder': True,
                'unfreeze_epoch': 10,
                'adaptation_strategy': 'full'
            },
            'wandb': {
                'project_name': 'Graph-Finetuning-Results',
                'run_name': f'eval-{dataset_name.lower()}-{model_artifact.split("/")[-1].replace(":", "_")}',
                'entity': 'your-wandb-entity'
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(template_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Template configuration created: {output_path}")


def validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    validator = ConfigValidator()
    return validator.validate_config_file(config_path)


def create_config_template(output_path: Union[str, Path], 
                          dataset_name: str = 'MUTAG',
                          model_artifact: str = 'entity/project/model:v1') -> None:
    """
    Convenience function to create a configuration template.
    
    Args:
        output_path: Path to save template
        dataset_name: Dataset name for template
        model_artifact: Model artifact path for template
    """
    validator = ConfigValidator()
    validator.create_template_config(output_path, dataset_name, model_artifact)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration validator for fine-tuning pipeline')
    parser.add_argument('--validate', type=str, help='Path to configuration file to validate')
    parser.add_argument('--create-template', type=str, help='Path to create template configuration')
    parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset name for template')
    parser.add_argument('--model-artifact', type=str, default='entity/project/model:v1', 
                       help='Model artifact path for template')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    if args.validate:
        try:
            config = validate_config(args.validate)
            print(f"✅ Configuration validation passed: {args.validate}")
        except ConfigValidationError as e:
            print(f"❌ Configuration validation failed: {e}")
            sys.exit(1)
    
    elif args.create_template:
        try:
            create_config_template(args.create_template, args.dataset, args.model_artifact)
            print(f"✅ Template configuration created: {args.create_template}")
        except Exception as e:
            print(f"❌ Failed to create template: {e}")
            sys.exit(1)
    
    else:
        parser.print_help() 
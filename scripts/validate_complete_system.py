#!/usr/bin/env python3
"""
Complete System Validation Script.

This script performs comprehensive validation of the entire GNN pre-training and
fine-tuning system to ensure everything is ready for production use.

Usage:
    python validate_complete_system.py [--quick] [--fix-issues]
"""

import argparse
import logging
import sys
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import importlib
import torch
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


class SystemValidator:
    """
    Comprehensive system validator for the GNN research pipeline.
    """
    
    def __init__(self, fix_issues: bool = False):
        """
        Initialize system validator.
        
        Args:
            fix_issues: Whether to attempt automatic fixes for detected issues
        """
        self.fix_issues = fix_issues
        self.validation_results = {}
        self.issues_found = []
        self.fixes_applied = []
        
        logger.info("System validator initialized")
        logger.info(f"Auto-fix mode: {self.fix_issues}")
    
    def validate_directory_structure(self) -> bool:
        """Validate project directory structure."""
        logger.info("Validating directory structure...")
        
        required_dirs = [
            'src',
            'src/models',
            'configs',
            'configs/pretrain',
            'configs/finetune',
            'data',
            'data/processed',
            'results'
        ]
        
        required_files = [
            'src/data_setup.py',
            'src/models/gnn.py',
            'src/models/heads.py',
            'src/models/pretrain_model.py',
            'src/main_pretrain.py',
            'src/main_finetune.py',
            'src/trainer.py',
            'src/losses.py',
            'src/config.py',
            'requirements.txt'
        ]
        
        missing_dirs = []
        missing_files = []
        
        # Check directories
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        # Check files
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_dirs or missing_files:
            logger.error("Missing required directories/files:")
            for d in missing_dirs:
                logger.error(f"  Missing directory: {d}")
            for f in missing_files:
                logger.error(f"  Missing file: {f}")
            
            if self.fix_issues:
                self._fix_directory_structure(missing_dirs, missing_files)
            
            return False
        
        logger.info("✅ Directory structure validation passed")
        return True
    
    def _fix_directory_structure(self, missing_dirs: List[str], missing_files: List[str]):
        """Attempt to fix directory structure issues."""
        logger.info("Attempting to fix directory structure...")
        
        # Create missing directories
        for dir_path in missing_dirs:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
                self.fixes_applied.append(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
        
        # Create missing __init__.py files for Python packages
        for file_path in missing_files:
            if file_path.endswith('__init__.py'):
                try:
                    Path(file_path).touch()
                    logger.info(f"Created file: {file_path}")
                    self.fixes_applied.append(f"Created file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to create file {file_path}: {e}")
    
    def validate_python_imports(self) -> bool:
        """Validate that all Python modules can be imported."""
        logger.info("Validating Python imports...")
        
        core_modules = [
            'src.config',
            'src.data_setup',
            'src.models.gnn',
            'src.models.heads',
            'src.models.pretrain_model',
            'src.trainer',
            'src.losses',
            'src.scheduler',
            'src.experiment_tracking',
            'src.checkpointing'
        ]
        
        optional_modules = [
            'src.model_adapter',
            'src.evaluation_metrics',
            'src.downstream_data_loading',
            'src.config_validator',
            'src.link_prediction'
        ]
        
        failed_imports = []
        optional_failed = []
        
        # Test core modules
        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                logger.debug(f"✅ {module_name}")
            except ImportError as e:
                logger.error(f"❌ {module_name}: {e}")
                failed_imports.append((module_name, str(e)))
        
        # Test optional modules
        for module_name in optional_modules:
            try:
                importlib.import_module(module_name)
                logger.debug(f"✅ {module_name} (optional)")
            except ImportError as e:
                logger.warning(f"⚠️  {module_name} (optional): {e}")
                optional_failed.append((module_name, str(e)))
        
        if failed_imports:
            logger.error("Critical import failures:")
            for module, error in failed_imports:
                logger.error(f"  {module}: {error}")
            return False
        
        if optional_failed:
            logger.info(f"Optional modules not available: {len(optional_failed)}")
            for module, error in optional_failed:
                logger.info(f"  {module}: {error}")
        
        logger.info("✅ Python imports validation passed")
        return True
    
    def validate_configurations(self) -> bool:
        """Validate configuration files."""
        logger.info("Validating configuration files...")
        
        # Check pre-training configs
        pretrain_configs = list(Path('configs/pretrain').glob('*.yaml'))
        finetune_configs = list(Path('configs/finetune').glob('**/*.yaml'))
        
        if not pretrain_configs:
            logger.error("No pre-training configuration files found")
            if self.fix_issues:
                self._create_sample_pretrain_configs()
            return False
        
        if not finetune_configs:
            logger.error("No fine-tuning configuration files found")
            if self.fix_issues:
                self._create_sample_finetune_configs()
            return False
        
        # Validate configuration syntax
        invalid_configs = []
        
        for config_file in pretrain_configs + finetune_configs:
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                logger.debug(f"✅ {config_file}")
            except Exception as e:
                logger.error(f"❌ {config_file}: {e}")
                invalid_configs.append((config_file, str(e)))
        
        if invalid_configs:
            logger.error("Invalid configuration files:")
            for config, error in invalid_configs:
                logger.error(f"  {config}: {error}")
            return False
        
        logger.info(f"✅ Configuration validation passed ({len(pretrain_configs)} pretrain, {len(finetune_configs)} finetune)")
        return True
    
    def _create_sample_pretrain_configs(self):
        """Create sample pre-training configuration files."""
        logger.info("Creating sample pre-training configurations...")
        
        # Create configs/pretrain directory
        Path('configs/pretrain').mkdir(parents=True, exist_ok=True)
        
        # Sample S5 config (most comprehensive)
        s5_config = {
            'run': {
                'project_name': 'Graph-Multitask-Learning',
                'run_name': 's5_domain_invariant',
                'tags': ['multi-task', 'domain-invariant']
            },
            'data': {
                'pretrain_datasets': ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES'],
                'batch_size': 32,
                'domain_balanced_sampling': True
            },
            'model': {
                'hidden_dim': 256,
                'num_layers': 5,
                'dropout_rate': 0.2
            },
            'training': {
                'max_steps': 100000,
                'patience': 10,
                'validation_metric': 'val/loss_total'
            },
            'tasks': {
                'node_feat_mask': {'enabled': True, 'weight': 1.0},
                'link_pred': {'enabled': True, 'weight': 1.0},
                'node_contrast': {'enabled': True, 'weight': 1.0},
                'graph_contrast': {'enabled': True, 'weight': 1.0},
                'graph_prop': {'enabled': True, 'weight': 1.0},
                'domain_adv': {'enabled': True, 'weight': 1.0}
            }
        }
        
        with open('configs/pretrain/s5_domain_invariant.yaml', 'w') as f:
            yaml.dump(s5_config, f, default_flow_style=False)
        
        self.fixes_applied.append("Created sample pre-training configuration")
    
    def _create_sample_finetune_configs(self):
        """Create sample fine-tuning configuration files."""
        logger.info("Creating sample fine-tuning configurations...")
        
        # Create directory structure
        for task_type in ['graph_classification', 'node_classification', 'link_prediction']:
            Path(f'configs/finetune/{task_type}').mkdir(parents=True, exist_ok=True)
        
        # Sample MUTAG config
        mutag_config = {
            'model_artifact': {'path': 'local-baseline'},
            'downstream_task': {
                'dataset_name': 'MUTAG',
                'task_type': 'graph_classification',
                'batch_size': 32
            },
            'training': {
                'epochs': 200,
                'optimizer': 'AdamW',
                'learning_rate': 0.0001
            },
            'fine_tuning_strategy': {
                'freeze_encoder': True,
                'unfreeze_epoch': 10
            },
            'wandb': {
                'project_name': 'gnn-finetuning',
                'run_name': 'mutag-test'
            }
        }
        
        with open('configs/finetune/graph_classification/mutag.yaml', 'w') as f:
            yaml.dump(mutag_config, f, default_flow_style=False)
        
        self.fixes_applied.append("Created sample fine-tuning configuration")
    
    def validate_data_processing(self) -> bool:
        """Validate data processing capabilities."""
        logger.info("Validating data processing...")
        
        try:
            # Check if processed data exists
            processed_dir = Path('data/processed')
            if not processed_dir.exists():
                logger.warning("Processed data directory does not exist")
                if self.fix_issues:
                    processed_dir.mkdir(parents=True, exist_ok=True)
                    self.fixes_applied.append("Created processed data directory")
            
            # Test data setup import
            from data_setup import save_processed_data, validate_processed_data
            logger.info("✅ Data processing modules imported successfully")
            
            # Check if we can create dummy processed data
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create dummy data
                dummy_data = [torch.randn(10, 5) for _ in range(5)]  # 5 graphs with 10 nodes, 5 features
                dummy_splits = {'train': torch.tensor([0, 1, 2]), 'val': torch.tensor([3, 4])}
                
                save_processed_data('TEST_DATASET', dummy_data, dummy_splits, None)
                logger.info("✅ Data processing validation passed")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Data processing validation failed: {e}")
            return False
    
    def validate_model_architecture(self) -> bool:
        """Validate model architecture components."""
        logger.info("Validating model architecture...")
        
        try:
            # Test model imports
            from models.gnn import InputEncoder, GINLayer, GIN_Backbone
            from models.heads import MLPHead, DotProductDecoder, BilinearDiscriminator
            from models.pretrain_model import PretrainableGNN, create_full_pretrain_model
            
            # Test model creation
            device = torch.device('cpu')
            model = create_full_pretrain_model(device=device)
            
            if model is None:
                logger.error("Failed to create pre-trainable model")
                return False
            
            # Test forward pass with dummy data
            dummy_data = type('Data', (), {
                'x': torch.randn(10, 7),  # 10 nodes, 7 features (like MUTAG)
                'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]]),  # Simple edges
                'num_nodes': 10
            })()
            
            outputs = model(dummy_data, 'MUTAG')
            
            if 'node_embeddings' not in outputs or 'graph_embedding' not in outputs:
                logger.error("Model output format is incorrect")
                return False
            
            logger.info("✅ Model architecture validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model architecture validation failed: {e}")
            return False
    
    def validate_training_components(self) -> bool:
        """Validate training pipeline components."""
        logger.info("Validating training components...")
        
        try:
            # Test training imports
            from trainer import PretrainTrainer
            from losses import MultiTaskLossComputer
            from scheduler import SchedulerManager
            from checkpointing import CheckpointManager
            from experiment_tracking import WandBTracker
            
            logger.info("✅ Training components imported successfully")
            
            # Test configuration loading
            from config import Config, load_config
            
            # Create temporary config for testing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                test_config = {
                    'run': {'project_name': 'test'},
                    'data': {'batch_size': 4},
                    'model': {'hidden_dim': 64},
                    'training': {'max_epochs': 1}
                }
                yaml.dump(test_config, f)
                temp_config_path = f.name
            
            try:
                config = load_config(temp_config_path)
                logger.info("✅ Configuration loading works")
            finally:
                Path(temp_config_path).unlink(missing_ok=True)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training components validation failed: {e}")
            return False
    
    def validate_evaluation_pipeline(self) -> bool:
        """Validate evaluation and fine-tuning pipeline."""
        logger.info("Validating evaluation pipeline...")
        
        try:
            # Test fine-tuning script existence
            finetune_scripts = [
                'src/main_finetune.py',
                'finetune_simplified.py'
            ]
            
            available_scripts = [script for script in finetune_scripts if Path(script).exists()]
            
            if not available_scripts:
                logger.error("No fine-tuning scripts found")
                return False
            
            logger.info(f"✅ Fine-tuning scripts available: {available_scripts}")
            
            # Test enhanced components (optional)
            optional_components = [
                'model_adapter',
                'evaluation_metrics',
                'downstream_data_loading',
                'link_prediction'
            ]
            
            available_optional = []
            for component in optional_components:
                try:
                    importlib.import_module(f'src.{component}')
                    available_optional.append(component)
                except ImportError:
                    continue
            
            if available_optional:
                logger.info(f"✅ Enhanced components available: {available_optional}")
            else:
                logger.warning("No enhanced evaluation components available")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Evaluation pipeline validation failed: {e}")
            return False
    
    def validate_system_integration(self) -> bool:
        """Validate complete system integration."""
        logger.info("Validating system integration...")
        
        try:
            # Test integration test script
            if Path('test_complete_evaluation.py').exists():
                logger.info("Running integration tests...")
                result = subprocess.run(
                    ['python', 'test_complete_evaluation.py'],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    logger.info("✅ Integration tests passed")
                    return True
                else:
                    logger.error(f"❌ Integration tests failed: {result.stderr}")
                    return False
            else:
                logger.warning("Integration test script not found, skipping")
                return True
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Integration tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ System integration validation failed: {e}")
            return False
    
    def run_comprehensive_validation(self, quick: bool = False) -> Dict[str, bool]:
        """
        Run comprehensive system validation.
        
        Args:
            quick: If True, skip time-consuming tests
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting comprehensive system validation...")
        
        validations = [
            ("Directory Structure", self.validate_directory_structure),
            ("Python Imports", self.validate_python_imports),
            ("Configuration Files", self.validate_configurations),
            ("Data Processing", self.validate_data_processing),
            ("Model Architecture", self.validate_model_architecture),
            ("Training Components", self.validate_training_components),
            ("Evaluation Pipeline", self.validate_evaluation_pipeline),
        ]
        
        if not quick:
            validations.append(("System Integration", self.validate_system_integration))
        
        results = {}
        
        for name, validation_func in validations:
            logger.info(f"\n--- {name} ---")
            try:
                results[name] = validation_func()
            except Exception as e:
                logger.error(f"Validation '{name}' crashed: {e}")
                results[name] = False
                self.issues_found.append(f"{name}: {str(e)}")
        
        self.validation_results = results
        return results
    
    def print_summary(self):
        """Print validation summary."""
        logger.info("\n" + "="*80)
        logger.info("SYSTEM VALIDATION SUMMARY")
        logger.info("="*80)
        
        total_validations = len(self.validation_results)
        passed_validations = sum(self.validation_results.values())
        
        logger.info(f"Total validations: {total_validations}")
        logger.info(f"Passed: {passed_validations}")
        logger.info(f"Failed: {total_validations - passed_validations}")
        logger.info(f"Success rate: {passed_validations/total_validations*100:.1f}%")
        
        # Print individual results
        logger.info("\nDetailed Results:")
        for name, result in self.validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {name:.<40} {status}")
        
        # Print issues found
        if self.issues_found:
            logger.info(f"\nIssues Found ({len(self.issues_found)}):")
            for issue in self.issues_found:
                logger.info(f"  - {issue}")
        
        # Print fixes applied
        if self.fixes_applied:
            logger.info(f"\nFixes Applied ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                logger.info(f"  - {fix}")
        
        # Overall status
        if passed_validations == total_validations:
            logger.info("\n🎉 SYSTEM IS READY FOR PRODUCTION!")
            logger.info("All validations passed. You can proceed with experiments.")
        else:
            logger.info(f"\n⚠️  SYSTEM NEEDS ATTENTION")
            logger.info(f"{total_validations - passed_validations} validation(s) failed.")
            logger.info("Please address the issues before running experiments.")
        
        logger.info("="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Validate complete GNN system')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation (skip time-consuming tests)')
    parser.add_argument('--fix-issues', action='store_true',
                       help='Attempt to automatically fix detected issues')
    
    args = parser.parse_args()
    
    # Create validator
    validator = SystemValidator(fix_issues=args.fix_issues)
    
    # Run validation
    results = validator.run_comprehensive_validation(quick=args.quick)
    
    # Print summary
    validator.print_summary()
    
    # Return appropriate exit code
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
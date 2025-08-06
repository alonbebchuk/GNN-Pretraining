#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for the Complete Research Study.

This script orchestrates the execution of all fine-tuning experiments as specified
in the research plan, including all pre-training schemes, downstream tasks, and
fine-tuning strategies.

Usage:
    python run_comprehensive_evaluation.py [--dry-run] [--subset] [--parallel]
"""

import argparse
import logging
import sys
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experimental schemes as defined in plan.md
PRE_TRAINING_SCHEMES = {
    'B1': {'name': 'From-Scratch', 'checkpoint': None},  # No pre-training
    'B2': {'name': 'Single-Task (Generative)', 'checkpoint': 'b2_single_generative'},
    'B3': {'name': 'Single-Task (Contrastive)', 'checkpoint': 'b3_single_contrastive'},
    'B4': {'name': 'Single-Domain (All Objectives)', 'checkpoint': 'b4_single_domain'},
    'S1': {'name': 'Multi-Task (Generative)', 'checkpoint': 's1_multi_generative'},
    'S2': {'name': 'Multi-Task (Contrastive)', 'checkpoint': 's2_multi_contrastive'},
    'S3': {'name': 'Multi-Task (All Self-Supervised)', 'checkpoint': 's3_all_self_supervised'},
    'S4': {'name': 'Multi-Task (All Objectives)', 'checkpoint': 's4_all_objectives'},
    'S5': {'name': 'Multi-Task (Domain-Invariant)', 'checkpoint': 's5_domain_invariant'}
}

# Downstream tasks as defined in plan.md
DOWNSTREAM_TASKS = {
    # Graph classification (in-domain and out-of-domain)
    'ENZYMES': {
        'dataset_name': 'ENZYMES',
        'task_type': 'graph_classification',
        'num_classes': 6,
        'in_domain': True,
        'batch_size': 32
    },
    'FRANKENSTEIN': {
        'dataset_name': 'FRANKENSTEIN',
        'task_type': 'graph_classification',
        'num_classes': 2,
        'in_domain': False,
        'batch_size': 16  # Smaller batch for large graphs
    },
    'PTC_MR': {
        'dataset_name': 'PTC_MR',
        'task_type': 'graph_classification',
        'num_classes': 2,
        'in_domain': False,
        'batch_size': 32
    },
    
    # Node classification (out-of-domain)
    'Cora': {
        'dataset_name': 'Cora',
        'task_type': 'node_classification',
        'num_classes': 7,
        'in_domain': False,
        'batch_size': 1  # Full-batch for node classification
    },
    'CiteSeer': {
        'dataset_name': 'CiteSeer',
        'task_type': 'node_classification',
        'num_classes': 6,
        'in_domain': False,
        'batch_size': 1  # Full-batch for node classification
    },
    
    # Link prediction (out-of-domain)
    'Cora_LP': {
        'dataset_name': 'Cora',
        'task_type': 'link_prediction',
        'num_classes': 2,  # Binary classification
        'in_domain': False,
        'batch_size': 16
    },
    'CiteSeer_LP': {
        'dataset_name': 'CiteSeer',
        'task_type': 'link_prediction',
        'num_classes': 2,  # Binary classification
        'in_domain': False,
        'batch_size': 16
    }
}

# Fine-tuning strategies as defined in plan.md
FINE_TUNING_STRATEGIES = ['full', 'linear']

# Random seeds for statistical robustness
RANDOM_SEEDS = [42, 84, 126]


class ExperimentRunner:
    """
    Manages the execution of comprehensive evaluation experiments.
    """
    
    def __init__(self, output_dir: str = 'results/comprehensive_evaluation',
                 checkpoint_dir: str = 'checkpoints',
                 config_dir: str = 'configs/finetune',
                 dry_run: bool = False):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory to save results
            checkpoint_dir: Directory containing pre-trained checkpoints
            config_dir: Directory containing fine-tuning configs
            dry_run: If True, only validate setup without running experiments
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_dir = Path(config_dir)
        self.dry_run = dry_run
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment tracking
        self.experiment_log = []
        self.failed_experiments = []
        
        logger.info(f"Experiment runner initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Config directory: {self.config_dir}")
        logger.info(f"Dry run mode: {self.dry_run}")
    
    def create_experiment_config(self, scheme_id: str, task_name: str, 
                               strategy: str, seed: int) -> Dict[str, Any]:
        """
        Create configuration for a specific experiment.
        
        Args:
            scheme_id: Pre-training scheme ID (B1, B2, ..., S5)
            task_name: Downstream task name
            strategy: Fine-tuning strategy
            seed: Random seed
            
        Returns:
            Configuration dictionary
        """
        scheme_info = PRE_TRAINING_SCHEMES[scheme_id]
        task_info = DOWNSTREAM_TASKS[task_name]
        
        # Base configuration
        config = {
            'model_artifact': {
                'path': f'{self.checkpoint_dir}/{scheme_info["checkpoint"]}/best_model.pt' if scheme_info['checkpoint'] else 'local-baseline'
            },
            'downstream_task': {
                'dataset_name': task_info['dataset_name'],
                'task_type': task_info['task_type'],
                'batch_size': task_info['batch_size'],
                'num_classes': task_info['num_classes'],
                'in_domain': task_info['in_domain']
            },
            'training': {
                'epochs': 200,
                'patience': 10,
                'optimizer': 'AdamW',
                'learning_rate': 1e-4,
                'weight_decay': 5e-4,
                'scheduler': {
                    'type': 'cosine',
                    'warmup_epochs': 20,
                    'min_lr_ratio': 0.01
                }
            },
            'fine_tuning_strategy': {
                'freeze_encoder': strategy == 'linear',
                'unfreeze_epoch': 50 if strategy == 'full' else None,
                'adaptation_method': strategy
            },
            'wandb': {
                'enabled': True,
                'project_name': 'gnn-comprehensive-evaluation',
                'run_name': f'{scheme_id}_{task_name}_{strategy}_seed{seed}',
                'tags': [scheme_id, task_name, strategy, f'seed{seed}']
            },
            'reproducibility': {
                'seed': seed
            }
        }
        
        # Task-specific adjustments
        if task_info['task_type'] == 'node_classification':
            config['training']['batch_size'] = 1  # Full-batch
            
        elif task_info['task_type'] == 'link_prediction':
            config['downstream_task']['link_prediction'] = {
                'negative_sampling_ratio': 1.0,
                'sampling_strategy': 'uniform'
            }
        
        # Strategy-specific adjustments
        if strategy == 'linear':
            config['training']['learning_rate'] = 1e-3  # Higher LR for linear probing
            config['fine_tuning_strategy']['freeze_encoder'] = True
        
        # In-domain vs out-of-domain adjustments
        if not task_info['in_domain']:
            # Out-of-domain tasks may need different settings
            config['fine_tuning_strategy']['pretrained_component_lr'] = 1e-5
            config['fine_tuning_strategy']['new_component_lr'] = 1e-3
        
        return config
    
    def run_single_experiment(self, scheme_id: str, task_name: str, 
                            strategy: str, seed: int) -> Dict[str, Any]:
        """
        Run a single fine-tuning experiment.
        
        Args:
            scheme_id: Pre-training scheme ID
            task_name: Downstream task name
            strategy: Fine-tuning strategy
            seed: Random seed
            
        Returns:
            Experiment results dictionary
        """
        experiment_id = f'{scheme_id}_{task_name}_{strategy}_seed{seed}'
        logger.info(f"Running experiment: {experiment_id}")
        
        start_time = time.time()
        
        try:
            # Create experiment configuration
            config = self.create_experiment_config(scheme_id, task_name, strategy, seed)
            
            # Save configuration
            config_path = self.output_dir / f'config_{experiment_id}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            if self.dry_run:
                logger.info(f"Dry run: {experiment_id} - configuration created")
                return {
                    'experiment_id': experiment_id,
                    'status': 'dry_run_success',
                    'config_path': str(config_path),
                    'duration': 0
                }
            
            # Run fine-tuning
            cmd = [
                'python', 'finetune_simplified.py',
                '--config', str(config_path)
            ]
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1 hour timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {experiment_id} completed successfully in {duration:.1f}s")
                
                # Parse results if available
                results_pattern = self.output_dir / f'results_{experiment_id}*.json'
                results_files = list(self.output_dir.glob(f'*{experiment_id}*.json'))
                
                parsed_results = {}
                if results_files:
                    try:
                        with open(results_files[0], 'r') as f:
                            parsed_results = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to parse results for {experiment_id}: {e}")
                
                return {
                    'experiment_id': experiment_id,
                    'status': 'success',
                    'duration': duration,
                    'config_path': str(config_path),
                    'results': parsed_results,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                logger.error(f"❌ {experiment_id} failed with return code {result.returncode}")
                return {
                    'experiment_id': experiment_id,
                    'status': 'failed',
                    'duration': duration,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {experiment_id} timed out")
            return {
                'experiment_id': experiment_id,
                'status': 'timeout',
                'duration': time.time() - start_time
            }
        
        except Exception as e:
            logger.error(f"❌ {experiment_id} crashed: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'status': 'crashed',
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def generate_experiment_list(self, subset: Optional[str] = None) -> List[tuple]:
        """
        Generate list of all experiments to run.
        
        Args:
            subset: Optional subset to run ('quick', 'baselines', 'schemes')
            
        Returns:
            List of (scheme_id, task_name, strategy, seed) tuples
        """
        experiments = []
        
        if subset == 'quick':
            # Quick test with minimal experiments
            schemes = ['B1', 'S5']
            tasks = ['MUTAG', 'Cora']
            strategies = ['full']
            seeds = [42]
        elif subset == 'baselines':
            # Just baseline experiments
            schemes = ['B1', 'B2', 'B3', 'B4']
            tasks = list(DOWNSTREAM_TASKS.keys())
            strategies = FINE_TUNING_STRATEGIES
            seeds = RANDOM_SEEDS
        elif subset == 'schemes':
            # Just the main schemes
            schemes = ['S1', 'S2', 'S3', 'S4', 'S5']
            tasks = list(DOWNSTREAM_TASKS.keys())
            strategies = FINE_TUNING_STRATEGIES
            seeds = RANDOM_SEEDS
        else:
            # Full experimental suite
            schemes = list(PRE_TRAINING_SCHEMES.keys())
            tasks = list(DOWNSTREAM_TASKS.keys())
            strategies = FINE_TUNING_STRATEGIES
            seeds = RANDOM_SEEDS
        
        # Generate all combinations
        for scheme, task, strategy, seed in itertools.product(schemes, tasks, strategies, seeds):
            experiments.append((scheme, task, strategy, seed))
        
        return experiments
    
    def run_experiments(self, subset: Optional[str] = None, 
                       parallel: bool = False, max_workers: int = 4):
        """
        Run all experiments.
        
        Args:
            subset: Optional subset to run
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers
        """
        experiments = self.generate_experiment_list(subset)
        
        logger.info(f"Generated {len(experiments)} experiments to run")
        logger.info(f"Parallel execution: {parallel}")
        if parallel:
            logger.info(f"Max workers: {max_workers}")
        
        if self.dry_run:
            logger.info("DRY RUN MODE - No actual training will be performed")
        
        # Run experiments
        start_time = time.time()
        
        if parallel and not self.dry_run:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_experiment = {
                    executor.submit(self.run_single_experiment, *exp): exp 
                    for exp in experiments
                }
                
                for future in as_completed(future_to_experiment):
                    experiment = future_to_experiment[future]
                    try:
                        result = future.result()
                        self.experiment_log.append(result)
                        
                        if result['status'] != 'success':
                            self.failed_experiments.append(result)
                    
                    except Exception as e:
                        logger.error(f"Experiment {experiment} generated an exception: {e}")
                        self.failed_experiments.append({
                            'experiment_id': '_'.join(map(str, experiment)),
                            'status': 'exception',
                            'error': str(e)
                        })
        else:
            # Sequential execution
            for i, experiment in enumerate(experiments, 1):
                logger.info(f"Progress: {i}/{len(experiments)}")
                result = self.run_single_experiment(*experiment)
                self.experiment_log.append(result)
                
                if result['status'] != 'success':
                    self.failed_experiments.append(result)
        
        total_time = time.time() - start_time
        
        # Save experiment log
        log_path = self.output_dir / 'experiment_log.json'
        with open(log_path, 'w') as f:
            json.dump({
                'experiments': self.experiment_log,
                'failed_experiments': self.failed_experiments,
                'total_experiments': len(experiments),
                'successful_experiments': len([e for e in self.experiment_log if e['status'] == 'success']),
                'failed_experiments_count': len(self.failed_experiments),
                'total_duration': total_time,
                'average_duration': total_time / len(experiments) if experiments else 0
            }, f, indent=2)
        
        # Print summary
        self.print_summary(total_time)
    
    def print_summary(self, total_time: float):
        """Print experiment summary."""
        total_experiments = len(self.experiment_log)
        successful = len([e for e in self.experiment_log if e['status'] == 'success'])
        failed = len(self.failed_experiments)
        
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total experiments: {total_experiments}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/total_experiments*100:.1f}%" if total_experiments > 0 else "N/A")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Average time per experiment: {total_time/total_experiments:.1f} seconds" if total_experiments > 0 else "N/A")
        
        if self.failed_experiments:
            logger.info("\nFailed experiments:")
            for exp in self.failed_experiments[:10]:  # Show first 10
                logger.info(f"  - {exp['experiment_id']}: {exp['status']}")
            if len(self.failed_experiments) > 10:
                logger.info(f"  ... and {len(self.failed_experiments) - 10} more")
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        logger.info("="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run comprehensive GNN evaluation')
    parser.add_argument('--output-dir', type=str, default='results/comprehensive_evaluation',
                       help='Output directory for results')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory containing pre-trained checkpoints')
    parser.add_argument('--config-dir', type=str, default='configs/finetune',
                       help='Directory for fine-tuning configs')
    parser.add_argument('--subset', type=str, choices=['quick', 'baselines', 'schemes'],
                       help='Run subset of experiments')
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate setup without running experiments')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        config_dir=args.config_dir,
        dry_run=args.dry_run
    )
    
    # Run experiments
    runner.run_experiments(
        subset=args.subset,
        parallel=args.parallel,
        max_workers=args.max_workers
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
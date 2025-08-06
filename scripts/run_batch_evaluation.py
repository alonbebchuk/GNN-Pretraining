#!/usr/bin/env python3
"""
Batch Evaluation Script for GNN Fine-tuning.

This script runs multiple fine-tuning experiments in batch mode,
supporting different execution strategies and comprehensive result collection.
"""

import sys
import argparse
import logging
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_config_files() -> Dict[str, List[Path]]:
    """Get all available fine-tuning configuration files."""
    configs_dir = Path('configs/finetune')
    
    config_files = {
        'graph_classification': [],
        'node_classification': [],
        'link_prediction': []
    }
    
    for task_type in config_files.keys():
        task_dir = configs_dir / task_type
        if task_dir.exists():
            config_files[task_type] = list(task_dir.glob('*.yaml'))
    
    return config_files


def run_single_experiment(config_path: Path, strategy: str, seed: int, 
                         pretrain_checkpoint: Optional[str] = None,
                         offline: bool = False, timeout: int = 3600) -> Dict[str, Any]:
    """Run a single fine-tuning experiment."""
    
    experiment_id = f"{config_path.stem}_{strategy}_seed{seed}"
    logger.info(f"Starting experiment: {experiment_id}")
    
    start_time = time.time()
    
    try:
        # Prepare command
        cmd = [
            sys.executable, 'finetune_simplified.py',
            '--config', str(config_path),
            '--strategy', strategy
        ]
        
        if pretrain_checkpoint:
            # We need to modify the config to include the checkpoint
            # For simplicity, we'll use the main script instead
            cmd = [
                sys.executable, 'src/main_finetune.py',
                '--downstream-config', str(config_path),
                '--strategy', strategy,
                '--pretrain-checkpoint', pretrain_checkpoint,
                '--seed', str(seed)
            ]
        else:
            cmd.extend(['--seed', str(seed)])
        
        if offline:
            cmd.append('--offline')
        
        # Run experiment
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse results
        experiment_result = {
            'experiment_id': experiment_id,
            'config_path': str(config_path),
            'strategy': strategy,
            'seed': seed,
            'pretrain_checkpoint': pretrain_checkpoint,
            'duration': duration,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            logger.info(f"✅ Completed: {experiment_id} ({duration:.1f}s)")
        else:
            logger.error(f"❌ Failed: {experiment_id} - {result.stderr[:200]}")
        
        return experiment_result
        
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Timeout: {experiment_id}")
        return {
            'experiment_id': experiment_id,
            'config_path': str(config_path),
            'strategy': strategy,
            'seed': seed,
            'success': False,
            'error': 'timeout',
            'duration': timeout
        }
    except Exception as e:
        logger.error(f"💥 Error: {experiment_id} - {e}")
        return {
            'experiment_id': experiment_id,
            'config_path': str(config_path),
            'strategy': strategy,
            'seed': seed,
            'success': False,
            'error': str(e),
            'duration': time.time() - start_time
        }


def create_experiment_plan(config_files: Dict[str, List[Path]], 
                          strategies: List[str],
                          seeds: List[int],
                          pretrain_checkpoints: List[Optional[str]],
                          subset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Create a plan for all experiments to run."""
    
    experiments = []
    
    # Flatten config files
    all_configs = []
    for task_configs in config_files.values():
        all_configs.extend(task_configs)
    
    # Apply subset filter
    if subset == 'quick':
        all_configs = all_configs[:2]  # Just first 2 configs
        seeds = seeds[:1]  # Just first seed
        strategies = strategies[:1]  # Just first strategy
    elif subset == 'single':
        all_configs = all_configs[:1]
        seeds = seeds[:1]
        strategies = strategies[:1]
    
    # Generate experiment combinations
    for config_path in all_configs:
        for strategy in strategies:
            for seed in seeds:
                for checkpoint in pretrain_checkpoints:
                    experiments.append({
                        'config_path': config_path,
                        'strategy': strategy,
                        'seed': seed,
                        'pretrain_checkpoint': checkpoint
                    })
    
    return experiments


def run_batch_evaluation(experiments: List[Dict[str, Any]], 
                        max_workers: int = 1,
                        offline: bool = False,
                        timeout: int = 3600) -> Dict[str, Any]:
    """Run batch evaluation of all experiments."""
    
    logger.info(f"Running {len(experiments)} experiments with {max_workers} workers")
    
    results = []
    start_time = time.time()
    
    if max_workers == 1:
        # Sequential execution
        for i, exp in enumerate(experiments):
            logger.info(f"Progress: {i+1}/{len(experiments)}")
            result = run_single_experiment(
                config_path=exp['config_path'],
                strategy=exp['strategy'],
                seed=exp['seed'],
                pretrain_checkpoint=exp['pretrain_checkpoint'],
                offline=offline,
                timeout=timeout
            )
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_exp = {
                executor.submit(
                    run_single_experiment,
                    exp['config_path'],
                    exp['strategy'],
                    exp['seed'],
                    exp['pretrain_checkpoint'],
                    offline,
                    timeout
                ): exp for exp in experiments
            }
            
            # Collect results
            for future in as_completed(future_to_exp):
                result = future.result()
                results.append(result)
                logger.info(f"Progress: {len(results)}/{len(experiments)}")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Compile summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    summary = {
        'total_experiments': len(experiments),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / len(experiments) if experiments else 0,
        'total_duration': total_duration,
        'average_duration': sum(r['duration'] for r in results) / len(results) if results else 0,
        'results': results
    }
    
    return summary


def save_results(summary: Dict[str, Any], output_file: Path):
    """Save batch evaluation results to file."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_file}")


def print_summary(summary: Dict[str, Any]):
    """Print batch evaluation summary."""
    
    print("\n" + "="*80)
    print("BATCH EVALUATION SUMMARY")
    print("="*80)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total duration: {summary['total_duration']:.1f} seconds")
    print(f"Average duration per experiment: {summary['average_duration']:.1f} seconds")
    
    if summary['failed'] > 0:
        print(f"\nFailed experiments:")
        for result in summary['results']:
            if not result['success']:
                print(f"  - {result['experiment_id']}: {result.get('error', 'Unknown error')}")
    
    print("="*80)


def main():
    """Main function for batch evaluation."""
    
    parser = argparse.ArgumentParser(
        description='Batch Fine-tuning Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_batch_evaluation.py
  
  # Quick test with subset
  python run_batch_evaluation.py --subset quick --parallel --max-workers 2
  
  # Run with specific strategies and seeds
  python run_batch_evaluation.py --strategies full linear --seeds 42 84 --offline
  
  # Dry run to see experiment plan
  python run_batch_evaluation.py --dry-run
        """
    )
    
    # Experiment selection
    parser.add_argument('--subset', choices=['quick', 'single', 'full'], default='full',
                       help='Subset of experiments to run')
    parser.add_argument('--strategies', nargs='+', choices=['full', 'linear'], 
                       default=['full', 'linear'], help='Fine-tuning strategies')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 84, 126],
                       help='Random seeds')
    parser.add_argument('--pretrain-checkpoints', nargs='+', default=[None],
                       help='Pre-training checkpoints (None for B1 baseline)')
    
    # Execution options
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel')
    parser.add_argument('--max-workers', type=int, default=2,
                       help='Maximum parallel workers')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout per experiment (seconds)')
    parser.add_argument('--offline', action='store_true',
                       help='Run without WandB logging')
    
    # Output options
    parser.add_argument('--output', type=str, default='results/batch_evaluation.json',
                       help='Output file for results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show experiment plan without running')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Get available configurations
        config_files = get_all_config_files()
        total_configs = sum(len(configs) for configs in config_files.values())
        
        if total_configs == 0:
            logger.error("No configuration files found in configs/finetune/")
            return 1
        
        logger.info(f"Found {total_configs} configuration files:")
        for task_type, configs in config_files.items():
            if configs:
                logger.info(f"  {task_type}: {len(configs)} configs")
        
        # Create experiment plan
        experiments = create_experiment_plan(
            config_files=config_files,
            strategies=args.strategies,
            seeds=args.seeds,
            pretrain_checkpoints=args.pretrain_checkpoints,
            subset=args.subset
        )
        
        logger.info(f"Generated {len(experiments)} experiments")
        
        if args.dry_run:
            print("\nEXPERIMENT PLAN:")
            print("="*50)
            for i, exp in enumerate(experiments[:10]):  # Show first 10
                print(f"{i+1:3d}. {exp['config_path'].name} | {exp['strategy']} | seed {exp['seed']}")
            if len(experiments) > 10:
                print(f"... and {len(experiments) - 10} more experiments")
            print(f"\nTotal: {len(experiments)} experiments")
            return 0
        
        # Run batch evaluation
        max_workers = args.max_workers if args.parallel else 1
        
        summary = run_batch_evaluation(
            experiments=experiments,
            max_workers=max_workers,
            offline=args.offline,
            timeout=args.timeout
        )
        
        # Save and display results
        output_path = Path(args.output)
        save_results(summary, output_path)
        print_summary(summary)
        
        # Return appropriate exit code
        return 0 if summary['success_rate'] > 0.8 else 1
        
    except KeyboardInterrupt:
        logger.info("Batch evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
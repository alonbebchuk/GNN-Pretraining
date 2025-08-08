#!/usr/bin/env python3
"""
Comprehensive experiment runner for GNN multi-task pre-training research.

This script automates the complete experimental pipeline:
1. Data preparation (run once)
2. Pre-training all 8 schemes with 3 different seeds
3. Downstream evaluation on all tasks with both strategies
4. Results aggregation and analysis

Based on Section 3.7 of the research plan:
- Pre-training: 8 models × 3 seeds = 24 runs (~67 hours)
- Fine-tuning: (8 models × 7 tasks × 2 strategies × 3 seeds) + (1 baseline × 7 tasks × 1 strategy × 3 seeds) = 357 runs
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json
from datetime import datetime
import os

try:
    # Use central reproducibility seeds
    from src.infrastructure.reproducibility import get_run_seeds
except Exception:
    from infrastructure.reproducibility import get_run_seeds

# Experimental configuration based on research plan
PRETRAIN_SCHEMES = [
    'b2_single_generative',
    'b3_single_contrastive', 
    'b4_single_domain',
    's1_multi_generative',
    's2_multi_contrastive',
    's3_all_self_supervised',
    's4_all_objectives',
    's5_domain_invariant'
]

DOWNSTREAM_TASKS = [
    ('graph_classification', 'enzymes', True),      # In-domain
    ('graph_classification', 'frankenstein', False), # Out-of-domain
    ('graph_classification', 'ptc_mr', False),      # Out-of-domain
    ('node_classification', 'cora', False),         # Out-of-domain
    ('node_classification', 'citeseer', False),     # Out-of-domain
    ('link_prediction', 'cora_lp', False),          # Out-of-domain
    ('link_prediction', 'citeseer_lp', False),      # Out-of-domain
]

FINETUNE_STRATEGIES = ['full', 'linear']
RANDOM_SEEDS = get_run_seeds()


class ExperimentRunner:
    """Manages the complete experimental pipeline."""
    
    def __init__(self, 
                 output_dir: str = 'results',
                 dry_run: bool = False,
                 skip_data_setup: bool = False,
                 skip_pretrain: bool = False,
                 skip_finetune: bool = False,
                 conda_env: str = 'gnn-simple'):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Base directory for all results
            dry_run: If True, only print commands without executing
            skip_data_setup: Skip data preparation step
            skip_pretrain: Skip pre-training step
            skip_finetune: Skip fine-tuning step
            conda_env: Conda environment name
        """
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.skip_data_setup = skip_data_setup
        self.skip_pretrain = skip_pretrain
        self.skip_finetune = skip_finetune
        self.conda_env = conda_env
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'pretrain').mkdir(exist_ok=True)
        (self.output_dir / 'finetune').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / 'logs' / f'experiment_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Track experiment progress
        self.progress = {
            'data_setup': False,
            'pretrain_completed': [],
            'finetune_completed': [],
            'total_time': 0,
            'start_time': None
        }
        
        self.logger.info(f"Experiment runner initialized")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
        self.logger.info(f"Dry run mode: {self.dry_run}")
    
    def run_command(self, cmd: List[str], description: str, 
                   capture_output: bool = False, timeout: int = None) -> subprocess.CompletedProcess:
        """
        Run a command with logging and error handling.
        
        Args:
            cmd: Command to run as list of strings
            description: Human-readable description of the command
            capture_output: Whether to capture stdout/stderr
            timeout: Command timeout in seconds
            
        Returns:
            CompletedProcess result
        """
        cmd_str = ' '.join(cmd)
        self.logger.info(f"Running: {description}")
        self.logger.debug(f"Command: {cmd_str}")
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would execute: {cmd_str}")
            return subprocess.CompletedProcess(cmd, 0)
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                check=True
            )
            duration = time.time() - start_time
            self.logger.info(f"Completed in {duration:.1f}s: {description}")
            return result
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {description}")
            self.logger.error(f"Exit code: {e.returncode}")
            if e.stdout:
                self.logger.error(f"Stdout: {e.stdout}")
            if e.stderr:
                self.logger.error(f"Stderr: {e.stderr}")
            raise
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Command timed out after {timeout}s: {description}")
            raise
    
    def setup_data(self):
        """Run data preparation step."""
        if self.skip_data_setup:
            self.logger.info("Skipping data setup (--skip-data-setup)")
            return
        
        self.logger.info("="*60)
        self.logger.info("STEP 1: DATA PREPARATION")
        self.logger.info("="*60)
        
        cmd = ['conda', 'run', '-n', self.conda_env, 'python', 'src/data_setup.py']
        self.run_command(cmd, "Data preparation", timeout=3600)  # 1 hour timeout
        
        self.progress['data_setup'] = True
        self.logger.info("✅ Data preparation completed successfully")
    
    def run_pretraining(self):
        """Run all pre-training experiments."""
        if self.skip_pretrain:
            self.logger.info("Skipping pre-training (--skip-pretrain)")
            return
        
        self.logger.info("="*60)
        self.logger.info("STEP 2: PRE-TRAINING")
        self.logger.info("="*60)
        
        total_runs = len(PRETRAIN_SCHEMES) * len(RANDOM_SEEDS)
        self.logger.info(f"Running {total_runs} pre-training experiments...")
        
        completed = 0
        for scheme in PRETRAIN_SCHEMES:
            for seed in RANDOM_SEEDS:
                run_id = f"{scheme}_seed_{seed}"
                self.logger.info(f"[{completed+1}/{total_runs}] Pre-training: {run_id}")
                
                # Create run-specific output directory
                run_output_dir = self.output_dir / 'pretrain' / run_id
                run_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Prepare command
                config_path = f'configs/pretrain/{scheme}.yaml'
                cmd = [
                    'conda', 'run', '-n', self.conda_env,
                    'python', 'src/main_pretrain.py',
                    '--config', config_path,
                    '--seed', str(seed)
                ]
                
                try:
                    # Run pre-training (estimated 2.8 hours per run)
                    self.run_command(
                        cmd, 
                        f"Pre-training {run_id}",
                        timeout=12000  # 3.3 hours timeout
                    )
                    
                    self.progress['pretrain_completed'].append(run_id)
                    completed += 1
                    
                    self.logger.info(f"✅ Completed pre-training: {run_id}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed pre-training: {run_id} - {str(e)}")
                    # Continue with other experiments
                    continue
        
        self.logger.info(f"Pre-training phase completed: {len(self.progress['pretrain_completed'])}/{total_runs} successful")
    
    def run_finetuning(self):
        """Run all downstream fine-tuning experiments."""
        if self.skip_finetune:
            self.logger.info("Skipping fine-tuning (--skip-finetune)")
            return
        
        self.logger.info("="*60)
        self.logger.info("STEP 3: DOWNSTREAM FINE-TUNING")
        self.logger.info("="*60)
        
        # Calculate total runs: (8 pretrained models × 7 tasks × 2 strategies × 3 seeds) + (1 baseline × 7 tasks × 1 strategy × 3 seeds)
        pretrained_runs = len(PRETRAIN_SCHEMES) * len(DOWNSTREAM_TASKS) * len(FINETUNE_STRATEGIES) * len(RANDOM_SEEDS)
        baseline_runs = len(DOWNSTREAM_TASKS) * len(RANDOM_SEEDS)  # Only full strategy for baseline
        total_runs = pretrained_runs + baseline_runs
        
        self.logger.info(f"Running {total_runs} fine-tuning experiments...")
        self.logger.info(f"  - Pre-trained models: {pretrained_runs} runs")
        self.logger.info(f"  - From-scratch baseline: {baseline_runs} runs")
        
        completed = 0
        
        # 1. Fine-tune pre-trained models
        for scheme in PRETRAIN_SCHEMES:
            for task_type, task_name, is_in_domain in DOWNSTREAM_TASKS:
                for strategy in FINETUNE_STRATEGIES:
                    for seed in RANDOM_SEEDS:
                        run_id = f"{scheme}_{task_name}_{strategy}_seed_{seed}"
                        completed += 1
                        
                        self.logger.info(f"[{completed}/{total_runs}] Fine-tuning: {run_id}")
                        
                        # Find pre-trained checkpoint
                        pretrain_run_id = f"{scheme}_seed_{seed}"
                        checkpoint_path = self.output_dir / 'pretrain' / pretrain_run_id / 'best_model.pt'
                        
                        if not checkpoint_path.exists():
                            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
                            continue
                        
                        # Create run-specific output directory
                        run_output_dir = self.output_dir / 'finetune' / run_id
                        run_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Prepare command
                        downstream_config = f'configs/finetune/{task_type}/{task_name}.yaml'
                        cmd = [
                            'conda', 'run', '-n', self.conda_env,
                            'python', 'src/main_finetune.py',
                            '--pretrain-checkpoint', str(checkpoint_path),
                            '--downstream-config', downstream_config,
                            '--strategy', strategy,
                            '--output-dir', str(run_output_dir),
                            '--seed', str(seed)
                        ]
                        
                        try:
                            # Run fine-tuning (estimated 14 hours total / 357 runs ≈ 2.4 minutes per run)
                            self.run_command(
                                cmd,
                                f"Fine-tuning {run_id}",
                                timeout=1800  # 30 minutes timeout per run
                            )
                            
                            self.progress['finetune_completed'].append(run_id)
                            self.logger.info(f"✅ Completed fine-tuning: {run_id}")
                            
                        except Exception as e:
                            self.logger.error(f"❌ Failed fine-tuning: {run_id} - {str(e)}")
                            continue
        
        # 2. From-scratch baseline (B1)
        for task_type, task_name, is_in_domain in DOWNSTREAM_TASKS:
            for seed in RANDOM_SEEDS:
                run_id = f"baseline_{task_name}_full_seed_{seed}"
                completed += 1
                
                self.logger.info(f"[{completed}/{total_runs}] Baseline: {run_id}")
                
                # Create run-specific output directory
                run_output_dir = self.output_dir / 'finetune' / run_id
                run_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Prepare command (no pre-trained checkpoint = from scratch B1)
                downstream_config = f'configs/finetune/{task_type}/{task_name}.yaml'
                cmd = [
                    'conda', 'run', '-n', self.conda_env,
                    'python', 'src/main_finetune.py',
                    '--downstream-config', downstream_config,
                    '--strategy', 'full',  # Always full training for baseline
                    '--output-dir', str(run_output_dir),
                    '--seed', str(seed)
                    # Note: No --pretrain-checkpoint argument = B1 from-scratch training
                ]
                
                try:
                    self.run_command(
                        cmd,
                        f"Baseline training {run_id}",
                        timeout=1800  # 30 minutes timeout
                    )
                    
                    self.progress['finetune_completed'].append(run_id)
                    self.logger.info(f"✅ Completed baseline: {run_id}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed baseline: {run_id} - {str(e)}")
                    continue
        
        self.logger.info(f"Fine-tuning phase completed: {len(self.progress['finetune_completed'])}/{total_runs} successful")
    
    def save_progress(self):
        """Save experiment progress to file."""
        progress_file = self.output_dir / 'experiment_progress.json'
        
        self.progress['total_time'] = time.time() - self.progress['start_time'] if self.progress['start_time'] else 0
        self.progress['timestamp'] = datetime.now().isoformat()
        
        with open(progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
        
        self.logger.info(f"Progress saved to: {progress_file}")
    
    def print_summary(self):
        """Print experiment summary."""
        self.logger.info("="*60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("="*60)
        
        total_time_hours = self.progress['total_time'] / 3600
        
        self.logger.info(f"Total experiment time: {total_time_hours:.1f} hours")
        self.logger.info(f"Data setup: {'✅' if self.progress['data_setup'] else '❌'}")
        self.logger.info(f"Pre-training runs completed: {len(self.progress['pretrain_completed'])}/{len(PRETRAIN_SCHEMES) * len(RANDOM_SEEDS)}")
        
        expected_finetune = (len(PRETRAIN_SCHEMES) * len(DOWNSTREAM_TASKS) * len(FINETUNE_STRATEGIES) * len(RANDOM_SEEDS) + 
                           len(DOWNSTREAM_TASKS) * len(RANDOM_SEEDS))
        self.logger.info(f"Fine-tuning runs completed: {len(self.progress['finetune_completed'])}/{expected_finetune}")
        
        if len(self.progress['pretrain_completed']) > 0:
            self.logger.info("Completed pre-training schemes:")
            for run_id in self.progress['pretrain_completed']:
                self.logger.info(f"  ✅ {run_id}")
        
        success_rate = (len(self.progress['pretrain_completed']) + len(self.progress['finetune_completed'])) / (len(PRETRAIN_SCHEMES) * len(RANDOM_SEEDS) + expected_finetune) * 100
        self.logger.info(f"Overall success rate: {success_rate:.1f}%")
        
        self.logger.info("="*60)
        
        if success_rate > 90:
            self.logger.info("🎉 Experiment pipeline completed successfully!")
        elif success_rate > 70:
            self.logger.info("⚠️ Experiment pipeline completed with some failures")
        else:
            self.logger.info("❌ Experiment pipeline had significant failures")
    
    def run_all(self):
        """Run the complete experimental pipeline."""
        self.progress['start_time'] = time.time()
        
        try:
            self.logger.info("🚀 Starting comprehensive GNN pre-training experiments")
            self.logger.info(f"Estimated total time: ~81 hours (67h pre-training + 14h fine-tuning)")
            
            # Step 1: Data preparation
            self.setup_data()
            
            # Step 2: Pre-training
            self.run_pretraining()
            
            # Step 3: Fine-tuning
            self.run_finetuning()
            
            # Summary
            self.print_summary()
            
        except KeyboardInterrupt:
            self.logger.info("Experiment interrupted by user")
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.save_progress()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run comprehensive GNN pre-training experiments')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for all results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing them')
    parser.add_argument('--skip-data-setup', action='store_true',
                       help='Skip data preparation step')
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='Skip pre-training step')
    parser.add_argument('--skip-finetune', action='store_true',
                       help='Skip fine-tuning step')
    parser.add_argument('--conda-env', type=str, default='gnn-simple',
                       help='Conda environment name')
    
    args = parser.parse_args()
    
    # Create and run experiment pipeline
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        skip_data_setup=args.skip_data_setup,
        skip_pretrain=args.skip_pretrain,
        skip_finetune=args.skip_finetune,
        conda_env=args.conda_env
    )
    
    runner.run_all()


if __name__ == '__main__':
    main() 
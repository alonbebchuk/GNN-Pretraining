#!/usr/bin/env python3
"""
Data Collection Script for Multi-Task GNN Pre-training Analysis

This script implements Step 1.1 and 1.2 of the analysis plan:
- Extract all fine-tuning experimental results from WandB
- Aggregate results across the 3 random seeds

Author: Analysis Pipeline
Date: 2024
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis/data_collection.log', mode='w'),  # 'w' mode overwrites
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
WANDB_PROJECT = "timoshka3-tel-aviv-university/gnn-pretraining-finetune"
WANDB_ENTITY = "timoshka3-tel-aviv-university"

# Expected experimental configuration
DOMAINS = ['ENZYMES', 'PTC_MR', 'Cora_NC', 'CiteSeer_NC', 'Cora_LP', 'CiteSeer_LP']
FINETUNE_STRATEGIES = ['linear_probe', 'full_finetune']
PRETRAINED_SCHEMES = ['b1', 'b2', 'b3', 's1', 's2', 's3', 's4', 's5', 'b4']
SEEDS = [42, 84, 126]

# Task type mapping
TASK_TYPES = {
    'ENZYMES': 'graph_classification',
    'PTC_MR': 'graph_classification',
    'Cora_NC': 'node_classification',
    'CiteSeer_NC': 'node_classification',
    'Cora_LP': 'link_prediction',
    'CiteSeer_LP': 'link_prediction'
}

# Expected test metrics to extract
TEST_METRICS = [
    'test/accuracy',
    'test/f1',
    'test/precision',
    'test/recall',
    'test/auc',
    'test/loss',
    'test/convergence_epochs',
    'test/training_time',
    'test/total_parameters',
    'test/trainable_parameters'
]

# Output paths
RESULTS_DIR = Path("analysis/results")
RAW_RESULTS_FILE = RESULTS_DIR / "raw_experimental_results.csv"
AGGREGATED_RESULTS_FILE = RESULTS_DIR / "aggregated_results.csv"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_run_name(run_name: str) -> Optional[Tuple[str, str, str, int]]:
    """
    Validate and parse run name according to actual patterns:
    Pattern 1 (6 parts): {domain_base}_{task_suffix}_{strategy_part1}_{strategy_part2}_{pretrained_scheme}_{seed}
    Pattern 2 (5 parts): {domain}_{strategy_part1}_{strategy_part2}_{pretrained_scheme}_{seed}
    
    Args:
        run_name: The WandB run name to validate
        
    Returns:
        Tuple of (domain, strategy, scheme, seed) if valid, None otherwise
    """
    try:
        parts = run_name.split('_')
        
        if len(parts) == 6:
            # Pattern 1: Domain with task suffix (e.g., CiteSeer_NC_full_finetune_b1_42)
            domain_base, task_suffix, strategy_part1, strategy_part2, scheme, seed_str = parts
            domain = f"{domain_base}_{task_suffix}"
            strategy = f"{strategy_part1}_{strategy_part2}"
            
        elif len(parts) == 5:
            # Pattern 2: Domain without task suffix (e.g., ENZYMES_full_finetune_b1_42)
            domain, strategy_part1, strategy_part2, scheme, seed_str = parts
            strategy = f"{strategy_part1}_{strategy_part2}"
            
        else:
            return None
        
        # Validate each component
        if (domain not in DOMAINS or 
            strategy not in FINETUNE_STRATEGIES or 
            scheme not in PRETRAINED_SCHEMES):
            return None
            
        try:
            seed = int(seed_str)
            if seed not in SEEDS:
                return None
        except ValueError:
            return None
            
        return domain, strategy, scheme, seed
        
    except Exception as e:
        logger.debug(f"Error parsing run name '{run_name}': {e}")
        return None


def extract_test_metrics(run) -> Dict[str, Any]:
    """
    Extract test metrics from a WandB run.
    
    Args:
        run: WandB run object
        
    Returns:
        Dictionary of extracted test metrics
    """
    metrics = {}
    
    try:
        # Get summary (final values)
        summary = run.summary
        
        for metric_key in TEST_METRICS:
            if metric_key in summary:
                metrics[metric_key.replace('test/', '')] = summary[metric_key]
            else:
                # Log missing metrics but don't fail
                logger.warning(f"Missing metric '{metric_key}' in run {run.name}")
                metrics[metric_key.replace('test/', '')] = None
                
    except Exception as e:
        logger.error(f"Error extracting metrics from run {run.name}: {e}")
        
    return metrics


def extract_all_finetune_results() -> pd.DataFrame:
    """
    Extract all fine-tuning experimental results from WandB.
    
    This implements Step 1.1 of the analysis plan.
    
    Returns:
        DataFrame with all experimental results
    """
    logger.info("Starting Step 1.1: Extract All Experimental Results from WandB")
    logger.info(f"Target project: {WANDB_PROJECT}")
    logger.info(f"Expected experiments: {len(DOMAINS)} domains × {len(PRETRAINED_SCHEMES)} schemes × {len(FINETUNE_STRATEGIES)} strategies × {len(SEEDS)} seeds = {len(DOMAINS) * len(PRETRAINED_SCHEMES) * len(FINETUNE_STRATEGIES) * len(SEEDS)} total")
    
    # Initialize WandB API
    try:
        api = wandb.Api()
        logger.info("Successfully initialized WandB API")
    except Exception as e:
        logger.error(f"Failed to initialize WandB API: {e}")
        raise
    
    # Fetch all runs from the project
    logger.info("Fetching runs from WandB project...")
    try:
        all_runs = api.runs(WANDB_PROJECT)
        logger.info(f"Found {len(all_runs)} total runs in project")
    except Exception as e:
        logger.error(f"Failed to fetch runs from project {WANDB_PROJECT}: {e}")
        raise
    
    # Filter and process fine-tuning runs
    valid_runs = []
    invalid_runs = []
    
    processed_count = 0
    for run in all_runs:
        try:
            run_name = run.name
            parsed = validate_run_name(run_name)
            
            if parsed is not None:
                domain, strategy, scheme, seed = parsed
                
                # Extract test metrics
                metrics = extract_test_metrics(run)
                
                # Create row for DataFrame
                row = {
                    'run_name': run_name,
                    'run_id': run.id,
                    'domain_name': domain,
                    'finetune_strategy': strategy,
                    'pretrained_scheme': scheme,
                    'seed': seed,
                    'task_type': TASK_TYPES[domain],
                    'run_state': run.state,
                    **metrics  # Add all extracted metrics
                }
                
                valid_runs.append(row)
                logger.debug(f"Processed valid run: {run_name}")
                
            else:
                invalid_runs.append(run_name)
                logger.debug(f"Skipped invalid run: {run_name}")
            
            processed_count += 1
            if processed_count % 50 == 0:
                logger.info(f"Processed {processed_count} runs... ({len(valid_runs)} valid)")
                
        except Exception as e:
            logger.warning(f"Error processing run: {e}")
            continue
    
    logger.info(f"Processed {len(valid_runs)} valid fine-tuning runs")
    logger.info(f"Skipped {len(invalid_runs)} invalid/irrelevant runs")
    
    if len(invalid_runs) > 0:
        logger.info(f"Examples of skipped runs: {invalid_runs[:10]}")
    
    # Convert to DataFrame
    df = pd.DataFrame(valid_runs)
    
    # Validate expected number of experiments
    expected_total = len(DOMAINS) * len(PRETRAINED_SCHEMES) * len(FINETUNE_STRATEGIES) * len(SEEDS)
    actual_total = len(df)
    
    logger.info(f"Expected experiments: {expected_total}")
    logger.info(f"Found experiments: {actual_total}")
    
    if actual_total < expected_total:
        logger.warning(f"Missing {expected_total - actual_total} experiments!")
        
        # Analyze missing combinations
        logger.info("Analyzing missing experiment combinations...")
        expected_combinations = set()
        for domain in DOMAINS:
            for scheme in PRETRAINED_SCHEMES:
                for strategy in FINETUNE_STRATEGIES:
                    for seed in SEEDS:
                        expected_combinations.add((domain, strategy, scheme, seed))
        
        found_combinations = set()
        for _, row in df.iterrows():
            found_combinations.add((row['domain_name'], row['finetune_strategy'], 
                                 row['pretrained_scheme'], row['seed']))
        
        missing_combinations = expected_combinations - found_combinations
        if missing_combinations:
            logger.warning(f"Missing {len(missing_combinations)} experiment combinations:")
            for combo in sorted(list(missing_combinations))[:10]:  # Show first 10
                logger.warning(f"  Missing: {combo}")
    
    # Data quality checks
    logger.info("Performing data quality checks...")
    
    # Only perform checks if we have valid runs
    if len(df) > 0:
        # Check for missing metrics
        for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc', 'loss']:
            if metric in df.columns:
                missing_count = df[metric].isna().sum()
                if missing_count > 0:
                    logger.warning(f"Missing {metric} values: {missing_count}/{len(df)} runs")
            else:
                logger.warning(f"Metric column '{metric}' not found in extracted data")
    else:
        logger.warning("No valid fine-tuning runs found - skipping data quality checks")
    
    # Check for failed runs (only if we have data)
    if len(df) > 0:
        failed_runs = df[df['run_state'] != 'finished']
        if len(failed_runs) > 0:
            logger.warning(f"Found {len(failed_runs)} non-finished runs:")
            for state in failed_runs['run_state'].value_counts().items():
                logger.warning(f"  {state[1]} runs in state: {state[0]}")
    
    # Save raw results
    logger.info(f"Saving raw results to {RAW_RESULTS_FILE}")
    df.to_csv(RAW_RESULTS_FILE, index=False)
    
    logger.info(f"Step 1.1 completed successfully. Extracted {len(df)} experimental results.")
    return df


def aggregate_results_across_seeds(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate experimental results across the 3 random seeds.
    
    This implements Step 1.2 of the analysis plan.
    
    Args:
        raw_df: DataFrame with raw experimental results
        
    Returns:
        DataFrame with aggregated statistics across seeds
    """
    logger.info("Starting Step 1.2: Aggregate Results Across Seeds")
    
    # Group by (domain_name, finetune_strategy, pretrained_scheme)
    groupby_cols = ['domain_name', 'finetune_strategy', 'pretrained_scheme', 'task_type']
    
    # Metrics to aggregate (exclude non-numeric columns)
    numeric_cols = ['accuracy', 'f1', 'precision', 'recall', 'auc', 'loss', 
                   'convergence_epochs', 'training_time', 'total_parameters', 'trainable_parameters']
    
    # Filter to only existing numeric columns
    available_numeric_cols = [col for col in numeric_cols if col in raw_df.columns]
    
    logger.info(f"Aggregating {len(available_numeric_cols)} numeric metrics across seeds")
    logger.info(f"Grouping by: {groupby_cols}")
    
    aggregated_rows = []
    
    for group_keys, group_df in raw_df.groupby(groupby_cols):
        domain_name, finetune_strategy, pretrained_scheme, task_type = group_keys
        
        # Check if we have all 3 seeds
        seeds_present = sorted(group_df['seed'].unique())
        expected_seeds = sorted(SEEDS)
        
        if seeds_present != expected_seeds:
            logger.warning(f"Missing seeds for {group_keys}: found {seeds_present}, expected {expected_seeds}")
        
        # Create aggregated row
        agg_row = {
            'domain_name': domain_name,
            'finetune_strategy': finetune_strategy,
            'pretrained_scheme': pretrained_scheme,
            'task_type': task_type,
            'num_seeds': len(group_df),
            'seeds_present': str(sorted(group_df['seed'].tolist()))
        }
        
        # Aggregate each numeric metric
        for metric in available_numeric_cols:
            values = group_df[metric].dropna()  # Remove NaN values
            
            if len(values) > 0:
                agg_row[f'{metric}_mean'] = values.mean()
                agg_row[f'{metric}_std'] = values.std()
                agg_row[f'{metric}_min'] = values.min()
                agg_row[f'{metric}_max'] = values.max()
                agg_row[f'{metric}_median'] = values.median()
                agg_row[f'{metric}_sem'] = values.sem()  # Standard error of mean
                agg_row[f'{metric}_count'] = len(values)  # Number of non-NaN values
            else:
                # All values are NaN
                for stat in ['mean', 'std', 'min', 'max', 'median', 'sem']:
                    agg_row[f'{metric}_{stat}'] = None
                agg_row[f'{metric}_count'] = 0
        
        aggregated_rows.append(agg_row)
    
    # Convert to DataFrame
    agg_df = pd.DataFrame(aggregated_rows)
    
    # Sort by domain, strategy, scheme for consistent ordering
    agg_df = agg_df.sort_values(['domain_name', 'finetune_strategy', 'pretrained_scheme'])
    
    logger.info(f"Created aggregated dataset with {len(agg_df)} combinations")
    
    # Expected number of combinations
    expected_combinations = len(DOMAINS) * len(FINETUNE_STRATEGIES) * len(PRETRAINED_SCHEMES)
    logger.info(f"Expected combinations: {expected_combinations}")
    logger.info(f"Found combinations: {len(agg_df)}")
    
    # Save aggregated results
    logger.info(f"Saving aggregated results to {AGGREGATED_RESULTS_FILE}")
    agg_df.to_csv(AGGREGATED_RESULTS_FILE, index=False)
    
    logger.info(f"Step 1.2 completed successfully. Created {len(agg_df)} aggregated combinations.")
    return agg_df


def main():
    """Main function to execute both steps of data collection."""
    logger.info("Starting Data Collection Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1.1: Extract raw results
        raw_df = extract_all_finetune_results()
        
        # Step 1.2: Aggregate across seeds
        agg_df = aggregate_results_across_seeds(raw_df)
        
        logger.info("=" * 60)
        logger.info("Data Collection Pipeline completed successfully!")
        logger.info(f"Raw results: {len(raw_df)} experiments → {RAW_RESULTS_FILE}")
        logger.info(f"Aggregated results: {len(agg_df)} combinations → {AGGREGATED_RESULTS_FILE}")
        
        return raw_df, agg_df
        
    except Exception as e:
        logger.error(f"Data collection pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

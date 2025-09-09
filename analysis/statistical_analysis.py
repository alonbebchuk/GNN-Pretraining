#!/usr/bin/env python3
"""
Statistical Analysis Script for Multi-Task GNN Pre-training

This script implements Step 2.1: RQ1 Pre-training Effectiveness Analysis
Analyzes whether pre-training improves performance compared to from-scratch training

Author: Analysis Pipeline
Date: 2024
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis/statistical_analysis.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path("analysis/results")
FIGURES_DIR = Path("analysis/figures")

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Task type to primary metric mapping
PRIMARY_METRICS = {
    'graph_classification': 'accuracy',
    'node_classification': 'accuracy',
    'link_prediction': 'auc'
}

# Pre-training schemes (excluding baseline)
PRETRAINED_SCHEMES = ['b2', 'b3', 's1', 's2', 's3', 's4', 's5', 'b4']
BASELINE_SCHEME = 'b1'


def load_aggregated_data() -> pd.DataFrame:
    """Load the aggregated results from Step 1.2."""
    filepath = RESULTS_DIR / "aggregated_results.csv"
    logger.info(f"Loading aggregated data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} aggregated combinations")
    return df


def load_raw_data() -> pd.DataFrame:
    """Load the raw results from Step 1.1 for statistical tests."""
    filepath = RESULTS_DIR / "raw_experimental_results.csv"
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} raw experiments")
    return df


def get_primary_metric_column(task_type: str, stat: str = 'mean') -> str:
    """Get the primary metric column name for a given task type."""
    primary_metric = PRIMARY_METRICS[task_type]
    return f"{primary_metric}_{stat}"


def calculate_improvement(pretrained_value: float, baseline_value: float) -> float:
    """Calculate percentage improvement of pretrained over baseline."""
    if baseline_value == 0:
        return 0.0
    return ((pretrained_value - baseline_value) / baseline_value) * 100


def cohens_d(group1: np.array, group2: np.array) -> float:
    """Calculate Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def perform_paired_ttest(raw_df: pd.DataFrame, domain: str, strategy: str, 
                        scheme: str, task_type: str) -> Dict[str, float]:
    """
    Perform paired t-test between a pre-trained scheme and baseline.
    
    Returns:
        Dictionary with p_value, t_statistic, and effect_size
    """
    primary_metric = PRIMARY_METRICS[task_type]
    
    # Get baseline values
    baseline_data = raw_df[
        (raw_df['domain_name'] == domain) & 
        (raw_df['finetune_strategy'] == strategy) & 
        (raw_df['pretrained_scheme'] == BASELINE_SCHEME)
    ][primary_metric].values
    
    # Get pretrained values
    pretrained_data = raw_df[
        (raw_df['domain_name'] == domain) & 
        (raw_df['finetune_strategy'] == strategy) & 
        (raw_df['pretrained_scheme'] == scheme)
    ][primary_metric].values
    
    # Check if we have matching seeds
    if len(baseline_data) != len(pretrained_data):
        logger.warning(f"Mismatched data sizes for {domain}/{strategy}/{scheme}: "
                      f"baseline={len(baseline_data)}, pretrained={len(pretrained_data)}")
        return {'p_value': np.nan, 't_statistic': np.nan, 'effect_size': np.nan}
    
    if len(baseline_data) < 2:
        logger.warning(f"Insufficient data for t-test: {domain}/{strategy}/{scheme}")
        return {'p_value': np.nan, 't_statistic': np.nan, 'effect_size': np.nan}
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(pretrained_data, baseline_data)
    
    # Calculate effect size (Cohen's d)
    effect_size = cohens_d(pretrained_data, baseline_data)
    
    return {
        'p_value': p_value,
        't_statistic': t_stat,
        'effect_size': effect_size
    }


def analyze_rq1_effectiveness(agg_df: pd.DataFrame, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main analysis for RQ1: Pre-training Effectiveness
    
    Returns:
        Tuple of (improvement_df, statistical_df, best_schemes_df)
    """
    logger.info("Starting RQ1 Analysis: Pre-training Effectiveness")
    
    improvement_results = []
    statistical_results = []
    
    # Get unique domains and strategies
    domains = agg_df['domain_name'].unique()
    strategies = agg_df['finetune_strategy'].unique()
    
    # Analyze each domain-strategy combination
    for domain in domains:
        for strategy in strategies:
            # Get task type for this domain
            task_type = agg_df[agg_df['domain_name'] == domain]['task_type'].iloc[0]
            primary_metric_col = get_primary_metric_column(task_type, 'mean')
            
            # Get baseline performance
            baseline_row = agg_df[
                (agg_df['domain_name'] == domain) & 
                (agg_df['finetune_strategy'] == strategy) & 
                (agg_df['pretrained_scheme'] == BASELINE_SCHEME)
            ]
            
            if baseline_row.empty:
                logger.warning(f"No baseline found for {domain}/{strategy}")
                continue
                
            baseline_value = baseline_row[primary_metric_col].iloc[0]
            
            # Analyze each pre-trained scheme
            for scheme in PRETRAINED_SCHEMES:
                pretrained_row = agg_df[
                    (agg_df['domain_name'] == domain) & 
                    (agg_df['finetune_strategy'] == strategy) & 
                    (agg_df['pretrained_scheme'] == scheme)
                ]
                
                if pretrained_row.empty:
                    logger.warning(f"No data found for {domain}/{strategy}/{scheme}")
                    continue
                
                pretrained_value = pretrained_row[primary_metric_col].iloc[0]
                
                # Calculate improvement
                improvement = calculate_improvement(pretrained_value, baseline_value)
                
                # Perform statistical test
                test_results = perform_paired_ttest(raw_df, domain, strategy, scheme, task_type)
                
                # Store improvement results
                improvement_results.append({
                    'domain': domain,
                    'strategy': strategy,
                    'scheme': scheme,
                    'task_type': task_type,
                    'baseline_value': baseline_value,
                    'pretrained_value': pretrained_value,
                    'improvement_percent': improvement,
                    'primary_metric': PRIMARY_METRICS[task_type]
                })
                
                # Store statistical results
                statistical_results.append({
                    'domain': domain,
                    'strategy': strategy,
                    'scheme': scheme,
                    'task_type': task_type,
                    'p_value': test_results['p_value'],
                    't_statistic': test_results['t_statistic'],
                    'effect_size': test_results['effect_size']
                })
    
    # Create DataFrames
    improvement_df = pd.DataFrame(improvement_results)
    statistical_df = pd.DataFrame(statistical_results)
    
    # Apply Bonferroni correction
    if len(statistical_df) > 0:
        p_values = statistical_df['p_value'].values
        reject, p_adjusted, _, _ = multipletests(
            p_values, 
            alpha=0.05, 
            method='bonferroni',
            is_sorted=False
        )
        statistical_df['p_value_corrected'] = p_adjusted
        statistical_df['significant'] = reject
    
    # Find best performing scheme per domain/strategy
    best_schemes = []
    for domain in domains:
        for strategy in strategies:
            domain_improvements = improvement_df[
                (improvement_df['domain'] == domain) & 
                (improvement_df['strategy'] == strategy)
            ]
            
            if not domain_improvements.empty:
                best_idx = domain_improvements['improvement_percent'].idxmax()
                best_row = domain_improvements.loc[best_idx]
                best_schemes.append({
                    'domain': domain,
                    'strategy': strategy,
                    'best_scheme': best_row['scheme'],
                    'improvement_percent': best_row['improvement_percent'],
                    'baseline_value': best_row['baseline_value'],
                    'best_value': best_row['pretrained_value']
                })
    
    best_schemes_df = pd.DataFrame(best_schemes)
    
    return improvement_df, statistical_df, best_schemes_df


def create_summary_tables(improvement_df: pd.DataFrame, statistical_df: pd.DataFrame, 
                         best_schemes_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create summary tables for RQ1 analysis."""
    logger.info("Creating summary tables")
    
    # Table 1: Mean improvement by scheme
    scheme_summary = improvement_df.groupby('scheme').agg({
        'improvement_percent': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    scheme_summary.columns = ['mean_improvement', 'std_improvement', 'min_improvement', 
                              'max_improvement', 'n_combinations']
    scheme_summary = scheme_summary.sort_values('mean_improvement', ascending=False)
    
    # Table 2: Statistical significance summary
    sig_summary = statistical_df.groupby('scheme').agg({
        'significant': 'sum',
        'effect_size': ['mean', 'std']
    })
    sig_summary.columns = ['n_significant', 'mean_effect_size', 'std_effect_size']
    sig_summary['total_tests'] = statistical_df.groupby('scheme').size()
    sig_summary['significance_rate'] = (sig_summary['n_significant'] / sig_summary['total_tests'] * 100).round(1)
    
    # Table 3: Domain-specific best schemes
    domain_summary = best_schemes_df.pivot(index='domain', columns='strategy', 
                                          values='best_scheme')
    
    return {
        'scheme_summary': scheme_summary,
        'significance_summary': sig_summary,
        'domain_best_schemes': domain_summary
    }


def save_results(improvement_df: pd.DataFrame, statistical_df: pd.DataFrame, 
                best_schemes_df: pd.DataFrame, summary_tables: Dict[str, pd.DataFrame]):
    """Save all results to CSV files."""
    logger.info("Saving results")
    
    # Save main analysis results
    improvement_df.to_csv(RESULTS_DIR / "rq1_improvement_analysis.csv", index=False)
    statistical_df.to_csv(RESULTS_DIR / "rq1_statistical_tests.csv", index=False)
    best_schemes_df.to_csv(RESULTS_DIR / "rq1_best_schemes.csv", index=False)
    
    # Save summary tables
    for name, table in summary_tables.items():
        table.to_csv(RESULTS_DIR / f"rq1_summary_{name}.csv")
    
    logger.info(f"Results saved to {RESULTS_DIR}")


def print_key_findings(improvement_df: pd.DataFrame, statistical_df: pd.DataFrame, 
                       summary_tables: Dict[str, pd.DataFrame]):
    """Print key findings from the analysis."""
    logger.info("\n" + "="*60)
    logger.info("KEY FINDINGS - RQ1: Pre-training Effectiveness")
    logger.info("="*60)
    
    # Overall improvement
    mean_improvement = improvement_df['improvement_percent'].mean()
    logger.info(f"\n1. Overall mean improvement: {mean_improvement:.2f}%")
    
    # Best performing scheme
    scheme_summary = summary_tables['scheme_summary']
    best_scheme = scheme_summary.index[0]
    best_improvement = scheme_summary.iloc[0]['mean_improvement']
    logger.info(f"2. Best performing scheme: {best_scheme} (mean improvement: {best_improvement:.2f}%)")
    
    # Statistical significance
    sig_summary = summary_tables['significance_summary']
    total_significant = sig_summary['n_significant'].sum()
    total_tests = sig_summary['total_tests'].sum()
    logger.info(f"3. Statistically significant improvements: {total_significant}/{total_tests} "
               f"({total_significant/total_tests*100:.1f}%)")
    
    # Schemes with positive improvement
    positive_schemes = improvement_df[improvement_df['improvement_percent'] > 0]
    positive_rate = len(positive_schemes) / len(improvement_df) * 100
    logger.info(f"4. Combinations with positive improvement: {len(positive_schemes)}/{len(improvement_df)} "
               f"({positive_rate:.1f}%)")
    
    # Average effect size
    mean_effect = statistical_df['effect_size'].mean()
    logger.info(f"5. Average effect size (Cohen's d): {mean_effect:.3f}")
    
    logger.info("="*60 + "\n")


def main():
    """Main function for RQ1 analysis."""
    logger.info("Starting RQ1 Pre-training Effectiveness Analysis")
    logger.info("="*60)
    
    try:
        # Load data
        agg_df = load_aggregated_data()
        raw_df = load_raw_data()
        
        # Perform main analysis
        improvement_df, statistical_df, best_schemes_df = analyze_rq1_effectiveness(agg_df, raw_df)
        
        # Create summary tables
        summary_tables = create_summary_tables(improvement_df, statistical_df, best_schemes_df)
        
        # Save results
        save_results(improvement_df, statistical_df, best_schemes_df, summary_tables)
        
        # Print key findings
        print_key_findings(improvement_df, statistical_df, summary_tables)
        
        logger.info("RQ1 Analysis completed successfully!")
        
        return improvement_df, statistical_df, best_schemes_df, summary_tables
        
    except Exception as e:
        logger.error(f"RQ1 Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()

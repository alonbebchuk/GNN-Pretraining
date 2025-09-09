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
        logging.FileHandler('statistical_analysis.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Task type to primary metric mapping
PRIMARY_METRICS = {
    'graph_classification': 'accuracy',
    'node_classification': 'accuracy',
    'link_prediction': 'auc'
}

# Constants from data collection
DOMAINS = ['ENZYMES', 'PTC_MR', 'Cora_NC', 'CiteSeer_NC', 'Cora_LP', 'CiteSeer_LP']
FINETUNE_STRATEGIES = ['linear_probe', 'full_finetune']
PRETRAINED_SCHEMES = ['b1', 'b2', 'b3', 's1', 's2', 's3', 's4', 's5', 'b4']
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


def get_effect_size_category(cohens_d: float) -> str:
    """Categorize Cohen's d effect size."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return 'negligible'
    elif abs_d < 0.5:
        return 'small'
    elif abs_d < 0.8:
        return 'medium'
    else:
        return 'large'


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


# ========================
# RQ2: Task Combination Analysis
# ========================

def get_task_combinations():
    """
    Define task combinations for pre-training schemes.
    
    Returns:
        dict: Mapping of scheme to task combination info
    """
    return {
        'b2': {'tasks': ['node_feat_mask'], 'type': 'single', 'count': 1},
        'b3': {'tasks': ['node_contrast'], 'type': 'single', 'count': 1},
        's1': {'tasks': ['node_feat_mask', 'link_pred'], 'type': 'generative', 'count': 2},
        's2': {'tasks': ['node_contrast', 'graph_contrast'], 'type': 'contrastive', 'count': 2},
        's3': {'tasks': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast'], 'type': 'combined', 'count': 4},
        's4': {'tasks': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop'], 'type': 'cross_domain', 'count': 5},
        's5': {'tasks': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop', 'domain_adv'], 'type': 'full_adv', 'count': 6},
        'b4': {'tasks': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop'], 'type': 'single_domain', 'count': 5}
    }


def analyze_task_combinations(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance across different task combinations.
    
    Args:
        agg_df: Aggregated results DataFrame
        
    Returns:
        DataFrame with task combination analysis
    """
    logger.info("Analyzing task combinations")
    
    task_combos = get_task_combinations()
    results = []
    
    for domain in DOMAINS:
        for strategy in FINETUNE_STRATEGIES:
            # Get baseline performance
            baseline_row = agg_df[
                (agg_df['domain_name'] == domain) & 
                (agg_df['finetune_strategy'] == strategy) & 
                (agg_df['pretrained_scheme'] == 'b1')
            ]
            
            if baseline_row.empty:
                logger.warning(f"No baseline found for {domain}-{strategy}")
                continue
                
            # Get primary metric for this task type
            task_type = baseline_row['task_type'].iloc[0]
            primary_metric = PRIMARY_METRICS[task_type]
            baseline_perf = baseline_row[f'{primary_metric}_mean'].iloc[0]
            
            # Analyze each scheme
            for scheme in ['b2', 'b3', 's1', 's2', 's3', 's4', 's5', 'b4']:
                scheme_row = agg_df[
                    (agg_df['domain_name'] == domain) & 
                    (agg_df['finetune_strategy'] == strategy) & 
                    (agg_df['pretrained_scheme'] == scheme)
                ]
                
                if scheme_row.empty:
                    continue
                    
                scheme_perf = scheme_row[f'{primary_metric}_mean'].iloc[0]
                improvement = ((scheme_perf - baseline_perf) / baseline_perf) * 100
                
                combo_info = task_combos[scheme]
                
                results.append({
                    'domain': domain,
                    'strategy': strategy,
                    'scheme': scheme,
                    'task_count': combo_info['count'],
                    'task_type': combo_info['type'],
                    'tasks': ', '.join(combo_info['tasks']),
                    'baseline_performance': baseline_perf,
                    'scheme_performance': scheme_perf,
                    'improvement_pct': improvement,
                    'mean_std': scheme_row[f'{primary_metric}_std'].iloc[0],
                    'mean_sem': scheme_row[f'{primary_metric}_sem'].iloc[0]
                })
    
    return pd.DataFrame(results)


def calculate_synergy_scores(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate task synergy scores.
    Synergy = performance(multi-task) - max(performance(single-tasks))
    
    Args:
        agg_df: Aggregated results DataFrame
        
    Returns:
        DataFrame with synergy analysis
    """
    logger.info("Calculating task synergy scores")
    
    results = []
    
    for domain in DOMAINS:
        for strategy in FINETUNE_STRATEGIES:
            # Get primary metric for this domain
            domain_rows = agg_df[agg_df['domain_name'] == domain]
            if domain_rows.empty:
                continue
            task_type = domain_rows['task_type'].iloc[0]
            primary_metric = PRIMARY_METRICS[task_type]
            
            # Get single-task performances (b2, b3)
            single_tasks = {}
            for scheme in ['b2', 'b3']:
                scheme_row = agg_df[
                    (agg_df['domain_name'] == domain) & 
                    (agg_df['finetune_strategy'] == strategy) & 
                    (agg_df['pretrained_scheme'] == scheme)
                ]
                if not scheme_row.empty:
                    single_tasks[scheme] = scheme_row[f'{primary_metric}_mean'].iloc[0]
            
            if len(single_tasks) < 2:
                continue
                
            max_single = max(single_tasks.values())
            
            # Calculate synergy for multi-task schemes
            for scheme in ['s1', 's2', 's3', 's4', 's5', 'b4']:
                scheme_row = agg_df[
                    (agg_df['domain_name'] == domain) & 
                    (agg_df['finetune_strategy'] == strategy) & 
                    (agg_df['pretrained_scheme'] == scheme)
                ]
                
                if scheme_row.empty:
                    continue
                    
                multi_perf = scheme_row[f'{primary_metric}_mean'].iloc[0]
                synergy = multi_perf - max_single
                synergy_pct = (synergy / max_single) * 100
                
                task_combos = get_task_combinations()
                
                results.append({
                    'domain': domain,
                    'strategy': strategy,
                    'scheme': scheme,
                    'task_count': task_combos[scheme]['count'],
                    'task_type': task_combos[scheme]['type'],
                    'max_single_performance': max_single,
                    'multi_task_performance': multi_perf,
                    'synergy_score': synergy,
                    'synergy_pct': synergy_pct,
                    'is_positive_synergy': synergy > 0
                })
    
    return pd.DataFrame(results)


def compare_progressive_combinations(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical comparison between progressive task combinations.
    
    Args:
        raw_df: Raw experimental results DataFrame
        
    Returns:
        DataFrame with statistical comparison results
    """
    logger.info("Performing progressive task combination comparisons")
    
    # Define progressive comparisons
    comparisons = [
        ('b2', 's1', 'Adding link prediction to node feature masking'),
        ('b3', 's2', 'Adding graph contrastive to node contrastive'),
        ('s1', 's3', 'Combining generative (s1) with contrastive tasks'),
        ('s2', 's3', 'Combining contrastive (s2) with generative tasks'),
        ('s3', 's4', 'Adding graph property prediction to 4-task'),
        ('s4', 's5', 'Adding domain adversarial to 5-task'),
        ('s4', 'b4', 'Cross-domain vs single-domain (5-task)')
    ]
    
    results = []
    
    for domain in DOMAINS:
        for strategy in FINETUNE_STRATEGIES:
            # Get primary metric for this domain
            domain_rows = raw_df[raw_df['domain_name'] == domain]
            if domain_rows.empty:
                continue
            task_type = domain_rows['task_type'].iloc[0]
            primary_metric = PRIMARY_METRICS[task_type]
            
            for scheme1, scheme2, description in comparisons:
                # Get data for both schemes
                data1 = raw_df[
                    (raw_df['domain_name'] == domain) & 
                    (raw_df['finetune_strategy'] == strategy) & 
                    (raw_df['pretrained_scheme'] == scheme1)
                ][primary_metric].values
                
                data2 = raw_df[
                    (raw_df['domain_name'] == domain) & 
                    (raw_df['finetune_strategy'] == strategy) & 
                    (raw_df['pretrained_scheme'] == scheme2)
                ][primary_metric].values
                
                if len(data1) == 0 or len(data2) == 0:
                    continue
                
                # Perform paired t-test
                from scipy.stats import ttest_rel
                
                try:
                    t_stat, p_value = ttest_rel(data2, data1)  # data2 - data1
                    
                    # Calculate effect size (Cohen's d for paired samples)
                    diff = data2 - data1
                    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                    
                    # Calculate mean improvement
                    mean1, mean2 = np.mean(data1), np.mean(data2)
                    improvement = ((mean2 - mean1) / mean1) * 100 if mean1 != 0 else 0
                    
                    results.append({
                        'domain': domain,
                        'strategy': strategy,
                        'comparison': f"{scheme1} vs {scheme2}",
                        'description': description,
                        'scheme1': scheme1,
                        'scheme2': scheme2,
                        'mean1': mean1,
                        'mean2': mean2,
                        'improvement_pct': improvement,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        't_statistic': t_stat,
                        'significant': p_value < 0.05,
                        'effect_size_category': get_effect_size_category(abs(cohens_d))
                    })
                    
                except Exception as e:
                    logger.warning(f"Statistical test failed for {domain}-{strategy} {scheme1} vs {scheme2}: {e}")
                    continue
    
    return pd.DataFrame(results)


def create_task_combination_summary(combo_df: pd.DataFrame, synergy_df: pd.DataFrame) -> dict:
    """
    Create summary tables for task combination analysis.
    
    Args:
        combo_df: Task combination analysis DataFrame
        synergy_df: Synergy analysis DataFrame
        
    Returns:
        Dictionary of summary tables
    """
    logger.info("Creating task combination summary tables")
    
    summaries = {}
    
    # 1. Best task combination per domain
    best_per_domain = combo_df.loc[combo_df.groupby(['domain', 'strategy'])['improvement_pct'].idxmax()]
    summaries['best_per_domain'] = best_per_domain[['domain', 'strategy', 'scheme', 'task_type', 'improvement_pct']].copy()
    
    # 2. Overall task type effectiveness
    task_type_summary = combo_df.groupby('task_type').agg({
        'improvement_pct': ['mean', 'std', 'count'],
        'scheme_performance': ['mean', 'std']
    }).round(3)
    task_type_summary.columns = ['_'.join(col).strip() for col in task_type_summary.columns]
    summaries['task_type_effectiveness'] = task_type_summary
    
    # 3. Synergy analysis summary
    synergy_summary = synergy_df.groupby('task_type').agg({
        'synergy_pct': ['mean', 'std', 'count'],
        'is_positive_synergy': 'sum'
    }).round(3)
    synergy_summary.columns = ['_'.join(col).strip() for col in synergy_summary.columns]
    synergy_summary['positive_synergy_rate'] = (synergy_summary['is_positive_synergy_sum'] / 
                                                synergy_summary['synergy_pct_count'] * 100).round(1)
    summaries['synergy_summary'] = synergy_summary
    
    # 4. Task count vs performance
    task_count_perf = combo_df.groupby('task_count').agg({
        'improvement_pct': ['mean', 'std', 'count'],
        'scheme_performance': ['mean', 'std']
    }).round(3)
    task_count_perf.columns = ['_'.join(col).strip() for col in task_count_perf.columns]
    summaries['task_count_performance'] = task_count_perf
    
    return summaries


def analyze_rq2(agg_df: pd.DataFrame, raw_df: pd.DataFrame) -> tuple:
    """
    Complete RQ2 analysis: Task Combination Analysis.
    
    Args:
        agg_df: Aggregated results DataFrame
        raw_df: Raw experimental results DataFrame
        
    Returns:
        Tuple of analysis results
    """
    logger.info("Starting RQ2 Analysis: Task Combination Analysis")
    logger.info("=" * 50)
    
    try:
        # 1. Task combination analysis
        combo_df = analyze_task_combinations(agg_df)
        
        # 2. Synergy analysis
        synergy_df = calculate_synergy_scores(agg_df)
        
        # 3. Progressive comparison analysis
        progressive_df = compare_progressive_combinations(raw_df)
        
        # 4. Apply Bonferroni correction to progressive comparisons
        if not progressive_df.empty:
            from statsmodels.stats.multitest import multipletests
            _, corrected_p, _, _ = multipletests(progressive_df['p_value'], method='bonferroni')
            progressive_df['p_value_corrected'] = corrected_p
            progressive_df['significant_corrected'] = corrected_p < 0.05
        
        # 5. Create summary tables
        summary_tables = create_task_combination_summary(combo_df, synergy_df)
        
        # 6. Save results
        logger.info("Saving RQ2 results")
        combo_df.to_csv(RESULTS_DIR / 'rq2_task_combination_analysis.csv', index=False)
        synergy_df.to_csv(RESULTS_DIR / 'rq2_synergy_scores.csv', index=False)
        progressive_df.to_csv(RESULTS_DIR / 'rq2_progressive_comparisons.csv', index=False)
        
        # Save summary tables
        for name, table in summary_tables.items():
            table.to_csv(RESULTS_DIR / f'rq2_summary_{name}.csv', index=True)
        
        logger.info("RQ2 results saved to analysis/results")
        
        # 7. Log key findings
        log_rq2_findings(combo_df, synergy_df, progressive_df, summary_tables)
        
        logger.info("RQ2 Analysis completed successfully!")
        
        return combo_df, synergy_df, progressive_df, summary_tables
        
    except Exception as e:
        logger.error(f"RQ2 Analysis failed: {e}")
        raise


def log_rq2_findings(combo_df: pd.DataFrame, synergy_df: pd.DataFrame, 
                     progressive_df: pd.DataFrame, summary_tables: dict):
    """Log key findings from RQ2 analysis."""
    logger.info("\n" + "="*50)
    logger.info("RQ2 KEY FINDINGS: Task Combination Analysis")
    logger.info("="*50)
    
    # Best task type overall
    task_type_eff = summary_tables['task_type_effectiveness']
    best_task_type = task_type_eff['improvement_pct_mean'].idxmax()
    best_improvement = task_type_eff.loc[best_task_type, 'improvement_pct_mean']
    
    logger.info(f"1. BEST TASK TYPE: {best_task_type} ({best_improvement:.2f}% avg improvement)")
    
    # Synergy analysis
    synergy_summary = summary_tables['synergy_summary']
    positive_synergy_rates = synergy_summary['positive_synergy_rate']
    best_synergy_type = positive_synergy_rates.idxmax()
    best_synergy_rate = positive_synergy_rates.max()
    
    logger.info(f"2. BEST SYNERGY: {best_synergy_type} ({best_synergy_rate:.1f}% positive synergy rate)")
    
    # Task count vs performance
    task_count_perf = summary_tables['task_count_performance']
    best_task_count = task_count_perf['improvement_pct_mean'].idxmax()
    best_count_improvement = task_count_perf.loc[best_task_count, 'improvement_pct_mean']
    
    logger.info(f"3. OPTIMAL TASK COUNT: {best_task_count} tasks ({best_count_improvement:.2f}% avg improvement)")
    
    # Significant progressive improvements
    if not progressive_df.empty:
        sig_improvements = progressive_df[progressive_df['significant_corrected'] & (progressive_df['improvement_pct'] > 0)]
        logger.info(f"4. SIGNIFICANT IMPROVEMENTS: {len(sig_improvements)} progressive combinations show significant gains")
        
        if not sig_improvements.empty:
            best_progression = sig_improvements.loc[sig_improvements['improvement_pct'].idxmax()]
            logger.info(f"   Best: {best_progression['comparison']} (+{best_progression['improvement_pct']:.2f}%)")
    
    # Overall synergy rate
    total_positive = synergy_df['is_positive_synergy'].sum()
    total_combinations = len(synergy_df)
    overall_synergy_rate = (total_positive / total_combinations * 100) if total_combinations > 0 else 0
    
    logger.info(f"5. OVERALL SYNERGY: {overall_synergy_rate:.1f}% of multi-task combinations show positive synergy")
    
    logger.info("="*50)


# ========================
# RQ3: Fine-tuning Strategy Comparison
# ========================

def compare_finetune_strategies(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare linear probing vs full fine-tuning strategies.
    
    Args:
        agg_df: Aggregated results DataFrame
        
    Returns:
        DataFrame with strategy comparison analysis
    """
    logger.info("Comparing fine-tuning strategies")
    
    results = []
    
    for domain in DOMAINS:
        # Get primary metric for this domain
        domain_rows = agg_df[agg_df['domain_name'] == domain]
        if domain_rows.empty:
            continue
        task_type = domain_rows['task_type'].iloc[0]
        primary_metric = PRIMARY_METRICS[task_type]
        
        for scheme in PRETRAINED_SCHEMES:
            # Get both strategies for this domain-scheme combination
            linear_row = agg_df[
                (agg_df['domain_name'] == domain) & 
                (agg_df['pretrained_scheme'] == scheme) & 
                (agg_df['finetune_strategy'] == 'linear_probe')
            ]
            
            full_row = agg_df[
                (agg_df['domain_name'] == domain) & 
                (agg_df['pretrained_scheme'] == scheme) & 
                (agg_df['finetune_strategy'] == 'full_finetune')
            ]
            
            if linear_row.empty or full_row.empty:
                continue
                
            # Extract performance metrics
            linear_perf = linear_row[f'{primary_metric}_mean'].iloc[0]
            full_perf = full_row[f'{primary_metric}_mean'].iloc[0]
            
            # Calculate differences
            abs_diff = full_perf - linear_perf
            rel_diff = (abs_diff / linear_perf) * 100 if linear_perf != 0 else 0
            
            # Extract efficiency metrics
            linear_time = linear_row['training_time_mean'].iloc[0] if 'training_time_mean' in linear_row.columns else None
            full_time = full_row['training_time_mean'].iloc[0] if 'training_time_mean' in full_row.columns else None
            
            linear_epochs = linear_row['convergence_epochs_mean'].iloc[0] if 'convergence_epochs_mean' in linear_row.columns else None
            full_epochs = full_row['convergence_epochs_mean'].iloc[0] if 'convergence_epochs_mean' in full_row.columns else None
            
            linear_params = linear_row['trainable_parameters_mean'].iloc[0] if 'trainable_parameters_mean' in linear_row.columns else None
            full_params = full_row['trainable_parameters_mean'].iloc[0] if 'trainable_parameters_mean' in full_row.columns else None
            
            results.append({
                'domain': domain,
                'scheme': scheme,
                'task_type': task_type,
                'linear_performance': linear_perf,
                'full_performance': full_perf,
                'absolute_difference': abs_diff,
                'relative_difference_pct': rel_diff,
                'better_strategy': 'full_finetune' if abs_diff > 0 else 'linear_probe',
                'linear_training_time': linear_time,
                'full_training_time': full_time,
                'time_ratio': full_time / linear_time if (linear_time and full_time and linear_time > 0) else None,
                'linear_epochs': linear_epochs,
                'full_epochs': full_epochs,
                'linear_trainable_params': linear_params,
                'full_trainable_params': full_params,
                'param_ratio': full_params / linear_params if (linear_params and full_params and linear_params > 0) else None,
                'linear_perf_per_param': linear_perf / linear_params if linear_params else None,
                'full_perf_per_param': full_perf / full_params if full_params else None
            })
    
    return pd.DataFrame(results)


def analyze_strategy_effectiveness(strategy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which pre-training schemes benefit more from full fine-tuning.
    
    Args:
        strategy_df: Strategy comparison DataFrame
        
    Returns:
        DataFrame with effectiveness analysis by scheme
    """
    logger.info("Analyzing strategy effectiveness by scheme")
    
    # Group by scheme and analyze strategy preferences
    scheme_analysis = strategy_df.groupby('scheme').agg({
        'relative_difference_pct': ['mean', 'std', 'count'],
        'better_strategy': lambda x: (x == 'full_finetune').sum(),
        'time_ratio': ['mean', 'std'],
        'param_ratio': ['mean', 'std']
    }).round(3)
    
    scheme_analysis.columns = ['_'.join(col).strip() for col in scheme_analysis.columns]
    
    # Calculate full fine-tuning preference rate
    scheme_analysis['full_finetune_preference_rate'] = (
        scheme_analysis['better_strategy_<lambda>'] / 
        scheme_analysis['relative_difference_pct_count'] * 100
    ).round(1)
    
    # Categorize schemes by their fine-tuning benefit
    scheme_analysis['benefit_category'] = scheme_analysis['relative_difference_pct_mean'].apply(
        lambda x: 'high_benefit' if x > 2 else 'medium_benefit' if x > 0 else 'low_benefit'
    )
    
    return scheme_analysis


def perform_strategy_statistical_tests(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform statistical tests comparing strategies for each domain-scheme combination.
    
    Args:
        raw_df: Raw experimental results DataFrame
        
    Returns:
        DataFrame with statistical test results
    """
    logger.info("Performing statistical tests for strategy comparisons")
    
    results = []
    
    for domain in DOMAINS:
        # Get primary metric for this domain
        domain_rows = raw_df[raw_df['domain_name'] == domain]
        if domain_rows.empty:
            continue
        task_type = domain_rows['task_type'].iloc[0]
        primary_metric = PRIMARY_METRICS[task_type]
        
        for scheme in PRETRAINED_SCHEMES:
            # Get data for both strategies
            linear_data = raw_df[
                (raw_df['domain_name'] == domain) & 
                (raw_df['pretrained_scheme'] == scheme) & 
                (raw_df['finetune_strategy'] == 'linear_probe')
            ][primary_metric].values
            
            full_data = raw_df[
                (raw_df['domain_name'] == domain) & 
                (raw_df['pretrained_scheme'] == scheme) & 
                (raw_df['finetune_strategy'] == 'full_finetune')
            ][primary_metric].values
            
            if len(linear_data) == 0 or len(full_data) == 0:
                continue
            
            # Perform paired t-test
            from scipy.stats import ttest_rel
            
            try:
                t_stat, p_value = ttest_rel(full_data, linear_data)  # full - linear
                
                # Calculate effect size (Cohen's d for paired samples)
                diff = full_data - linear_data
                cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                
                # Calculate mean improvement
                mean_linear, mean_full = np.mean(linear_data), np.mean(full_data)
                improvement = ((mean_full - mean_linear) / mean_linear) * 100 if mean_linear != 0 else 0
                
                results.append({
                    'domain': domain,
                    'scheme': scheme,
                    'task_type': task_type,
                    'linear_mean': mean_linear,
                    'full_mean': mean_full,
                    'improvement_pct': improvement,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    't_statistic': t_stat,
                    'significant': p_value < 0.05,
                    'effect_size_category': get_effect_size_category(abs(cohens_d)),
                    'better_strategy': 'full_finetune' if improvement > 0 else 'linear_probe'
                })
                
            except Exception as e:
                logger.warning(f"Statistical test failed for {domain}-{scheme}: {e}")
                continue
    
    # Apply Bonferroni correction
    if results:
        results_df = pd.DataFrame(results)
        from statsmodels.stats.multitest import multipletests
        _, corrected_p, _, _ = multipletests(results_df['p_value'], method='bonferroni')
        results_df['p_value_corrected'] = corrected_p
        results_df['significant_corrected'] = corrected_p < 0.05
        return results_df
    
    return pd.DataFrame()


def create_efficiency_analysis(strategy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create efficiency analysis comparing strategies.
    
    Args:
        strategy_df: Strategy comparison DataFrame
        
    Returns:
        DataFrame with efficiency analysis
    """
    logger.info("Creating efficiency analysis")
    
    # Filter out rows with missing efficiency data
    efficiency_df = strategy_df.dropna(subset=['time_ratio', 'param_ratio']).copy()
    
    if efficiency_df.empty:
        logger.warning("No efficiency data available")
        return pd.DataFrame()
    
    # Calculate efficiency metrics
    efficiency_df['time_cost'] = efficiency_df['time_ratio'] - 1  # Additional time cost
    efficiency_df['param_cost'] = efficiency_df['param_ratio'] - 1  # Additional parameter cost
    
    # Performance improvement per unit cost
    efficiency_df['perf_per_time_cost'] = efficiency_df['relative_difference_pct'] / efficiency_df['time_cost']
    efficiency_df['perf_per_param_cost'] = efficiency_df['relative_difference_pct'] / efficiency_df['param_cost']
    
    # Efficiency categories
    efficiency_df['efficiency_category'] = efficiency_df.apply(
        lambda row: 'highly_efficient' if row['relative_difference_pct'] > 1 and row['time_ratio'] < 2
        else 'moderately_efficient' if row['relative_difference_pct'] > 0 and row['time_ratio'] < 5
        else 'inefficient', axis=1
    )
    
    return efficiency_df


def create_rq3_summary_tables(strategy_df: pd.DataFrame, effectiveness_df: pd.DataFrame, 
                             statistical_df: pd.DataFrame, efficiency_df: pd.DataFrame) -> dict:
    """
    Create summary tables for RQ3 analysis.
    
    Args:
        strategy_df: Strategy comparison DataFrame
        effectiveness_df: Effectiveness analysis DataFrame
        statistical_df: Statistical test results DataFrame
        efficiency_df: Efficiency analysis DataFrame
        
    Returns:
        Dictionary of summary tables
    """
    logger.info("Creating RQ3 summary tables")
    
    summaries = {}
    
    # 1. Overall strategy preference
    total_combinations = len(strategy_df)
    full_preferred = (strategy_df['better_strategy'] == 'full_finetune').sum()
    summaries['overall_preference'] = pd.DataFrame({
        'strategy': ['linear_probe', 'full_finetune'],
        'preferred_count': [total_combinations - full_preferred, full_preferred],
        'preference_rate': [(total_combinations - full_preferred) / total_combinations * 100,
                           full_preferred / total_combinations * 100]
    })
    
    # 2. Strategy preference by task type
    task_preference = strategy_df.groupby('task_type')['better_strategy'].apply(
        lambda x: pd.Series({
            'linear_probe_count': (x == 'linear_probe').sum(),
            'full_finetune_count': (x == 'full_finetune').sum(),
            'total': len(x)
        })
    ).unstack(fill_value=0)
    task_preference['full_finetune_rate'] = (task_preference['full_finetune_count'] / 
                                           task_preference['total'] * 100).round(1)
    summaries['task_type_preference'] = task_preference
    
    # 3. Statistical significance summary
    if not statistical_df.empty:
        sig_summary = statistical_df.groupby('better_strategy').agg({
            'significant_corrected': 'sum',
            'improvement_pct': ['mean', 'std'],
            'cohens_d': ['mean', 'std']
        }).round(3)
        sig_summary.columns = ['_'.join(col).strip() for col in sig_summary.columns]
        summaries['significance_summary'] = sig_summary
    
    # 4. Efficiency summary
    if not efficiency_df.empty:
        eff_summary = efficiency_df.groupby('efficiency_category').agg({
            'relative_difference_pct': ['mean', 'count'],
            'time_ratio': ['mean', 'std'],
            'perf_per_time_cost': ['mean', 'std']
        }).round(3)
        eff_summary.columns = ['_'.join(col).strip() for col in eff_summary.columns]
        summaries['efficiency_summary'] = eff_summary
    
    return summaries


def analyze_rq3(agg_df: pd.DataFrame, raw_df: pd.DataFrame) -> tuple:
    """
    Complete RQ3 analysis: Fine-tuning Strategy Comparison.
    
    Args:
        agg_df: Aggregated results DataFrame
        raw_df: Raw experimental results DataFrame
        
    Returns:
        Tuple of analysis results
    """
    logger.info("Starting RQ3 Analysis: Fine-tuning Strategy Comparison")
    logger.info("=" * 50)
    
    try:
        # 1. Strategy comparison analysis
        strategy_df = compare_finetune_strategies(agg_df)
        
        # 2. Effectiveness analysis by scheme
        effectiveness_df = analyze_strategy_effectiveness(strategy_df)
        
        # 3. Statistical significance testing
        statistical_df = perform_strategy_statistical_tests(raw_df)
        
        # 4. Efficiency analysis
        efficiency_df = create_efficiency_analysis(strategy_df)
        
        # 5. Create summary tables
        summary_tables = create_rq3_summary_tables(strategy_df, effectiveness_df, 
                                                  statistical_df, efficiency_df)
        
        # 6. Save results
        logger.info("Saving RQ3 results")
        strategy_df.to_csv(RESULTS_DIR / 'rq3_strategy_comparison.csv', index=False)
        effectiveness_df.to_csv(RESULTS_DIR / 'rq3_effectiveness_analysis.csv', index=True)
        if not statistical_df.empty:
            statistical_df.to_csv(RESULTS_DIR / 'rq3_statistical_tests.csv', index=False)
        if not efficiency_df.empty:
            efficiency_df.to_csv(RESULTS_DIR / 'rq3_efficiency_analysis.csv', index=False)
        
        # Save summary tables
        for name, table in summary_tables.items():
            table.to_csv(RESULTS_DIR / f'rq3_summary_{name}.csv', index=True)
        
        logger.info("RQ3 results saved to analysis/results")
        
        # 7. Log key findings
        log_rq3_findings(strategy_df, effectiveness_df, statistical_df, efficiency_df, summary_tables)
        
        logger.info("RQ3 Analysis completed successfully!")
        
        return strategy_df, effectiveness_df, statistical_df, efficiency_df, summary_tables
        
    except Exception as e:
        logger.error(f"RQ3 Analysis failed: {e}")
        raise


def log_rq3_findings(strategy_df: pd.DataFrame, effectiveness_df: pd.DataFrame, 
                     statistical_df: pd.DataFrame, efficiency_df: pd.DataFrame, 
                     summary_tables: dict):
    """Log key findings from RQ3 analysis."""
    logger.info("\n" + "="*50)
    logger.info("RQ3 KEY FINDINGS: Fine-tuning Strategy Comparison")
    logger.info("="*50)
    
    # Overall strategy preference
    overall_pref = summary_tables['overall_preference']
    full_rate = overall_pref[overall_pref['strategy'] == 'full_finetune']['preference_rate'].iloc[0]
    
    logger.info(f"1. OVERALL STRATEGY PREFERENCE: Full fine-tuning preferred in {full_rate:.1f}% of cases")
    
    # Best improvement
    best_improvement = strategy_df['relative_difference_pct'].max()
    best_case = strategy_df.loc[strategy_df['relative_difference_pct'].idxmax()]
    
    logger.info(f"2. BEST IMPROVEMENT: {best_improvement:.2f}% for {best_case['domain']}-{best_case['scheme']}")
    
    # Statistical significance
    if not statistical_df.empty:
        sig_improvements = statistical_df[statistical_df['significant_corrected'] & (statistical_df['improvement_pct'] > 0)]
        logger.info(f"3. SIGNIFICANT IMPROVEMENTS: {len(sig_improvements)} combinations show significant gains")
    
    # Efficiency insights
    if not efficiency_df.empty:
        efficient_count = (efficiency_df['efficiency_category'] == 'highly_efficient').sum()
        total_eff = len(efficiency_df)
        logger.info(f"4. EFFICIENCY: {efficient_count}/{total_eff} combinations are highly efficient")
        
        avg_time_cost = efficiency_df['time_ratio'].mean()
        logger.info(f"5. AVERAGE TIME COST: {avg_time_cost:.1f}x training time for full fine-tuning")
    
    # Task type preferences
    task_pref = summary_tables['task_type_preference']
    best_task_for_full = task_pref['full_finetune_rate'].idxmax()
    best_rate = task_pref.loc[best_task_for_full, 'full_finetune_rate']
    
    logger.info(f"6. TASK TYPE PREFERENCE: {best_task_for_full} benefits most from full fine-tuning ({best_rate:.1f}%)")
    
    logger.info("="*50)


def main():
    logger.info("Starting Comprehensive Statistical Analysis")
    logger.info("=" * 50)

    # Load data
    agg_df = load_aggregated_data()
    raw_df = load_raw_data()

    # Perform RQ1 analysis
    improvement_df, statistical_df, best_schemes_df = analyze_rq1_effectiveness(agg_df, raw_df)
    
    # Perform RQ2 analysis
    combo_df, synergy_df, progressive_df, rq2_summary_tables = analyze_rq2(agg_df, raw_df)
    
    # Perform RQ3 analysis
    strategy_df, effectiveness_df, rq3_statistical_df, efficiency_df, rq3_summary_tables = analyze_rq3(agg_df, raw_df)
    
    logger.info("All analyses completed successfully!")


if __name__ == "__main__":
    main()

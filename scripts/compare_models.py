#!/usr/bin/env python3
"""
Model Comparison and Analysis Script.

This script compares the performance of different models (pre-trained vs from-scratch)
and provides statistical analysis of the results.

Usage:
    python compare_models.py --results-dir evaluation_results/
    python compare_models.py --wandb-project Graph-Finetuning-Results
"""

import argparse
import logging
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelComparator:
    """
    Compare and analyze different model performances.
    """
    
    def __init__(self, results_dir: Optional[str] = None, wandb_project: Optional[str] = None):
        """
        Initialize model comparator.
        
        Args:
            results_dir: Directory containing result files
            wandb_project: WandB project to fetch results from
        """
        self.results_dir = Path(results_dir) if results_dir else None
        self.wandb_project = wandb_project
        self.results_data = []
        self.comparison_df = None
    
    def load_results_from_files(self):
        """Load results from local files."""
        if not self.results_dir or not self.results_dir.exists():
            logging.warning(f"Results directory not found: {self.results_dir}")
            return
        
        # Look for result files
        result_files = list(self.results_dir.glob("*.yaml")) + list(self.results_dir.glob("*.json"))
        
        for result_file in result_files:
            try:
                if result_file.suffix == '.yaml':
                    with open(result_file, 'r') as f:
                        data = yaml.safe_load(f)
                elif result_file.suffix == '.json':
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                else:
                    continue
                
                self.results_data.append({
                    'source': 'file',
                    'file_path': str(result_file),
                    'data': data
                })
                
                logging.info(f"Loaded results from {result_file}")
                
            except Exception as e:
                logging.warning(f"Failed to load {result_file}: {e}")
    
    def load_results_from_wandb(self):
        """Load results from WandB project."""
        if not self.wandb_project:
            return
        
        try:
            api = wandb.Api()
            runs = api.runs(self.wandb_project)
            
            for run in runs:
                # Get run metrics and config
                run_data = {
                    'run_id': run.id,
                    'run_name': run.name,
                    'config': run.config,
                    'summary': run.summary._json_dict,
                    'state': run.state,
                    'tags': run.tags
                }
                
                self.results_data.append({
                    'source': 'wandb',
                    'run_id': run.id,
                    'data': run_data
                })
            
            logging.info(f"Loaded {len(runs)} runs from WandB project: {self.wandb_project}")
            
        except Exception as e:
            logging.error(f"Failed to load from WandB: {e}")
    
    def parse_results_to_dataframe(self) -> pd.DataFrame:
        """Parse loaded results into a structured DataFrame."""
        parsed_results = []
        
        for result_entry in self.results_data:
            data = result_entry['data']
            
            if result_entry['source'] == 'wandb':
                # Parse WandB run data
                run_data = data
                
                # Extract key information
                model_type = 'from_scratch' if 'from-scratch' in run_data.get('tags', []) else 'pretrained'
                dataset = self._extract_dataset_from_name(run_data['run_name'])
                
                # Extract metrics
                metrics = run_data['summary']
                
                parsed_result = {
                    'source': 'wandb',
                    'run_id': run_data['run_id'],
                    'run_name': run_data['run_name'],
                    'model_type': model_type,
                    'dataset': dataset,
                    'final_test_accuracy': metrics.get('final_test_accuracy', metrics.get('test_accuracy')),
                    'final_test_f1': metrics.get('final_test_f1', metrics.get('test_f1')),
                    'final_test_auc': metrics.get('final_test_auc', metrics.get('test_auc')),
                    'best_val_accuracy': metrics.get('best_val_accuracy'),
                    'training_time_seconds': metrics.get('training_time_seconds'),
                    'config': run_data['config']
                }
                
                parsed_results.append(parsed_result)
            
            elif result_entry['source'] == 'file':
                # Parse file-based results
                if 'results' in data:
                    # Evaluation suite results
                    for model_name, model_results in data['results'].items():
                        for task_name, task_result in model_results.items():
                            if task_result['status'] == 'success':
                                parsed_result = {
                                    'source': 'file',
                                    'file_path': result_entry['file_path'],
                                    'model_name': model_name,
                                    'task_name': task_name,
                                    'model_type': 'pretrained',
                                    'dataset': self._extract_dataset_from_task(task_name),
                                    'status': task_result['status'],
                                    'execution_time': task_result['execution_time']
                                }
                                parsed_results.append(parsed_result)
        
        self.comparison_df = pd.DataFrame(parsed_results)
        return self.comparison_df
    
    def _extract_dataset_from_name(self, run_name: str) -> str:
        """Extract dataset name from run name."""
        datasets = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES', 'FRANKENSTEIN', 'PTC_MR', 'Cora', 'CiteSeer']
        run_name_upper = run_name.upper()
        
        for dataset in datasets:
            if dataset in run_name_upper:
                return dataset
        
        return 'unknown'
    
    def _extract_dataset_from_task(self, task_name: str) -> str:
        """Extract dataset name from task name."""
        if 'MUTAG' in task_name:
            return 'MUTAG'
        elif 'ENZYMES' in task_name:
            return 'ENZYMES'
        elif 'FRANKENSTEIN' in task_name:
            return 'FRANKENSTEIN'
        elif 'Cora' in task_name:
            return 'Cora'
        elif 'CiteSeer' in task_name:
            return 'CiteSeer'
        else:
            return 'unknown'
    
    def compute_statistical_comparison(self) -> Dict[str, Any]:
        """Compute statistical comparison between model types."""
        if self.comparison_df is None or self.comparison_df.empty:
            logging.warning("No comparison data available")
            return {}
        
        results = {}
        
        # Group by dataset and model type
        grouped = self.comparison_df.groupby(['dataset', 'model_type'])
        
        for dataset in self.comparison_df['dataset'].unique():
            if dataset == 'unknown':
                continue
            
            dataset_data = self.comparison_df[self.comparison_df['dataset'] == dataset]
            
            # Get pretrained and from-scratch results
            pretrained_data = dataset_data[dataset_data['model_type'] == 'pretrained']
            scratch_data = dataset_data[dataset_data['model_type'] == 'from_scratch']
            
            dataset_results = {
                'dataset': dataset,
                'pretrained_count': len(pretrained_data),
                'scratch_count': len(scratch_data),
                'metrics_comparison': {}
            }
            
            # Compare metrics
            for metric in ['final_test_accuracy', 'final_test_f1', 'best_val_accuracy']:
                if metric in self.comparison_df.columns:
                    pretrained_values = pretrained_data[metric].dropna()
                    scratch_values = scratch_data[metric].dropna()
                    
                    if len(pretrained_values) > 0 and len(scratch_values) > 0:
                        # Compute statistics
                        pretrained_mean = pretrained_values.mean()
                        scratch_mean = scratch_values.mean()
                        improvement = pretrained_mean - scratch_mean
                        improvement_pct = (improvement / scratch_mean) * 100 if scratch_mean > 0 else 0
                        
                        # Statistical test
                        if len(pretrained_values) > 1 and len(scratch_values) > 1:
                            statistic, p_value = stats.ttest_ind(pretrained_values, scratch_values)
                        else:
                            statistic, p_value = None, None
                        
                        dataset_results['metrics_comparison'][metric] = {
                            'pretrained_mean': pretrained_mean,
                            'pretrained_std': pretrained_values.std(),
                            'scratch_mean': scratch_mean,
                            'scratch_std': scratch_values.std(),
                            'improvement': improvement,
                            'improvement_pct': improvement_pct,
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < 0.05 if p_value is not None else False
                        }
            
            results[dataset] = dataset_results
        
        return results
    
    def create_comparison_plots(self, output_dir: str = 'comparison_plots'):
        """Create visualization plots for model comparison."""
        if self.comparison_df is None or self.comparison_df.empty:
            logging.warning("No data available for plotting")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Accuracy comparison by dataset
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Filter out unknown datasets and missing accuracy values
        plot_data = self.comparison_df[
            (self.comparison_df['dataset'] != 'unknown') & 
            (self.comparison_df['final_test_accuracy'].notna())
        ]
        
        if not plot_data.empty:
            sns.boxplot(data=plot_data, x='dataset', y='final_test_accuracy', hue='model_type', ax=ax)
            ax.set_title('Test Accuracy Comparison: Pre-trained vs From-Scratch')
            ax.set_ylabel('Test Accuracy')
            ax.set_xlabel('Dataset')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved accuracy comparison plot to {output_path / 'accuracy_comparison.png'}")
        
        # 2. Training time comparison
        time_data = self.comparison_df[
            (self.comparison_df['training_time_seconds'].notna()) &
            (self.comparison_df['dataset'] != 'unknown')
        ]
        
        if not time_data.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=time_data, x='dataset', y='training_time_seconds', hue='model_type', ax=ax)
            ax.set_title('Training Time Comparison: Pre-trained vs From-Scratch')
            ax.set_ylabel('Training Time (seconds)')
            ax.set_xlabel('Dataset')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved training time comparison plot to {output_path / 'training_time_comparison.png'}")
        
        # 3. Performance improvement heatmap
        stats_results = self.compute_statistical_comparison()
        if stats_results:
            datasets = list(stats_results.keys())
            metrics = ['final_test_accuracy', 'final_test_f1', 'best_val_accuracy']
            
            improvement_matrix = []
            for dataset in datasets:
                row = []
                for metric in metrics:
                    if metric in stats_results[dataset]['metrics_comparison']:
                        improvement = stats_results[dataset]['metrics_comparison'][metric]['improvement_pct']
                        row.append(improvement)
                    else:
                        row.append(0)
                improvement_matrix.append(row)
            
            if improvement_matrix:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(improvement_matrix, 
                           xticklabels=[m.replace('final_test_', '').replace('best_val_', 'val_') for m in metrics],
                           yticklabels=datasets,
                           annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax)
                ax.set_title('Performance Improvement (%) of Pre-trained vs From-Scratch')
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Datasets')
                plt.tight_layout()
                plt.savefig(output_path / 'improvement_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logging.info(f"Saved improvement heatmap to {output_path / 'improvement_heatmap.png'}")
    
    def generate_comparison_report(self, output_file: str = 'model_comparison_report.md'):
        """Generate a comprehensive comparison report."""
        stats_results = self.compute_statistical_comparison()
        
        report_lines = [
            "# Model Comparison Report",
            "",
            "This report compares the performance of pre-trained models vs from-scratch baselines.",
            "",
            f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Experiments:** {len(self.comparison_df) if self.comparison_df is not None else 0}",
            "",
            "## Summary",
            ""
        ]
        
        if stats_results:
            # Overall summary
            total_improvements = 0
            significant_improvements = 0
            
            for dataset, results in stats_results.items():
                for metric, metric_results in results['metrics_comparison'].items():
                    if metric_results['improvement'] > 0:
                        total_improvements += 1
                    if metric_results.get('significant', False) and metric_results['improvement'] > 0:
                        significant_improvements += 1
            
            report_lines.extend([
                f"- **Datasets Analyzed:** {len(stats_results)}",
                f"- **Metrics Showing Improvement:** {total_improvements}",
                f"- **Statistically Significant Improvements:** {significant_improvements}",
                "",
                "## Detailed Results by Dataset",
                ""
            ])
            
            # Detailed results for each dataset
            for dataset, results in stats_results.items():
                report_lines.extend([
                    f"### {dataset}",
                    "",
                    f"- **Pre-trained Experiments:** {results['pretrained_count']}",
                    f"- **From-scratch Experiments:** {results['scratch_count']}",
                    ""
                ])
                
                # Metrics table
                if results['metrics_comparison']:
                    report_lines.extend([
                        "| Metric | Pre-trained | From-scratch | Improvement | P-value | Significant |",
                        "|--------|-------------|--------------|-------------|---------|-------------|"
                    ])
                    
                    for metric, metric_results in results['metrics_comparison'].items():
                        pretrained_mean = metric_results['pretrained_mean']
                        scratch_mean = metric_results['scratch_mean']
                        improvement_pct = metric_results['improvement_pct']
                        p_value = metric_results['p_value']
                        significant = metric_results['significant']
                        
                        report_lines.append(
                            f"| {metric} | {pretrained_mean:.3f} | {scratch_mean:.3f} | "
                            f"{improvement_pct:+.1f}% | {p_value:.3f if p_value else 'N/A'} | "
                            f"{'✓' if significant else '✗'} |"
                        )
                    
                    report_lines.append("")
        
        # Conclusions
        report_lines.extend([
            "## Conclusions",
            "",
            "Based on the statistical analysis:",
            ""
        ])
        
        if stats_results:
            # Generate conclusions based on results
            avg_improvements = []
            for dataset, results in stats_results.items():
                for metric, metric_results in results['metrics_comparison'].items():
                    if metric_results['improvement_pct'] != 0:
                        avg_improvements.append(metric_results['improvement_pct'])
            
            if avg_improvements:
                avg_improvement = np.mean(avg_improvements)
                if avg_improvement > 5:
                    report_lines.append("- **Pre-trained models show substantial performance improvements** over from-scratch baselines.")
                elif avg_improvement > 0:
                    report_lines.append("- Pre-trained models show modest improvements over from-scratch baselines.")
                else:
                    report_lines.append("- Pre-trained models show mixed results compared to from-scratch baselines.")
            
            # Count significant results
            sig_count = sum(1 for dataset, results in stats_results.items() 
                          for metric, metric_results in results['metrics_comparison'].items()
                          if metric_results.get('significant', False))
            
            if sig_count > 0:
                report_lines.append(f"- **{sig_count} statistically significant differences** were found.")
            else:
                report_lines.append("- No statistically significant differences were found.")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logging.info(f"Generated comparison report: {output_file}")
        
        return '\n'.join(report_lines)
    
    def run_complete_analysis(self, output_dir: str = 'model_analysis'):
        """Run complete model comparison analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logging.info("Starting complete model analysis...")
        
        # Load results
        if self.results_dir:
            self.load_results_from_files()
        
        if self.wandb_project:
            self.load_results_from_wandb()
        
        if not self.results_data:
            logging.error("No results data loaded. Check your data sources.")
            return
        
        # Parse to DataFrame
        df = self.parse_results_to_dataframe()
        logging.info(f"Parsed {len(df)} result entries")
        
        # Statistical analysis
        stats_results = self.compute_statistical_comparison()
        logging.info(f"Computed statistics for {len(stats_results)} datasets")
        
        # Generate plots
        plot_dir = output_path / 'plots'
        self.create_comparison_plots(str(plot_dir))
        
        # Generate report
        report_file = output_path / 'comparison_report.md'
        self.generate_comparison_report(str(report_file))
        
        # Save processed data
        if df is not None and not df.empty:
            df.to_csv(output_path / 'comparison_data.csv', index=False)
            logging.info(f"Saved processed data to {output_path / 'comparison_data.csv'}")
        
        # Save statistical results
        if stats_results:
            with open(output_path / 'statistical_results.json', 'w') as f:
                json.dump(stats_results, f, indent=2, default=str)
            logging.info(f"Saved statistical results to {output_path / 'statistical_results.json'}")
        
        logging.info(f"Complete analysis saved to {output_path}")
        
        return {
            'dataframe': df,
            'statistical_results': stats_results,
            'output_directory': str(output_path)
        }


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(description='Compare model performances')
    parser.add_argument('--results-dir', type=str,
                       help='Directory containing result files')
    parser.add_argument('--wandb-project', type=str,
                       help='WandB project to fetch results from')
    parser.add_argument('--output-dir', type=str, default='model_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    if not args.results_dir and not args.wandb_project:
        logging.error("Must specify either --results-dir or --wandb-project")
        return 1
    
    # Initialize comparator
    comparator = ModelComparator(args.results_dir, args.wandb_project)
    
    try:
        # Run complete analysis
        results = comparator.run_complete_analysis(args.output_dir)
        
        if results:
            logging.info("Model comparison analysis completed successfully!")
            return 0
        else:
            logging.error("Analysis failed - no results generated")
            return 1
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    exit(main()) 
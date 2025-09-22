import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_experiment_data():
    """Load experimental data."""
    results_dir = Path(__file__).parent / "results"
    experiment_data = pd.read_csv(results_dir / "experiment_results.csv")
    
    improvements = []
    for (domain, strategy), group in experiment_data.groupby(['domain_name', 'finetune_strategy']):
        baseline_data = group[group['pretrained_scheme'] == 'b1']
        if len(baseline_data) == 0:
            continue

        baseline_metric = baseline_data['accuracy' if domain not in ['CiteSeer_LP', 'Cora_LP'] else 'auc'].mean()

        for scheme in group['pretrained_scheme'].unique():
            if scheme == 'b1':
                continue

            scheme_data = group[group['pretrained_scheme'] == scheme]
            if len(scheme_data) == 0:
                continue

            metric_col = 'accuracy' if domain not in ['CiteSeer_LP', 'Cora_LP'] else 'auc'

            for _, row in scheme_data.iterrows():
                improvement_pct = ((row[metric_col] - baseline_metric) / baseline_metric) * 100
                improvements.append({
                    'domain': domain,
                    'scheme': scheme,
                    'strategy': strategy,
                    'improvement_percent': improvement_pct,
                    'task_type': get_task_type(domain),
                    'seed': row['seed']
                })

    return pd.DataFrame(improvements), experiment_data


def get_task_type(domain):
    """Map domain names to task types."""
    if domain in ['ENZYMES', 'PTC_MR']:
        return 'Graph Classification'
    elif domain in ['Cora_NC', 'CiteSeer_NC']:
        return 'Node Classification'
    elif domain in ['Cora_LP', 'CiteSeer_LP']:
        return 'Link Prediction'
    return 'unknown'


def create_domain_heatmap(improvements_df, output_dir):
    """Create Figure 1: Domain performance heatmap."""
    improvements_df['scheme_strategy'] = improvements_df['scheme'] + '_' + improvements_df['strategy'].str.replace('_probe', '').str.replace('_finetune', '').str.replace('linear', 'LP').str.replace('full', 'FT')
    
    pivot_data = improvements_df.groupby(['domain', 'scheme_strategy'])['improvement_percent'].mean().reset_index()
    heatmap_data = pivot_data.pivot(index='domain', columns='scheme_strategy', values='improvement_percent')
    
    domain_order = ['ENZYMES', 'PTC_MR', 'Cora_NC', 'CiteSeer_NC', 'Cora_LP', 'CiteSeer_LP']
    scheme_order = ['b2_LP', 'b2_FT', 'b3_LP', 'b3_FT', 's1_LP', 's1_FT', 's2_LP', 's2_FT', 's3_LP', 's3_FT', 's4_LP', 's4_FT', 's5_LP', 's5_FT']
    
    heatmap_data = heatmap_data.reindex(index=domain_order, columns=[col for col in scheme_order if col in heatmap_data.columns])
    
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdBu_r', center=0, cbar_kws={'label': 'Mean Improvement (%)'}, annot_kws={'size': 10})
    
    from matplotlib.patches import Circle
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not pd.isna(value) and value > 0:
                circle = Circle((j + 0.5, i + 0.5), 0.4, fill=False, edgecolor='red', linewidth=3)
                ax.add_patch(circle)
    
    plt.title('Mean Performance Improvements by Domain and Pre-training Scheme-Strategy', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Pre-training Scheme + Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Downstream Domain', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / "domain_performance_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated Figure 1: {output_path}")


def create_task_type_heatmap(improvements_df, output_dir):
    """Create Figure 2: Task type performance heatmap."""
    improvements_df['scheme_strategy'] = improvements_df['scheme'] + '_' + improvements_df['strategy'].str.replace('_probe', '').str.replace('_finetune', '').str.replace('linear', 'LP').str.replace('full', 'FT')
    
    pivot_data = improvements_df.groupby(['task_type', 'scheme_strategy'])['improvement_percent'].mean().reset_index()
    heatmap_data = pivot_data.pivot(index='task_type', columns='scheme_strategy', values='improvement_percent')
    
    task_order = ['Graph Classification', 'Node Classification', 'Link Prediction']
    scheme_order = ['b2_LP', 'b2_FT', 'b3_LP', 'b3_FT', 's1_LP', 's1_FT', 's2_LP', 's2_FT', 's3_LP', 's3_FT', 's4_LP', 's4_FT', 's5_LP', 's5_FT']
    
    heatmap_data = heatmap_data.reindex(index=task_order, columns=[col for col in scheme_order if col in heatmap_data.columns])
    
    plt.figure(figsize=(14, 6))
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdBu_r', center=0, cbar_kws={'label': 'Mean Improvement (%)'}, annot_kws={'size': 12})
    
    from matplotlib.patches import Circle
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not pd.isna(value) and value > 0:
                circle = Circle((j + 0.5, i + 0.5), 0.4, fill=False, edgecolor='red', linewidth=3)
                ax.add_patch(circle)
    
    plt.title('Mean Performance Improvements by Task Type and Pre-training Scheme-Strategy', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Pre-training Scheme + Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Task Type', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / "task_type_performance_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated Figure 2: {output_path}")


def create_table1_full_finetuning(experiment_data, output_dir):
    """Generate Table 1: Full Fine-tuning Efficiency."""
    domains = experiment_data['domain_name'].unique()
    schemes = ['b2', 'b3', 's1', 's2', 's3', 's4', 's5']
    
    results = []
    
    for domain in domains:
        domain_data = experiment_data[experiment_data['domain_name'] == domain]
        
        baseline_ft = domain_data[
            (domain_data['pretrained_scheme'] == 'b1') & 
            (domain_data['finetune_strategy'] == 'full_finetune')
        ]
        
        if len(baseline_ft) == 0:
            continue
            
        baseline_time = baseline_ft['training_time'].mean()
        baseline_epochs = baseline_ft['convergence_epochs'].mean()
        baseline_time_per_epoch = baseline_time / baseline_epochs if baseline_epochs > 0 else 0
        
        for scheme in schemes:
            scheme_ft = domain_data[
                (domain_data['pretrained_scheme'] == scheme) & 
                (domain_data['finetune_strategy'] == 'full_finetune')
            ]
            
            if len(scheme_ft) > 0:
                scheme_time = scheme_ft['training_time'].mean()
                scheme_epochs = scheme_ft['convergence_epochs'].mean()
                scheme_time_per_epoch = scheme_time / scheme_epochs if scheme_epochs > 0 else 0
                
                time_speedup = baseline_time_per_epoch / scheme_time_per_epoch if scheme_time_per_epoch > 0 else 1.0
                conv_speedup = baseline_epochs / scheme_epochs if scheme_epochs > 0 else 1.0
                overall_speedup = time_speedup * conv_speedup
                
                results.append({
                    'Domain': domain,
                    'Scheme': scheme,
                    'Overall_Speedup': overall_speedup
                })
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "table1_full_finetuning.csv", index=False)
    print(f"✓ Generated Table 1 data: table1_full_finetuning.csv")
    return df


def create_table2_linear_probing(experiment_data, output_dir):
    """Generate Table 2: Linear Probing Efficiency."""
    domains = experiment_data['domain_name'].unique()
    schemes = ['b2', 'b3', 's1', 's2', 's3', 's4', 's5']
    
    results = []
    
    for domain in domains:
        domain_data = experiment_data[experiment_data['domain_name'] == domain]
        
        baseline_ft = domain_data[
            (domain_data['pretrained_scheme'] == 'b1') & 
            (domain_data['finetune_strategy'] == 'full_finetune')
        ]
        baseline_lp = domain_data[
            (domain_data['pretrained_scheme'] == 'b1') & 
            (domain_data['finetune_strategy'] == 'linear_probe')
        ]
        
        if len(baseline_ft) == 0 or len(baseline_lp) == 0:
            continue
            
        baseline_ft_time = baseline_ft['training_time'].mean()
        baseline_ft_epochs = baseline_ft['convergence_epochs'].mean()
        baseline_ft_time_per_epoch = baseline_ft_time / baseline_ft_epochs if baseline_ft_epochs > 0 else 0
        baseline_ft_params = baseline_ft['trainable_parameters'].mean()
        
        for scheme in schemes:
            scheme_lp = domain_data[
                (domain_data['pretrained_scheme'] == scheme) & 
                (domain_data['finetune_strategy'] == 'linear_probe')
            ]
            
            if len(scheme_lp) > 0:
                scheme_time = scheme_lp['training_time'].mean()
                scheme_epochs = scheme_lp['convergence_epochs'].mean()
                scheme_time_per_epoch = scheme_time / scheme_epochs if scheme_epochs > 0 else 0
                scheme_lp_params = scheme_lp['trainable_parameters'].mean()
                
                time_speedup = baseline_ft_time_per_epoch / scheme_time_per_epoch if scheme_time_per_epoch > 0 else 1.0
                conv_speedup = baseline_ft_epochs / scheme_epochs if scheme_epochs > 0 else 1.0
                overall_speedup = time_speedup * conv_speedup
                param_efficiency = baseline_ft_params / scheme_lp_params if scheme_lp_params > 0 else 1.0
                
                results.append({
                    'Domain': domain,
                    'Scheme': scheme,
                    'Overall_Speedup': overall_speedup,
                    'Parameter_Efficiency': param_efficiency
                })
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "table2_linear_probing.csv", index=False)
    print(f"✓ Generated Table 2 data: table2_linear_probing.csv")
    return df


def create_table3_full_finetune_performance(experiment_data, output_dir):
    """Generate Table 3: Full Fine-tuning Performance Analysis - EXACT MATCH with Figure 1."""
    results = {}
    all_schemes = ['b2', 'b3', 'b4', 's1', 's2', 's3', 's4', 's5']
    
    # Use EXACT same computation as Figure 1 heatmap
    for domain in ['PTC_MR', 'ENZYMES', 'Cora_NC', 'CiteSeer_NC', 'Cora_LP', 'CiteSeer_LP']:
        # Get baseline data for this domain with full_finetune strategy
        baseline_data = experiment_data[
            (experiment_data['domain_name'] == domain) & 
            (experiment_data['pretrained_scheme'] == 'b1') & 
            (experiment_data['finetune_strategy'] == 'full_finetune')
        ]
        
        if len(baseline_data) == 0:
            continue
            
        # Use same metric selection as Figure 1
        metric_col = 'accuracy' if domain not in ['CiteSeer_LP', 'Cora_LP'] else 'auc'
        baseline_metric = baseline_data[metric_col].mean()
        
        for scheme in all_schemes:
            # Get scheme data for full_finetune strategy
            scheme_data = experiment_data[
                (experiment_data['domain_name'] == domain) & 
                (experiment_data['pretrained_scheme'] == scheme) & 
                (experiment_data['finetune_strategy'] == 'full_finetune')
            ]
            
            if len(scheme_data) > 0:
                # Compute improvement exactly like Figure 1
                scheme_metric = scheme_data[metric_col].mean()
                improvement_pct = ((scheme_metric - baseline_metric) / baseline_metric) * 100
                results[f'{scheme}_FT_{domain}'] = improvement_pct
    
    # Save results
    import json
    with open(output_dir / "table3_full_finetune_performance.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Generated Table 3 data: table3_full_finetune_performance.json")
    return results


def create_table4_linear_probe_performance(experiment_data, output_dir):
    """Generate Table 4: Linear Probing Performance Analysis - EXACT MATCH with Figure 1."""
    results = {}
    all_schemes = ['b2', 'b3', 'b4', 's1', 's2', 's3', 's4', 's5']
    
    # Use EXACT same computation as Figure 1 heatmap
    for domain in ['PTC_MR', 'ENZYMES', 'Cora_NC', 'CiteSeer_NC', 'Cora_LP', 'CiteSeer_LP']:
        # Get baseline data for this domain with LINEAR_PROBE strategy (CRITICAL FIX!)
        baseline_data = experiment_data[
            (experiment_data['domain_name'] == domain) & 
            (experiment_data['pretrained_scheme'] == 'b1') & 
            (experiment_data['finetune_strategy'] == 'linear_probe')
        ]
        
        if len(baseline_data) == 0:
            continue
            
        # Use same metric selection as Figure 1
        metric_col = 'accuracy' if domain not in ['CiteSeer_LP', 'Cora_LP'] else 'auc'
        baseline_metric = baseline_data[metric_col].mean()
        
        for scheme in all_schemes:
            # Get scheme data for linear_probe strategy
            scheme_data = experiment_data[
                (experiment_data['domain_name'] == domain) & 
                (experiment_data['pretrained_scheme'] == scheme) & 
                (experiment_data['finetune_strategy'] == 'linear_probe')
            ]
            
            if len(scheme_data) > 0:
                # Compute improvement exactly like Figure 1
                scheme_metric = scheme_data[metric_col].mean()
                improvement_pct = ((scheme_metric - baseline_metric) / baseline_metric) * 100
                results[f'{scheme}_LIN_{domain}'] = improvement_pct
    
    # Save results
    import json
    with open(output_dir / "table4_linear_probe_performance.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Generated Table 4 data: table4_linear_probe_performance.json")
    return results


def main():
    results_dir = Path(__file__).parent / "results"
    
    # Load data
    improvements_df, experiment_data = load_experiment_data()
    
    # Generate exact outputs used in paper:
    print("Generating paper data files...")
    
    # Figures 1 & 2
    create_domain_heatmap(improvements_df.copy(), results_dir)
    create_task_type_heatmap(improvements_df.copy(), results_dir)
    
    # Tables 1, 2, 3, 4
    create_table1_full_finetuning(experiment_data, results_dir)
    create_table2_linear_probing(experiment_data, results_dir)
    create_table3_full_finetune_performance(experiment_data, results_dir)
    create_table4_linear_probe_performance(experiment_data, results_dir)
    
    print("\n" + "="*60)
    print("✅ ALL PAPER DATA FILES GENERATED SUCCESSFULLY!")
    print("="*60)
    print("📁 Files created in analysis/results/:")
    print("   - domain_performance_heatmap.png (Figure 1)")
    print("   - task_type_performance_heatmap.png (Figure 2)")
    print("   - table1_full_finetuning.csv (Table 1 data)")
    print("   - table2_linear_probing.csv (Table 2 data)")
    print("   - table3_full_finetune_performance.json (Table 3 data)")
    print("   - table4_linear_probe_performance.json (Table 4 data)")


if __name__ == "__main__":
    main()
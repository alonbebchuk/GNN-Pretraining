import wandb
import pandas as pd
import numpy as np
from collections import defaultdict

# Initialize wandb API
api = wandb.Api()

# Define the runs to analyze - mapping names to actual run IDs
run_mapping = {
    'b2_42': 'vesia2mn',
    'b3_42': 'n4ew4pbk', 
    'b4_42': 'jlt3mpx4',
    's1_42': 'jgw2das3',
    's2_42': 'iwvi5sa4',
    's3_42': '2l2inzyc',
    's4_42': 'hgizj0vm',
    's5_42': 'cwph8hc8'
}
project_path = "timoshka3-tel-aviv-university/GNN"

def fetch_run_data(run_name, run_id):
    """Fetch comprehensive data for a given run"""
    try:
        run = api.run(f"{project_path}/{run_id}")
        
        # Get run configuration
        config = run.config
        
        # Get summary metrics (final values)
        summary = run.summary
        
        # Get full history
        history = run.scan_history()
        history_df = pd.DataFrame(history)
        
        return {
            'run_name': run_name,
            'run_id': run_id,
            'config': config,
            'summary': summary,
            'history': history_df,
            'state': run.state,
            'tags': run.tags
        }
    except Exception as e:
        print(f"Error fetching {run_name} ({run_id}): {e}")
        return None

# Fetch data for all runs
print("Fetching run data...")
run_data = {}
for run_name, run_id in run_mapping.items():
    print(f"Fetching {run_name} ({run_id})...")
    data = fetch_run_data(run_name, run_id)
    if data:
        run_data[run_name] = data

print(f"Successfully fetched data for {len(run_data)} runs")

# Analyze the data
print("\n" + "="*80)
print("PRETRAINING RUNS ANALYSIS")
print("="*80)

for run_name, data in run_data.items():
    print(f"\n--- RUN {run_name} ---")
    print(f"State: {data['state']}")
    print(f"Tags: {data.get('tags', [])}")
    
    # Configuration analysis
    config = data['config']
    print(f"Config keys: {list(config.keys())}")
    
    # Summary metrics analysis  
    summary = data['summary']
    print(f"Summary keys: {list(summary.keys())}")
    
    # History analysis
    history = data['history']
    print(f"History shape: {history.shape}")
    print(f"History columns: {list(history.columns)}")
    
    # Key metrics overview
    if 'val/loss/total' in summary:
        print(f"Final validation loss: {summary['val/loss/total']:.4f}")
    if 'train/loss/total' in summary:
        print(f"Final training loss: {summary['train/loss/total']:.4f}")
    
    # Check for convergence indicators
    if len(history) > 0:
        print(f"Training epochs: {len(history)}")
        if 'train/progress/epoch' in history.columns:
            max_epoch = history['train/progress/epoch'].max()
            print(f"Max epoch reached: {max_epoch}")
        
        # Look for loss trends
        for loss_col in ['train/loss/total', 'val/loss/total']:
            if loss_col in history.columns:
                loss_values = history[loss_col].dropna()
                if len(loss_values) > 1:
                    initial = loss_values.iloc[0]
                    final = loss_values.iloc[-1] 
                    improvement = ((initial - final) / initial) * 100
                    print(f"{loss_col} - Initial: {initial:.4f}, Final: {final:.4f}, Improvement: {improvement:.2f}%")

# Deep analysis of task-specific losses
print("\n" + "="*80)
print("TASK-SPECIFIC LOSS ANALYSIS")
print("="*80)

task_names = ['node_feature_masking', 'link_prediction', 'node_contrastive', 
              'graph_contrastive', 'graph_property_prediction', 'domain_adv']
domain_names = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']

for run_name, data in run_data.items():
    print(f"\n--- TASK BREAKDOWN FOR {run_name} ---")
    history = data['history']
    summary = data['summary']
    
    # Check task-level losses
    for task in task_names:
        train_key = f'train/loss/{task}'
        val_key = f'val/loss/{task}'
        
        if train_key in summary:
            print(f"{task}: Train={summary[train_key]:.4f}, Val={summary.get(val_key, 'N/A')}")
    
    # Check domain-task losses
    print("Domain-Task Losses:")
    for domain in domain_names:
        domain_losses = []
        for task in task_names[:-1]:  # Exclude domain_adv
            key = f'train/loss/{domain}/{task}'
            if key in summary:
                domain_losses.append(f"{task[:3]}: {summary[key]:.3f}")
        if domain_losses:
            print(f"  {domain}: {', '.join(domain_losses)}")

# Training dynamics analysis
print("\n" + "="*80)
print("TRAINING DYNAMICS ANALYSIS")
print("="*80)

for run_name, data in run_data.items():
    print(f"\n--- DYNAMICS FOR {run_name} ---")
    history = data['history']
    
    # Gradient norms
    if 'train/gradients/model_grad_norm' in history.columns:
        grad_norms = history['train/gradients/model_grad_norm'].dropna()
        if len(grad_norms) > 0:
            print(f"Gradient norm - Mean: {grad_norms.mean():.4f}, Std: {grad_norms.std():.4f}, Max: {grad_norms.max():.4f}")
    
    # Domain adversarial lambda (for s5)
    if 'val/domain_adv/lambda' in history.columns:
        lambdas = history['val/domain_adv/lambda'].dropna()
        if len(lambdas) > 0:
            print(f"Domain adversarial lambda - Final: {lambdas.iloc[-1]:.4f}")
    
    # Check for training instability
    if 'train/loss/total' in history.columns:
        train_losses = history['train/loss/total'].dropna()
        if len(train_losses) > 10:
            # Look for large jumps or oscillations
            loss_diffs = train_losses.diff().abs()
            large_jumps = (loss_diffs > loss_diffs.quantile(0.95)).sum()
            print(f"Training stability - Large loss jumps: {large_jumps}, Loss variance: {train_losses.var():.6f}")

print(f"\nAnalysis complete. Data saved for {len(run_data)} runs.")

#!/usr/bin/env python3

import wandb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

def analyze_completed_runs():
    """Analyze the completed pretraining runs: b2_42, b3_42, b4_42, s1_42"""
    
    # Initialize WandB
    api = wandb.Api()
    project = "alonbebchuk-tel-aviv-university/gnn-pretraining-pretrain"
    
    # Target runs to analyze - all pretraining schemes
    target_runs = ["b2_42", "b3_42", "b4_42", "s1_42", "s2_42", "s3_42", "s4_42", "s5_42"]
    
    print("🔍 ANALYZING COMPLETED PRETRAINING RUNS")
    print("=" * 50)
    
    # Fetch runs
    runs_data = {}
    for run_name in target_runs:
        try:
            runs = api.runs(project, filters={"display_name": run_name})
            if runs:
                run = runs[0]  # Get the first (most recent) run with this name
                runs_data[run_name] = run
                print(f"✅ Found {run_name}: {run.state}")
            else:
                print(f"❌ Run {run_name} not found")
        except Exception as e:
            print(f"❌ Error fetching {run_name}: {e}")
    
    if not runs_data:
        print("❌ No runs found to analyze")
        return
    
    # Analyze each run
    analysis_results = {}
    
    for run_name, run in runs_data.items():
        print(f"\n📊 ANALYZING {run_name.upper()}")
        print("-" * 30)
        
        try:
            # Get run history (metrics over time)
            history = run.scan_history()
            df = pd.DataFrame(history)
            
            if df.empty:
                print(f"⚠️  No history data for {run_name}")
                continue
            
            # Extract key metrics
            analysis = analyze_run_metrics(df, run_name)
            analysis_results[run_name] = analysis
            
            # Print run summary
            print_run_summary(analysis, run_name)
            
        except Exception as e:
            print(f"❌ Error analyzing {run_name}: {e}")
    
    # Comparative analysis
    print("\n🔬 COMPARATIVE ANALYSIS")
    print("=" * 50)
    comparative_analysis(analysis_results)
    
    # Research insights
    print("\n🎯 RESEARCH INSIGHTS")
    print("=" * 50)
    research_insights(analysis_results)
    
    return analysis_results

def analyze_run_metrics(df: pd.DataFrame, run_name: str) -> Dict:
    """Analyze metrics for a single run"""
    
    analysis = {
        'run_name': run_name,
        'total_epochs': len(df),
        'final_metrics': {},
        'training_dynamics': {},
        'stability': {}
    }
    
    # Get final epoch metrics
    if not df.empty:
        final_row = df.iloc[-1]
        
        # Extract final validation loss
        val_loss_cols = [col for col in df.columns if 'val' in col.lower() and 'loss' in col.lower()]
        if val_loss_cols:
            analysis['final_metrics']['val_loss'] = final_row.get(val_loss_cols[0], None)
        
        # Extract final training loss
        train_loss_cols = [col for col in df.columns if 'train' in col.lower() and 'loss' in col.lower() and 'weight' not in col.lower()]
        if train_loss_cols:
            analysis['final_metrics']['train_loss'] = final_row.get(train_loss_cols[0], None)
    
    # Analyze training dynamics
    for col in df.columns:
        if 'loss' in col.lower() and df[col].dtype in [np.float64, np.float32]:
            try:
                values = df[col].dropna()
                if len(values) > 1:
                    analysis['training_dynamics'][col] = {
                        'initial': float(values.iloc[0]),
                        'final': float(values.iloc[-1]),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'improvement': float(values.iloc[0] - values.iloc[-1]),
                        'improvement_pct': float((values.iloc[0] - values.iloc[-1]) / values.iloc[0] * 100) if values.iloc[0] != 0 else 0
                    }
            except:
                continue
    
    # Analyze stability (coefficient of variation for loss)
    train_loss_col = None
    for col in df.columns:
        if 'train' in col.lower() and 'loss' in col.lower() and 'weight' not in col.lower():
            train_loss_col = col
            break
    
    if train_loss_col and train_loss_col in df.columns:
        values = df[train_loss_col].dropna()
        if len(values) > 5:  # Need enough points for stability analysis
            # Coefficient of Variation (CV) - lower is more stable
            cv = float(values.std() / values.mean()) if values.mean() != 0 else float('inf')
            analysis['stability']['cv'] = cv
            
            # Check for convergence (last 20% of training)
            last_20pct = values.iloc[int(0.8 * len(values)):]
            if len(last_20pct) > 1:
                analysis['stability']['late_cv'] = float(last_20pct.std() / last_20pct.mean()) if last_20pct.mean() != 0 else float('inf')
    
    return analysis

def print_run_summary(analysis: Dict, run_name: str):
    """Print summary for a single run"""
    
    # Map run names to task descriptions
    task_descriptions = {
        'b2_42': 'Node Feature Masking (Generative)',
        'b3_42': 'Node Contrastive Learning',
        'b4_42': 'All 5 Tasks (Mixed)',
        's1_42': 'NFM + Link Prediction (Both Generative)',
        's2_42': 'Node + Graph Contrastive (Both Contrastive)',
        's3_42': 'Generative + Contrastive Mix (4 Tasks)',
        's4_42': 'All 5 Core Tasks (Maximum Multi-task)',
        's5_42': 'All 5 Tasks + Domain Adversarial (Ultimate)'
    }
    
    print(f"📋 {task_descriptions.get(run_name, run_name)}")
    print(f"   Epochs: {analysis['total_epochs']}")
    
    # Final metrics
    if 'val_loss' in analysis['final_metrics'] and analysis['final_metrics']['val_loss'] is not None:
        print(f"   Final Val Loss: {analysis['final_metrics']['val_loss']:.4f}")
    
    if 'train_loss' in analysis['final_metrics'] and analysis['final_metrics']['train_loss'] is not None:
        print(f"   Final Train Loss: {analysis['final_metrics']['train_loss']:.4f}")
    
    # Training stability
    if 'cv' in analysis['stability']:
        stability_rating = "Excellent" if analysis['stability']['cv'] < 0.1 else \
                          "Good" if analysis['stability']['cv'] < 0.3 else \
                          "Fair" if analysis['stability']['cv'] < 0.5 else "Poor"
        print(f"   Training Stability: {stability_rating} (CV: {analysis['stability']['cv']:.3f})")
    
    # Best improvement
    best_improvement = 0
    best_metric = None
    for metric, stats in analysis['training_dynamics'].items():
        if stats['improvement'] > best_improvement:
            best_improvement = stats['improvement']
            best_metric = metric
    
    if best_metric:
        print(f"   Best Improvement: {best_improvement:.4f} in {best_metric}")

def comparative_analysis(results: Dict):
    """Compare performance across runs"""
    
    if len(results) < 2:
        print("⚠️  Need at least 2 runs for comparison")
        return
    
    # Extract validation losses for comparison
    val_losses = {}
    train_losses = {}
    stabilities = {}
    
    for run_name, analysis in results.items():
        if 'val_loss' in analysis['final_metrics'] and analysis['final_metrics']['val_loss'] is not None:
            val_losses[run_name] = analysis['final_metrics']['val_loss']
        
        if 'train_loss' in analysis['final_metrics'] and analysis['final_metrics']['train_loss'] is not None:
            train_losses[run_name] = analysis['final_metrics']['train_loss']
        
        if 'cv' in analysis['stability']:
            stabilities[run_name] = analysis['stability']['cv']
    
    # Rank by validation loss (lower is better)
    if val_losses:
        print("🏆 VALIDATION LOSS RANKING (Lower is Better):")
        sorted_val = sorted(val_losses.items(), key=lambda x: x[1])
        for i, (run_name, loss) in enumerate(sorted_val, 1):
            print(f"   {i}. {run_name}: {loss:.4f}")
    
    # Rank by stability (lower CV is better)
    if stabilities:
        print("\n📊 TRAINING STABILITY RANKING (Lower CV is Better):")
        sorted_stability = sorted(stabilities.items(), key=lambda x: x[1])
        for i, (run_name, cv) in enumerate(sorted_stability, 1):
            stability_rating = "Excellent" if cv < 0.1 else \
                              "Good" if cv < 0.3 else \
                              "Fair" if cv < 0.5 else "Poor"
            print(f"   {i}. {run_name}: CV={cv:.3f} ({stability_rating})")

def research_insights(results: Dict):
    """Generate research insights from the analysis"""
    
    task_mapping = {
        'b2_42': {'tasks': ['node_feat_mask'], 'type': 'generative', 'complexity': 'single'},
        'b3_42': {'tasks': ['node_contrast'], 'type': 'contrastive', 'complexity': 'single'},
        'b4_42': {'tasks': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop'], 'type': 'mixed', 'complexity': 'multi'},
        's1_42': {'tasks': ['node_feat_mask', 'link_pred'], 'type': 'generative', 'complexity': 'multi'},
        's2_42': {'tasks': ['node_contrast', 'graph_contrast'], 'type': 'contrastive', 'complexity': 'multi'},
        's3_42': {'tasks': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast'], 'type': 'mixed', 'complexity': 'multi'},
        's4_42': {'tasks': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop'], 'type': 'mixed', 'complexity': 'multi'},
        's5_42': {'tasks': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop', 'domain_adv'], 'type': 'mixed_adv', 'complexity': 'multi'}
    }
    
    val_losses = {}
    stabilities = {}
    
    for run_name, analysis in results.items():
        if 'val_loss' in analysis['final_metrics'] and analysis['final_metrics']['val_loss'] is not None:
            val_losses[run_name] = analysis['final_metrics']['val_loss']
        if 'cv' in analysis['stability']:
            stabilities[run_name] = analysis['stability']['cv']
    
    print("🔍 Key Findings:")
    
    # Single task vs multi-task comparison
    single_tasks = ['b2_42', 'b3_42']
    multi_tasks = ['b4_42', 's1_42', 's2_42', 's3_42', 's4_42', 's5_42']
    
    single_losses = [val_losses.get(task) for task in single_tasks if task in val_losses]
    multi_losses = [val_losses.get(task) for task in multi_tasks if task in val_losses]
    
    if single_losses and multi_losses:
        avg_single = np.mean(single_losses)
        avg_multi = np.mean(multi_losses)
        print(f"   • Single-task avg validation loss: {avg_single:.4f}")
        print(f"   • Multi-task avg validation loss: {avg_multi:.4f}")
        
        if avg_multi < avg_single:
            print(f"   ✅ Multi-task pretraining shows {((avg_single - avg_multi)/avg_single*100):.1f}% improvement!")
        else:
            print(f"   ⚠️  Single-task outperforms multi-task by {((avg_multi - avg_single)/avg_single*100):.1f}%")
    
    # Generative vs contrastive
    if 'b2_42' in val_losses and 'b3_42' in val_losses:
        generative_loss = val_losses['b2_42']
        contrastive_loss = val_losses['b3_42']
        print(f"   • Generative (b2): {generative_loss:.4f} vs Contrastive (b3): {contrastive_loss:.4f}")
        
        if generative_loss < contrastive_loss:
            print(f"   ✅ Generative pretraining outperforms contrastive by {((contrastive_loss - generative_loss)/generative_loss*100):.1f}%")
        else:
            print(f"   ✅ Contrastive pretraining outperforms generative by {((generative_loss - contrastive_loss)/contrastive_loss*100):.1f}%")
    
    # Multi-task contrastive performance (s2_42)
    if 's2_42' in val_losses:
        print(f"   • Multi-contrastive (s2): {val_losses['s2_42']:.4f} - Fixed crash issues")
    
    # Domain adversarial analysis
    if 's4_42' in val_losses and 's5_42' in val_losses:
        base_loss = val_losses['s4_42']
        domain_adv_loss = val_losses['s5_42']
        print(f"   • Domain Adversarial Impact: s4 ({base_loss:.4f}) vs s5 ({domain_adv_loss:.4f})")
        
        if domain_adv_loss < base_loss:
            print(f"   ✅ Domain adversarial improves performance by {((base_loss - domain_adv_loss)/base_loss*100):.1f}%")
        else:
            print(f"   ⚠️ Domain adversarial reduces performance by {((domain_adv_loss - base_loss)/base_loss*100):.1f}%")
    
    # Task complexity analysis
    complexity_groups = {
        'single_gen': ['b2_42'],
        'single_con': ['b3_42'], 
        'multi_gen': ['s1_42'],
        'multi_con': ['s2_42'],
        'multi_mixed': ['s3_42', 'b4_42', 's4_42'],
        'multi_adv': ['s5_42']
    }
    
    for group_name, schemes in complexity_groups.items():
        group_losses = [val_losses.get(scheme) for scheme in schemes if scheme in val_losses]
        if group_losses:
            avg_loss = np.mean(group_losses)
            print(f"   • {group_name.replace('_', ' ').title()}: {avg_loss:.4f} (avg of {len(group_losses)} schemes)")
    
    # Stability insights
    if stabilities:
        most_stable = min(stabilities.items(), key=lambda x: x[1])
        least_stable = max(stabilities.items(), key=lambda x: x[1])
        print(f"   • Most stable training: {most_stable[0]} (CV: {most_stable[1]:.3f})")
        print(f"   • Least stable training: {least_stable[0]} (CV: {least_stable[1]:.3f})")
    
    print("\n💡 Recommendations:")
    
    # Best performer
    if val_losses:
        best_run = min(val_losses.items(), key=lambda x: x[1])
        best_config = task_mapping.get(best_run[0], {})
        print(f"   • Best performing configuration: {best_run[0]} ({best_config.get('type', 'unknown')} tasks)")
        print(f"   • Consider this as baseline for downstream finetuning")
    
    # Stability recommendations  
    if stabilities:
        stable_runs = [run for run, cv in stabilities.items() if cv < 0.3]
        if stable_runs:
            print(f"   • Most reliable configurations for production: {', '.join(stable_runs)}")
        
        unstable_runs = [run for run, cv in stabilities.items() if cv > 0.5]
        if unstable_runs:
            print(f"   • Configurations needing improvement: {', '.join(unstable_runs)}")

if __name__ == "__main__":
    try:
        results = analyze_completed_runs()
        
        print("\n" + "="*60)
        print("🎯 ANALYSIS COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

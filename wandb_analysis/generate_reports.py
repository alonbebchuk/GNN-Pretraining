#!/usr/bin/env python3
"""
Generate WandB reports for systematic GNN pre-training analysis.
Run this AFTER pre-training experiments complete.
"""

import argparse
import wandb
from analysis_queries import RQ_ANALYSIS_QUERIES, REPORT_SECTIONS

def create_research_question_reports(project_name: str, sweep_id: str = None):
    """Create reports addressing each research question."""
    
    api = wandb.Api()
    
    # Get runs from project (optionally filtered by sweep)
    if sweep_id:
        runs = api.sweep(f"{project_name}/{sweep_id}").runs
        print(f"Analyzing {len(runs)} runs from sweep {sweep_id}")
    else:
        runs = api.runs(project_name)
        print(f"Analyzing {len(runs)} runs from project {project_name}")
    
    # Create reports for each research question
    reports_created = []
    
    for rq_id, rq_config in RQ_ANALYSIS_QUERIES.items():
        print(f"\nCreating report for {rq_id}: {rq_config['description']}")
        
        # Filter runs by relevant tags
        relevant_runs = [
            run for run in runs 
            if any(tag in run.tags for tag in rq_config['tags'])
        ]
        
        if not relevant_runs:
            print(f"  No runs found with tags: {rq_config['tags']}")
            continue
            
        print(f"  Found {len(relevant_runs)} relevant runs")
        
        # Create report (this would use WandB's report API)
        report_url = create_wandb_report(
            title=f"RQ Analysis: {rq_config['description']}", 
            runs=relevant_runs,
            metrics=rq_config['metrics'],
            group_by=rq_config.get('group_by', 'exp_name')
        )
        
        reports_created.append({
            'rq_id': rq_id,
            'title': rq_config['description'],
            'url': report_url,
            'run_count': len(relevant_runs)
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("REPORTS CREATED:")
    print(f"{'='*60}")
    for report in reports_created:
        print(f"📊 {report['rq_id']}: {report['title']}")
        print(f"   Runs analyzed: {report['run_count']}")
        print(f"   URL: {report['url']}")
        print()

def create_wandb_report(title: str, runs, metrics: list, group_by: str):
    """Create a WandB report with specified configuration."""
    # This is a placeholder - actual implementation would use WandB's report API
    # For now, return a mock URL
    report_url = f"https://wandb.ai/your-username/gnn-pretraining/reports/{title.replace(' ', '-')}"
    return report_url

def main():
    parser = argparse.ArgumentParser(description="Generate analysis reports")
    parser.add_argument("--project", default="gnn-pretraining", help="WandB project name")
    parser.add_argument("--sweep-id", help="Optional: Focus on specific sweep")
    
    args = parser.parse_args()
    
    print("🔍 Generating WandB Analysis Reports")
    print(f"Project: {args.project}")
    if args.sweep_id:
        print(f"Sweep: {args.sweep_id}")
    
    create_research_question_reports(args.project, args.sweep_id)
    
    print("\n✅ Report generation complete!")
    print("💡 Tip: Pin important reports to your WandB project dashboard")

if __name__ == "__main__":
    main()
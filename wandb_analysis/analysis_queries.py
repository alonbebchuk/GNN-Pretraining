"""
WandB analysis queries for systematic GNN pre-training study.
These queries support the research questions outlined in plan.md.
"""

# Research Question Analysis Queries
RQ_ANALYSIS_QUERIES = {
    "RQ1_pretrain_vs_scratch": {
        "description": "Compare all pre-trained models vs from-scratch baseline",
        "tags": ["pretrained", "from_scratch_baseline"],
        "metrics": ["val/selection_total", "best_val_total"],
        "group_by": "exp_name"
    },
    
    "RQ2_task_combinations": {
        "description": "Compare different task combination strategies",
        "tags": ["single_task_baselines", "multi_task_schemes"],
        "metrics": ["val/selection_total", "convergence_epoch"],
        "group_by": ["paradigm", "scheme_type"]
    },
    
    "RQ2_paradigm_comparison": {
        "description": "Generative vs Contrastive paradigms (S1 vs S2)",
        "tags": ["paradigm_comparison"],
        "metrics": ["val/selection_total", "timing/time_to_convergence_hours"],
        "group_by": "paradigm"
    },
    
    "RQ2_cross_domain_benefit": {
        "description": "Multi-domain vs single-domain (S4 vs B4)",
        "tags": ["cross_domain_comparison"],
        "metrics": ["val/selection_total", "val/balanced_total"],
        "group_by": "is_multi_domain"
    },
    
    "convergence_analysis": {
        "description": "Training efficiency and convergence patterns",
        "tags": ["phase:pretrain"],
        "metrics": [
            "timing/time_to_convergence_hours",
            "efficiency/training_efficiency", 
            "best_epoch",
            "early_stopped"
        ],
        "group_by": "exp_name"
    },
    
    "domain_balance_analysis": {
        "description": "Domain loss balance during training",
        "tags": ["phase:pretrain"],
        "metrics": [
            "train/domain_balance/mean",
            "train/domain_balance/std", 
            "train/domain_balance/cv"
        ],
        "group_by": ["exp_name", "has_adversarial"]
    }
}

# Comparison Groups for Statistical Analysis
COMPARISON_GROUPS = {
    "baseline_vs_pretrained": {
        "baseline": ["from_scratch_baseline"],
        "treatment": ["pretrained"]
    },
    "single_vs_multi_task": {
        "baseline": ["single_task_baselines"],
        "treatment": ["multi_task_schemes"]
    },
    "generative_vs_contrastive": {
        "group1": ["S1"],  # generative
        "group2": ["S2"]   # contrastive
    },
    "complexity_progression": {
        "groups": ["S3", "S4", "S5"]  # increasing complexity
    }
}

# Report Templates
REPORT_SECTIONS = {
    "executive_summary": {
        "title": "Executive Summary",
        "description": "Key findings and recommendations",
        "charts": ["summary_performance", "best_schemes_ranking"]
    },
    "rq1_pretrain_value": {
        "title": "RQ1: Value of Pre-training",
        "description": "When does pre-training help vs hurt?",
        "charts": ["pretrain_vs_scratch", "negative_transfer_analysis"]
    },
    "rq2_task_synergies": {
        "title": "RQ2: Task Combination Effects", 
        "description": "Which task combinations work best?",
        "charts": ["paradigm_comparison", "complexity_progression", "cross_domain_benefit"]
    },
    "efficiency_analysis": {
        "title": "Training Efficiency Analysis",
        "description": "Convergence speed and computational costs",
        "charts": ["convergence_patterns", "training_efficiency", "early_stopping_analysis"]
    },
    "domain_analysis": {
        "title": "Domain Adaptation Analysis",
        "description": "Cross-domain transfer and domain balance",
        "charts": ["domain_balance", "ood_performance", "adversarial_impact"]
    }
}
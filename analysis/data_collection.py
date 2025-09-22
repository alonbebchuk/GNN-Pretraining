import os
import sys
import pandas as pd
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

WANDB_PROJECT = "gnn-pretraining-finetune"
WANDB_ENTITY = "timoshka3-tel-aviv-university"

DOMAINS = ['ENZYMES', 'PTC_MR', 'Cora_NC', 'CiteSeer_NC', 'Cora_LP', 'CiteSeer_LP']
FINETUNE_STRATEGIES = ['linear_probe', 'full_finetune']
PRETRAINED_SCHEMES = ['b1', 'b2', 'b3', 's1', 's2', 's3', 's4', 's5', 'b4']
SEEDS = [42, 84, 126]

RESULTS_DIR = Path(__file__).parent / "results"
EXPERIMENT_RESULTS_FILE = RESULTS_DIR / "experiment_results.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_run_name(run_name: str) -> Optional[Dict[str, str]]:
    try:
        parts = run_name.split('_')
        if len(parts) < 4:
            return None

        seed = int(parts[-1])
        if seed not in SEEDS:
            return None

        pretrained_scheme = parts[-2]
        if pretrained_scheme not in PRETRAINED_SCHEMES:
            return None

        strategy_end_idx = len(parts) - 2

        for strategy_start_idx in range(1, strategy_end_idx):
            potential_strategy = '_'.join(parts[strategy_start_idx:strategy_end_idx])
            if potential_strategy in FINETUNE_STRATEGIES:
                domain = '_'.join(parts[:strategy_start_idx])
                if domain in DOMAINS:
                    return {
                        'domain_name': domain,
                        'finetune_strategy': potential_strategy,
                        'pretrained_scheme': pretrained_scheme,
                        'seed': seed
                    }

        return None

    except (ValueError, IndexError):
        return None


def extract_test_metrics(run) -> Dict[str, Any]:
    target_metrics = [
        'test/accuracy',
        'test/f1', 
        'test/precision',
        'test/recall',
        'test/auc',
        'test/convergence_epochs',
        'test/training_time',
        'test/trainable_parameters'
    ]

    metrics = {}
    for metric_key in target_metrics:
        if metric_key in run.summary:
            value = run.summary[metric_key]
            clean_key = metric_key.replace('test/', '')
            metrics[clean_key] = value
        else:
            clean_key = metric_key.replace('test/', '')
            metrics[clean_key] = None

    return metrics


def extract_all_finetune_results() -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

    extracted_data = []

    for run in runs:
        parsed_name = validate_run_name(run.name)

        if parsed_name is None:
            continue

        if run.state != "finished":
            continue

        test_metrics = extract_test_metrics(run)

        row_data = {
            **parsed_name,
            **test_metrics
        }

        extracted_data.append(row_data)

    df = pd.DataFrame(extracted_data)
    df = df.sort_values(['domain_name', 'finetune_strategy', 'pretrained_scheme', 'seed'])
    df.to_csv(EXPERIMENT_RESULTS_FILE, index=False)

    return df


def main():
    results_df = extract_all_finetune_results()
    return results_df


if __name__ == "__main__":
    main()

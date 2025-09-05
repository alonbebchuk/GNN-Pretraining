#!/bin/bash
# Ben's VM3 (gnn-l4-03) - Finetuning Script
# Domains: CiteSeer_LP (54 experiments: 2 strategies × 9 schemes × 3 seeds)

set -e
echo "=== Ben VM3 (gnn-l4-03) - Finetuning Phase ==="
echo "Running: CiteSeer_LP domain (54 experiments)"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate ./.conda

# Set wandb configuration
export WANDB_ENTITY=alon-bebchuk-tel-aviv-university
export WANDB_PROJECT=gnn-pretraining

# Wait for pretraining to complete (check wandb for required models)
echo "Waiting for pretraining models to be available..."
echo "Check https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining for progress"

# Run domain-specific finetuning sweep
echo "Starting CiteSeer_LP finetuning experiments..."
python run_finetune.py --domain_sweep CiteSeer_LP

echo "=== Ben VM3 Finetuning Complete ==="
echo "End time: $(date)"
echo "Results: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining"

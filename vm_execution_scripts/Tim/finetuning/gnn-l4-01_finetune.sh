#!/bin/bash
# Tim's VM1 (gnn-l4-01) - Finetuning Script
# Domains: ENZYMES (54 experiments: 2 strategies × 9 schemes × 3 seeds)

set -e
echo "=== Tim VM1 (gnn-l4-01) - Finetuning Phase ==="
echo "Running: ENZYMES domain (54 experiments)"
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
echo "Starting ENZYMES finetuning experiments..."
python run_finetune.py --domain_sweep ENZYMES

echo "=== Tim VM1 Finetuning Complete ==="
echo "End time: $(date)"
echo "Results: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining"

#!/bin/bash
# Ben's VM2 (gnn-l4-02) - Finetuning Script
# Domains: Cora_LP (54 experiments: 2 strategies × 9 schemes × 3 seeds)

set -e
echo "=== Ben VM2 (gnn-l4-02) - Finetuning Phase ==="
echo "Running: Cora_LP domain (54 experiments)"
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
echo "Starting Cora_LP finetuning experiments..."
python run_finetune.py --domain_sweep Cora_LP

echo "=== Ben VM2 Finetuning Complete ==="
echo "End time: $(date)"
echo "Results: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining"

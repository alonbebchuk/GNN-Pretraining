#!/bin/bash
# Ben's VM1 (gnn-l4-01) - Pretraining Script
# Schemes: b2 (all seeds)

set -e
echo "=== Ben VM1 (gnn-l4-01) - Pretraining Phase ==="
echo "Running: b2 scheme with all seeds"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate ./.conda

# Set wandb configuration
export WANDB_ENTITY=alon-bebchuk-tel-aviv-university
export WANDB_PROJECT=gnn-pretraining

# Run experiments in parallel
echo "Starting b2 experiments..."
python run_pretrain.py --exp_name b2 --seed 42 &
python run_pretrain.py --exp_name b2 --seed 84 &
python run_pretrain.py --exp_name b2 --seed 126 &

# Wait for all to complete
wait

echo "=== Ben VM1 Pretraining Complete ==="
echo "End time: $(date)"
echo "Results: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining"

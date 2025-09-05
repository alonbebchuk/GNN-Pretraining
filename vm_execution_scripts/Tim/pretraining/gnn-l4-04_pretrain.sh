#!/bin/bash
# Tim's VM4 (gnn-l4-04) - Pretraining Script
# Schemes: b4 (all seeds)

set -e
echo "=== Tim VM4 (gnn-l4-04) - Pretraining Phase ==="
echo "Running: b4 scheme with all seeds"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate ./.conda

# Set wandb configuration
export WANDB_ENTITY=alon-bebchuk-tel-aviv-university
export WANDB_PROJECT=gnn-pretraining

# Run experiments in parallel
echo "Starting b4 experiments..."
python run_pretrain.py --exp_name b4 --seed 42 &
python run_pretrain.py --exp_name b4 --seed 84 &
python run_pretrain.py --exp_name b4 --seed 126 &

# Wait for all to complete
wait

echo "=== Tim VM4 Pretraining Complete ==="
echo "End time: $(date)"
echo "Results: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining"

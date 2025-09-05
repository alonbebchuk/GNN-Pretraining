#!/bin/bash
# Ben's VM3 (gnn-l4-03) - Pretraining Script
# Schemes: s1 (all seeds)

set -e
echo "=== Ben VM3 (gnn-l4-03) - Pretraining Phase ==="
echo "Running: s1 scheme with all seeds"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate ./.conda

# Set wandb configuration
export WANDB_ENTITY=alon-bebchuk-tel-aviv-university
export WANDB_PROJECT=gnn-pretraining

# Run experiments in parallel
echo "Starting s1 experiments..."
python run_pretrain.py --exp_name s1 --seed 42 &
python run_pretrain.py --exp_name s1 --seed 84 &
python run_pretrain.py --exp_name s1 --seed 126 &

# Wait for all to complete
wait

echo "=== Ben VM3 Pretraining Complete ==="
echo "End time: $(date)"
echo "Results: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining"

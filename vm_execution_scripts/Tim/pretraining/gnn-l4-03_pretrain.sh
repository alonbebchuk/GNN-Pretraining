#!/bin/bash
# Tim's VM3 (gnn-l4-03) - Pretraining Script
# Schemes: s5 (all seeds)

set -e
echo "=== Tim VM3 (gnn-l4-03) - Pretraining Phase ==="
echo "Running: s5 scheme with all seeds"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate ./.conda

# Set wandb configuration
export WANDB_ENTITY=alon-bebchuk-tel-aviv-university
export WANDB_PROJECT=gnn-pretraining

# Run experiments in parallel
echo "Starting s5 experiments..."
python run_pretrain.py --exp_name s5 --seed 42 &
python run_pretrain.py --exp_name s5 --seed 84 &
python run_pretrain.py --exp_name s5 --seed 126 &

# Wait for all to complete
wait

echo "=== Tim VM3 Pretraining Complete ==="
echo "End time: $(date)"
echo "Results: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining"

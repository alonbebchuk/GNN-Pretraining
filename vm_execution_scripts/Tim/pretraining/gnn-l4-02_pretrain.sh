#!/bin/bash
# Tim's VM2 (gnn-l4-02) - Pretraining Script
# Schemes: s4 (all seeds)

set -e
echo "=== Tim VM2 (gnn-l4-02) - Pretraining Phase ==="
echo "Running: s4 scheme with all seeds"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate ./.conda

# Set wandb configuration
export WANDB_ENTITY=alon-bebchuk-tel-aviv-university
export WANDB_PROJECT=gnn-pretraining

# Run experiments in parallel
echo "Starting s4 experiments..."
python run_pretrain.py --exp_name s4 --seed 42 &
python run_pretrain.py --exp_name s4 --seed 84 &
python run_pretrain.py --exp_name s4 --seed 126 &

# Wait for all to complete
wait

echo "=== Tim VM2 Pretraining Complete ==="
echo "End time: $(date)"
echo "Results: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining"

#!/bin/bash
# Single VM - Run s5 Only
# This script runs only s5 scheme with all 3 seeds to parallelize with other VM

set -e
echo "=== Single VM - s5 Scheme Only ==="
echo "Running: s5 scheme (3 experiments: 3 seeds)"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate gnn-pretraining

# Set wandb configuration
export WANDB_ENTITY=timoshka3-tel-aviv-university
export WANDB_PROJECT=GNN

echo "Starting s5 pretraining experiments..."
echo "This will run 3 experiments (s5 with seeds 42, 84, 126)"
echo ""

# Run s5 with all seeds sequentially (or you could parallelize further)
python run_pretrain.py --exp_name s5 --seed 42
python run_pretrain.py --exp_name s5 --seed 84
python run_pretrain.py --exp_name s5 --seed 126

echo "=== s5 Pretraining Complete ==="
echo "Completed: 3 experiments (s5 with seeds 42, 84, 126)"
echo "End time: $(date)"
echo "Results: https://wandb.ai/timoshka3-tel-aviv-university/GNN"
echo ""
echo "Note: Other VM should complete remaining schemes (s4, etc.)"

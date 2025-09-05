#!/bin/bash
# Ben's VM - Complete Pretraining Script
# Schemes: b2, b3, s1, s2 (12 experiments: 4 schemes × 3 seeds)

set -e
echo "=== Ben's VM - Pretraining Phase ==="
echo "Running: b2, b3, s1, s2 schemes (12 experiments total)"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate ./.conda

# Set wandb configuration
export WANDB_ENTITY=timoshka3-tel-aviv-university
export WANDB_PROJECT=GNN

echo "Starting Ben's pretraining experiments..."

# Run all Ben's experiments in parallel
for scheme in b2 b3 s1 s2; do
    for seed in 42 84 126; do
        echo "Starting: $scheme seed $seed"
        python run_pretrain.py --exp_name $scheme --seed $seed &
    done
done

# Wait for all experiments to complete
wait

echo "=== Ben's VM Pretraining Complete ==="
echo "Completed: 12 experiments (b2, b3, s1, s2 × 3 seeds each)"
echo "End time: $(date)"
echo "Results: https://wandb.ai/timoshka3-tel-aviv-university/GNN"

#!/bin/bash
# Tim's VM - Complete Pretraining Script
# Schemes: s3, s4, s5, b4 (12 experiments: 4 schemes × 3 seeds)

set -e
echo "=== Tim's VM - Pretraining Phase ==="
echo "Running: s3, s4, s5, b4 schemes (12 experiments total)"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate ./.conda

# Set wandb configuration
export WANDB_ENTITY=timoshka3-tel-aviv-university
export WANDB_PROJECT=GNN

echo "Starting Tim's pretraining experiments..."

# Run all Tim's experiments in parallel
for scheme in s3 s4 s5 b4; do
    for seed in 42 84 126; do
        echo "Starting: $scheme seed $seed"
        python run_pretrain.py --exp_name $scheme --seed $seed &
    done
done

# Wait for all experiments to complete
wait

echo "=== Tim's VM Pretraining Complete ==="
echo "Completed: 12 experiments (s3, s4, s5, b4 × 3 seeds each)"
echo "End time: $(date)"
echo "Results: https://wandb.ai/timoshka3-tel-aviv-university/GNN"

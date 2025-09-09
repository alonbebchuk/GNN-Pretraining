#!/bin/bash
# Ben's VM - Complete Finetuning Script
# Domains: CiteSeer_NC, CiteSeer_LP, ENZYMES (162 experiments: 3 domains × 2 strategies × 9 schemes × 3 seeds)

set -e
echo "=== Ben's VM - Finetuning Phase ==="
echo "Running: CiteSeer_NC, CiteSeer_LP, ENZYMES domains (162 experiments total)"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate /home/benc6116/workspace/GNN-Pretraining/.conda

# Set wandb configuration
export WANDB_ENTITY=timoshka3-tel-aviv-university
export WANDB_PROJECT=GNN

# Wait for pretraining to complete (check wandb for required models)
echo "Waiting for pretraining models to be available..."
echo "Check https://wandb.ai/timoshka3-tel-aviv-university/GNN for progress"
echo "Ensure all 24 pretraining runs are complete before proceeding"
read -p "Press Enter when all pretraining is complete..."

echo "Starting Ben's finetuning experiments..."

# Run Ben's domain sweeps sequentially (to avoid overwhelming single GPU)
echo "Running CiteSeer_NC domain..."
python run_finetune.py --domain_sweep CiteSeer_NC

echo "Running CiteSeer_LP domain..."
python run_finetune.py --domain_sweep CiteSeer_LP

echo "Running ENZYMES domain..."
python run_finetune.py --domain_sweep ENZYMES

echo "=== Ben's VM Finetuning Complete ==="
echo "Completed: 162 experiments across 3 domains"
echo "End time: $(date)"
echo "Results: https://wandb.ai/timoshka3-tel-aviv-university/GNN"

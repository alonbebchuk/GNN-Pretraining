#!/bin/bash
# Tim's VM - Complete Finetuning Script
# Domains: Cora_NC, Cora_LP, PTC_MR (162 experiments: 3 domains × 2 strategies × 9 schemes × 3 seeds)

set -e
echo "=== Tim's VM - Finetuning Phase ==="
echo "Running: Cora_NC, Cora_LP, PTC_MR domains (162 experiments total)"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate gnn-pretraining

# Set wandb configuration
export WANDB_ENTITY=timoshka3-tel-aviv-university
export WANDB_PROJECT=GNN

# Wait for pretraining to complete (check wandb for required models)
echo "Waiting for pretraining models to be available..."
echo "Check https://wandb.ai/timoshka3-tel-aviv-university/GNN for progress"
echo "Ensure all 24 pretraining runs are complete before proceeding"
read -p "Press Enter when all pretraining is complete..."

echo "Starting Tim's finetuning experiments..."

# Run Tim's domain sweeps sequentially (to avoid overwhelming single GPU)
echo "Running Cora_NC domain..."
python run_finetune.py --domain_sweep Cora_NC

echo "Running Cora_LP domain..."
python run_finetune.py --domain_sweep Cora_LP

echo "Running PTC_MR domain..."
python run_finetune.py --domain_sweep PTC_MR

echo "=== Tim's VM Finetuning Complete ==="
echo "Completed: 162 experiments across 3 domains"
echo "End time: $(date)"
echo "Results: https://wandb.ai/timoshka3-tel-aviv-university/GNN"

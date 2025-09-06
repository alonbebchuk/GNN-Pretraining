#!/bin/bash
# Resume Script - Run Remaining Experiments After Manual Stop
# Run this on the other VM after stopping the original script
# Runs: s3_126, s4_42, s4_84, s4_126 (s5 will be handled by other VM)

set -e
echo "=== Resume Remaining Pretraining Experiments ==="
echo "Running: s3_126 + all s4 experiments (4 total)"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate gnn-pretraining

# Set wandb configuration
export WANDB_ENTITY=timoshka3-tel-aviv-university
export WANDB_PROJECT=GNN

echo "Starting remaining pretraining experiments..."
echo "Will run: s3_126, s4_42, s4_84, s4_126"
echo "Note: s5 experiments will be handled by other VM"
echo ""

# Run remaining s3
echo "Running remaining s3 experiment..."
python run_pretrain.py --exp_name s3 --seed 126

# Run all s4 experiments
echo "Running all s4 experiments..."
python run_pretrain.py --exp_name s4 --seed 42
python run_pretrain.py --exp_name s4 --seed 84
python run_pretrain.py --exp_name s4 --seed 126

echo "=== Resume Pretraining Complete ==="
echo "Completed: 4 experiments (s3_126 + all s4)"
echo "End time: $(date)"
echo "Results: https://wandb.ai/timoshka3-tel-aviv-university/GNN"
echo ""
echo "Next: s5 experiments should be completed on other VM"

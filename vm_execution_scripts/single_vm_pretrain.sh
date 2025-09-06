#!/bin/bash
# Single VM - Complete Pretraining Script
# All Schemes: b2, b3, s1, s2, s3, s4, s5, b4 (24 experiments: 8 schemes × 3 seeds)

set -e
echo "=== Single VM - Complete Pretraining Phase ==="
echo "Running: ALL 8 schemes (24 experiments total)"
echo "Start time: $(date)"

cd ~/workspace/GNN-Pretraining

# Activate conda environment
source ~/miniconda/bin/activate
conda activate gnn-pretraining

# Set wandb configuration
export WANDB_ENTITY=timoshka3-tel-aviv-university
export WANDB_PROJECT=GNN

echo "Starting all pretraining experiments using built-in sweep with progress tracking..."
echo "This will take approximately 17 GPU hours (~17-20 hours wall time)"
echo "Progress will be displayed automatically by run_pretrain.py"
echo ""

# Use the built-in sweep functionality which includes:
# - Automatic parallelization based on available GPUs
# - Built-in progress tracking: "Progress: X/24 (✓completed ✗failed)" 
# - Error handling and reporting
# - All 8 schemes × 3 seeds = 24 experiments
python run_pretrain.py --sweep

echo "=== Single VM Pretraining Complete ==="
echo "Completed: ALL 24 experiments (8 schemes × 3 seeds each)"
echo "End time: $(date)"
echo "Results: https://wandb.ai/timoshka3-tel-aviv-university/GNN"
echo ""
echo "Next steps:"
echo "1. Verify all 24 pretraining runs completed successfully in wandb"
echo "2. Both teammates can now run their respective finetuning scripts:"
echo "   - Ben: ./vm_execution_scripts/Ben/ben_vm_finetune.sh"
echo "   - Tim: ./vm_execution_scripts/Tim/tim_vm_finetune.sh"

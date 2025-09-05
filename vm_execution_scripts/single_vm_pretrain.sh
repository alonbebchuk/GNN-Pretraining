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
conda activate ./.conda

# Set wandb configuration
export WANDB_ENTITY=timoshka3-tel-aviv-university
export WANDB_PROJECT=GNN

echo "Starting all pretraining experiments on single VM..."
echo "This will take approximately 17 GPU hours (~17-20 hours wall time)"

# Run all experiments sequentially to avoid GPU memory issues
for scheme in b2 b3 s1 s2 s3 s4 s5 b4; do
    for seed in 42 84 126; do
        echo "Starting: $scheme seed $seed"
        python run_pretrain.py --exp_name $scheme --seed $seed
        echo "Completed: $scheme seed $seed"
    done
    echo "Completed all seeds for scheme: $scheme"
done

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

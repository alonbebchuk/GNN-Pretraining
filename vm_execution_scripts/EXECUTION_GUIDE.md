# GNN Pretraining - Distributed Execution Guide

> **Updated for GPU Quota Constraints**: This guide has been redesigned for **2 VMs total (1 per teammate)** due to GPU quota limitations.

## Overview

This guide explains how to run the complete GNN pretraining experiments using **2 VMs (1 per teammate)** due to GPU quota limitations.

**Total Experiments:**
- **Pretraining**: 24 experiments (8 schemes × 3 seeds) = ~17 GPU hours
- **Finetuning**: 324 experiments (6 domains × 2 strategies × 9 schemes × 3 seeds) = ~24 GPU hours
- **Sequential time**: ~41 GPU hours
- **With 2 VMs**: ~20-21 hours total (parallel pretraining + parallel finetuning)

## Team Setup

### VM Naming Convention
**Pretraining**: Either teammate creates 1 VM: `shared-gnn-l4`
**Finetuning**: Each teammate creates 1 VM:
- **Ben**: `ben-gnn-l4` 
- **Tim**: `tim-gnn-l4`

### Results Location
All experiments log to the shared wandb project:
**https://wandb.ai/timoshka3-tel-aviv-university/GNN**

---

## Phase 1: VM Setup (Do Once Per VM)

### 1. Create VMs
- **Machine type**: `g2-standard-8` (8 vCPU, 32 GB RAM)
- **GPU**: NVIDIA L4
- **Image**: Deep Learning VM for PyTorch 2.4 with CUDA 12.4
- **Disk**: 150 GB
- **Region**: `us-east1`

### 2. Initial Setup on Each VM
```bash
# SSH into each VM and run:
ssh gnn-l4-01  # (or 02, 03, 04)

# Follow the bootstrap script from vm_setup.md
~/bootstrap_gnn_vm.sh

# Activate environment
source ~/.bashrc
cd ~/workspace/GNN-Pretraining
conda activate gnn-pretraining

# Set correct wandb entity for shared project
export WANDB_ENTITY=timoshka3-tel-aviv-university
export WANDB_PROJECT=GNN

# Make it permanent in bashrc
echo "export WANDB_ENTITY=timoshka3-tel-aviv-university" >> ~/.bashrc
echo "export WANDB_PROJECT=GNN" >> ~/.bashrc

# Login to wandb
wandb login  # Paste your API key

# Verify setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
echo "Results will go to: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
```

### 3. Copy Execution Scripts
On each VM, copy the appropriate scripts from this repository:
```bash
# Copy all scripts
git pull  # Get latest version with scripts

# Verify scripts exist
ls vm_execution_scripts/single_vm_pretrain.sh  # Pretraining: Single VM
ls vm_execution_scripts/Ben/ben_vm_finetune.sh # Finetuning: Ben's script
ls vm_execution_scripts/Tim/tim_vm_finetune.sh # Finetuning: Tim's script
```

---

## Phase 2: Pretraining (Single VM Only)

### Single VM Pretraining Setup

Either Ben or Tim creates one VM and runs all pretraining:

#### Execution:
```bash
# Either Ben or Tim creates one VM (e.g., shared-gnn-l4) and runs:
./vm_execution_scripts/single_vm_pretrain.sh
```

#### Distribution:
| Setup | VM | Schemes | Experiments |
|-------|----|---------| ------------|
| Single VM | shared-gnn-l4 | ALL: b2, b3, s1, s2, s3, s4, s5, b4 | 24 (8 schemes × 3 seeds each) |

**Expected Duration**: 17-20 hours (uses built-in sweep with automatic parallelization)

**Features**: 
- **Built-in progress tracking**: Shows "Progress: X/24 (✓completed ✗failed)"
- **Automatic parallelization**: Uses available GPU cores efficiently  
- **Error handling**: Reports failed experiments with details
- **Coordination**: Monitor progress at wandb, both teammates proceed to finetuning after completion

---

## Phase 3: Finetuning (Start After Pretraining)

### Wait Condition
⚠️ **Do NOT start finetuning until pretraining is complete!** 

Monitor progress at: https://wandb.ai/timoshka3-tel-aviv-university/GNN

Look for **24 completed pretraining runs** before proceeding.

### Execution Order: **BOTH VMs START SIMULTANEOUSLY**

#### Ben's VM:
```bash
# Ben's VM (ben-gnn-l4)
./vm_execution_scripts/Ben/ben_vm_finetune.sh
```

#### Tim's VM:
```bash
# Tim's VM (tim-gnn-l4)
./vm_execution_scripts/Tim/tim_vm_finetune.sh
```

### Finetuning Distribution:
| Person | VM | Domains | Experiments |
|--------|----|---------| ------------|
| Ben | ben-gnn-l4 | CiteSeer_NC, Cora_LP, CiteSeer_LP | 162 (3 domains × 54 exp each) |
| Tim | tim-gnn-l4 | ENZYMES, PTC_MR, Cora_NC | 162 (3 domains × 54 exp each) |

**Total**: Ben = 162 experiments, Tim = 162 experiments (perfectly balanced!)  
**Expected Duration**: 12-15 hours (162 experiments per VM sequentially)

---

## Monitoring and Results

### Real-time Monitoring
- **Wandb Dashboard**: https://wandb.ai/timoshka3-tel-aviv-university/GNN
- **Expected runs**: 24 pretraining + 324 finetuning = 348 total runs
- **Run naming**: `{domain}_{strategy}_{scheme}_{seed}`

### VM Management
```bash
# Check GPU usage
nvidia-smi

# Monitor script progress
tail -f nohup.out  # If running with nohup

# Stop VM when idle (IMPORTANT for cost savings)
# In GCP Console: Compute Engine > VM Instances > Stop
```

### Troubleshooting
```bash
# If a script fails, check logs:
echo "Check wandb for partial results"

# Restart specific experiments:
python run_pretrain.py --exp_name b2 --seed 42
python run_finetune.py --domain_name ENZYMES --finetune_strategy full_finetune --pretrained_scheme b2 --seed 42

# Check environment:
conda list | grep torch
echo $WANDB_ENTITY
echo $WANDB_PROJECT
```

---

## Timeline Summary

| Phase | Duration | Parallelization | Status |
|-------|----------|-----------------|---------|
| **VM Setup** | 30 min | Per person | One-time |
| **Pretraining** | 8-10 hours | 2 VMs (12 exp/VM) | Required first |
| **Finetuning** | 12-15 hours | 2 VMs (162 exp/VM) | After pretraining |
| **Total** | **20-25 hours** | vs 41 hours sequential | 2x speedup! |

## Cost Estimate
- **L4 GPU**: ~$0.60/hour
- **2 VMs × 25 hours**: ~$30 total
- **vs Sequential**: 41 hours × $0.60 = ~$25 (but takes 2+ days)

---

## Success Criteria

✅ **Pretraining Complete**: 24 runs in wandb  
✅ **Finetuning Complete**: 324 runs in wandb  
✅ **Total Results**: 348 runs with comprehensive metrics  
✅ **Ready for Analysis**: Statistical significance testing, performance tables, plots

**Final Results Location**: https://wandb.ai/timoshka3-tel-aviv-university/GNN

---

*This distributed execution strategy reduces total runtime from 41 hours to 6-10 hours while maintaining full experimental rigor!*

# GNN Pretraining - Distributed Execution Guide

## Overview

This guide explains how to run the complete GNN pretraining experiments using **8 VMs (4 per teammate)** for maximum parallel efficiency.

**Total Experiments:**
- **Pretraining**: 24 experiments (8 schemes × 3 seeds) = ~17 GPU hours
- **Finetuning**: 324 experiments (6 domains × 2 strategies × 9 schemes × 3 seeds) = ~24 GPU hours
- **Sequential time**: ~41 GPU hours
- **With 8 VMs**: ~6-8 hours total

## Team Setup

### VM Naming Convention
Each teammate creates 4 VMs with these exact names:
- **Ben**: `gnn-l4-01`, `gnn-l4-02`, `gnn-l4-03`, `gnn-l4-04`
- **Tim**: `gnn-l4-01`, `gnn-l4-02`, `gnn-l4-03`, `gnn-l4-04`

### Results Location
All experiments log to the shared wandb project:
**https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining**

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
conda activate ./.conda

# Set correct wandb entity for shared project
export WANDB_ENTITY=alon-bebchuk-tel-aviv-university
export WANDB_PROJECT=gnn-pretraining

# Make it permanent in bashrc
echo "export WANDB_ENTITY=alon-bebchuk-tel-aviv-university" >> ~/.bashrc
echo "export WANDB_PROJECT=gnn-pretraining" >> ~/.bashrc

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
ls vm_execution_scripts/Ben/{pretraining,finetuning}/
ls vm_execution_scripts/Tim/{pretraining,finetuning}/
```

---

## Phase 2: Pretraining (Run in Parallel)

### Execution Order: **ALL VMs START SIMULTANEOUSLY**

#### Ben's VMs:
```bash
# VM gnn-l4-01 (Ben)
./vm_execution_scripts/Ben/pretraining/gnn-l4-01_pretrain.sh

# VM gnn-l4-02 (Ben)  
./vm_execution_scripts/Ben/pretraining/gnn-l4-02_pretrain.sh

# VM gnn-l4-03 (Ben)
./vm_execution_scripts/Ben/pretraining/gnn-l4-03_pretrain.sh

# VM gnn-l4-04 (Ben)
./vm_execution_scripts/Ben/pretraining/gnn-l4-04_pretrain.sh
```

#### Tim's VMs:
```bash
# VM gnn-l4-01 (Tim)
./vm_execution_scripts/Tim/pretraining/gnn-l4-01_pretrain.sh

# VM gnn-l4-02 (Tim)
./vm_execution_scripts/Tim/pretraining/gnn-l4-02_pretrain.sh

# VM gnn-l4-03 (Tim)
./vm_execution_scripts/Tim/pretraining/gnn-l4-03_pretrain.sh

# VM gnn-l4-04 (Tim)
./vm_execution_scripts/Tim/pretraining/gnn-l4-04_pretrain.sh
```

### Pretraining Distribution:
| Person | VM | Scheme | Experiments |
|--------|----|---------| ------------|
| Ben | gnn-l4-01 | b2 | 3 (seeds: 42, 84, 126) |
| Ben | gnn-l4-02 | b3 | 3 (seeds: 42, 84, 126) |
| Ben | gnn-l4-03 | s1 | 3 (seeds: 42, 84, 126) |
| Ben | gnn-l4-04 | s2 | 3 (seeds: 42, 84, 126) |
| Tim | gnn-l4-01 | s3 | 3 (seeds: 42, 84, 126) |
| Tim | gnn-l4-02 | s4 | 3 (seeds: 42, 84, 126) |
| Tim | gnn-l4-03 | s5 | 3 (seeds: 42, 84, 126) |
| Tim | gnn-l4-04 | b4 | 3 (seeds: 42, 84, 126) |

**Expected Duration**: 2-4 hours (3 experiments per VM in parallel)

---

## Phase 3: Finetuning (Start After Pretraining)

### Wait Condition
⚠️ **Do NOT start finetuning until pretraining is complete!** 

Monitor progress at: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining

Look for **24 completed pretraining runs** before proceeding.

### Execution Order: **ALL VMs START SIMULTANEOUSLY**

#### Ben's VMs:
```bash
# VM gnn-l4-01 (Ben) - CiteSeer_NC domain
./vm_execution_scripts/Ben/finetuning/gnn-l4-01_finetune.sh

# VM gnn-l4-02 (Ben) - Cora_LP domain
./vm_execution_scripts/Ben/finetuning/gnn-l4-02_finetune.sh

# VM gnn-l4-03 (Ben) - CiteSeer_LP domain
./vm_execution_scripts/Ben/finetuning/gnn-l4-03_finetune.sh

# VM gnn-l4-04 (Ben) - Standby (can be stopped to save costs)
./vm_execution_scripts/Ben/finetuning/gnn-l4-04_finetune.sh
```

#### Tim's VMs:
```bash
# VM gnn-l4-01 (Tim) - ENZYMES domain
./vm_execution_scripts/Tim/finetuning/gnn-l4-01_finetune.sh

# VM gnn-l4-02 (Tim) - PTC_MR domain
./vm_execution_scripts/Tim/finetuning/gnn-l4-02_finetune.sh

# VM gnn-l4-03 (Tim) - Cora_NC domain
./vm_execution_scripts/Tim/finetuning/gnn-l4-03_finetune.sh

# VM gnn-l4-04 (Tim) - Standby (can be stopped to save costs)
./vm_execution_scripts/Tim/finetuning/gnn-l4-04_finetune.sh
```

### Finetuning Distribution:
| Person | VM | Domain | Experiments |
|--------|----|---------| ------------|
| Ben | gnn-l4-01 | CiteSeer_NC | 54 (2 strategies × 9 schemes × 3 seeds) |
| Ben | gnn-l4-02 | Cora_LP | 54 |
| Ben | gnn-l4-03 | CiteSeer_LP | 54 |
| Ben | gnn-l4-04 | Standby | 0 (can stop to save costs) |
| Tim | gnn-l4-01 | ENZYMES | 54 |
| Tim | gnn-l4-02 | PTC_MR | 54 |
| Tim | gnn-l4-03 | Cora_NC | 54 |
| Tim | gnn-l4-04 | Standby | 0 (can stop to save costs) |

**Total**: Ben = 162 experiments, Tim = 162 experiments (perfectly balanced!)  
**Expected Duration**: 4-6 hours (54 experiments per active VM)

---

## Monitoring and Results

### Real-time Monitoring
- **Wandb Dashboard**: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining
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
| **Pretraining** | 2-4 hours | 8 VMs (3 exp/VM) | Required first |
| **Finetuning** | 4-6 hours | 6 VMs (54 exp/VM) | After pretraining |
| **Total** | **6-10 hours** | vs 41 hours sequential | 4x speedup! |

## Cost Estimate
- **L4 GPU**: ~$0.60/hour
- **8 VMs × 8 hours**: ~$38 total
- **vs Sequential**: 41 hours × $0.60 = ~$25 (but takes 2+ days)

---

## Success Criteria

✅ **Pretraining Complete**: 24 runs in wandb  
✅ **Finetuning Complete**: 324 runs in wandb  
✅ **Total Results**: 348 runs with comprehensive metrics  
✅ **Ready for Analysis**: Statistical significance testing, performance tables, plots

**Final Results Location**: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining

---

*This distributed execution strategy reduces total runtime from 41 hours to 6-10 hours while maintaining full experimental rigor!*

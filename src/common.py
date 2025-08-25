# =============================================================================
# COMMON UTILITIES
# =============================================================================
# 
# This file now contains only the most general utilities that are used
# across multiple modules. Most constants have been moved to their
# logical locations:
#
# - GNN constants -> src/model/gnn.py
# - Model head constants -> src/model/heads.py  
# - Token initialization -> src/model/pretrain_model.py
# - Task constants -> src/pretraining/tasks.py
# - Loss constants -> src/pretraining/losses.py
# - Dataset constants -> src/data/data_setup.py
# - Training/monitoring constants -> src/pretraining/main_pretrain.py
# - Augmentation constants -> src/pretraining/augmentations.py
#
# =============================================================================

from src.data.data_setup import DATASET_SIZES

# Step-based training system constants (used by multiple training schemes)
SAMPLES_PER_DOMAIN_PER_BATCH = 32 // len(DATASET_SIZES)  # 8 samples per domain
LARGEST_DOMAIN_SIZE = max(DATASET_SIZES.values())        # 4110 samples (largest domain)  
STEPS_PER_EPOCH_MULTI_DOMAIN = LARGEST_DOMAIN_SIZE // SAMPLES_PER_DOMAIN_PER_BATCH  # ~513 steps/epoch

STEPS_PER_EPOCH_SINGLE_DOMAIN = DATASET_SIZES['ENZYMES'] // 32  # ~19 steps/epoch

EQUIVALENT_EPOCHS = 30  # Target: 30 epochs worth of training
PRETRAIN_MAX_STEPS_MULTI_DOMAIN = EQUIVALENT_EPOCHS * STEPS_PER_EPOCH_MULTI_DOMAIN    # ~15390 steps
PRETRAIN_MAX_STEPS_SINGLE_DOMAIN = EQUIVALENT_EPOCHS * STEPS_PER_EPOCH_SINGLE_DOMAIN  # ~570 steps

PRETRAIN_EVAL_EVERY_STEPS_MULTI_DOMAIN = STEPS_PER_EPOCH_MULTI_DOMAIN // 2   # ~256 steps
PRETRAIN_EVAL_EVERY_STEPS_SINGLE_DOMAIN = STEPS_PER_EPOCH_SINGLE_DOMAIN // 2 # ~10 steps

PRETRAIN_PATIENCE_STEPS_MULTI_DOMAIN = int(0.25 * PRETRAIN_MAX_STEPS_MULTI_DOMAIN)   # ~3847 steps
PRETRAIN_PATIENCE_STEPS_SINGLE_DOMAIN = int(0.25 * PRETRAIN_MAX_STEPS_SINGLE_DOMAIN) # ~143 steps

PRETRAIN_WARMUP_STEPS_MULTI_DOMAIN = int(0.1 * PRETRAIN_MAX_STEPS_MULTI_DOMAIN)   # ~1539 steps
PRETRAIN_WARMUP_STEPS_SINGLE_DOMAIN = int(0.1 * PRETRAIN_MAX_STEPS_SINGLE_DOMAIN)  # ~57 steps

PRETRAIN_LOG_EVERY_STEPS_MULTI_DOMAIN = int(0.25 * STEPS_PER_EPOCH_MULTI_DOMAIN)   # Every ~0.25 epochs
PRETRAIN_LOG_EVERY_STEPS_SINGLE_DOMAIN = int(0.25 * STEPS_PER_EPOCH_SINGLE_DOMAIN) # Every ~0.25 epochs

# Mathematical Constants
GRAD_NORM_EXPONENT = 0.5  # Exponent for gradient norm (sqrt)

# Scheme Classification Function  
def get_training_scheme(exp_name: str, active_tasks: list) -> str:
    """
    Determine training scheme from experiment name and active tasks.
    Based on analysis of your current experiment naming patterns.
    """
    exp_name_lower = exp_name.lower()

    # Single domain experiments
    if 'single_domain' in exp_name_lower or len(set(['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']) & set(exp_name.upper().split('_'))) == 1:
        return 'single_domain'

    # Multi-task contrastive (your s2 problem case)
    if 'contrastive' in exp_name_lower or ('multi_task' in exp_name_lower and any(task in ['node_contrast', 'graph_contrast'] for task in active_tasks)):
        return 'multi_task_contrastive'

    # Multi-task generative
    if 'generative' in exp_name_lower or ('multi_task' in exp_name_lower and any(task in ['node_feat_mask', 'graph_prop'] for task in active_tasks)):
        return 'multi_task_generative'

    # All self-supervised (comprehensive training)
    if 'all_self_supervised' in exp_name_lower or len(active_tasks) >= 4:
        return 'all_self_supervised'

    return 'default'
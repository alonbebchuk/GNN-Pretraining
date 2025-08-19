"""
Central hub for all experiment hyperparameters and configuration.
"""

# -----------------------------------------------------------------------------
# Global & Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED = 0

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------
DROPOUT_RATE = 0.2
GNN_HIDDEN_DIM = 256
GNN_NUM_LAYERS = 5

CONTRASTIVE_PROJ_DIM = 128
GRAPH_PROP_HEAD_HIDDEN_DIM = 512
DOMAIN_ADV_HEAD_HIDDEN_DIM = 128
DOMAIN_ADV_HEAD_OUT_DIM = 4

MASK_TOKEN_INIT_MEAN = 0.0
MASK_TOKEN_INIT_STD = 0.02

# -----------------------------------------------------------------------------
# Data Processing & Splits
# -----------------------------------------------------------------------------
DATA_ROOT_DIR = '/kaggle/working/gnn-pretraining/data'

VAL_FRACTION = 0.1
VAL_TEST_FRACTION = 0.2
VAL_TEST_SPLIT_RATIO = 0.5

NORMALIZATION_EPS = 1e-8
NORMALIZATION_STD_FALLBACK = 1.0

# -----------------------------------------------------------------------------
# Graph Augmentations (GraphCL-style)
# -----------------------------------------------------------------------------
AUGMENTATION_ATTR_MASK_PROB = 0.5
AUGMENTATION_ATTR_MASK_RATE = 0.15
AUGMENTATION_EDGE_DROP_PROB = 0.5
AUGMENTATION_EDGE_DROP_RATE = 0.15
AUGMENTATION_NODE_DROP_PROB = 0.5
AUGMENTATION_NODE_DROP_RATE = 0.15
AUGMENTATION_MIN_NODES_PER_GRAPH = 2

# -----------------------------------------------------------------------------
# Pretraining Tasks
# -----------------------------------------------------------------------------
NODE_FEATURE_MASKING_MASK_RATE = 0.15

NODE_CONTRASTIVE_TEMPERATURE = float(0.1)

GRAPH_PROPERTY_DIM = 15

# -----------------------------------------------------------------------------
# Loss Weighting & Schedulers
# -----------------------------------------------------------------------------
UNCERTAINTY_LOSS_COEF = 0.5
LOGSIGMA_TO_SIGMA_SCALE = 0.5

GRL_GAMMA = 10.0
GRL_LAMBDA_MIN = 0.0
GRL_LAMBDA_MAX = 1.0

# -----------------------------------------------------------------------------
# Datasets & Domains
# -----------------------------------------------------------------------------
PRETRAIN_TUDATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']
DOWNSTREAM_TUDATASETS = ['ENZYMES', 'FRANKENSTEIN', 'PTC_MR']
OVERLAP_TUDATASETS = sorted(list(set(PRETRAIN_TUDATASETS).intersection(set(DOWNSTREAM_TUDATASETS))))
ALL_TUDATASETS = sorted(list(set(PRETRAIN_TUDATASETS + DOWNSTREAM_TUDATASETS)))
ALL_PLANETOID_DATASETS = ['Cora', 'CiteSeer']

FEATURE_TYPES = {
    'MUTAG': 'categorical',
    'PROTEINS': 'categorical',
    'NCI1': 'categorical',
    'ENZYMES': 'continuous',
    'FRANKENSTEIN': 'categorical',
    'PTC_MR': 'categorical',
    'Cora': 'bow',
    'CiteSeer': 'bow'
}

DOMAIN_DIMENSIONS = {
    'MUTAG': 7,
    'PROTEINS': 4,
    'NCI1': 37,
    'ENZYMES': 21,
    'FRANKENSTEIN': 780,
    'PTC_MR': 18,
    'Cora': 1433,
    'CiteSeer': 3703
}

# -----------------------------------------------------------------------------
# Pretraining - Training Hyperparameters & Tracking (shared, not per-run)
# -----------------------------------------------------------------------------

PRETRAIN_EPOCHS = 20
PATIENCE = 20
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_EVAL_EVERY_EPOCHS = 1
PRETRAIN_LOG_EVERY_STEPS = 10
PRETRAIN_NUM_WORKERS = 4
PRETRAIN_LR_WARMUP_FRACTION = 0.1
PRETRAIN_LR_MIN_FACTOR = 0.01

PRETRAIN_LR_MODEL = 3e-4
PRETRAIN_LR_UNCERTAINTY = 5e-3
PRETRAIN_ADAM_BETAS = (0.9, 0.999)
PRETRAIN_ADAM_EPS = 1e-8
PRETRAIN_MODEL_WEIGHT_DECAY = 0.01
PRETRAIN_UNCERTAINTY_WEIGHT_DECAY = 0.0

# -----------------------------------------------------------------------------
# Scheme-Specific Hyperparameters (Literature-Informed)
# -----------------------------------------------------------------------------

# Learning rates by training scheme (based on academic literature analysis)
SCHEME_SPECIFIC_LR_MODEL = {
    'multi_task_contrastive': 1e-4,      # Lower LR for contrastive learning (MA-GCL, GraphCL-DTA)
    'single_domain': 3e-4,               # Standard rate for single domain
    'multi_task_generative': 2e-4,       # Slightly lower for generative tasks  
    'all_self_supervised': 1e-3,         # Higher rate for comprehensive training (GSR)
    'default': 3e-4
}

SCHEME_SPECIFIC_LR_UNCERTAINTY = {
    'multi_task_contrastive': 1e-3,      # Reduced uncertainty LR for contrastive
    'single_domain': 5e-3,               # Keep current for single domain
    'multi_task_generative': 3e-3,       # Moderate reduction for generative
    'all_self_supervised': 5e-3,         # Keep standard for comprehensive
    'default': 5e-3
}

# Learning rate decay factors by scheme
SCHEME_SPECIFIC_LR_MIN_FACTOR = {
    'multi_task_contrastive': 0.1,       # Less aggressive decay for contrastive (MA-GCL)
    'single_domain': 0.01,               # Current setting
    'multi_task_generative': 0.05,       # Moderate decay
    'all_self_supervised': 0.01,         # Standard decay
    'default': 0.01
}

# Warmup fractions by scheme  
SCHEME_SPECIFIC_LR_WARMUP_FRACTION = {
    'multi_task_contrastive': 0.2,       # Longer warmup for contrastive (GraphCL-DTA)
    'single_domain': 0.1,                # Current setting
    'multi_task_generative': 0.15,       # Moderate warmup
    'all_self_supervised': 0.1,          # Standard warmup
    'default': 0.1
}

# Early stopping patience by scheme
SCHEME_SPECIFIC_PATIENCE = {
    'multi_task_contrastive': 8,         # Shorter patience due to overfitting tendency
    'single_domain': 15,                 # Moderate patience
    'multi_task_generative': 12,         # Moderate patience  
    'all_self_supervised': 20,           # Longer patience for comprehensive training
    'default': 20
}

# Dropout rates by scheme (for contrastive regularization)
SCHEME_SPECIFIC_DROPOUT = {
    'multi_task_contrastive': 0.4,       # Higher dropout for contrastive (MA-GCL, RHCO)
    'single_domain': 0.2,                # Current setting
    'multi_task_generative': 0.3,        # Moderate regularization
    'all_self_supervised': 0.2,          # Standard dropout
    'default': 0.2
}

# Weight decay by scheme
SCHEME_SPECIFIC_MODEL_WEIGHT_DECAY = {
    'multi_task_contrastive': 0.05,      # Stronger regularization for contrastive
    'single_domain': 0.01,               # Current setting
    'multi_task_generative': 0.02,       # Moderate regularization
    'all_self_supervised': 0.01,         # Standard regularization
    'default': 0.01
}

WANDB_PROJECT = 'gnn-pretraining'
PRETRAIN_OUTPUT_DIR = '/kaggle/working/gnn-pretraining/outputs/pretrain'

PRETRAIN_PIN_MEMORY = True
PRETRAIN_DROP_LAST = False

# -----------------------------------------------------------------------------
# Scheme Classification Functions
# -----------------------------------------------------------------------------

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

def get_scheme_hyperparameter(scheme: str, param_dict: dict, default_key: str = 'default'):
    """Get hyperparameter value for a specific training scheme."""
    return param_dict.get(scheme, param_dict.get(default_key, None))

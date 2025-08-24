"""
Central hub for all experiment hyperparameters and configuration.

This module organizes all hyperparameters by purpose and usage:
1. Core System Settings
2. Data & Preprocessing
3. Model Architecture
4. Step-Based Training Configuration
5. Task-Specific Settings
6. Augmentation & Regularization
7. Dataset Definitions
8. Task Balancing & Monitoring System
9. Scheme-Specific Hyperparameters
10. Utility Functions for Setup & Monitoring
"""

# =============================================================================
# 1. CORE SYSTEM SETTINGS
# =============================================================================

# Reproducibility
RANDOM_SEED = 0

# Paths & Environment
DATA_ROOT_DIR = '/kaggle/working/gnn-pretraining/data'
PRETRAIN_OUTPUT_DIR = '/kaggle/working/gnn-pretraining/outputs/pretrain'
WANDB_PROJECT = 'gnn-pretraining'

# System Configuration
PRETRAIN_PIN_MEMORY = True
PRETRAIN_DROP_LAST = False

# =============================================================================
# 2. DATA & PREPROCESSING
# =============================================================================

# Dataset Splits
VAL_FRACTION = 0.1                # Validation fraction for pretraining
VAL_TEST_FRACTION = 0.2           # Val+test fraction for downstream
VAL_TEST_SPLIT_RATIO = 0.5        # Split ratio between val and test

# Mathematical Constants
EPSILON = 1e-8                    # Avoid division by zero
PERCENTAGE_CONVERSION = 100       # Convert ratios to percentages

# =============================================================================
# 3. MODEL ARCHITECTURE
# =============================================================================

# Core GNN Architecture
GNN_HIDDEN_DIM = 256              # Hidden dimension for GNN layers
GNN_NUM_LAYERS = 5                # Number of GNN layers
DROPOUT_RATE = 0.2                # Base dropout rate

# Task-Specific Head Dimensions
CONTRASTIVE_PROJ_DIM = 128        # Contrastive projection dimension
GRAPH_PROP_HEAD_HIDDEN_DIM = 512  # Graph property prediction head
DOMAIN_ADV_HEAD_HIDDEN_DIM = 128  # Domain adversarial head
DOMAIN_ADV_HEAD_OUT_DIM = 4       # Number of domains for adversarial training

# Token Initialization
MASK_TOKEN_INIT_MEAN = 0.0        # Mean for mask token initialization
MASK_TOKEN_INIT_STD = 0.02        # Std for mask token initialization

# =============================================================================
# 4. TRAINING CONFIGURATION
# =============================================================================

# =============================================================================
# STEP-BASED TRAINING SYSTEM (Fair & Robust)
# =============================================================================

# Dataset sizes for step calculation
DATASET_SIZES = {
    'MUTAG': 188,
    'PROTEINS': 1113,
    'NCI1': 4110,
    'ENZYMES': 600,
    'total_multi_domain': 6011,  # Sum of pretraining datasets
    'batch_size': 32  # From PRETRAIN_BATCH_SIZE
}

# Calculate steps per epoch for different training modes
# ~188 steps/epoch
STEPS_PER_EPOCH_MULTI_DOMAIN = DATASET_SIZES['total_multi_domain'] // DATASET_SIZES['batch_size']
# ~19 steps/epoch
STEPS_PER_EPOCH_SINGLE_DOMAIN = DATASET_SIZES['ENZYMES'] // DATASET_SIZES['batch_size']

# Training Duration: Translate 20-50 epochs to steps
# Using multi-domain as baseline (more conservative estimate)
EQUIVALENT_EPOCHS = 30  # Target: 30 epochs worth of training
PRETRAIN_MAX_STEPS = EQUIVALENT_EPOCHS * \
    STEPS_PER_EPOCH_MULTI_DOMAIN  # ~5640 steps

# Evaluation & Early Stopping
# ~94 steps (0.5 epoch equivalent)
PRETRAIN_EVAL_EVERY_STEPS = STEPS_PER_EPOCH_MULTI_DOMAIN // 2
# 25% of total steps (~1410 steps)
PRETRAIN_PATIENCE_STEPS = int(0.25 * PRETRAIN_MAX_STEPS)

# Warmup & Scheduling
# 10% warmup (~564 steps)
PRETRAIN_WARMUP_STEPS = int(0.1 * PRETRAIN_MAX_STEPS)

# Logging frequency
PRETRAIN_LOG_EVERY_STEPS = 50  # More frequent logging for step-based training

# Base Learning Rates (DEPRECATED - use scheme-specific values)
# These are kept for backward compatibility but schemes should be used instead
PRETRAIN_LR_MODEL = 3e-4          # ⚠️  DEPRECATED: Use get_scheme_hyperparameters()
PRETRAIN_LR_UNCERTAINTY = 5e-3    # ⚠️  DEPRECATED: Use get_scheme_hyperparameters()
# ⚠️  DEPRECATED: Use get_scheme_hyperparameters()
PRETRAIN_MODEL_WEIGHT_DECAY = 0.01
PRETRAIN_UNCERTAINTY_WEIGHT_DECAY = 0.0  # No L2 for uncertainty weights

# Optimizer Configuration (shared across all schemes)
PRETRAIN_LR_MIN_FACTOR = 0.01     # Min LR = initial_lr * 0.01 (1% decay floor)
PRETRAIN_ADAM_BETAS = (0.9, 0.999)  # Adam optimizer betas
PRETRAIN_ADAM_EPS = 1e-8          # Adam epsilon

# Training Loop Configuration
PRETRAIN_BATCH_SIZE = 32          # Batch size
PRETRAIN_NUM_WORKERS = 2          # DataLoader workers

# =============================================================================
# 5. TASK-SPECIFIC SETTINGS
# =============================================================================

# Pretraining Task Configuration
NODE_FEATURE_MASKING_MASK_RATE = 0.15  # Fraction of node features to mask
NODE_CONTRASTIVE_TEMPERATURE = 0.1     # Temperature for contrastive learning
GRAPH_PROPERTY_DIM = 15                 # Number of graph properties to predict

# Loss Weighting & Uncertainty
UNCERTAINTY_LOSS_COEF = 0.5       # Weight for uncertainty regularization
LOGSIGMA_TO_SIGMA_SCALE = 0.5     # Scale factor for log(σ) → σ conversion

# Domain Adversarial Learning
GRL_GAMMA = 10.0                  # Gradient reversal layer strength
GRL_LAMBDA_MIN = 0.0              # Min domain adversarial weight
GRL_LAMBDA_MAX = 1.0              # Max domain adversarial weight

# Scheduler Constants
GRL_CORE_NUMERATOR = 2.0          # Core numerator for GRL scheduling
COSINE_MULTIPLIER = 0.5           # Cosine schedule multiplier
COSINE_OFFSET = 1.0               # Cosine schedule offset
SCHEDULER_PROGRESS_CLAMP_MIN = 0.0  # Min progress for schedulers
SCHEDULER_PROGRESS_CLAMP_MAX = 1.0  # Max progress for schedulers

# =============================================================================
# 6. AUGMENTATION & REGULARIZATION
# =============================================================================

# Graph Augmentation Probabilities (GraphCL-style)
AUGMENTATION_ATTR_MASK_PROB = 0.5    # Probability of attribute masking
AUGMENTATION_EDGE_DROP_PROB = 0.5    # Probability of edge dropping
AUGMENTATION_NODE_DROP_PROB = 0.5    # Probability of node dropping

# Augmentation Rates
AUGMENTATION_ATTR_MASK_RATE = 0.15   # Fraction of attributes to mask
AUGMENTATION_EDGE_DROP_RATE = 0.15   # Fraction of edges to drop
AUGMENTATION_NODE_DROP_RATE = 0.15   # Fraction of nodes to drop

# Graph Structure Constraints
AUGMENTATION_MIN_NODES_PER_GRAPH = 2  # Min nodes after augmentation
# WHY 2? Because 1 node = isolated node with no edges = destroys graph structure
# 2 nodes allows ≥1 potential edge, preserving minimal graph semantics
# Based on GraphCL literature and GNN requirements for connectivity

# Edge Structure Constraints
AUGMENTATION_MIN_EDGES_PER_GRAPH = 1      # Always keep at least 1 edge if graph had edges
SMALL_GRAPH_EDGE_THRESHOLD = 8            # Graphs with ≤8 edges need conservative dropping  
MIN_EDGE_RETENTION_RATE = 0.5             # Keep at least 50% edges for small graphs
# WHY these values? Small molecular/social graphs need structural preservation
# Dropping >50% edges from graphs with ≤8 edges often destroys meaningful connectivity

# Graph Processing Constraints
MIN_NODES_AFTER_DROP = 2          # Min nodes to keep after node dropping
SMALL_GRAPH_THRESHOLD = 5         # Threshold for small graphs
MAX_SINGLE_NODE_DROP = 1          # Max nodes to drop from small graphs

# Mathematical Constants for Graph Metrics
GRAD_NORM_EXPONENT = 0.5          # Exponent for gradient norm (sqrt)

# Monitoring Constants
TIMING_STEPS_WINDOW = 10          # Number of recent steps for timing
CONTRASTIVE_FALLBACK_LOSS = 0.1   # Fallback loss for edge cases
TASK_ZERO_LOSS = 0.0              # Zero loss for empty batches
DEFAULT_TASK_SCALE = 1.0          # Default scaling factor

# =============================================================================
# 7. DATASET DEFINITIONS
# =============================================================================

# Dataset Collections
PRETRAIN_TUDATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']
DOWNSTREAM_TUDATASETS = ['ENZYMES', 'FRANKENSTEIN', 'PTC_MR']
OVERLAP_TUDATASETS = list(set(PRETRAIN_TUDATASETS).intersection(set(DOWNSTREAM_TUDATASETS)))
ALL_TUDATASETS = list(set(PRETRAIN_TUDATASETS + DOWNSTREAM_TUDATASETS))
ALL_PLANETOID_DATASETS = ['Cora', 'CiteSeer']

# Dataset Feature Types
FEATURE_TYPES = {
    'MUTAG': 'categorical',      # One-hot encoded molecular features
    'PROTEINS': 'categorical',   # One-hot encoded protein features
    'NCI1': 'categorical',       # One-hot encoded chemical features
    'ENZYMES': 'continuous',     # Continuous enzyme features (scaled)
    'FRANKENSTEIN': 'categorical',  # One-hot encoded features
    'PTC_MR': 'categorical',     # One-hot encoded chemical features
    'Cora': 'bow',              # Bag-of-words text features (normalized)
    'CiteSeer': 'bow'           # Bag-of-words text features (normalized)
}

# Input Feature Dimensions by Dataset
DOMAIN_DIMENSIONS = {
    'MUTAG': 7,          # 7 node features for molecular graphs
    'PROTEINS': 4,       # 4 node features for protein structure
    'NCI1': 37,          # 37 chemical atom features
    'ENZYMES': 21,       # 21 continuous enzyme properties
    'FRANKENSTEIN': 780,  # 780 molecular descriptors
    'PTC_MR': 18,        # 18 chemical atom features
    'Cora': 1433,        # 1433 text features (papers)
    'CiteSeer': 3703     # 3703 text features (papers)
}

# Domain-specific Preprocessing
CONTINUOUS_FEATURE_SCALE_FACTOR = 0.5  # Scale down continuous features to match categorical variance

# -----------------------------------------------------------------------------
# Pretraining - Training Hyperparameters & Tracking (shared, not per-run)
# -----------------------------------------------------------------------------

# PRETRAIN_EPOCHS = 20
PRETRAIN_EPOCHS = 5
PATIENCE = 5
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

# =============================================================================
# 8. TASK BALANCING & MONITORING SYSTEM
# =============================================================================

# WHY DUAL MONITORING? Loss scales + uncertainty weighting work together:
# 1. Loss scales: Provide manual control based on domain knowledge
# 2. Uncertainty weighting: Adaptive balancing that learns from data
# 3. Together: Best of both worlds - informed initialization + adaptive learning

TASK_LOSS_SCALES = {
    'node_feat_mask': 0.1,     # Scale DOWN - was dominating due to high dimensionality
    'graph_prop': 0.3,         # Scale down moderately - medium complexity task
    'node_contrast': 1.0,      # Reference task (keep as baseline)
    'graph_contrast': 2.0,     # Scale UP - was too weak, needs more signal
    'link_pred': 1.5,          # Scale UP slightly - important structural task
    'domain_adv': 1.0,         # Reference task (domain adaptation baseline)
}

# Comprehensive monitoring metrics for Wandb logging
MONITORING_METRICS = {
    # Task Performance Metrics
    'task_losses': ['node_feat_mask_loss', 'graph_prop_loss', 'node_contrast_loss',
                    'graph_contrast_loss', 'link_pred_loss', 'domain_adv_loss'],

    # Task Balancing Metrics
    'uncertainty_weights': ['unc_weight_node_feat_mask', 'unc_weight_graph_prop', 'unc_weight_node_contrast',
                            'unc_weight_graph_contrast', 'unc_weight_link_pred', 'unc_weight_domain_adv'],

    # Loss Scale Effects
    'scaled_losses': ['scaled_node_feat_mask', 'scaled_graph_prop', 'scaled_node_contrast',
                      'scaled_graph_contrast', 'scaled_link_pred', 'scaled_domain_adv'],

    # Task Contributions (scaled_loss * uncertainty_weight)
    'task_contributions': ['contrib_node_feat_mask', 'contrib_graph_prop', 'contrib_node_contrast',
                           'contrib_graph_contrast', 'contrib_link_pred', 'contrib_domain_adv'],

    # Balance Analysis
    'balance_metrics': ['total_loss', 'uncertainty_loss', 'dominant_task', 'task_balance_ratio',
                        'contribution_entropy', 'uncertainty_adaptation_rate'],

    # Training Progress
    'training_metrics': ['step', 'lr_model', 'lr_uncertainty', 'grad_norm_model', 'grad_norm_uncertainty'],

    # System Metrics
    'system_metrics': ['step_time', 'memory_usage', 'gpu_utilization']
}

# =============================================================================
# 9. SCHEME-SPECIFIC HYPERPARAMETERS (Clean Design)
# =============================================================================

# WHY SCHEME-SPECIFIC HYPERPARAMS? Different training paradigms need different approaches:
# - Contrastive learning: Prone to representation collapse → needs higher dropout, lower LR
# - Generative tasks: Different convergence patterns → different patience/warmup
# - Multi-task: Complex optimization landscape → specialized scheduling
# Based on literature analysis (MA-GCL, GraphCL-DTA, GSR, RHCO papers)

# Clean design: All scheme hyperparameters in one place
SCHEME_HYPERPARAMETERS = {
    'multi_task_contrastive': {
        # Contrastive learning requires careful tuning to prevent collapse
        # LOWER - prevents representation collapse (MA-GCL, GraphCL-DTA)
        'lr_model': 1e-4,
        'lr_uncertainty': 1e-3,        # Reduced - uncertainty adapts slowly with contrastive
        # LESS aggressive decay - avoids optimization cliff (MA-GCL)
        'lr_min_factor': 0.1,
        # LONGER warmup - critical for stability (GraphCL-DTA)
        'warmup_steps_fraction': 0.2,
        # SHORTER patience - overfitting tendency with contrastive
        'patience_fraction': 0.15,
        # HIGHER - prevents overfitting (MA-GCL, RHCO)
        'dropout_rate': 0.4,
        'weight_decay': 0.05,          # STRONGER - regularization critical for contrastive
    },
    'single_domain': {
        # Focused training on single dataset
        'lr_model': 3e-4,              # Standard rate for focused training
        'lr_uncertainty': 5e-3,        # Standard for single-domain focus
        'lr_min_factor': 0.01,         # Standard 1% minimum
        'warmup_steps_fraction': 0.1,  # Standard warmup
        'patience_fraction': 0.3,      # Moderate patience for focused training
        'dropout_rate': 0.2,           # Standard regularization
        'weight_decay': 0.01,          # Standard L2 regularization
    },
    'multi_task_generative': {
        # Generative tasks (masking, graph properties)
        'lr_model': 2e-4,              # Moderate - generative tasks need stability
        'lr_uncertainty': 3e-3,        # Moderate adaptation rate
        'lr_min_factor': 0.05,         # Moderate decay to 5%
        'warmup_steps_fraction': 0.15,  # Moderate warmup for stability
        'patience_fraction': 0.25,     # Moderate patience
        'dropout_rate': 0.3,           # Moderate regularization
        'weight_decay': 0.02,          # Moderate regularization
    },
    'all_self_supervised': {
        # Comprehensive training with all tasks
        'lr_model': 1e-3,              # HIGHER - comprehensive training (GSR)
        'lr_uncertainty': 5e-3,        # Standard for comprehensive training
        'lr_min_factor': 0.01,         # Standard decay for stability
        'warmup_steps_fraction': 0.1,  # Standard warmup
        'patience_fraction': 0.35,     # LONGER - complex training needs more time
        'dropout_rate': 0.2,           # Standard for comprehensive training
        'weight_decay': 0.01,          # Standard regularization
    },
    'default': {
        # Fallback configuration
        'lr_model': 3e-4,
        'lr_uncertainty': 5e-3,
        'lr_min_factor': 0.01,
        'warmup_steps_fraction': 0.1,
        'patience_fraction': 0.25,
        'dropout_rate': 0.2,
        'weight_decay': 0.01,
    }
}

# =============================================================================
# 10. SCHEME CLASSIFICATION & UTILITY FUNCTIONS
# =============================================================================


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


def get_scheme_hyperparameters(scheme: str) -> dict:
    """
    Get all hyperparameters for a specific training scheme.

    Args:
        scheme: Training scheme name

    Returns:
        dict: Complete hyperparameter configuration for the scheme
    """
    scheme_config = SCHEME_HYPERPARAMETERS.get(
        scheme, SCHEME_HYPERPARAMETERS['default']).copy()

    # Convert fractional values to absolute step values
    scheme_config['warmup_steps'] = int(
        scheme_config['warmup_steps_fraction'] * PRETRAIN_MAX_STEPS)
    scheme_config['patience_steps'] = int(
        scheme_config['patience_fraction'] * PRETRAIN_MAX_STEPS)

    # Add common training configuration
    scheme_config.update({
        'max_steps': PRETRAIN_MAX_STEPS,
        'eval_every_steps': PRETRAIN_EVAL_EVERY_STEPS,
        'log_every_steps': PRETRAIN_LOG_EVERY_STEPS,
        'batch_size': PRETRAIN_BATCH_SIZE,
        'num_workers': PRETRAIN_NUM_WORKERS,
        'training_mode': 'steps'
    })

    return scheme_config


def get_scheme_hyperparameter(scheme: str, param_name: str, default_scheme: str = 'default'):
    """Get specific hyperparameter value for a training scheme (backward compatibility)."""
    scheme_config = SCHEME_HYPERPARAMETERS.get(
        scheme, SCHEME_HYPERPARAMETERS.get(default_scheme, {}))
    return scheme_config.get(param_name)


def setup_training_config(scheme: str, experiment_name: str = None) -> dict:
    """
    Complete training setup for a given scheme with all configurations.

    Args:
        scheme: Training scheme name
        experiment_name: Optional experiment name for logging

    Returns:
        dict: Complete training configuration
    """
    config = get_scheme_hyperparameters(scheme)

    # Add experiment metadata
    config.update({
        'scheme': scheme,
        'experiment_name': experiment_name or f"{scheme}_experiment",
        'task_loss_scales': TASK_LOSS_SCALES.copy(),
        'monitoring_metrics': MONITORING_METRICS.copy(),
        'dataset_info': {
            'sizes': DATASET_SIZES,
            'steps_per_epoch_multi': STEPS_PER_EPOCH_MULTI_DOMAIN,
            'steps_per_epoch_single': STEPS_PER_EPOCH_SINGLE_DOMAIN,
            'equivalent_epochs': EQUIVALENT_EPOCHS
        }
    })

    return config


def get_monitoring_config() -> dict:
    """Get comprehensive monitoring configuration for Wandb."""
    return {
        'log_every_steps': PRETRAIN_LOG_EVERY_STEPS,
        'eval_every_steps': PRETRAIN_EVAL_EVERY_STEPS,
        'metrics': MONITORING_METRICS,
        'task_names': list(TASK_LOSS_SCALES.keys()),
        'track_gradients': True,
        'track_system_metrics': True,
        # Save less frequently than eval
        'save_checkpoints_every_steps': PRETRAIN_EVAL_EVERY_STEPS * 2
    }


def calculate_training_progress(current_step: int) -> dict:
    """Calculate training progress metrics."""
    progress_fraction = min(current_step / PRETRAIN_MAX_STEPS, 1.0)

    return {
        'progress_fraction': progress_fraction,
        'current_step': current_step,
        'max_steps': PRETRAIN_MAX_STEPS,
        'steps_remaining': max(PRETRAIN_MAX_STEPS - current_step, 0),
        'equivalent_epochs_completed': (current_step / STEPS_PER_EPOCH_MULTI_DOMAIN),
        'is_in_warmup': current_step < PRETRAIN_WARMUP_STEPS,
        'warmup_progress': min(current_step / PRETRAIN_WARMUP_STEPS, 1.0) if PRETRAIN_WARMUP_STEPS > 0 else 1.0
    }

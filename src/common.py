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
DATA_ROOT_DIR = '/content/data'

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

PRETRAIN_EPOCHS = 100
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_EVAL_EVERY_EPOCHS = 1
PRETRAIN_LOG_EVERY_STEPS = 10
PRETRAIN_NUM_WORKERS = 4
PRETRAIN_LR_WARMUP_FRACTION = 0.1
PRETRAIN_LR_MIN_FACTOR = 0.0

PRETRAIN_LR_MODEL = 3e-4
PRETRAIN_LR_UNCERTAINTY = 5e-3
PRETRAIN_ADAM_BETAS = (0.9, 0.999)
PRETRAIN_ADAM_EPS = 1e-8
PRETRAIN_MODEL_WEIGHT_DECAY = 0.01
PRETRAIN_UNCERTAINTY_WEIGHT_DECAY = 0.0

WANDB_PROJECT = 'gnn-pretraining'
PRETRAIN_OUTPUT_DIR = '/content/outputs/pretrain'

PRETRAIN_PIN_MEMORY = True
PRETRAIN_DROP_LAST = False

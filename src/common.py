"""
Central hub for all experiment hyperparameters and configuration.

Organized by logical groups to make it easy to configure experiments
without searching through the codebase.
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

# Head dimensions (relative to backbone hidden dim)
CONTRASTIVE_PROJ_DIM_FACTOR = 0.5          # output dim for node contrastive projection head
GRAPH_PROP_HEAD_HIDDEN_FACTOR = 2.0         # hidden dim for graph property head
DOMAIN_ADV_HEAD_HIDDEN_FACTOR = 0.5         # hidden dim for domain adversarial head

# Mask token initialization for node feature masking (BERT-style)
MASK_TOKEN_INIT_MEAN = 0.0
MASK_TOKEN_INIT_STD = 0.02

# -----------------------------------------------------------------------------
# Data Processing & Splits
# -----------------------------------------------------------------------------
CV_N_SPLITS = 1
PRETRAIN_VAL_TEST_FRACTION = 0.2            # for TU pretrain: 80/20 → val/test split next
VAL_TEST_SPLIT_RATIO = 0.5                  # splits the 20% equally into val/test

# Numerical stability and normalization
NORMALIZATION_EPS = 1e-8
NORMALIZATION_STD_FALLBACK = 1.0
BOW_ROW_SUM_FALLBACK = 1.0

# TU dataset download options
TUDATASET_USE_NODE_ATTR = True

# -----------------------------------------------------------------------------
# Graph Augmentations (GraphCL-style)
# -----------------------------------------------------------------------------
AUGMENTATION_ATTR_MASK_PROB = 0.5
AUGMENTATION_ATTR_MASK_RATE = 0.15
AUGMENTATION_EDGE_DROP_PROB = 0.5
AUGMENTATION_EDGE_DROP_RATE = 0.15
AUGMENTATION_SUBGRAPH_PROB = 0.5
AUGMENTATION_WALK_LENGTH = 10
AUGMENTATION_MIN_NODES_RATIO = 0.3          # fraction of start nodes for random walks

# Minimums for discrete sampling operations
AUGMENTATION_MIN_ATTR_MASK_DIM = 1          # at least one feature dim masked
AUGMENTATION_MIN_START_NODES = 1            # at least one starting node for random walks

# -----------------------------------------------------------------------------
# Pretraining Tasks
# -----------------------------------------------------------------------------
# Node feature masking
NODE_FEATURE_MASKING_MASK_RATE = 0.15
NODE_FEATURE_MASKING_MIN_NODES = 1

# Node-level contrastive learning
NODE_CONTRASTIVE_TEMPERATURE = float(0.1)
CONTRASTIVE_SYMMETRY_COEF = 0.5             # 0.5*(L12+L21)

# Link prediction
NEGATIVE_SAMPLING_RATIO = 1.0               # negatives per positive edge

# Graph property prediction
GRAPH_PROPERTY_DIM = 15

# -----------------------------------------------------------------------------
# Loss Weighting & Schedulers
# -----------------------------------------------------------------------------
# Uncertainty weighting (Kendall & Gal-style multi-task weighting)
UNCERTAINTY_LOSS_COEF = 0.5
LOGSIGMA_TO_SIGMA_SCALE = 0.5               # sigma = exp(0.5 * log_sigma_sq)

# GRL Lambda schedule (DANN)
GRL_GAMMA = 10.0
GRL_LAMBDA_MIN = 0.0
GRL_LAMBDA_MAX = 1.0

# -----------------------------------------------------------------------------
# Datasets & Domains
# -----------------------------------------------------------------------------
PRETRAIN_TUDATASETS = ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']
DOWNSTREAM_TUDATASETS = ['ENZYMES', 'FRANKENSTEIN', 'PTC_MR']
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

# Domain-specific input feature dimensions (PyG datasets)
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


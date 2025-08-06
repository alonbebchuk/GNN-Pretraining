# GNN Multi-Task Pre-Training and Evaluation Pipeline

This document provides a comprehensive guide to using the GNN multi-task pre-training and evaluation pipeline implemented according to the research plan specifications.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n gnn-simple python=3.9 -y
conda activate gnn-simple

# Install PyTorch and PyTorch Geometric
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install remaining dependencies
pip install torch-geometric scikit-learn tqdm pytest pyyaml wandb matplotlib
```

### 2. Data Preparation

```bash
# Run data setup (only needed once)
python src/data_setup.py
```

This will:
- Download all TUDatasets (MUTAG, PROTEINS, NCI1, ENZYMES, FRANKENSTEIN, PTC_MR)
- Download Planetoid datasets (Cora, CiteSeer)
- Apply preprocessing (z-score normalization, one-hot encoding, row-normalization)
- Generate train/validation/test splits with proper stratification
- Save processed data to `data/processed/`

### 3. Pre-Training

```bash
# Quick test run (recommended first)
python src/main_pretrain.py --config configs/quick_test.yaml --offline --log-level INFO

# Full training run
python src/main_pretrain.py --config configs/default.yaml --log-level INFO

# Run specific pre-training scheme
python src/main_pretrain.py --config configs/pretrain/s5_domain_invariant.yaml
```

### 4. Fine-Tuning and Evaluation

```bash
# Evaluate a pre-trained model on downstream task
python finetune.py --config configs/finetune_mutag_on_model_v3.yaml

# Run comprehensive evaluation suite
python run_evaluation_suite.py --model-artifact "your-entity/project/model:best"

# Evaluate multiple models
python run_evaluation_suite.py --model-list \
    "entity/project/s5-model:best" \
    "entity/project/s4-model:best"
```

### 5. Complete Experimental Pipeline

```bash
# Run all pre-training schemes
python run_all_experiments.py

# Run full evaluation suite after pre-training
python run_evaluation_suite.py --config evaluation_config.yaml
```

## 📊 **Evaluation Pipeline Overview**

The evaluation pipeline provides comprehensive assessment of pre-trained models through fine-tuning on downstream tasks:

### **Key Components:**
1. **`finetune.py`** - Main fine-tuning script for individual evaluations
2. **Configuration files** - YAML configs for different downstream tasks  
3. **`run_evaluation_suite.py`** - Batch evaluation script for comprehensive assessment

### **Supported Tasks:**
- **Graph Classification**: MUTAG, PROTEINS, NCI1, ENZYMES (in-domain), FRANKENSTEIN, PTC_MR (out-of-domain)
- **Node Classification**: Cora, CiteSeer (out-of-domain)

### **Evaluation Strategy:**
- **In-Domain Tasks**: Reuse pre-trained domain-specific encoders, faster convergence
- **Out-of-Domain Tasks**: Test transfer learning capabilities across different domains
- **Two-Phase Fine-tuning**: Initial head-only training, then end-to-end fine-tuning

### **Metrics Tracked:**
- Classification accuracy, F1-score, AUC-ROC
- Convergence speed and training time
- Transfer learning effectiveness

For detailed evaluation guide, see `EVALUATION_GUIDE.md`.

## 📋 Configuration System

The training pipeline is driven by YAML configuration files that specify all hyperparameters and settings.

### Configuration Structure

```yaml
run:                    # Experiment tracking settings
  project_name: 'Graph-Multitask-Learning'
  run_name: 'experiment-name'
  entity: null          # WandB entity (optional)
  tags: []              # Experiment tags
  notes: 'Description'  # Experiment notes

data:                   # Data loading settings
  pretrain_datasets: ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']
  batch_size: 32
  domain_balanced_sampling: true
  num_workers: 4
  pin_memory: true

model:                  # Model architecture
  hidden_dim: 256       # Hidden dimension (as per research plan)
  num_layers: 5         # Number of GIN layers
  dropout_rate: 0.2     # Dropout rate
  enable_augmentations: true

optimizer:              # Optimizer settings
  name: 'AdamW'
  lr: 5.0e-4           # Learning rate
  weight_decay: 0.01

scheduler:              # Learning rate scheduling
  type: 'cosine'        # cosine, linear, step, none
  warmup_epochs: 10
  min_lr_ratio: 0.01

training:               # Training process
  max_epochs: 100
  max_steps: 100000     # Use steps if > 0, otherwise epochs
  validation_freq: 1    # Validate every N epochs
  patience: 10          # Early stopping patience
  validation_metric: 'val/loss_total'
  metric_mode: 'min'    # min or max

domain_adversarial:     # Domain adversarial training
  enabled: true
  schedule_type: 'dann' # dann, linear, constant
  initial_lambda: 0.0
  final_lambda: 1.0
  gamma: 10.0

tasks:                  # Multi-task configuration
  node_feat_mask:       # Node feature masking
    enabled: true
    mask_rate: 0.15
  link_pred:            # Link prediction
    enabled: true
    negative_sampling_ratio: 1.0
  node_contrast:        # Node contrastive learning
    enabled: true
    temperature: 0.1
  graph_contrast:       # Graph contrastive learning
    enabled: true
  graph_prop:           # Graph property prediction
    enabled: true
  domain_adv:           # Domain adversarial
    enabled: true
```

## 🎯 Pre-Training Tasks

The pipeline implements all 6 pre-training tasks from the research plan:

### 1. Node Feature Masking (Generative)
- **Objective**: Reconstruct masked node features
- **Implementation**: 15% of nodes masked with learnable [MASK] tokens
- **Loss**: MSE between predicted and original h₀ embeddings
- **Head**: MLP (256 → 256 → 256)

### 2. Link Prediction (Generative)
- **Objective**: Predict existence of edges
- **Implementation**: 1:1 positive to negative edge sampling
- **Loss**: Binary cross-entropy
- **Head**: Dot-product decoder

### 3. Node Contrastive Learning (GraphCL-style)
- **Objective**: Learn node representations via contrastive learning
- **Implementation**: NT-Xent loss with graph augmentations
- **Augmentations**: Attribute masking (15%), edge dropping (15%), subgraph sampling
- **Head**: MLP projection to 128-dim space

### 4. Graph Contrastive Learning (InfoGraph-style)
- **Objective**: Maximize mutual information between nodes and graph summary
- **Implementation**: Bilinear discriminator for (node, graph) pairs
- **Loss**: Binary cross-entropy
- **Head**: Bilinear discriminator

### 5. Graph Property Prediction (Auxiliary Supervised)
- **Objective**: Predict structural properties
- **Properties**: Number of nodes, edges, average clustering coefficient
- **Loss**: MSE (z-score normalized targets)
- **Head**: MLP (256 → 256 → 3)

### 6. Domain Adversarial Training
- **Objective**: Learn domain-invariant representations
- **Implementation**: Gradient reversal layer with scheduled λ
- **Loss**: Cross-entropy for domain classification
- **Head**: MLP (256 → 128 → N_domains) with no dropout

## 🔧 Advanced Features

### Uncertainty Weighting
- **Purpose**: Automatically balance multi-task losses
- **Method**: Learnable uncertainty parameters σᵢ per task
- **Formula**: L_total = Σ[(1/2σᵢ²)Lᵢ + log σᵢ]

### Domain-Balanced Sampling
- **Purpose**: Ensure equal representation from all domains
- **Method**: First select domain uniformly, then sample graph from that domain
- **Benefit**: Prevents domain imbalance in batches

### Graph Augmentations
- **Attribute Masking**: Set 15% of feature dimensions to zero
- **Edge Dropping**: Remove 15% of edges randomly
- **Subgraph Sampling**: Extract subgraph via random walks (length 10)
- **Application**: Each augmentation applied with 50% probability

### Gradient Reversal Layer
- **Forward**: Identity function
- **Backward**: Multiply gradients by -λ
- **Schedule**: λ = 2/(1 + exp(-γp)) - 1 (DANN schedule)

## 📊 Experiment Tracking

### Weights & Biases Integration
- **Automatic**: Model architecture, hyperparameters, system metrics
- **Metrics**: All losses (individual and weighted), learning rates, λ schedule
- **Artifacts**: Model checkpoints with versioning
- **Visualization**: Learning curves, confusion matrices, histograms

### Logged Metrics
```
Training:
- train/loss_total, train/loss_node_feat_mask, train/loss_link_pred, etc.
- train/lr, train/lambda_da
- train/batch_time, train/epoch_time

Validation:
- val/loss_total, val/loss_node_feat_mask, val/loss_link_pred, etc.

System:
- system/gpu_memory_allocated_gb, system/gpu_memory_cached_gb

Uncertainty (if enabled):
- train/uncertainty_node_feat_mask, train/uncertainty_link_pred, etc.
```

## 💾 Checkpointing

### Automatic Checkpointing
- **Regular**: Every N epochs (configurable)
- **Best**: Based on validation metric
- **Latest**: Always maintains latest checkpoint
- **Final**: Saved at training completion

### Checkpoint Contents
```python
{
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss_computer_state_dict': uncertainty_params,
    'metrics': current_metrics,
    'best_metric_value': best_score,
    'timestamp': datetime.now().isoformat()
}
```

### Resuming Training
```bash
# Resume from specific checkpoint
python src/main_pretrain.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_0050.pt

# Resume from latest checkpoint (automatic)
python src/main_pretrain.py --config configs/default.yaml --resume checkpoints/latest_checkpoint.pt
```

## 🛠️ Command Line Options

```bash
python src/main_pretrain.py [OPTIONS]

Required:
  --config PATH              Configuration YAML file

Optional:
  --resume PATH              Resume from checkpoint
  --create-default-config    Create default config and exit
  --log-level LEVEL          DEBUG, INFO, WARNING, ERROR
  --offline                  Run without WandB logging
  --dry-run                  Validate setup without training
```

## 📈 Monitoring Training

### Real-time Monitoring
1. **Console**: Progress bars, loss values, learning rates
2. **WandB Dashboard**: Comprehensive metrics and visualizations
3. **Checkpoints**: Regular saves for resumption

### Key Metrics to Watch
- **Total Loss**: Should decrease steadily
- **Individual Task Losses**: Check for task-specific issues
- **Validation Loss**: Monitor for overfitting
- **Learning Rate**: Verify scheduler is working
- **Domain Adversarial λ**: Should increase according to schedule
- **Uncertainty Parameters**: Should stabilize during training

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Or use absolute imports in scripts
```

#### 2. CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `hidden_dim` or `num_layers`
- Set `pin_memory: false`

#### 3. Data Not Found
```bash
# Ensure data setup was run
python src/data_setup.py
```

#### 4. WandB Issues
- Use `--offline` flag for local-only runs
- Set `WANDB_MODE=offline` environment variable

#### 5. Slow Training
- Increase `num_workers` for data loading
- Use `pin_memory: true` for GPU training
- Reduce logging frequency (`log_freq`)

### Debug Mode
```bash
# Run with debug logging and small config
python src/main_pretrain.py --config configs/quick_test.yaml --log-level DEBUG --offline
```

## 📚 Research Plan Compliance

This implementation fully complies with the research plan specifications:

- ✅ **Architecture**: 5-layer GIN with 256-dim hidden space
- ✅ **Tasks**: All 6 pre-training tasks implemented
- ✅ **Losses**: Correct loss functions and weighting
- ✅ **Scheduling**: DANN schedule for domain adversarial λ
- ✅ **Data**: Domain-balanced sampling, proper preprocessing
- ✅ **Regularization**: 20% dropout, uncertainty weighting
- ✅ **Tracking**: Comprehensive experiment logging
- ✅ **Reproducibility**: Fixed seeds, deterministic operations

## 🔄 Next Steps

After pre-training completion:
1. **Model Evaluation**: Use saved checkpoints for downstream tasks
2. **Analysis**: Examine WandB logs for insights
3. **Fine-tuning**: Adapt pre-trained models to specific tasks
4. **Comparison**: Compare against from-scratch baselines

## 📞 Support

For issues or questions:
1. Check this README and configuration examples
2. Review console logs and WandB dashboard
3. Use `--dry-run` to validate setup
4. Test with `configs/quick_test.yaml` first 
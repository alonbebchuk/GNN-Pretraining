# Complete GNN Evaluation Pipeline Guide

## 🎯 Overview

This guide provides comprehensive instructions for running the complete Graph Neural Network (GNN) evaluation pipeline, from data preparation through final analysis. The system implements the research plan described in `plan.md` with enhanced robustness and production-ready features.

## 🏗️ System Architecture

The evaluation pipeline consists of several integrated components:

1. **Data Processing** (`src/data_setup.py`)
2. **Model Architecture** (`src/models/`)
3. **Pre-training Pipeline** (`src/main_pretrain.py`, `src/trainer.py`)
4. **Fine-tuning Pipeline** (`src/main_finetune.py`, `finetune_simplified.py`)
5. **Enhanced Evaluation** (`src/model_adapter.py`, `src/evaluation_metrics.py`)
6. **Comprehensive Analysis** (`compare_models.py`, `run_comprehensive_evaluation.py`)

## 🚀 Quick Start

### Step 1: System Validation

Before running any experiments, validate your system setup:

```bash
# Full validation
python validate_complete_system.py

# Quick validation (faster)
python validate_complete_system.py --quick

# Auto-fix detected issues
python validate_complete_system.py --fix-issues
```

### Step 2: Data Preparation

Prepare all datasets (run once):

```bash
# Activate your conda environment
conda activate gnn-simple

# Process all datasets
python src/data_setup.py
```

### Step 3: Run Integration Tests

Verify all components work together:

```bash
python test_complete_evaluation.py
```

## 📊 Evaluation Options

### Option 1: Simplified Fine-tuning (Recommended for Testing)

Use the robust simplified fine-tuning script:

```bash
# Example: Fine-tune on MUTAG
python finetune_simplified.py --config configs/finetune/template_enhanced.yaml
```

### Option 2: Full Fine-tuning Pipeline

Use the complete fine-tuning pipeline with all features:

```bash
# Example: Fine-tune with pre-trained model
python src/main_finetune.py \
    --pretrain-checkpoint checkpoints/s5_domain_invariant/best_model.pt \
    --downstream-config configs/finetune/graph_classification/mutag.yaml \
    --strategy full

# Example: B1 baseline (from scratch)
python src/main_finetune.py \
    --downstream-config configs/finetune/graph_classification/mutag.yaml \
    --strategy full
```

### Option 3: Comprehensive Evaluation Suite

Run all experiments as specified in the research plan:

```bash
# Quick test (minimal experiments)
python run_comprehensive_evaluation.py --subset quick --dry-run

# Full experimental suite
python run_comprehensive_evaluation.py

# Parallel execution (faster)
python run_comprehensive_evaluation.py --parallel --max-workers 4
```

## 🔧 Configuration System

### Pre-training Configurations

Located in `configs/pretrain/`:
- `b2_single_generative.yaml` - Single-task generative
- `b3_single_contrastive.yaml` - Single-task contrastive
- `s5_domain_invariant.yaml` - Multi-task domain-invariant
- etc.

### Fine-tuning Configurations

Located in `configs/finetune/`:
- `graph_classification/` - Graph-level tasks
- `node_classification/` - Node-level tasks
- `link_prediction/` - Edge-level tasks
- `template_enhanced.yaml` - Comprehensive template

### Configuration Structure

```yaml
# Model artifact (pre-trained checkpoint)
model_artifact:
  path: 'checkpoints/s5_domain_invariant/best_model.pt'

# Downstream task
downstream_task:
  dataset_name: 'MUTAG'
  task_type: 'graph_classification'
  batch_size: 32

# Training settings
training:
  epochs: 200
  optimizer: 'AdamW'
  learning_rate: 0.0001

# Fine-tuning strategy
fine_tuning_strategy:
  freeze_encoder: true
  unfreeze_epoch: 10
  adaptation_method: 'full'

# Experiment tracking
wandb:
  project_name: 'gnn-evaluation'
  run_name: 'mutag-experiment'
```

## 🧪 Experimental Schemes

As defined in `plan.md`, the system supports:

### Baseline Schemes
- **B1**: From-scratch training
- **B2**: Single-task generative (Node Feature Masking)
- **B3**: Single-task contrastive (Node Contrastive)
- **B4**: Single-domain all objectives (ENZYMES only)

### Multi-task Schemes
- **S1**: Multi-task generative (NFM + LP)
- **S2**: Multi-task contrastive (NC + GC)
- **S3**: All self-supervised (NFM + LP + NC + GC)
- **S4**: All objectives (S3 + GPP)
- **S5**: Domain-invariant (S4 + DA)

### Downstream Tasks
- **Graph Classification**: ENZYMES (in-domain), FRANKENSTEIN, PTC_MR (out-of-domain)
- **Node Classification**: Cora, CiteSeer (out-of-domain)
- **Link Prediction**: Cora_LP, CiteSeer_LP (out-of-domain)

## 🎛️ Advanced Features

### Enhanced Model Adapter

The `src/model_adapter.py` provides:
- Automatic model loading with fallbacks
- Domain-specific encoder handling
- Task-specific head creation
- Robust error handling

### Comprehensive Evaluation Metrics

The `src/evaluation_metrics.py` provides:
- Task-specific metrics (accuracy, F1, AUC, etc.)
- Link prediction metrics (Precision@K, Recall@K, Hit Rate)
- Training progress analysis
- Statistical significance testing

### Enhanced Link Prediction

The `src/link_prediction.py` provides:
- Advanced negative sampling strategies
- Multiple prediction methods
- Comprehensive evaluation metrics

## 📈 Results Analysis

### Individual Experiment Results

Results are saved in JSON format with comprehensive metrics:

```json
{
  "best_val_accuracy": 0.85,
  "final_test_metrics": {
    "test_accuracy": 0.82,
    "test_f1": 0.81,
    "test_auc_roc": 0.88
  },
  "training_history": {
    "train_losses": [...],
    "val_accuracies": [...]
  }
}
```

### Comprehensive Analysis

Use the analysis scripts:

```bash
# Compare models statistically
python compare_models.py --results-dir results/comprehensive_evaluation

# Generate comparison plots and reports
python compare_models.py --results-dir results --generate-plots --create-report
```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **Import Errors**
   ```bash
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   
   # Or use the validation script
   python validate_complete_system.py --fix-issues
   ```

2. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Enable gradient accumulation
   - Use mixed precision training

3. **Missing Dependencies**
   ```bash
   # Install/update requirements
   pip install -r requirements.txt
   
   # Or use conda
   conda install pytorch torchvision torchaudio -c pytorch
   conda install pyg -c pyg
   ```

4. **Configuration Errors**
   - Use the enhanced template: `configs/finetune/template_enhanced.yaml`
   - Validate with: `python src/config_validator.py`

5. **Data Processing Issues**
   - Re-run data setup: `python src/data_setup.py`
   - Check processed data: `ls data/processed/`

### Debug Mode

Enable verbose logging:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -u finetune_simplified.py --config your_config.yaml 2>&1 | tee debug.log
```

## 🔬 Advanced Usage

### Custom Configurations

Create custom configurations based on templates:

```bash
# Copy and modify template
cp configs/finetune/template_enhanced.yaml configs/finetune/my_experiment.yaml

# Edit the configuration
nano configs/finetune/my_experiment.yaml

# Run experiment
python finetune_simplified.py --config configs/finetune/my_experiment.yaml
```

### Batch Processing

Process multiple configurations:

```bash
# Create a batch script
for config in configs/finetune/graph_classification/*.yaml; do
    echo "Running $config"
    python finetune_simplified.py --config "$config"
done
```

### Integration with Compute Clusters

For SLURM clusters:

```bash
#!/bin/bash
#SBATCH --job-name=gnn-eval
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

conda activate gnn-simple
python run_comprehensive_evaluation.py --subset schemes
```

## 📊 Performance Optimization

### Memory Optimization
- Use `dynamic_batch_sizing: true` in configs
- Enable `mixed_precision: true`
- Set appropriate `max_memory_usage`

### Speed Optimization
- Use `--parallel` for batch experiments
- Enable `pin_memory: true` in data configs
- Use appropriate `num_workers`

### Storage Optimization
- Set `keep_n_checkpoints` to limit saved models
- Use compression for large result files
- Clean up temporary files regularly

## 🎯 Best Practices

1. **Always validate your system first**
   ```bash
   python validate_complete_system.py
   ```

2. **Start with quick tests**
   ```bash
   python run_comprehensive_evaluation.py --subset quick --dry-run
   ```

3. **Use version control for configurations**
   ```bash
   git add configs/
   git commit -m "Add experiment configurations"
   ```

4. **Monitor resource usage**
   - Use `nvidia-smi` for GPU monitoring
   - Set appropriate memory limits
   - Use profiling tools when needed

5. **Document your experiments**
   - Use descriptive run names
   - Add tags and notes to WandB
   - Keep experiment logs

## 📚 Additional Resources

- **Research Plan**: `plan.md` - Complete research methodology
- **Architecture Guide**: `TRAINING_PIPELINE_README.md` - Pre-training details
- **Configuration Reference**: `configs/finetune/template_enhanced.yaml`
- **API Documentation**: Individual module docstrings
- **Troubleshooting**: `validate_complete_system.py` for diagnostics

## 🤝 Support

For issues or questions:

1. **Check validation**: `python validate_complete_system.py`
2. **Review logs**: Check output logs for error details
3. **Test components**: Use `test_complete_evaluation.py`
4. **Verify setup**: Ensure all dependencies are installed

---

**Status**: ✅ **PRODUCTION READY**

This evaluation pipeline is comprehensive, robust, and ready for conducting the complete research study as specified in `plan.md`.
# Multi-Task Graph Neural Network Pre-training: A Systematic Study

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/314023516_323855288_212052708.pdf)
[![Code](https://img.shields.io/badge/Code-Python-blue)](src/)
[![Data](https://img.shields.io/badge/Analysis-Results-green)](analysis/results/)

> **Evidence-based optimization of GNN pre-training through systematic multi-task evaluation across domains and transfer strategies.**

This repository contains the complete implementation and experimental results for our systematic study of multi-task Graph Neural Network pre-training. We evaluate 96 controlled experiments across 6 domains and 8 pre-training schemes, revealing critical insights about computational efficiency and performance patterns.

## 🔬 Key Findings

- **Parameter Efficiency Paradox**: 40.3x parameter reduction yields only 1.06x-1.37x speedup due to forward pass bottleneck
- **Heterogeneous Efficiency**: Best speedups reach 1.54x (full fine-tuning) and 1.70x (linear probing) with domain-scheme specificity  
- **Selective Performance**: +35.8% peak gains require precise domain-scheme matching
- **Predictable Failures**: Four systematic failure modes enable avoidance strategies

## 📁 Repository Structure

```
├── paper/                          # Research paper and figures
│   ├── 314023516_323855288_212052708.tex    # LaTeX source
│   ├── 314023516_323855288_212052708.pdf    # Final paper
│   ├── gnn_pretrain.bib                     # Bibliography
│   └── figs/                                # Paper figures
├── src/                            # Source code
│   ├── data/                       # Data loading and preprocessing
│   ├── models/                     # GNN architectures and heads
│   ├── pretrain/                   # Pre-training tasks and utilities
│   └── finetune/                   # Fine-tuning implementations
├── analysis/                       # Data analysis and visualization
│   ├── data_analysis.py           # Main analysis script
│   └── results/                   # Generated tables and figures
├── outputs/                        # Trained models
│   ├── pretrain/                  # Pre-trained checkpoints
│   └── finetune/                  # Fine-tuned models
├── vm_execution_scripts/          # Distributed training scripts
├── run_pretrain.py                # Main pre-training entry point
├── run_finetune.py                # Main fine-tuning entry point
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/alonbebchuk/gnn-multitask-pretraining.git
cd gnn-multitask-pretraining

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

The framework automatically downloads and preprocesses datasets:
- **Molecular**: MUTAG, PROTEINS, NCI1, ENZYMES (TUDataset)
- **Citation Networks**: Cora, CiteSeer, PubMed (Planetoid)

### Basic Usage

#### 1. Run Pre-training

```bash
# Single experiment
python run_pretrain.py --experiment s1 --seed 42

# Multiple schemes with parallel execution
python run_pretrain.py --experiments s1 s2 s3 --seeds 42 84 126 --parallel
```

#### 2. Run Fine-tuning

```bash
# Single fine-tuning experiment
python run_finetune.py --domain Cora_NC --strategy linear_probe --scheme s1 --seed 42

# Full evaluation across all combinations
python run_finetune.py --all --parallel
```

#### 3. Generate Analysis

```bash
cd analysis
python data_analysis.py
```

This generates:
- `results/table1_full_finetuning.csv` - Full fine-tuning efficiency metrics
- `results/table2_linear_probing.csv` - Linear probing efficiency metrics  
- `results/table3_scheme_analysis.json` - Performance improvement analysis
- `results/domain_performance_heatmap.png` - Cross-domain transfer visualization
- `results/task_type_performance_heatmap.png` - Task-type performance patterns

## 🔧 Pre-training Schemes

| Scheme | Tasks | Domain Coverage | Description |
|--------|--------|-----------------|-------------|
| `b1` | None | - | From-scratch baseline (no pre-training) |
| `b2` | NFM + LP | All | Basic 2-task combination |
| `b3` | NFM + LP + NC | All | 3-task with node contrastive |
| `b4` | NFM + LP + NC + GC + GP | ENZYMES only | 5-task single-domain |
| `s1` | NFM + LP + NC + GC | All | 4-task multi-domain |
| `s2` | NFM + LP + NC + GP | All | 4-task with graph properties |
| `s3` | NFM + LP + NC + GC | All | Same as s1 (ablation) |
| `s4` | NFM + LP + NC + GC + GP | All | 5-task comprehensive |
| `s5` | NFM + LP + NC + GC + GP + DA | All | 6-task with domain adversarial |

**Task Abbreviations:**
- **NFM**: Node Feature Masking
- **LP**: Link Prediction  
- **NC**: Node Contrastive Learning
- **GC**: Graph Contrastive Learning
- **GP**: Graph Property Prediction
- **DA**: Domain Adversarial Training

## 📊 Evaluation Domains

| Domain | Task Type | Nodes | Edges | Classes | Metric |
|--------|-----------|--------|--------|---------|--------|
| Cora_NC | Node Classification | 2,708 | 5,429 | 7 | Accuracy |
| CiteSeer_NC | Node Classification | 3,327 | 4,732 | 6 | Accuracy |
| Cora_LP | Link Prediction | 2,708 | 5,429 | 2 | AUC |
| CiteSeer_LP | Link Prediction | 3,327 | 4,732 | 2 | AUC |
| ENZYMES | Graph Classification | 32.5* | 62.1* | 6 | Accuracy |
| PTC_MR | Graph Classification | 14.3* | 14.7* | 2 | Accuracy |

*Average per graph

## 🔬 Experimental Framework

### Multi-Task Learning Components

- **PCGrad**: Gradient surgery for conflicting objectives
- **Adaptive Loss Balancing**: Dynamic loss weighting
- **Temperature Scheduling**: Contrastive learning optimization
- **Gradient Reversal Layer**: Domain-invariant feature learning

### Transfer Strategies

1. **Full Fine-tuning**: Update all parameters (backbone + heads)
2. **Linear Probing**: Freeze backbone, train only task-specific heads

### Efficiency Metrics

- **Time Per Epoch Speedup**: Training time comparison
- **Convergence Speedup**: Epochs to convergence ratio
- **Overall Speedup**: Combined time × convergence benefit
- **Parameter Efficiency**: Trainable parameter ratio

## 📈 Reproducing Results

### Complete Experimental Pipeline

```bash
# 1. Run all pre-training experiments (estimated: 24-48 hours on V100)
python run_pretrain.py --all --parallel

# 2. Run all fine-tuning experiments (estimated: 12-24 hours)
python run_finetune.py --all --parallel

# 3. Generate analysis and paper figures
cd analysis
python data_analysis.py

# 4. Compile paper (requires LaTeX)
cd ../paper
pdflatex 314023516_323855288_212052708.tex
bibtex 314023516_323855288_212052708
pdflatex 314023516_323855288_212052708.tex
pdflatex 314023516_323855288_212052708.tex
```

### Hardware Requirements

- **Minimum**: 16GB RAM, CUDA-capable GPU (8GB+ VRAM)
- **Recommended**: 32GB+ RAM, V100/A100 GPU (16GB+ VRAM)
- **Distributed**: Multi-GPU setup supported via `vm_execution_scripts/`

### Expected Outputs

The complete pipeline generates:
- 25 pre-trained models (3 seeds × 8 schemes + baseline)
- 288 fine-tuning results (6 domains × 8 schemes × 2 strategies × 3 seeds)
- Comprehensive efficiency and performance analysis
- Publication-ready figures and tables

## 🛠 Advanced Usage

### Custom Pre-training Schemes

Define new schemes in `src/pretrain/pretrain.py`:

```python
def create_custom_scheme(model):
    tasks = [
        NodeFeatureMaskingTask(model, mask_rate=0.3),
        LinkPredictionTask(model),
        # Add your custom tasks
    ]
    return tasks
```

### Distributed Training

Use the provided VM scripts for large-scale experiments:

```bash
# Setup distributed environment
bash vm_execution_scripts/vm_setup.md

# Run distributed pre-training
bash vm_execution_scripts/single_vm_pretrain.sh
```

### Custom Datasets

Extend `src/data/data_setup.py` to include new datasets:

```python
def load_custom_dataset():
    # Your dataset loading logic
    return dataset
```

## 📄 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{multitask_gnn_2024,
  title={Multi-Task Graph Neural Network Pre-training: A Systematic Study},
  author={[Author Names]},
  journal={[Conference/Journal]},
  year={2024},
  url={https://github.com/alonbebchuk/gnn-multitask-pretraining}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙋‍♂️ Support

- **Issues**: [GitHub Issues](https://github.com/alonbebchuk/gnn-multitask-pretraining/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alonbebchuk/gnn-multitask-pretraining/discussions)
- **Email**: [contact information]

## 🏆 Acknowledgments

- **Datasets**: TUDataset and Planetoid dataset providers
- **Frameworks**: PyTorch Geometric team
- **Compute**: [Acknowledge compute resources if applicable]

---

<div align="center">

**[🔗 Paper](paper/314023516_323855288_212052708.pdf) • [📊 Results](analysis/results/) • [💻 Code](src/) • [🎯 Issues](https://github.com/alonbebchuk/gnn-multitask-pretraining/issues)**

</div>

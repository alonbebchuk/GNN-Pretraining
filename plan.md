# **Multi-Task, Cross-Domain Pre-training for Graph Neural Networks**

**Paper Title:** A Systematic Analysis of Multi-Task, Cross-Domain Pre-training for Graph Neural Networks

**Abstract:** We present a systematic analysis of multi-task, cross-domain pre-training for Graph Neural Networks (GNNs). We evaluate combinations of node, link, and graph-level objectives on a diverse set of downstream tasks, comparing against from-scratch, single-task, and single-domain baselines to establish best practices with rigorous statistical analysis.

---

## **1. Introduction**

Pre-training remains underdeveloped for Graph Neural Networks (GNNs). Most GNNs are trained from scratch for specific tasks, and existing pre-training research is often narrow in scope. This work provides the first systematic study of multi-task, cross-domain pre-training for GNNs.

**Key Contributions:**
1. **Systematic Evaluation:** Evaluate the impact of node, link, and graph-level pre-training task combinations.
2. **Cross-Domain Analysis:** Analyze the cross-domain transferability of learned representations.
3. **Statistical Rigor:** Apply proper statistical testing with multiple comparison correction and effect size reporting.
4. **Actionable Guidelines:** Provide evidence-based best practices for GNN pre-training.

---

## **2. Methodology**

### **2.1. Core Architecture**

#### **2.1.1. Evidence-Based Design Decisions**
Based on extensive experimental validation:

- **✅ Adaptive Loss Balancing:** Multi-task interference resolved with inverse-magnitude weighting and domain adversarial safety constraints
- **✅ Gradient Surgery (PCGrad):** Multi-task gradient conflicts resolved through conflict detection and orthogonal projection, applied uniformly to all multi-task schemes for fair experimental comparison
- **✅ Constant Learning Rate:** Cosine annealing added complexity without benefit; constant LR=1e-5 achieves stable convergence  
- **✅ Conservative Hyperparameters:** BATCH_SIZE=32, WEIGHT_DECAY=1e-5, PATIENCE=50% for optimal stability
- **✅ Gradient Clipping:** max_norm=0.5 prevents gradient explosions while preserving learning dynamics
- **✅ Feature Normalization:** [-3.0, 3.0] clipping stabilizes continuous datasets (PROTEINS, ENZYMES)
- **✅ Enhanced Input Encoding:** BatchNorm1d + Dropout improves cross-domain robustness
- **✅ GRL Scheduler:** Kept for domain adversarial training (s5) with lambda scheduling

#### **2.1.2. Building Blocks**
- **GNN Backbone:** 5-layer GIN architecture with residual connections and batch normalization
- **Input Encoder:** `Linear(D_in, 256) -> BatchNorm1d -> ReLU -> Dropout(0.2)`
- **Task Heads:** Standardized MLP: `Linear(dim_in, dim_hidden) -> ReLU -> Dropout -> Linear(dim_hidden, dim_out)`

#### **2.1.3. Dataset Configuration**
**Sources:** Graph classification (`MUTAG`, `PROTEINS`, `NCI1`, `ENZYMES`, `PTC_MR`); Node/link tasks (`Cora`, `CiteSeer`)

**Input Dimensions:** `MUTAG`: 7, `PROTEINS`: 4, `NCI1`: 37, `ENZYMES`: 3, `PTC_MR`: 18, `Cora`: 1433, `CiteSeer`: 3703

#### **2.1.4. Latest Architecture Updates (2024)**

**Enhanced Multi-Task Optimization:**
- **✅ Task-Specific Learning Rates:** Link prediction (1e-6), others (1e-5), domain adversarial (5e-6) for stability
- **✅ Gradient Surgery Separation:** Domain adversarial excluded from PCGrad (preserves adversarial dynamics)
- **✅ Improved GRL Integration:** Proper gradient reversal with progressive scheduling (λ: 0→0.01)
- **✅ Temperature Scheduling:** Contrastive learning with exact progress tracking (0.5→0.05)
- **✅ Order-Invariant Link Prediction:** Symmetric features (sum, product, difference) for undirected graphs
- **✅ Flexible MLP Architecture:** Configurable dropout rates with proper validation
- **✅ Separate Loss Optimization:** Main tasks cooperatively optimized, domain adversarial separately

### **2.2. Pre-training Setup**

**Dataset Pool:** Combined training splits of `MUTAG`, `PROTEINS`, `NCI1`, and `ENZYMES`

**Pre-training Tasks:**

1. **Node Feature Masking (NFM)** - Mask 15% of node embeddings, reconstruct with MSE loss
2. **Link Prediction (LP)** - Binary edge classification with 1:1 positive:negative sampling  
3. **Node Contrastive (NC)** - SimCLR-style contrastive learning with graph augmentations
4. **Graph Contrastive (GC)** - InfoGraph-style mutual information maximization
5. **Graph Property Prediction (GPP)** - Regression on 12 structural properties (nodes, edges, density, etc.)
6. **Domain Adversarial (DA)** - Domain classifier with gradient reversal layer (GRL)

**Adaptive Loss Balancing Function:**
$$\mathcal{L}_{\text{total}} = \sum_{i \in \text{Tasks}} w_i \mathcal{L}_i - \lambda(p) \mathcal{L}_{\text{domain\_adv}}$$

**where** $w_i = \frac{1/|\mathcal{L}_i|}{\sum_j 1/|\mathcal{L}_j|}$ **(inverse magnitude weighting)** and $\lambda(p) = \frac{2}{1 + e^{-\gamma p}} - 1$ **(GRL scheduling)**

**Rationale:** Multi-task training suffered from severe task interference with simple averaging. Adaptive weighting using inverse loss magnitudes provides stable multi-task optimization while domain adversarial loss includes safety constraints to prevent negative total losses.

**Gradient Surgery for Multi-Task Optimization:**

For schemes with multiple tasks (b4, s1, s2, s3, s4, s5), **PCGrad (Projecting Conflicting Gradients)** is applied to resolve gradient conflicts:

$$\mathbf{g}_i^{(t+1)} = \mathbf{g}_i^{(t)} - \sum_{j \neq i} \max(0, \mathbf{g}_i^{(t)} \cdot \mathbf{g}_j^{(t)}) \frac{\mathbf{g}_j^{(t)}}{||\mathbf{g}_j^{(t)}||^2}$$

**where** $\mathbf{g}_i^{(t)}$ represents the gradient for task $i$ at step $t$, and conflicting gradients (negative dot product) are projected onto the normal plane of interfering task gradients.

**Application Logic:**
- **Multi-task schemes:** Apply PCGrad gradient surgery for conflict resolution
- **Single-task schemes:** Standard backward propagation (no conflicts to resolve)

**Rationale:** While adaptive loss balancing handles loss scale differences, gradient surgery addresses directional conflicts between task gradients, ensuring fair experimental comparison across all multi-task schemes while leveraging beneficial optimization techniques.

### **2.3. Fine-tuning Setup**

**Fine-tuning Strategies:**
1. **Linear Probing (B2):** Freeze backbone, train head only
2. **Full Fine-tuning (B3):** Train all parameters with differentiated learning rates

**Baselines:**
- **B1 (From-scratch):** Train target model without pre-training
- **B2 (Linear Probing):** Freeze pre-trained backbone
- **B3 (Full Fine-tuning):** Fine-tune all parameters

---

## **3. Research Questions**

**RQ1:** Does multi-task, cross-domain pre-training improve downstream performance compared to from-scratch training?

**RQ2:** What task combinations are most effective for pre-training?

**RQ3:** How do different fine-tuning strategies compare in terms of performance and efficiency?

**RQ4:** Which pre-training tasks show the strongest affinity for specific downstream domains?

---

## **4. Metrics and Analysis Framework**

### **4.1. Pre-training Metrics**

**Training Metrics (logged per step):**
- **Domain-Task Losses:** `train/loss/{domain}/{task}` (20 metrics: 4 domains × 5 tasks)
- **Task-Aggregated Losses:** `train/loss/{task_name}` (6 metrics: 5 regular + domain_adv)
- **Domain-Aggregated Losses:** `train/loss/{domain_name}` (4 metrics)
- **Total Loss:** `train/loss/total` (optimization target)
- **Loss Balancer Weights:** `train/loss_balancer/weight/{task_name}` (adaptive weighting coefficients)
- **Gradient Surgery:** `gradient_surgery/{total_conflicts,total_projections,conflict_ratio}` (multi-task only)
- **Domain Adversarial:** `train/domain_adv/lambda` (when active)
- **Training Dynamics:** `train/gradients/model_grad_norm`
- **Progress Tracking:** `train/progress/epoch`

**Validation Metrics (logged per epoch):**
- **Domain-Task Losses:** `val/loss/{domain}/{task}` (20 metrics)
- **Task-Aggregated Losses:** `val/loss/{task_name}` (6 metrics)
- **Domain-Aggregated Losses:** `val/loss/{domain_name}` (4 metrics)
- **Total Loss:** `val/loss/total` (model selection metric)
- **Domain Adversarial:** `val/domain_adv/lambda` (when active)

### **4.2. Fine-tuning Metrics**

**Training Metrics (logged per step):**
- **Performance:** `train/{accuracy,f1,precision,recall,auc,loss}`
- **Learning Rates:** `train/lr/{parameter_group}` (backbone, encoder, head)
- **Training Dynamics:** `train/gradients/model_grad_norm`, `train/system/time_per_step`
- **Progress Tracking:** `train/progress/epoch`, `train/progress/step`

**Validation Metrics (logged per epoch):**
- **Performance:** `val/{accuracy,f1,precision,recall,auc,loss}`

**Test Metrics (logged at end):**
- **Performance:** `test/{accuracy,f1,precision,recall,auc,loss}`
- **Efficiency:** `test/convergence_epochs`, `test/training_time`
- **Model Size:** `test/total_parameters`, `test/trainable_parameters`
- **Progress Tracking:** `test/progress/epoch`

### **4.3. Analysis Framework**

#### **4.3.1. RQ1: Pre-training Effectiveness**
**Metrics Used:**
- Primary: `test/{accuracy,f1,auc}` across all schemes vs B1 baseline
- Efficiency: `test/{convergence_epochs,training_time}`

**Analysis Methods:**
- Paired t-tests with Bonferroni correction for multiple comparisons
- Effect size computation (Cohen's d) for practical significance
- Improvement rate calculation: `(pretrained - baseline) / baseline`

#### **4.3.2. RQ2: Task Combination Analysis** 
**Metrics Used:**
- Learning dynamics: `train/loss/{task_name}_raw` over time
- Task interactions: `train/loss/{domain}/{task}_raw` correlation matrix  
- Gradient conflicts: `gradient_surgery/{conflict_ratio,total_conflicts}` (multi-task schemes)
- Final performance: `test/*` by pre-training scheme

**Analysis Methods:**
- Task synergy detection via correlation analysis
- Gradient conflict analysis: conflict frequency and resolution effectiveness
- Ablation studies comparing single-task vs multi-task schemes  
- Multi-task optimization stability: gradient surgery impact on training dynamics
- Domain-task affinity heat maps

#### **4.3.3. RQ3: Fine-tuning Strategy Comparison**
**Metrics Used:**
- Performance: `test/*` by fine-tuning strategy
- Efficiency: `test/{convergence_epochs,training_time,*_parameters}`
- Training dynamics: `train/lr/{parameter_group}`

**Analysis Methods:**
- Strategy effectiveness comparison (B2 vs B3)
- Parameter efficiency analysis (performance per parameter)
- Computational cost-benefit analysis

#### **4.3.4. RQ4: Task Affinity Patterns**
**Metrics Used:**
- Pre-training losses: `train/loss/{domain}/{task}_raw`
- Transfer performance: `test/*` by downstream task
- Domain adversarial impact: `train/domain_adv/lambda` correlation with transfer

**Analysis Methods:**
- Cross-domain transfer matrix construction
- Specialization vs generalization pattern detection
- Domain adaptation effectiveness quantification

### **4.4. Key Visualizations**

#### **Main Paper Figures:**
1. **Pre-training Value (RQ1):** Box plots with significance markers showing improvement over baseline
2. **Task Synergy Heat Map (RQ2):** Task correlation matrix with clustering
3. **Learning Curves:** Multi-panel plots showing convergence patterns across schemes
4. **Fine-tuning Strategy Comparison (RQ3):** Performance vs efficiency scatter plots

#### **Supplementary Figures:**
5. **Domain-Task Affinity Matrix:** Heat map of domain-task performance patterns
6. **Training Dynamics:** Gradient norms and timing analysis
7. **Computational Efficiency:** Performance vs cost trade-off analysis

### **4.5. Statistical Rigor**

**Requirements Met:**
- Multiple random seeds (N=3) for statistical robustness
- Proper statistical testing with family-wise error control
- Effect size reporting for practical significance
- Literature-grounded baseline comparisons

**Experimental Metadata:**
- Extracted from WandB run names: `{domain}_{strategy}_{scheme}_{seed}`
- No redundant logging of metadata in metrics

---

## **5. Current Implementation Status**

### **5.1. Architecture Files (Fully Implemented)**

#### **Core Model Components:**
- **`src/models/pretrain_model.py`**: Main `PretrainableGNN` model with domain-specific input encoders
- **`src/models/heads.py`**: Task-specific heads with GRL, flexible MLPHead, order-invariant link predictor
- **`src/models/gnn.py`**: GIN backbone with standardized InputEncoder

#### **Training Infrastructure:**
- **`src/pretrain/pretrain.py`**: Main training orchestration with separate domain adversarial optimization
- **`src/pretrain/schedulers.py`**: Temperature and GRL schedulers with exact progress tracking
- **`src/pretrain/optimizers.py`**: Task-specific learning rates with automatic parameter grouping
- **`src/pretrain/gradient_surgery.py`**: PCGrad implementation for multi-task optimization
- **`src/pretrain/tasks.py`**: All pretraining tasks with hard negative mining and enhanced contrastive learning
- **`src/pretrain/adaptive_loss_balancer.py`**: Inverse magnitude loss weighting

#### **Data Processing:**
- **`src/pretrain/augmentations.py`**: Proven graph augmentation strategies (node/edge drop, attribute masking)

### **5.2. Key Implementation Features**

#### **🔧 Multi-Task Learning Architecture:**
```python
# Separate optimization paths:
main_task_losses = {k: v for k, v in per_task_losses.items() if k != 'domain_adv'}
domain_adv_loss = per_task_losses.get('domain_adv', torch.tensor(0.0))

# PCGrad applied to cooperative tasks only
gradient_surgery.apply_gradient_surgery(model, main_task_losses, task_names)
main_total_loss.backward(retain_graph=True)

# Domain adversarial gets separate backward pass (preserves adversarial dynamics)
domain_adv_loss.backward()
```

#### **🎯 Task-Specific Optimization:**
```python
TASK_SPECIFIC_LR = {
    'link_pred': 1e-6,      # Reduced for stability (S1_42 instability fix)
    'node_feat_mask': 1e-5,
    'node_contrast': 1e-5,
    'graph_contrast': 1e-5,
    'graph_prop': 1e-5,
    'domain_adv': 5e-6,     # Slightly reduced for softer adversarial training
}
```

#### **⚡ Progressive Scheduling:**
```python
# Temperature scheduling for contrastive learning
TemperatureScheduler: 0.5 → 0.05 (exact step tracking)

# GRL scheduling for domain adversarial
GRLScheduler: λ = 0 (first 40% epochs) → 0.01 (progressive ramp-up)
```

#### **🔄 Order-Invariant Link Prediction:**
```python
# Symmetric edge features for undirected graphs
h_sum = h_src + h_dst      # Commutative
h_product = h_src * h_dst  # Commutative  
h_diff = torch.abs(h_src - h_dst)  # Symmetric
edge_features = torch.cat([h_sum, h_product, h_diff], dim=1)
```

### **5.3. Implementation Quality Assurance**

#### **✅ Correctness Verification:**
- **Syntax Validation:** All Python files compile successfully
- **Import Dependencies:** All cross-module imports verified
- **Edge Case Handling:** Division by zero protection in schedulers
- **Type Safety:** Proper error handling in flexible constructors

#### **✅ Architectural Consistency:**
- **Domain Encoding:** Standard `InputEncoder` used across all domains (maximum specificity)
- **Task Heads:** Unified `MLPHead` with configurable dropout rates
- **Loss Computation:** Proper gradient flow for adversarial and cooperative objectives
- **Scheduler Interface:** Consistent `__call__()` and `step()` methods

#### **✅ Scientific Soundness:**
- **Gradient Surgery:** PCGrad applied only to compatible cooperative tasks
- **Domain Adversarial:** Proper GRL with separate optimization (maintains adversarial dynamics)
- **Contrastive Learning:** Hard negative mining with proven augmentation strategies
- **Learning Rates:** Task-specific rates based on empirical stability analysis

---

## **6. Experimental Design**

### **6.1. Pre-training Experimental Schemes**

**Systematic Multi-Task Design:**

| Scheme | Tasks | Domain | Research Purpose |
|--------|-------|--------|------------------|
| **b2** | node_feat_mask | Cross | Generative baseline |
| **b3** | node_contrast | Cross | Contrastive baseline |
| **s1** | node_feat_mask, link_pred | Cross | Generative multi-task |
| **s2** | node_contrast, graph_contrast | Cross | Contrastive multi-task |
| **s3** | node_feat_mask, link_pred, node_contrast, graph_contrast | Cross | Combined 4-task |
| **s4** | All 5 tasks | Cross | Full cross-domain |
| **s5** | All 5 tasks + domain_adv | Cross | Full + domain adversarial |
| **b4** | All 5 tasks | Single (ENZYMES) | Single-domain comprehensive |

**Optimization Strategy:**
- **Multi-task schemes** (b4, s1, s2, s3, s4, s5): PCGrad gradient surgery + adaptive loss balancing
- **Single-task schemes** (b2, b3): Standard optimization (no multi-task conflicts)

**Research Questions Addressed:**
- **RQ1 (Effectiveness):** All schemes vs from-scratch baselines
- **RQ2 (Task Combinations):** Systematic progression b2/b3 → s1/s2 → s3 → s4 → s5
- **RQ3 (Fine-tuning):** Cross-cutting across all pretrained models  
- **RQ4 (Task Affinity):** Critical comparison s4 (cross-domain) vs b4 (single-domain) with identical tasks

### **6.2. Experimental Scope**
**Total Runs:** ~150 experiments
- **Pre-training:** 24 runs (8 schemes × 3 seeds)
- **Fine-tuning:** ~125 runs (various configurations × 3 seeds)

**Computational Budget:**
- **Pre-training:** ~17 GPU hours  
- **Fine-tuning:** ~24 GPU hours
- **Total:** ~41 GPU hours

### **6.3. Quality Assurance**
- Reproducible random seed management (seeds: 42, 84, 126)
- Model selection excludes ENZYMES (prevents overfitting)
- Comprehensive logging for all research questions
- Automated statistical analysis pipeline

### **6.4. Execution Method**
**Simplified Command-Line Interface:**
- Direct CLI argument passing to main scripts
- Parallel GPU execution for sweep operations

**Example Execution:**
```bash
# Single experiments
python src/pretrain/pretrain.py --exp_name b2 --seed 42
python src/finetune/finetune.py --domain_name ENZYMES --finetune_strategy full_finetune --pretrained_scheme b2 --seed 42

# Parameter sweeps with GPU parallelization
python run_pretrain.py --sweep
python run_finetune.py --domain_sweep ENZYMES
```

### **6.5. Expected Deliverables**
1. **Performance Tables** with statistical significance testing
2. **Training Dynamics Plots** with confidence intervals  
3. **Ablation Study Results** with effect sizes
4. **Task Affinity Analysis** with correlation matrices
5. **Computational Efficiency Analysis** with cost-benefit ratios

---

## **7. Metric Coverage Verification**

### **7.1. Analysis Requirements vs Logged Metrics**

✅ **RQ1 Analysis Coverage:**
- Pre-training improvement: `test/{accuracy,f1,auc}` ✓
- Convergence advantage: `test/{convergence_epochs,training_time}` ✓
- Statistical testing: Multiple seeds + test metrics ✓

✅ **RQ2 Analysis Coverage:**
- Task synergy: `train/loss/{domain}/{task}_raw` correlation ✓
- Learning dynamics: `train/loss/{task_name}_raw` over time ✓
- Optimal combinations: Performance by scheme ✓

✅ **RQ3 Analysis Coverage:**
- Strategy effectiveness: `test/*` by strategy ✓
- Parameter efficiency: `test/{total,trainable}_parameters` ✓
- Training dynamics: `train/lr/*` ✓

✅ **RQ4 Analysis Coverage:**
- Task affinity: `train/loss/{domain}/{task}_raw` matrix ✓
- Transfer patterns: Cross-domain performance comparison ✓
- Domain adaptation: `train/domain_adv/lambda` effectiveness ✓

**All research questions fully supported by logged metrics with no redundancy.**

---

## **8. Expected Contributions**

1. **Multi-task pre-training effectiveness** - Statistical validation of improvement over baselines
2. **Task synergy identification** - Evidence-based task combination recommendations  
3. **Fine-tuning strategy optimization** - Performance vs efficiency trade-off analysis
4. **Domain-task affinity patterns** - Empirically validated specialization insights

**Open Science Impact:** Complete codebase, pre-trained models, experimental logs, and analysis pipelines will be released for community use.

---

*This plan provides a comprehensive framework for systematic GNN pre-training analysis with rigorous statistical validation and complete metric coverage.*

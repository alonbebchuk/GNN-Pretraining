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

#### **2.1.1. Building Blocks**
- **GNN Backbone:** 3-layer GIN architecture with residual connections and batch normalization
- **Input Encoder:** Domain-specific encoder: `Linear(D_in, 256) -> ReLU`
- **Task Heads:** Standardized MLP: `Linear(dim_in, dim_hidden) -> ReLU -> Dropout -> Linear(dim_hidden, dim_out)`

#### **2.1.2. Dataset Configuration**
**Sources:** Graph classification (`MUTAG`, `PROTEINS`, `NCI1`, `ENZYMES`, `PTC_MR`); Node/link tasks (`Cora`, `CiteSeer`)

**Input Dimensions:** `MUTAG`: 7, `PROTEINS`: 4, `NCI1`: 37, `ENZYMES`: 3, `PTC_MR`: 18, `Cora`: 1433, `CiteSeer`: 3703

### **2.2. Pre-training Setup**

**Dataset Pool:** Combined training splits of `MUTAG`, `PROTEINS`, `NCI1`, and `ENZYMES`

**Pre-training Tasks:**

1. **Node Feature Masking (NFM)** - Mask 15% of node embeddings, reconstruct with MSE loss
2. **Link Prediction (LP)** - Binary edge classification with 1:1 positive:negative sampling  
3. **Node Contrastive (NC)** - SimCLR-style contrastive learning with graph augmentations
4. **Graph Contrastive (GC)** - InfoGraph-style mutual information maximization
5. **Graph Property Prediction (GPP)** - Regression on 12 structural properties (nodes, edges, density, etc.)
6. **Domain Adversarial (DA)** - Domain classifier with gradient reversal layer (GRL)

**Combined Loss Function (Kendall et al., 2018):**
$$\mathcal{L}_{\text{total}} = \sum_{i \in \text{Tasks}} \left( \frac{1}{2\sigma_i^2}\mathcal{L}_i + \frac{1}{2}\log\sigma_i^2 \right) - \lambda \mathcal{L}_{\text{domain}}$$

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
- **Granular Raw Losses:** `train/loss/{domain}/{task}_raw` (20 metrics: 4 domains × 5 tasks)
- **Task-Aggregated Losses:** `train/loss/{task_name}_raw` (6 metrics: 5 regular + domain_adv)
- **Domain-Weighted Losses:** `train/loss/{domain_name}_weighted` (4 metrics)
- **Total Weighted Loss:** `train/loss/total_weighted` (optimization target)
- **Uncertainty Parameters:** `train/uncertainty/{task_name}_sigma` (6 metrics)
- **Learning Rates:** `train/lr/model`, `train/lr/uncertainty`
- **Domain Adversarial:** `train/domain_adv/lambda` (when active)
- **Training Dynamics:** `train/gradients/model_grad_norm`, `train/system/time_per_step`
- **Progress Tracking:** `train/epoch`, `train/step`

**Validation Metrics (logged per epoch):**
- **Granular Raw Losses:** `val/loss/{domain}/{task}_raw` (20 metrics)
- **Task-Aggregated Losses:** `val/loss/{task_name}_raw` (6 metrics)
- **Domain-Weighted Losses:** `val/loss/{domain_name}_weighted` (4 metrics)
- **Total Weighted Loss:** `val/loss/total_weighted` (model selection metric)
- **Uncertainty Parameters:** `val/uncertainty/{task_name}_sigma` (6 metrics)
- **Domain Adversarial:** `val/domain_adv/lambda` (when active)
- **Progress Tracking:** `val/epoch`

### **4.2. Fine-tuning Metrics**

**Training Metrics (logged per step):**
- **Performance:** `finetune/train/{accuracy,f1,precision,recall,auc,loss}`
- **Learning Rates:** `finetune/train/lr/{parameter_group}` (backbone, encoder, head)
- **Progress Tracking:** `finetune/train/epoch`, `finetune/train/step`
- **System Timing:** `finetune/system/time_per_step`

**Validation Metrics (logged per epoch):**
- **Performance:** `finetune/val/{accuracy,f1,precision,recall,auc,loss}`
- **Progress Tracking:** `finetune/val/epoch`

**Test Metrics (logged at end):**
- **Performance:** `finetune/test/{accuracy,f1,precision,recall,auc,loss}`
- **Efficiency:** `finetune/test/convergence_epochs`, `finetune/test/training_time`
- **Model Size:** `finetune/test/total_parameters`, `finetune/test/trainable_parameters`
- **Progress Tracking:** `finetune/test/epoch`

### **4.3. Analysis Framework**

#### **4.3.1. RQ1: Pre-training Effectiveness**
**Metrics Used:**
- Primary: `finetune/test/{accuracy,f1,auc}` across all schemes vs B1 baseline
- Efficiency: `finetune/test/{convergence_epochs,training_time}`

**Analysis Methods:**
- Paired t-tests with Bonferroni correction for multiple comparisons
- Effect size computation (Cohen's d) for practical significance
- Improvement rate calculation: `(pretrained - baseline) / baseline`

#### **4.3.2. RQ2: Task Combination Analysis** 
**Metrics Used:**
- Learning dynamics: `train/loss/{task_name}_raw` over time
- Task interactions: `train/loss/{domain}/{task}_raw` correlation matrix  
- Final performance: `finetune/test/*` by pre-training scheme

**Analysis Methods:**
- Task synergy detection via correlation analysis
- Ablation studies comparing single-task vs multi-task schemes
- Domain-task affinity heat maps

#### **4.3.3. RQ3: Fine-tuning Strategy Comparison**
**Metrics Used:**
- Performance: `finetune/test/*` by fine-tuning strategy
- Efficiency: `finetune/test/{convergence_epochs,training_time,*_parameters}`
- Training dynamics: `finetune/train/lr/{parameter_group}`

**Analysis Methods:**
- Strategy effectiveness comparison (B2 vs B3)
- Parameter efficiency analysis (performance per parameter)
- Computational cost-benefit analysis

#### **4.3.4. RQ4: Task Affinity Patterns**
**Metrics Used:**
- Pre-training losses: `train/loss/{domain}/{task}_raw`
- Transfer performance: `finetune/test/*` by downstream task
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
6. **Uncertainty Evolution:** Task uncertainty weights over training steps
7. **Training Dynamics:** Gradient norms and timing analysis
8. **Computational Efficiency:** Performance vs cost trade-off analysis

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

## **5. Implementation Details**

### **5.1. Experimental Scope**
**Total Runs:** ~150 experiments
- **Pre-training:** 24 runs (8 schemes × 3 seeds)
- **Fine-tuning:** ~125 runs (various configurations × 3 seeds)

**Computational Budget:**
- **Pre-training:** ~17 GPU hours  
- **Fine-tuning:** ~24 GPU hours
- **Total:** ~41 GPU hours

### **5.2. Quality Assurance**
- Reproducible random seed management (seeds: 42, 84, 126)
- Model selection excludes ENZYMES (prevents overfitting)
- Comprehensive logging for all research questions
- Automated statistical analysis pipeline

### **5.3. Expected Deliverables**
1. **Performance Tables** with statistical significance testing
2. **Training Dynamics Plots** with confidence intervals  
3. **Ablation Study Results** with effect sizes
4. **Task Affinity Analysis** with correlation matrices
5. **Computational Efficiency Analysis** with cost-benefit ratios

---

## **6. Metric Coverage Verification**

### **6.1. Analysis Requirements vs Logged Metrics**

✅ **RQ1 Analysis Coverage:**
- Pre-training improvement: `finetune/test/{accuracy,f1,auc}` ✓
- Convergence advantage: `finetune/test/{convergence_epochs,training_time}` ✓
- Statistical testing: Multiple seeds + test metrics ✓

✅ **RQ2 Analysis Coverage:**
- Task synergy: `train/loss/{domain}/{task}_raw` correlation ✓
- Learning dynamics: `train/loss/{task_name}_raw` over time ✓
- Optimal combinations: Performance by scheme ✓

✅ **RQ3 Analysis Coverage:**
- Strategy effectiveness: `finetune/test/*` by strategy ✓
- Parameter efficiency: `finetune/test/{total,trainable}_parameters` ✓
- Training dynamics: `finetune/train/lr/*` ✓

✅ **RQ4 Analysis Coverage:**
- Task affinity: `train/loss/{domain}/{task}_raw` matrix ✓
- Transfer patterns: Cross-domain performance comparison ✓
- Domain adaptation: `train/domain_adv/lambda` effectiveness ✓

### **6.2. Comprehensive Metric Summary**

**Pre-training Metrics Total:** ~44 per training step, ~34 per validation epoch
**Fine-tuning Metrics Total:** ~12 per training step, ~7 per validation epoch, ~9 at test

**All research questions fully supported by logged metrics with no redundancy.**

---

## **7. Expected Contributions**

1. **Multi-task pre-training effectiveness** - Statistical validation of improvement over baselines
2. **Task synergy identification** - Evidence-based task combination recommendations  
3. **Fine-tuning strategy optimization** - Performance vs efficiency trade-off analysis
4. **Domain-task affinity patterns** - Empirically validated specialization insights

**Open Science Impact:** Complete codebase, pre-trained models, experimental logs, and analysis pipelines will be released for community use.

---

*This plan provides a comprehensive framework for systematic GNN pre-training analysis with rigorous statistical validation and complete metric coverage.*

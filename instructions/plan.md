### **Paper Title:** A Large-Scale Analysis of Multi-Task, Cross-Domain Pre-training for Graph Neural Networks

**Abstract:** We present a large-scale, systematic analysis of multi-task, cross-domain pre-training for Graph Neural Networks (GNNs). We evaluate combinations of node, link, and graph-level objectives on a diverse set of downstream tasks, comparing against from-scratch, single-task, and single-domain baselines to establish best practices.

---

### **1. Introduction**

Pre-training remains underdeveloped for Graph Neural Networks (GNNs). Most GNNs are trained from scratch for specific tasks, and existing pre-training research is often narrow in scope. This work provides the first large-scale, systematic study of multi-task, cross-domain pre-training for GNNs.

**Key Contributions:**
1.  **Systematic Evaluation:** Evaluate the impact of node, link, and graph-level pre-training task combinations.
2.  **Cross-Domain Analysis:** Analyze the cross-domain transferability of learned representations.
3.  **Actionable Guidelines:** Provide a set of best practices for GNN pre-training.
4.  **Open Science:** Release all code, models, and logs to the community.

### **2. Methodology**

*   **2.1. Core Architecture and Components**
    *   **2.1.1. Reusable Building Blocks**
        *   **Standardized MLP Head:** A standard 2-layer MLP for prediction tasks, defined as `Linear(dim_in, dim_hidden) -> ReLU -> Dropout -> Linear(dim_hidden, dim_out)`.
            *   **Note:** This is a general-purpose block for producing final predictions.
        *   **Dot Product Decoder:** A non-parametric decoder computes edge probability: `p(edge) = sigmoid(h_u^T * h_v)`.
        *   **Bilinear Discriminator:** A bilinear discriminator scores pairs: `D(x, y) = sigmoid(x^T * W * y)`.
    *   **2.1.2. GNN Model Definition**
        *   **Input Encoder:** A domain-specific linear encoder maps raw node features (`D_in`) to a shared 256-d hidden space (`h_0`): `Linear(D_in, 256) -> LayerNorm -> ReLU -> Dropout`.
        *   **GNN Backbone:** A stack of 5 GIN layers. Each GIN block updates a node's embedding, $$h_v^{(l)}$$, through a sequence of operations designed for high expressiveness. The update from layer $$l$$ to $$l+1$$ is computed as follows:
            1.  **Neighbor Aggregation:** First, aggregate features from the node itself and its neighbors. The central node's importance is weighted by a learnable parameter, `ε`.
                $$ a_v^{(l)} = (1+\epsilon) h_v^{(l)} + \sum_{u \in \mathcal{N}(v)} h_u^{(l)} $$
            2.  **GIN Convolution:** The aggregated features are passed through a simple 2-layer MLP (`Linear(256, 256) -> ReLU -> Linear(256, 256)`) to transform the representation.
                $$ h'_{\text{conv}} = \text{MLP}_{\text{GIN}} \left( a_v^{(l)} \right) $$
            3.  **Residual Connection:** To improve gradient flow, the input node embedding is added back to the transformed output.
                $$ h'_{\text{res}} = h'_{\text{conv}} + h_v^{(l)} $$
            4.  **Post-Activation:** The result is passed through a final sequence of Layer Normalization, a ReLU activation, and Dropout for regularization.
                $$ h_v^{(l+1)} = \text{Dropout} \left( \text{ReLU} \left( \text{LayerNorm} \left( h'_{\text{res}} \right) \right) \right) $$
    *   **2.1.3. Regularization**
        *   A fixed dropout rate of `p=0.2` is used in all `Dropout` layers throughout the architecture.
    *   **2.1.4. Initialization and Defaults**
        *   All layers use the default PyTorch weight initializations and hyperparameters.

*   **2.2. Datasets and Experimental Setups**
    *   **2.2.1. Datasets**
        *   **Sources:** Graph classification datasets (`MUTAG`, `PROTEINS`, `NCI1`, `ENZYMES`, `FRANKENSTEIN`, `PTC_MR`) from TUDatasets; node/link datasets (`Cora`, `CiteSeer`) from Planetoid.
        *   **Input Features and Preprocessing:** Continuous features are z-score standardized, categorical features are one-hot encoded, and bag-of-words features are row-normalized.
        *   **Input Dimensions (`D_in`):** `MUTAG`: 7, `NCI1`: 37, `PROTEINS`: 4, `ENZYMES`: 21, `FRANKENSTEIN`: 780, `PTC_MR`: 18, `Cora`: 1433, `CiteSeer`: 3703.
        *   **Data Splits:**
            *   **Pre-training:** To ensure reproducibility, each source dataset is first individually split into 80% training and 20% validation sets (no test set) using a fixed random seed. These splits are then saved and reused for all experiments. The training splits form a combined pool, and to counteract domain imbalance, batches are formed using a weighted sampling strategy: a domain is first chosen uniformly, and then a graph is sampled (with replacement) from that domain’s specific training set. The validation sets from each domain are kept separate for a balanced evaluation.
            *   **Downstream Tasks:** For the TUDatasets, a fixed 80% training, 10% validation, and 10% test stratified split is generated once with a fixed random seed and reused across all runs. `Cora` and `CiteSeer` use their standard public splits, which are inherently fixed.
    *   **2.2.2. Pre-training Setup**
        *   **Dataset Pool:** The combined training splits of `MUTAG`, `PROTEINS`, `NCI1`, and `ENZYMES`.
        *   **Pre-training Tasks and Losses:**
            *   **i. Self-Supervised (Generative)**
                *   **Node Feature Masking:** For 15% of nodes in a graph, their input features are replaced by a single, shared, learnable `[MASK]` embedding. The model's goal is to reconstruct the original 256-dim embeddings (`h_0`) of these masked nodes.
                    *   **Implementation Note:** Because input feature dimensions (`D_in`) vary by domain, a separate learnable `[MASK]` vector must be created for each pre-training domain.
                    *   **Head:** `MLPHead(dim_in=256, dim_hidden=256, dim_out=256)`. Loss: MSE.
                *   **Link Prediction:** Classify edges. For each existing (positive) edge, one non-existent (negative) edge is uniformly sampled from the set of all non-edges in the graph, creating a 1:1 positive-to-negative ratio.
                    *   **Head:** `Dot Product Decoder`. Loss: BCE.
            *   **ii. Self-Supervised (Contrastive)**
                *   **Node-level (GraphCL-style):** This task contrasts augmented views of nodes. For each graph, two augmented versions (`G'` and `G''`) are created. Each augmentation is a sequential composition of the following transformations, each applied with a `p=0.5` probability: attribute masking (15% of feature dimensions), edge dropping (15% of edges), and subgraph sampling (via random walks of length 10). A positive pair consists of the same node viewed through these two different augmentations (`h_v'` and `h_v''`). For a given node's embedding `h_v'`, negative pairs are formed with the embeddings of all other nodes in the batch.
                    *   **Projection Head:** `MLPHead(dim_in=256, dim_hidden=256, dim_out=128)`. Loss: NT-Xent (Normalized Temperature-scaled Cross-Entropy) using cosine similarity and a fixed temperature `τ=0.1`.
                *   **Graph-Level (InfoGraph-style):** This task maximizes the mutual information between a graph's summary vector `s` and its node embeddings `h_v`. The summary vector `s` is computed by taking the mean of the final-layer node embeddings for all nodes in the graph.
                    *   **Positive Pairs:** For a graph `G`, pairs are `(s, h_v)` for all nodes `v` in `G`.
                    *   **Negative Pairs & Ratio:** For the same graph summary `s`, negative pairs are `(s, h_u)` where `u` is a node from any *other* graph in the same batch. This means all nodes from other graphs in the batch serve as negatives, making explicit negative sampling unnecessary. The positive-to-negative ratio for a given graph is therefore `N` to `(M - N)`, where `N` is the number of nodes in the graph and `M` is the total number of nodes in the batch.
                    *   **Head:** `Bilinear Discriminator` scoring `(node_embedding, graph_embedding)` pairs. Loss: BCE.
            *   **iii. Supervised Auxiliary Task**
                *   **Graph Property Prediction:** Regress on 15 comprehensive structural properties including basic structure (nodes, edges, density), degree statistics (average, variance, maximum), clustering measures (global clustering, transitivity, triangles), connectivity (components, diameter, assortativity), and centralization metrics. To prevent scale dominance, all target properties are z-score standardized across the training set before computing the loss.
                    *   **Head:** `MLPHead(dim_in=256, dim_hidden=512, dim_out=15)`. Loss: MSE.
            *   **iv. Domain-Adversarial Objective (to promote transferability)**
                *   To explicitly encourage the GNN to learn domain-invariant features, we use an adversarial setup with a **Gradient Reversal Layer (GRL)**. A dedicated classifier head is trained to predict a graph's source domain, while the GRL ensures that the GNN backbone learns to produce embeddings that confuse this classifier, thus promoting domain-invariance.
                *   **GRL Mechanism:** During the forward pass, the GRL acts as an identity function, allowing normal domain classification. During backpropagation, the GRL reverses and scales the gradient by `-lambda`, effectively training the GNN backbone to maximize the domain classifier's loss (confusion).
                *   The classifier head does not use dropout (`p=0`) to ensure it is as strong as possible, providing the sharpest training signal for the GNN backbone.
                    *   **Head:** `MLPHead(dim_in=256, dim_hidden=128, dim_out=N_domains)`. Loss: Cross-Entropy.
        *   **Combined Pre-training Loss:**
            *   The total loss function is optimized from the perspective of the GNN backbone. Tasks are balanced using a common heuristic form of uncertainty weighting (which works well for both regression and classification tasks), and the domain-adversarial objective is incorporated via a negatively-weighted term:
                $$ \mathcal{L}_{\text{total}} = \sum_{i \in \text{Tasks}} \left( \frac{1}{2\sigma_i^2}\mathcal{L}_i + \log\sigma_i \right) - \lambda \mathcal{L}_{\text{domain}} $$
            *   The negative sign on the domain loss term (`L_domain`) is crucial. By minimizing `L_total`, the GNN backbone is implicitly trained to *maximize* the domain classifier's loss, forcing it to produce domain-invariant embeddings. The domain classifier itself is trained to *minimize* `L_domain` through the normal forward pass, while the GRL ensures the backbone receives the adversarial signal. Instead of a fixed hyperparameter, `λ` is gradually increased during training using a standard schedule: `λ_p = (2 / (1 + exp(-γ * p))) - 1`, where `p` is the training progress from 0 to 1 and `γ` is a fixed parameter set to 10.
    *   **2.2.3. Downstream Evaluation Setup**
        *   **i. Graph Classification:** Datasets: `ENZYMES` (in-domain); `FRANKENSTEIN`, `PTC_MR` (out-of-domain). A global graph embedding is computed via mean pooling over the final node embeddings. This embedding is then fed into an `MLPHead(dim_in=256, dim_out=N_classes)`. Loss: BCE for binary tasks, Cross-Entropy for multi-class.
        *   **ii. Node Classification:** Datasets: `Cora`, `CiteSeer` (out-of-domain). Final node embeddings are fed into an `MLPHead(dim_in=256, dim_out=N_classes)`. Loss: Cross-Entropy.
        *   **iii. Link Prediction:** Datasets: `Cora`, `CiteSeer` (out-of-domain). For both training and evaluation, each existing (positive) edge is paired with one uniformly sampled non-existent (negative) edge for a 1:1 ratio. Final node embeddings are scored by a `Dot Product Decoder`. Loss: BCE.

### **3. Experimental Design, Evaluation, and Analysis**

This section outlines the comprehensive design for this study, from pre-training strategies to final analysis, ensuring a controlled and thorough investigation.

#### **3.1. Research Questions (RQs)**
*   **RQ1:** When does multi-task pre-training surpass from-scratch training, and how can we mitigate **negative transfer**?
*   **RQ2:** Which combination of pre-training tasks yields the most generalizable representations?
*   **RQ3:** How can we most effectively adapt pre-trained models to downstream tasks (e.g., full fine-tuning vs. linear probing)?
*   **RQ4:** Is the optimal pre-training strategy dependent on the downstream task type?

#### **3.2. Pre-training Schemes & Rationale**

We will implement several pre-training schemes to systematically investigate different paradigms. While RQs 1-3 are addressed by direct comparisons between the schemes below, RQ4 is addressed by analyzing the results from all schemes across the different downstream task types (graph, node, and link prediction) to identify any task-specific affinities.

**Pre-training Data:**
*   A single, aggregated dataset of four TUDatasets: `MUTAG`, `PROTEINS`, `NCI1`, and `ENZYMES`.

**Pre-training Tasks:**
*   **Generative:** Node Feature Masking (NFM), Link Prediction (LP).
*   **Contrastive:** Node-level Contrastive (NC, GraphCL-style), Graph-level Contrastive (GC, InfoGraph-style).
*   **Auxiliary Supervised:** Graph Property Prediction (GPP).
*   **Adversarial:** Domain-Adversarial objective (DA).

**Experimental Schemes:**

| Scheme ID | Name | Tasks Included | Purpose & Research Question Addressed |
| :--- | :--- | :--- | :--- |
| **B1** | From-Scratch | None | **(Baseline)** Establishes baseline performance. Answers **RQ1**. |
| **B2** | Single-Task (Generative) | NFM | **(Baseline)** A representative generative single-task model. For **RQ1 \& RQ2**. |
| **B3** | Single-Task (Contrastive) | NC | **(Baseline)** A representative contrastive single-task model. For **RQ1 \& RQ2**. |
| **B4** | Single-Domain (All Objectives) | NFM + LP + NC + GC + GPP | **(Baseline)** Isolate the benefit of multi-domain data by pre-training on the `ENZYMES` dataset only. For **RQ2**. |
| **S1** | Multi-Task (Generative) | NFM + LP | Tests the synergy of purely generative tasks. Addresses **RQ2**. |
| **S2** | Multi-Task (Contrastive) | NC + GC | Tests the synergy of purely contrastive tasks. Addresses **RQ2**. |
| **S3** | Multi-Task (All Self-Supervised) | NFM + LP + NC + GC | Tests the synergy of all self-supervised paradigms. For **RQ2**. |
| **S4** | Multi-Task (All Objectives) | NFM + LP + NC + GC + GPP | Combines all objectives (self-supervised + auxiliary) to establish a performance ceiling. For **RQ2**. |
| **S5** | Multi-Task (Domain-Invariant) | NFM + LP + NC + GC + GPP + DA | Extends the all-objective model with domain-adversarial training to improve transferability and mitigate negative transfer. For **RQ1 & RQ2**. |

#### **3.3. Pre-training Protocol**

All pre-training schemes will follow a consistent protocol to ensure fair comparison.

*   **Model & Optimizer:** 5-layer GIN backbone (256-d hidden), optimized with AdamW (`lr=5e-4`, `β1=0.9`, `β2=0.999`, `wd=0.01`).
*   **Training Details:** Trained for 100k steps with a batch size of 32. Batches are formed using the domain-balanced sampling strategy described in Sec. 2.2.1. A cosine annealing schedule with a 10% linear warm-up period is used. Multi-task losses are balanced via uncertainty weighting.
*   **Validation & Regularization:** The best model is selected based on a domain-balanced validation metric. We calculate the validation loss on each domain's validation set separately and then average these per-domain losses. Early stopping (patience=10) is applied to this final averaged metric.

#### **3.4. Downstream Evaluation & Finetuning Protocol**

We will evaluate pre-trained models using two distinct fine-tuning strategies. The specific components that are trained or frozen depend on whether the downstream task is considered "in-domain" or "out-of-domain" for the model being evaluated. A dataset is considered "in-domain" if the pre-trained model was trained on it. Otherwise, it is considered "out-of-domain".

**Fine-tuning Strategies:**

*   **1. Full Fine-tuning:** The pre-trained GNN backbone is unfrozen and fine-tuned with a small learning rate (`lr=1e-4`). A new prediction head is always trained from scratch (`lr=1e-3`). The handling of the input encoder differs by case:
    *   **In-Domain:** The corresponding pre-trained `Input Encoder` is also unfrozen and fine-tuned with `lr=1e-4`.
    *   **Out-of-Domain:** A new `Input Encoder` is created and trained from scratch (`lr=1e-3`), as the pre-trained encoder is incompatible with the new dataset's feature dimensions.

*   **2. Linear Probing:** The entire pre-trained GNN backbone is frozen. A new prediction head is always trained from scratch (`lr=1e-3`).
    *   **In-Domain:** The pre-trained `Input Encoder` is also frozen and used.
    *   **Out-of-Domain:** A new `Input Encoder` is created and trained from scratch (`lr=1e-3`), as the pre-trained encoder is incompatible with the new dataset's feature dimensions.

**Common Training Parameters:**
*   **Optimizer:** AdamW with `β1=0.9`, `β2=0.999` and a weight decay of `0.05`.
*   **LR Schedule:** Cosine annealing with a linear warm-up over the first 10% of steps for all trainable parameters.
*   **Batch Size:** 32 for graph tasks; full-batch for node tasks (`Cora`, `CiteSeer`).
*   **Training Duration:** Up to 200 epochs, with early stopping (patience=10) on the validation loss.

#### **3.5. Core Analytical Comparisons**

To answer our research questions, we will perform the following specific comparisons, with each comparison being conducted across all relevant downstream tasks and fine-tuning strategies.

*   **1. The Value of Pre-training (addressing RQ1):**
    *   **Comparison:** The primary baseline, **B1 (From-Scratch)**, will be compared against every pre-trained scheme (**B2-B4** and **S1-S5**).
    *   **Purpose:** To determine the fundamental conditions under which any form of pre-training provides a benefit.

*   **2. Single-Task vs. Multi-Task Pre-training (addressing RQ1 & RQ2):**
    *   **Comparison:** The performance of the single-task models (**B2, B3**) will be compared as a group against the multi-task models (**S1-S5**).
    *   **Purpose:** To establish whether combining tasks provides synergistic benefits over pre-training on individual tasks.

*   **3. Generative vs. Contrastive Paradigms (addressing RQ2):**
    *   **Comparison:** The purely generative scheme **S1 (NFM + LP)** will be directly compared against the purely contrastive scheme **S2 (NC + GC)**. Their performance will also be compared against the combined scheme **S3**.
    *   **Purpose:** To understand the relative strengths of the two main self-supervised learning philosophies and the benefit of their combination.

*   **4. Impact of Auxiliary & Adversarial Objectives (addressing RQ2 & Negative Transfer in RQ1):**
    *   **Comparison:** This is a sequential analysis of the most complex schemes:
        *   **S3 (Multi-Task (All Self-Supervised))** vs. **S4 (Multi-Task (All Objectives))** to measure the impact of adding a supervised auxiliary task.
        *   **S4 (Multi-Task (All Objectives))** vs. **S5 (Multi-Task (Domain-Invariant))** to measure the impact of adding the domain-adversarial objective.
    *   **Purpose:** To perform an ablation study on the most powerful models to see how each component contributes to performance and generalization.

*   **5. The Benefit of Cross-Domain Data (addressing RQ2):**
    *   **Comparison:** The multi-domain **S4 (Multi-Task (All Objectives))** model will be compared against the **B4 (Single-Domain)** baseline (which uses the same task combination).
    *   **Purpose:** To directly and rigorously measure the performance gain attributable to using a diverse, multi-domain pre-training dataset.

*   **6. Task-Type Affinity (addressing RQ4):**
    *   **Comparison:** This is an analysis *within* each scheme's results. For each pre-trained model, we will analyze its relative performance across the three task types: Graph Classification, Node Classification, and Link Prediction.
    *   **Purpose:** To identify if certain pre-training strategies (e.g., node-focused like NFM vs. graph-focused like GPP) create an affinity for specific downstream task types.

*   **7. Fine-tuning Strategy Effectiveness (addressing RQ3):**
    *   **Comparison:** For the best overall pre-trained model (e.g., **S5 (Multi-Task (Domain-Invariant))**), we will directly compare the results from **Full Fine-tuning** vs. **Linear Probing**.
    *   **Purpose:** To determine the most effective adaptation method between the two core strategies.

#### **3.6. Evaluation Metrics and Analysis Plan**

The analysis will focus on core performance and efficiency to provide clear, actionable insights. All results will be reported as the mean and standard deviation over 3 runs using a pre-defined set of random seeds (`[42, 84, 126]`) to ensure statistical robustness.
*   **1. Core Performance Metrics:**
    *   The primary evaluation will use standard metrics for each task type: Accuracy, F1-Score, and AUC-ROC.
    *   Statistical significance of performance differences between key models will be verified with paired tests (e.g., paired t-test or Wilcoxon signed-rank test, using a significance level of `p < 0.05`).
*   **2. Efficiency and Convergence Analysis:**
    *   **Convergence Speed:** We will measure the number of training epochs required for each fine-tuning run to reach the best score on its validation set. This will help quantify how pre-training accelerates downstream adaptation.
    *   **Computational Cost:** The wall-clock training time for each fine-tuning strategy will be reported to provide a practical comparison of their computational expense.
#### **3.7. Experimental Scope and Budget**

The study involves a substantial number of experimental runs to ensure statistical robustness. All results will be averaged over 3 runs with different random seeds.

*   **Pre-training:** We will train 8 unique models (B2-B4, S1-S5) for 100k steps each.
    *   **Total Runs:** `8 models × 3 seeds = 24 runs`.
    *   **Estimated Time:** `24 runs × ~2.8 hours/run ≈ 67 hours`.
*   **Fine-tuning:** We will evaluate the 8 pre-trained models and 1 from-scratch baseline on 7 downstream tasks.
    *   **Total Runs:** `(8 models × 7 tasks × 2 strategies × 3 seeds) + (1 baseline × 7 tasks × 1 strategy × 3 seeds) = 336 + 21 = 357 runs`.
    *   **Estimated Time:** Fine-tuning is significantly faster, estimated at `~14 hours` total.

### **4. Conclusion & Future Work**

*   We will provide clear recommendations on effective multi-task pre-training strategies for GNNs, expecting to show that a multi-task, multi-domain approach is superior for generalization and robustness.
*   **Future Work:** This framework can be extended by exploring dynamic task-weighting schemes, applying insights to even larger and more complex graph transformer architectures, and investigating pre-training for evolving or temporal graphs. The more complex analyses deferred from this study—such as robustness evaluation and interpretability analysis—also provide clear avenues for future research.

### **5. Implementation Guide for Software Provider**

This section provides a detailed blueprint for a software provider to implement the research plan described above. The goal is to ensure the codebase is modular, reproducible, and easy to extend.

#### **5.1. Tech Stack & Environment**

The project should be developed using the following technologies. Please provide a `requirements.txt` file with pinned versions.

*   **Language:** Python (version 3.9+)
*   **Core Libraries:**
    *   `PyTorch` (version 1.12+ for CUDA support)
    *   `torch-geometric` (for GNN layers, datasets, and utilities)
    *   `numpy` (for numerical operations)
    *   `scikit-learn` (for metrics like F1-Score, AUC-ROC, and preprocessing)
    *   `pandas` (for results aggregation)
*   **Configuration:** `PyYAML` (for parsing YAML config files)
*   **Experiment Tracking (Recommended):** `wandb` (for logging metrics and visualizations)
*   **Utilities:** `tqdm` (for progress bars)
*   **Analysis:** `matplotlib` (for plotting), `scipy` (for statistical tests)

#### **5.2. Recommended Project Structure**

A well-organized project structure is critical. Please use the following layout:

```
/
├── configs/                  # YAML configuration files for each experiment
│   ├── pretrain/
│   │   ├── b2_nfm.yaml
│   │   ├── b3_nc.yaml
│   │   ├── ...
│   │   └── s5_all_da.yaml
│   └── finetune/
│       ├── graph_classification/
│       │   ├── enzymes.yaml
│       │   └── ...
│       ├── node_classification/
│       │   └── ...
│       └── ...
├── data/                     # Raw and processed data
│   ├── raw/
│   └── processed/            # Saved preprocessed graphs and data splits
├── notebooks/                # Jupyter notebooks for analysis and visualization
├── results/                  # Output directory for logs, metrics, and model checkpoints
│   ├── pretrain/
│   └── finetune/
├── src/                      # Source code
│   ├── data_setup.py         # Data loading, preprocessing, and splitting logic
│   ├── datasets.py           # Custom PyG Dataset classes
│   ├── layers.py             # Custom model layers (e.g., Gradient Reversal Layer)
│   ├── models/
│   │   ├── gnn.py            # Core GNN backbone and input encoder
│   │   └── heads.py          # Prediction heads (MLP, decoders)
│   ├── tasks.py              # Logic for each pre-training task (loss calculation, etc.)
│   ├── trainer.py            # Training and evaluation loops (pre-training and fine-tuning)
│   ├── utils.py              # Utility functions (e.g., config loaders, loggers)
│   ├── main_pretrain.py      # Main script to run pre-training
│   ├── main_finetune.py      # Main script to run fine-tuning and evaluation
│   └── results.py            # Script for aggregating results and running analysis
└── README.md                 # Project overview and instructions
└── requirements.txt          # Pinned Python dependencies
```

#### **5.3. Configuration Management**

All experimental parameters should be managed via YAML files to ensure reproducibility.

*   Create a separate YAML config file for each experimental scheme (B2-S5) and each downstream task.
*   The main scripts (`main_pretrain.py`, `main_finetune.py`) should take a path to a config file as a command-line argument.

**Example `configs/pretrain/s5_all_da.yaml`:**
```yaml
# Scheme S5: Multi-Task (Domain-Invariant)
training:
  seed: 42
  batch_size: 32
  num_steps: 100000
  patience: 10 # for early stopping
  
optimizer:
  lr: 5.0e-4
  beta1: 0.9
  beta2: 0.999
  weight_decay: 0.01

lr_scheduler:
  warmup_fraction: 0.1

model:
  type: 'GIN'
  num_layers: 5
  hidden_dim: 256
  dropout: 0.2

tasks:
  - name: 'NodeFeatureMasking'
    weight: 1.0 # Initial weight for uncertainty weighting
    params:
      mask_rate: 0.15
  - name: 'LinkPrediction'
    weight: 1.0
    params:
      neg_sample_ratio: 1.0
  - name: 'NodeContrastive'
    weight: 1.0
    params:
      temperature: 0.1
      augmentations:
        - type: 'AttributeMask'
          p: 0.5
          mask_rate: 0.15
        - type: 'EdgeDrop'
          p: 0.5
          drop_rate: 0.15
        - type: 'SubgraphSample'
          p: 0.5
          walk_length: 10
  - name: 'GraphContrastive'
    weight: 1.0
  - name: 'GraphPropertyPrediction'
    weight: 1.0
  - name: 'DomainAdversarial'
    weight: 1.0 # This is lambda, will be scheduled
    params:
      gamma: 10.0 # for lambda schedule

pretrain_datasets: ['MUTAG', 'PROTEINS', 'NCI1', 'ENZYMES']
```

#### **5.4. Data Preparation (`src/data_setup.py`)**

*   **Download:** Use `torch_geometric.datasets` to download all required datasets (`TUDatasets`, `Planetoid`).
*   **Preprocessing:** Implement the specified preprocessing logic (z-score for continuous, one-hot for categorical).
*   **Splits:**
    *   Implement the data splitting logic as described in **Sec 2.2.1**.
    *   Use a fixed global random seed of **0** for all splits to ensure reproducibility.
    *   Save the processed `Data` objects and the train/val/test indices to disk (e.g., using `torch.save`) in the `data/processed/` directory. This script should be run only once before any training.

#### **5.5. Core Implementation Details**

*   **Gradient Reversal Layer (`src/layers.py`):**
    *   Implement this as a custom `torch.autograd.Function`.
    *   The `forward` pass should be an identity function.
    *   The `backward` pass should multiply the incoming gradient by `-lambda`. The `lambda` value should be passed to the function and updated by the trainer according to the schedule in **Sec 2.2.2**.

*   **Domain Adversarial Training with GRL (`src/trainer.py`):**
    *   The domain-adversarial training uses a **Gradient Reversal Layer (GRL)** approach, which is a well-established method for domain adversarial training. This approach is simpler and more efficient than the two-step optimization process while achieving the same goal of domain invariance.
    *   **Implementation Details:**
        1.  **Forward Pass:** During the forward pass, the GRL acts as an identity function, allowing normal computation through the domain classifier.
        2.  **Backward Pass:** During backpropagation, the GRL reverses and scales the gradient by `-lambda`, effectively training the GNN backbone to produce embeddings that confuse the domain classifier.
        3.  **Single Optimizer:** All parameters (GNN backbone, task heads, and domain classifier) are updated using a single optimizer, with the GRL automatically handling the adversarial training signal.
    *   **Advantages:** This approach is computationally efficient, numerically stable, and widely used in domain adversarial training literature. It achieves the same domain invariance objective as the two-step approach but with simpler implementation. The GRL approach is the standard method used in domain adversarial training and has been successfully applied in numerous studies.

*   **Uncertainty Weighting (`src/trainer.py`):**
    *   For the combined loss in **Sec 2.2.2**, the `σ_i` terms are learnable parameters.
    *   Define `log_sigma_sq = torch.nn.Parameter(torch.zeros(num_tasks))` for each task.
    *   The loss for a task `i` becomes `0.5 * torch.exp(-log_sigma_sq[i]) * raw_loss_i + 0.5 * log_sigma_sq[i]`.
    *   These parameters should be part of the model's optimizer and will be learned automatically during training.

*   **Logging (`src/utils.py`, `src/trainer.py`):**
    *   Integrate `WandB`.
    *   Log all hyperparameters from the config file at the start of a run.
    *   **During pre-training:** Log total loss, and individual task losses per step. Log the domain-balanced validation metric periodically.
    *   **During fine-tuning:** Log training and validation loss per epoch.
    *   **At the end of each run:** Save model checkpoints and a summary file (e.g., `results.json`) containing final metrics (Accuracy, F1, AUC-ROC), convergence epoch, and total wall-clock time.

#### **5.6. Execution Scripts & Workflow**

1.  **`python src/data_setup.py`**: Run once to prepare all data.
2.  **`python src/main_pretrain.py --config configs/pretrain/s5_all_da.yaml`**: Run for each pre-training scheme. The script should load the config, build the model, and run the training loop, saving checkpoints and logs to a unique directory (e.g., `results/pretrain/s5_all_da/seed_1/`).
3.  **`python src/main_finetune.py --pretrain_ckpt_path <path> --downstream_config <path> --strategy <full|linear>`**: Run for each combination of pre-trained model, downstream task, and finetuning strategy.
    *   The from-scratch baseline (B1) is a special case where `pretrain_ckpt_path` is omitted.
    *   The script saves its results to a unique directory (e.g., `results/finetune/ENZYMES/from_s5/full_finetune/seed_1/`).

#### **5.7. Final Deliverables**

1.  **Full Source Code:** All contents of the `src/` directory.
2.  **Configuration Files:** All YAML files in `configs/`.
3.  **Requirements File:** A `requirements.txt` file.
4.  **README:** A `README.md` file explaining how to set up the environment, prepare the data, and run the pre-training and fine-tuning experiments.
5.  **Results:** A directory containing all saved model checkpoints and log files from every experimental run, organized by scheme and seed. This is crucial for the final analysis.
6.  **Analysis Script (`src/results.py`):** A Python script that loads the saved `results.json` files from all experimental runs, performs the statistical tests described in **Sec 3.5 & 3.6**, and generates the final tables and figures for the paper.
7.  **(Optional but Recommended) Analysis Notebook:** A Jupyter notebook in `notebooks/` that uses the functions from `src/results.py` to present the analysis in a more narrative format.
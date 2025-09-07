# Research Design Justifications and Citation Guide

**For: Multi-Task, Cross-Domain Pre-training for Graph Neural Networks**

---

## Initial Context Prompt

Use this prompt first to establish context, then follow with specific decision prompts:

```
I am writing a research paper titled "A Systematic Analysis of Multi-Task, Cross-Domain Pre-training for Graph Neural Networks" for ACL. This work provides the first systematic study of multi-task, cross-domain pre-training for GNNs, evaluating combinations of node, link, and graph-level objectives on diverse downstream tasks.

Our research framework:
- Architecture: 5-layer GIN backbone with domain-specific input encoders that map to 256 hidden dimensions
- Pre-training tasks: 6 tasks including node feature masking (15%), link prediction, node/graph contrastive learning, graph property prediction, and domain adversarial training
- Multi-task framework: Uses PCGrad gradient surgery and adaptive loss balancing with inverse magnitude weighting
- Experimental design: 8 pre-training schemes (single-task baselines → multi-task progressions), cross-domain vs single-domain comparison, 3 seeds with proper statistical validation
- Datasets: TU datasets (MUTAG, PROTEINS, NCI1, ENZYMES) + Planetoid (Cora, CiteSeer) for comprehensive evaluation
- Research questions: (1) Does multi-task pre-training improve performance? (2) What task combinations are most effective? (3) How do fine-tuning strategies compare? (4) Which tasks show strongest domain affinity?

I will send you specific prompts about individual design decisions. For each query, please provide: (1) Direct citations with full bibliographic details, (2) Specific findings that support our choices, (3) Any contradictory evidence I should address, (4) Theoretical frameworks that explain why our approach works.

Please wait for my specific questions - do not respond to this context-setting message yet.
```

---

## Architecture Decisions

### 1. **GNN Backbone: 5-Layer GIN Architecture**

**Decision**: Used 5-layer Graph Isomorphism Network (GIN) as backbone
**Code Reference**: `src/models/gnn.py:8` - `GNN_NUM_LAYERS = 5`

```
I'm conducting systematic research on multi-task, cross-domain pre-training for Graph Neural Networks (GNNs) for an ACL paper. We chose a 5-layer Graph Isomorphism Network (GIN) as our backbone architecture for pre-training on 6 different tasks across multiple graph domains (molecular graphs, citation networks). Our experimental setup involves 8 different pre-training schemes and evaluation on diverse downstream tasks.

SPECIFIC RESEARCH QUESTIONS I need citations for:
1. What is the optimal GNN depth (number of layers) to balance expressiveness and over-smoothing, particularly for multi-task pre-training scenarios?
2. Why is GIN architecture specifically advantageous compared to GCN, GAT, GraphSAGE, or Graph Transformers for graph-level representation learning?
3. How does network depth affect multi-task learning and cross-domain transfer in GNNs?
4. What empirical evidence exists for 5-layer networks being optimal for diverse graph tasks?
5. How do Graph Transformers compare to traditional GNNs like GIN, and why might GIN be more suitable for multi-domain, multi-task scenarios?

Please provide:
- Comprehensive literature review on GNN depth analysis and over-smoothing phenomena
- Comparative studies between GIN and other GNN architectures (GCN, GAT, GraphSAGE, Graph Transformers) showing GIN's advantages
- Theoretical justification for why GIN's message passing is superior for multi-domain scenarios
- Analysis of Graph Transformers vs traditional GNNs: computational complexity, scalability, and suitability for multi-task learning
- Empirical studies showing optimal layer numbers for different graph tasks
- Evidence for when Graph Transformers are beneficial vs when traditional GNNs like GIN are preferable
- Any contradictory evidence about deeper vs shallower networks I should address
- Specific citations from top-tier venues (NeurIPS, ICML, ICLR) with exact findings and page numbers where possible

Focus particularly on papers that analyze architectural choices for transfer learning and multi-task scenarios in graph domains, including recent Graph Transformer literature.
```

---

### 2. **Hidden Dimension: 256**

**Decision**: Set GNN hidden dimension to 256
**Code Reference**: `src/models/gnn.py:7` - `GNN_HIDDEN_DIM = 256`

```
For our systematic multi-task, cross-domain GNN pre-training research (ACL submission), we set the hidden dimension to 256 across all layers. This choice needs to be justified for a diverse set of graph domains (molecular: MUTAG, PROTEINS, NCI1, ENZYMES; citation: Cora, CiteSeer) and 6 different pre-training tasks (node masking, link prediction, contrastive learning, etc.).

SPECIFIC RESEARCH QUESTIONS needing citations:
1. What factors determine optimal hidden dimension sizes in GNNs for representation learning vs computational efficiency?
2. How does hidden dimensionality affect cross-domain transfer performance in multi-task scenarios?
3. Is 256 dimensions supported by empirical studies as effective for diverse graph tasks?
4. How does hidden dimension choice interact with multi-task learning and gradient interference?
5. What theoretical frameworks exist for choosing representation dimensionality in graph neural networks?

Please provide:
- Empirical studies analyzing hidden dimension effects on GNN performance across different tasks
- Theoretical analysis of representation capacity vs dimensionality in graph embeddings
- Comparative studies showing performance vs computational cost trade-offs for different dimensions
- Papers specifically validating 256 as effective dimension size in graph learning
- Analysis of how dimensionality affects transfer learning and multi-task performance
- Evidence for dimensional choices in successful graph pre-training systems
- Any studies showing negative effects of this dimension choice I should address

Prioritize citations from recent top-tier venues that specifically address multi-task or transfer learning scenarios in graph domains.
```

---

### 3. **Input Encoding Strategy**

**Decision**: Domain-specific input encoders with BatchNorm + ReLU + Dropout
**Code Reference**: `src/models/gnn.py:11-23` - `InputEncoder` class

```
We use domain-specific input encoders for our multi-domain GNN pre-training, where each domain (MUTAG: 7D, PROTEINS: 4D, NCI1: 37D, ENZYMES: 21D, Cora: 1433D, CiteSeer: 3703D) has its own Linear→BatchNorm1d→ReLU→Dropout(0.2) encoder that maps to our standard 256-dimensional hidden space.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. Why are domain-specific input encoders necessary for cross-domain graph neural network training?
2. How does BatchNorm1d specifically benefit GNN training stability and cross-domain transfer?
3. What is the theoretical and empirical justification for standardizing disparate input dimensions to a common embedding space?
4. How does input normalization affect gradient flow and training dynamics in multi-task GNN scenarios?
5. What are the best practices for handling vastly different input dimensionalities (4D to 1433D) in transfer learning?
6. How does the choice of activation function (ReLU) in input encoding affect representation learning?

Please provide:
- Studies on BatchNorm effectiveness in graph neural networks and its impact on training stability
- Theoretical analysis of why domain-specific encoders enable cross-domain knowledge transfer
- Empirical evidence for input normalization improving cross-domain performance in GNNs
- Comparative studies of different input encoding strategies (shared vs domain-specific)
- Analysis of how input dimensionality differences affect transfer learning in graph domains
- Studies on optimal input encoding architectures for multi-domain graph learning
- Evidence for ReLU vs other activations in input encoding layers
- Research on handling heterogeneous feature spaces in graph neural networks
```

---

## Pre-training Task Design

### 4. **Node Feature Masking (15% mask rate)**

**Decision**: Mask 15% of node embeddings for reconstruction
**Code Reference**: `src/models/pretrain_model.py:20` - `NODE_FEATURE_MASKING_MASK_RATE = 0.15`

```
Our multi-task GNN pre-training framework includes node feature masking as one of 6 pre-training objectives, where we randomly mask 15% of node embeddings and train the model to reconstruct them using MSE loss. This is inspired by masked language modeling but adapted for graph structured data across multiple domains (molecular and citation graphs).

SPECIFIC RESEARCH QUESTIONS needing comprehensive citations:
1. How does the principle of masked reconstruction from BERT/language modeling translate to graph neural networks?
2. What is the optimal masking rate for graph-based self-supervised learning, and how does 15% compare to other rates?
3. Why is node-level feature masking effective for learning graph representations that transfer across domains?
4. How does masked node prediction interact with other pre-training objectives in multi-task scenarios?
5. What are the theoretical foundations for why reconstructive pre-training improves downstream performance?

Please provide:
- Original BERT and masked language modeling papers showing the effectiveness of reconstruction-based pre-training
- Graph-specific adaptations of masked reconstruction (e.g., GraphMAE, other graph masked autoencoders)
- Empirical studies comparing different masking rates (10%, 15%, 20%, etc.) in graph self-supervised learning
- Theoretical analysis of why reconstructive objectives learn useful representations
- Studies showing how masked reconstruction complements contrastive learning in multi-task settings
- Applications of masked node prediction in molecular and citation graph domains
- Analysis of masking strategies (random vs structured) and their effects
- Evidence for optimal masking rates across different graph types and tasks

Focus particularly on papers that validate 15% as an effective masking rate and studies showing synergy between reconstructive and contrastive pre-training objectives.
```

---

### 5. **Contrastive Learning with Temperature Scheduling**

**Decision**: Temperature scheduling from 0.5 → 0.05 for contrastive tasks
**Code Reference**: `src/pretrain/schedulers.py` - `INITIAL_TEMP = 0.5`, `FINAL_TEMP = 0.05`

```
Our framework includes both node-level and graph-level contrastive learning tasks that use progressive temperature scheduling from 0.5 to 0.05 throughout training. We apply SimCLR-style node contrastive learning and InfoGraph-style graph contrastive learning with hard negative mining, using graph augmentations (node/edge dropping, attribute masking) to create positive pairs.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. What is the theoretical foundation for temperature scheduling in contrastive learning and why does it improve performance?
2. How do SimCLR principles translate from computer vision to graph neural network domains?
3. Why is InfoGraph-style mutual information maximization effective for graph-level representation learning?
4. What are the optimal temperature ranges and scheduling strategies for graph contrastive learning?
5. How does temperature affect the balance between positive and negative similarities in graph contrastive objectives?
6. What is the relationship between temperature scheduling and curriculum learning in self-supervised representation learning?

Please provide:
- Original SimCLR paper and theoretical analysis of temperature's role in contrastive learning
- InfoGraph paper and its theoretical foundations for graph-level contrastive learning
- Empirical studies on optimal temperature values and scheduling strategies
- Analysis of how temperature affects contrastive learning dynamics and representation quality
- Studies comparing different temperature scheduling approaches (linear, cosine, exponential)
- Graph-specific contrastive learning papers validating temperature scheduling effectiveness
- Theoretical analysis of why starting with higher temperature and annealing helps learning
- Comparative studies of node-level vs graph-level contrastive learning in multi-task settings
- Evidence for the specific temperature range (0.5→0.05) we use or recommendations for alternatives
```

---

### 6. **Hard Negative Mining Strategy**

**Decision**: Mine 30% hard negatives with minimum 8 negatives
**Code Reference**: `src/pretrain/tasks.py:18-19` - `HARD_NEGATIVE_RATIO = 0.3`, `MIN_HARD_NEGATIVES = 8`

```
We implement hard negative mining for both contrastive learning tasks and link prediction, where we identify 30% of negatives as "hard" (highest similarity among all negatives) with a minimum of 8 hard negatives per sample. This is applied to node contrastive learning, graph contrastive learning, and link prediction tasks to improve the quality of learned representations.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. What is the theoretical basis for why hard negative mining improves contrastive learning performance?
2. How does the ratio of hard vs random negatives (30% in our case) affect representation quality and training dynamics?
3. What empirical evidence exists for optimal hard negative ratios and minimum thresholds in graph domains?
4. How does hard negative mining specifically benefit link prediction tasks in graph neural networks?
5. What are the computational trade-offs between hard negative mining effectiveness and training efficiency?
6. How does hard negative mining interact with multi-task learning when applied across different objective types?

Please provide:
- Theoretical analysis of hard negative mining and its role in improving contrastive learning
- Empirical studies comparing hard vs random negative sampling in contrastive learning
- Specific applications of hard negative mining in graph neural networks and link prediction
- Analysis of optimal hard negative ratios and their effects on representation quality
- Studies on hard negative mining in multi-task scenarios and its interaction with other objectives
- Evidence for specific thresholds (30% ratio, minimum 8 negatives) or guidelines for setting these parameters
- Computational analysis of hard negative mining overhead vs performance benefits
- Comparative studies showing hard negative mining effectiveness across different graph tasks
- Research on curriculum learning aspects of hard negative mining (starting easy, getting harder)
```

---

## Multi-Task Learning Framework

### 7. **Gradient Surgery (PCGrad)**

**Decision**: Applied PCGrad for multi-task gradient conflict resolution
**Code Reference**: `src/pretrain/gradient_surgery.py` - PCGrad implementation

```
Our systematic multi-task GNN pre-training framework uses PCGrad (Projecting Conflicting Gradients) to resolve gradient conflicts between our 6 pre-training tasks across multiple graph domains. This is a critical component since we train on combinations of node-level (masking, contrastive), edge-level (link prediction), and graph-level (property prediction, graph contrastive) objectives simultaneously, plus domain adversarial training.

SPECIFIC RESEARCH QUESTIONS requiring comprehensive citations:
1. How does PCGrad specifically resolve gradient conflicts in multi-task learning, and what is the theoretical foundation?
2. Why is gradient surgery superior to simple loss balancing or uncertainty weighting approaches for conflicting objectives?
3. What empirical evidence exists for PCGrad's effectiveness in graph neural network contexts or representation learning?
4. How does gradient conflict resolution affect convergence and final performance in multi-task scenarios?
5. Are there alternative gradient surgery methods, and how does PCGrad compare?
6. What are the computational overheads and practical considerations of gradient surgery?

Please provide:
- The original PCGrad paper (Yu et al., NeurIPS 2020) with detailed explanation of the method
- Theoretical analysis of why gradient conflicts hurt multi-task learning performance
- Empirical comparisons between PCGrad and other multi-task optimization approaches
- Applications of PCGrad or similar techniques in graph neural networks or representation learning
- Studies showing when gradient surgery is necessary vs when simple approaches suffice
- Analysis of PCGrad's interaction with different task types (generative, contrastive, adversarial)
- Any limitations or failure cases of PCGrad I should acknowledge
- Recent improvements or variants of gradient surgery techniques

Focus on providing both theoretical justification and empirical validation, particularly for scenarios involving diverse task types like ours (reconstruction, contrastive, adversarial).
```

---

### 8. **Adaptive Loss Balancing**

**Decision**: Inverse magnitude weighting for loss balancing
**Code Reference**: `src/pretrain/adaptive_loss_balancer.py:32-34` - inverse magnitude computation

```
Our multi-task framework uses adaptive loss balancing with inverse magnitude weighting: w_i = (1/|L_i|) / Σ(1/|L_j|), where tasks with smaller loss magnitudes receive higher weights. This addresses the challenge of balancing 6 different pre-training objectives (reconstruction, contrastive, adversarial) that naturally have different scales and convergence rates.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. Why is adaptive loss balancing critical for successful multi-task learning, especially with diverse objective types?
2. How does inverse magnitude weighting compare to other approaches like uncertainty weighting, gradient-based weighting, or equal weighting?
3. What theoretical foundations justify inverse magnitude weighting as an effective loss balancing strategy?
4. How do different loss balancing approaches affect multi-task optimization convergence and final performance?
5. What are the interactions between loss balancing and gradient surgery techniques in multi-task scenarios?
6. How does loss balancing effectiveness vary across different combinations of task types (generative, contrastive, adversarial)?

Please provide:
- Comprehensive survey of multi-task loss balancing strategies and their theoretical foundations
- Comparative studies between uncertainty weighting, gradient-based weighting, and magnitude-based approaches
- Theoretical analysis of why inverse magnitude weighting helps balance task contributions
- Empirical evidence for adaptive loss balancing improving multi-task performance
- Studies on loss balancing in scenarios with diverse objective types (like reconstruction + contrastive + adversarial)
- Analysis of how loss balancing interacts with gradient surgery and other multi-task optimization techniques
- Evidence for when simple equal weighting fails and adaptive approaches become necessary
- Research on loss balancing warm-up strategies and their effect on training stability
- Applications of loss balancing specifically in graph neural networks or representation learning contexts
```

---

### 9. **Domain Adversarial Training**

**Decision**: Domain adversarial training with gradient reversal layer (GRL)
**Code Reference**: `src/models/heads.py:16-32` - `GradientReversalFunction`

```
Our scheme s5 includes domain adversarial training as the 6th pre-training objective, using a gradient reversal layer to learn domain-invariant representations across 4 diverse graph domains (MUTAG, PROTEINS, NCI1, ENZYMES). The domain classifier attempts to predict source domain while the backbone learns to fool it, encouraging domain-invariant features for better cross-domain transfer.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. What is the theoretical foundation of domain adversarial training and how does the gradient reversal layer work mathematically?
2. Why does domain adversarial training improve cross-domain transfer performance compared to standard multi-domain training?
3. How effective is DANN specifically in graph neural network contexts or structured data domains?
4. How does domain adversarial training interact with other multi-task objectives (contrastive, reconstructive) in joint optimization?
5. What are the optimal scheduling strategies for the adversarial loss weight (lambda parameter) during training?
6. When does domain adversarial training help vs hurt in multi-domain scenarios, and what factors determine success?

Please provide:
- The original DANN paper (Ganin et al., JMLR 2016) with detailed theoretical explanation
- Theoretical analysis of why domain adversarial training learns transferable representations
- Empirical studies showing DANN effectiveness across different domains and tasks
- Applications of domain adversarial training in graph neural networks or structured domains
- Analysis of gradient reversal layer implementation and its theoretical properties
- Studies on optimal adversarial loss scheduling and lambda parameter tuning strategies
- Comparisons between domain adversarial training and other domain adaptation techniques
- Evidence for when domain adversarial training is beneficial vs harmful in multi-domain settings
- Analysis of domain adversarial training in multi-task learning contexts and its interaction with other objectives
```

---

## Training Configuration

### 10. **Learning Rate Strategy**

**Decision**: Task-specific learning rates (1e-5 main, 1e-6 link pred, 5e-6 domain adv)
**Code Reference**: `src/pretrain/optimizers.py` - `TASK_SPECIFIC_LR`

```
We use differentiated learning rates across our 6 pre-training tasks: 1e-5 for node masking/contrastive/graph tasks, 1e-6 for link prediction (for stability), and 5e-6 for domain adversarial training. This task-specific approach addresses the different convergence characteristics and sensitivity of diverse objective types in our multi-task framework.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. Why do different task types (reconstruction, contrastive, adversarial) require different learning rates for optimal performance?
2. How does learning rate sensitivity vary across different graph neural network tasks and objectives?
3. What empirical evidence exists for task-specific learning rates improving multi-task learning outcomes?
4. Why might link prediction tasks specifically require lower learning rates compared to other graph objectives?
5. How do learning rate differences affect gradient flow and task interference in multi-task optimization?
6. What are the theoretical foundations for adaptive or task-specific learning rate strategies?

Please provide:
- Studies on learning rate sensitivity across different types of neural network objectives
- Empirical analysis of task-specific learning rates in multi-task learning scenarios
- Research on optimal learning rates for specific graph neural network tasks (link prediction, node classification, etc.)
- Theoretical analysis of how learning rate affects convergence in different objective landscapes
- Evidence for why certain tasks (like link prediction) might require more conservative learning rates
- Studies on adaptive learning rate strategies in multi-task optimization
- Analysis of learning rate effects on gradient interference and task balance
- Comparative studies of uniform vs differentiated learning rates in multi-task settings
```

---

### 11. **Batch Size Selection**

**Decision**: Batch size 32 for pre-training, varying for fine-tuning
**Code Reference**: `src/pretrain/pretrain.py:27` - `BATCH_SIZE = 32`

```
We use batch size 32 for pre-training across all domains and tasks, with task-specific batch sizes for fine-tuning (32 for graph classification, 256 for link prediction, -1 for full-batch node classification). This choice balances memory constraints, gradient noise, and training stability across our diverse multi-task, multi-domain framework.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. How does batch size affect training dynamics and convergence in graph neural networks?
2. What are the optimal batch sizes for different types of graph learning tasks (node, edge, graph-level)?
3. How does batch size interact with multi-task learning and gradient interference patterns?
4. What is the relationship between batch size and gradient noise in graph neural network training?
5. How do memory constraints and computational efficiency factor into batch size selection for graph tasks?
6. What empirical evidence exists for batch size 32 being effective for graph neural network pre-training?

Please provide:
- Empirical studies on batch size effects in graph neural network training
- Theoretical analysis of batch size impact on gradient estimation and convergence in GNNs
- Comparative studies of different batch sizes across various graph learning tasks
- Research on batch size optimization for multi-task learning scenarios
- Analysis of memory vs performance trade-offs in graph neural network batch size selection
- Studies on how batch size affects different types of graph objectives (reconstruction, contrastive, adversarial)
- Evidence for optimal batch sizes in self-supervised graph representation learning
- Research on batch size effects on gradient interference in multi-task optimization
```

---

### 12. **Early Stopping Strategy**

**Decision**: Patience = 50% of total epochs
**Code Reference**: `src/pretrain/pretrain.py:30` - `PATIENCE_FRACTION = 0.5`

```
We implement early stopping with patience equal to 50% of total training epochs (25 out of 50 epochs for pre-training, varying for fine-tuning). This strategy prevents overfitting while allowing sufficient training time for our complex multi-task objectives to converge across diverse graph domains.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. What are the theoretical foundations for early stopping and its effect on generalization in neural networks?
2. How should patience values be set relative to total training epochs for optimal performance?
3. What empirical evidence exists for early stopping effectiveness in graph neural network training?
4. How does early stopping interact with multi-task learning where different tasks may converge at different rates?
5. What are the optimal early stopping criteria for self-supervised representation learning?
6. How does early stopping affect transfer learning and pre-training effectiveness?

Please provide:
- Theoretical analysis of early stopping and its regularization effects in deep learning
- Empirical studies on optimal patience values and early stopping strategies
- Research on early stopping in graph neural networks and its effect on generalization
- Studies on early stopping in multi-task learning scenarios with diverse convergence rates
- Analysis of early stopping criteria for self-supervised and representation learning
- Evidence for relative patience values (e.g., 50% of total epochs) in deep learning
- Research on early stopping effects on transfer learning and pre-training quality
- Comparative studies of different early stopping strategies and validation criteria
```

---

## Experimental Design

### 13. **Dataset Selection**

**Decision**: TU datasets (MUTAG, PROTEINS, NCI1, ENZYMES) + Planetoid (Cora, CiteSeer)
**Code Reference**: `src/data/data_setup.py:24-29` - dataset definitions

```
We selected a diverse collection of graph datasets spanning molecular graphs (MUTAG: 188 graphs, PROTEINS: 1113 graphs, NCI1: 4110 graphs, ENZYMES: 600 graphs) and citation networks (Cora: 2708 nodes, CiteSeer: 3327 nodes) with varying sizes, structural properties, and feature dimensions (4D to 1433D) to comprehensively evaluate cross-domain transfer learning.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. Why are TU datasets (MUTAG, PROTEINS, NCI1, ENZYMES) considered standard benchmarks for graph classification?
2. What makes Cora and CiteSeer appropriate benchmark datasets for node classification and link prediction?
3. How do these datasets provide sufficient diversity for comprehensive cross-domain transfer evaluation?
4. What are the key structural and statistical properties of these datasets that make them suitable for pre-training research?
5. What precedent exists for using these specific datasets in graph neural network transfer learning studies?
6. How do the size and complexity differences across these datasets affect transfer learning evaluation?

Please provide:
- Papers establishing TU datasets as standard benchmarks in graph neural network research
- Studies using Cora and CiteSeer as evaluation benchmarks for graph learning tasks
- Analysis of dataset properties and their suitability for transfer learning evaluation
- Comparative studies that have used these datasets for pre-training or representation learning
- Research on dataset diversity requirements for comprehensive transfer learning evaluation
- Studies analyzing the structural properties and characteristics of these benchmark datasets
- Evidence for these datasets providing sufficient coverage of different graph domains
- Previous work using similar dataset combinations for cross-domain graph learning evaluation
```

---

### 14. **Cross-Domain vs Single-Domain Comparison**

**Decision**: Compare s4 (cross-domain) vs b4 (single-domain ENZYMES)
**Code Reference**: `src/pretrain/pretrain.py:32-41` - scheme definitions

```
A critical aspect of our systematic GNN pre-training analysis is comparing cross-domain (s4: trained on MUTAG, PROTEINS, NCI1, ENZYMES) vs single-domain (b4: trained only on ENZYMES) pre-training using identical 5-task combinations. Both schemes use the same multi-task framework (PCGrad, adaptive loss balancing) but differ in domain diversity. This comparison directly addresses whether domain diversity helps or hurts transfer learning.

SPECIFIC RESEARCH QUESTIONS requiring comprehensive citations:
1. When does cross-domain pre-training outperform single-domain pre-training in transfer learning scenarios?
2. What theoretical frameworks explain why domain diversity can either help or hurt transfer performance?
3. How does domain similarity/dissimilarity affect the success of cross-domain transfer learning?
4. What factors determine whether negative transfer occurs in multi-domain pre-training?
5. Are there studies specifically comparing single-domain vs multi-domain pre-training in graph neural networks?
6. How does the number and diversity of source domains affect transfer learning performance?

Please provide:
- Theoretical analysis of positive vs negative transfer in multi-domain learning scenarios
- Empirical studies comparing single-domain vs multi-domain pre-training across different fields
- Analysis of domain similarity measures and their relationship to transfer success
- Studies on negative transfer and when domain diversity hurts performance
- Research on optimal domain selection strategies for cross-domain pre-training
- Graph-specific studies on cross-domain transfer (if available)
- Theoretical frameworks for predicting when cross-domain pre-training will be beneficial
- Analysis of the relationship between domain diversity and representation quality
- Studies on curriculum learning or domain ordering effects in multi-domain training
- Evidence for domain adaptation techniques that mitigate negative transfer

Focus particularly on providing both theoretical explanations and empirical evidence for when cross-domain pre-training is expected to succeed vs fail, especially in structured data domains.
```

---

### 15. **Statistical Validation Framework**

**Decision**: 3 seeds (42, 84, 126) with proper statistical testing
**Code Reference**: Multiple files - consistent seed usage

```
We conduct all experiments with 3 different random seeds (42, 84, 126) to ensure statistical robustness, applying proper significance testing with Bonferroni correction for multiple comparisons and reporting effect sizes (Cohen's d) alongside p-values. This addresses the reproducibility crisis in machine learning research.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. How many random seeds are sufficient for robust statistical validation in machine learning experiments?
2. What are the appropriate statistical tests for comparing model performance across multiple conditions?
3. Why is effect size reporting critical beyond p-values, and what constitutes meaningful effect sizes?
4. How should multiple comparison correction be applied in systematic ML evaluation studies?
5. What are the best practices for reproducible machine learning research and statistical reporting?
6. How does seed selection and number of seeds affect the reliability of experimental conclusions?

Please provide:
- Studies on statistical best practices for machine learning research and appropriate seed numbers
- Research on proper statistical testing methods for model comparison and significance testing
- Analysis of effect size importance and interpretation in machine learning contexts
- Guidelines for multiple comparison correction in ML evaluation studies
- Studies addressing the reproducibility crisis and proper experimental design in ML
- Evidence for 3 seeds being sufficient vs recommendations for higher numbers
- Research on proper statistical reporting standards for machine learning papers
- Analysis of how random seed selection affects experimental conclusions and reproducibility
```

---

## Evaluation Framework

### 16. **Fine-tuning Strategy Comparison**

**Decision**: Compare from-scratch, linear probing, and full fine-tuning
**Code Reference**: `src/models/finetune_model.py:50-58` - strategy implementation

```
We systematically compare three fine-tuning strategies: (1) from-scratch training (b1 baseline), (2) linear probing (freeze backbone, train only task head), and (3) full fine-tuning (train all parameters with differentiated learning rates). This comparison reveals when and why pre-training helps across different transfer scenarios.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. When is linear probing vs full fine-tuning more effective for transfer learning?
2. What theoretical frameworks explain the effectiveness of different fine-tuning strategies?
3. How do fine-tuning strategies interact with the amount and type of pre-training data?
4. What factors determine whether linear probing or full fine-tuning achieves better transfer performance?
5. How do computational costs compare between different fine-tuning approaches?
6. What are the empirical guidelines for choosing fine-tuning strategies in different domains?

Please provide:
- Comprehensive comparison studies of linear probing vs full fine-tuning across different domains
- Theoretical analysis of when different fine-tuning strategies are optimal
- Studies on fine-tuning strategy effectiveness in graph neural networks specifically
- Research on the relationship between pre-training quality and optimal fine-tuning strategy
- Analysis of computational trade-offs between different fine-tuning approaches
- Evidence for fine-tuning strategy selection based on dataset size and task similarity
- Studies on fine-tuning strategies in self-supervised representation learning contexts
- Empirical guidelines for choosing fine-tuning approaches in transfer learning
```

---

### 17. **Metric Selection**

**Decision**: Task-specific metrics (accuracy, F1, AUC) with statistical testing
**Code Reference**: `src/finetune/metrics.py` - comprehensive metrics

```
We use comprehensive task-specific evaluation metrics: accuracy/F1/precision/recall for classification tasks, AUC for link prediction, with additional efficiency metrics (convergence epochs, training time, parameters). All metrics include statistical testing to ensure robust conclusions about transfer learning effectiveness.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. What are the standard evaluation metrics for different types of graph learning tasks?
2. How should transfer learning effectiveness be properly measured and compared?
3. Why are multiple complementary metrics necessary for comprehensive evaluation?
4. What efficiency metrics are important for evaluating pre-training and transfer learning?
5. How should statistical significance be properly assessed in transfer learning evaluation?
6. What are the best practices for metric selection in systematic machine learning studies?

Please provide:
- Studies establishing standard evaluation metrics for graph classification, node classification, and link prediction
- Research on proper evaluation methodologies for transfer learning and pre-training effectiveness
- Analysis of why multiple metrics provide more comprehensive evaluation than single metrics
- Studies on efficiency metrics and their importance in transfer learning evaluation
- Guidelines for statistical testing and significance assessment in ML evaluation
- Best practices for metric selection in systematic machine learning research
- Research on evaluation bias and proper experimental design in transfer learning studies
- Studies on metric reliability and validity in graph neural network evaluation
```

---

## Implementation Choices

### 18. **Graph Augmentation Strategy**

**Decision**: Node/edge dropping and attribute masking
**Code Reference**: `src/pretrain/augmentations.py` - augmentation methods

```
We use graph augmentation strategies for contrastive learning: node dropping (random removal), edge dropping (random edge removal), and attribute masking (random feature masking). These augmentations create positive pairs for contrastive objectives while preserving essential graph structure and semantics.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. What graph augmentation strategies effectively preserve graph semantics while providing useful training signal?
2. How do different augmentation types (structural vs feature-based) affect contrastive learning performance?
3. What are the theoretical foundations for graph augmentation design in self-supervised learning?
4. How do augmentation strategies interact with different types of graphs (molecular vs citation networks)?
5. What empirical evidence exists for optimal augmentation parameters and combinations?
6. How do graph augmentations affect the quality of learned representations across domains?

Please provide:
- Comprehensive studies on graph augmentation techniques for contrastive learning
- Theoretical analysis of what augmentations preserve vs destroy in graph structure
- Empirical comparisons of different augmentation strategies across graph types
- Research on augmentation parameter selection and optimization
- Studies on augmentation effectiveness across different graph domains
- Analysis of how augmentations affect representation quality and transfer learning
- Principled approaches to graph augmentation design and selection
- Research on combining multiple augmentation strategies effectively
```

---

### 19. **Gradient Clipping**

**Decision**: Max gradient norm = 0.5
**Code Reference**: `src/pretrain/pretrain.py:29` - `MAX_GRAD_NORM = 0.5`

```
We apply gradient clipping with maximum norm 0.5 to prevent gradient explosions in our multi-task training framework. This is particularly important given the combination of diverse objectives (reconstruction, contrastive, adversarial) that can create unstable gradient dynamics.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. Why is gradient clipping necessary in graph neural network training and what problems does it solve?
2. How should optimal gradient clipping values be determined for different architectures and tasks?
3. How does gradient clipping interact with multi-task learning and gradient surgery techniques?
4. What empirical evidence exists for specific clipping values (like 0.5) in graph neural networks?
5. How does gradient clipping affect convergence and final performance in deep learning?
6. What are the theoretical foundations for gradient clipping and its regularization effects?

Please provide:
- Studies on gradient explosion problems in graph neural networks and deep learning
- Research on optimal gradient clipping values and selection strategies
- Analysis of gradient clipping effects in multi-task learning scenarios
- Empirical studies validating specific clipping values in neural network training
- Theoretical analysis of gradient clipping and its effect on optimization dynamics
- Studies on gradient clipping interaction with other optimization techniques
- Research on gradient clipping necessity and effectiveness in different architectures
- Evidence for gradient clipping benefits vs potential drawbacks in representation learning
```

---

### 20. **Feature Normalization**

**Decision**: StandardScaler with [-3, 3] clipping for continuous features
**Code Reference**: `src/data/data_setup.py:17-18` - normalization constants

```
We apply StandardScaler normalization followed by [-3, 3] clipping to continuous features (PROTEINS, ENZYMES datasets). This two-step process handles extreme outliers while maintaining normalized distributions, which is critical for stable cross-domain training with vastly different feature scales.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. How does feature normalization affect cross-domain transfer learning in graph neural networks?
2. What are the optimal normalization strategies for handling heterogeneous feature scales across graph domains?
3. Why is clipping important in addition to standardization, and what are optimal clipping ranges?
4. How do normalization strategies affect gradient flow and training stability in multi-domain scenarios?
5. What empirical evidence exists for specific normalization approaches in graph neural networks?
6. How do outliers and extreme values affect cross-domain transfer if not properly handled?

Please provide:
- Studies on feature normalization strategies for graph neural networks
- Research on cross-domain transfer with heterogeneous feature scales
- Analysis of standardization vs other normalization approaches in graph learning
- Studies on outlier handling and clipping strategies in neural network training
- Evidence for specific clipping ranges (like [-3, 3]) in machine learning
- Research on normalization effects on gradient flow and training stability
- Analysis of how feature scale differences affect multi-domain learning
- Studies on preprocessing strategies for robust cross-domain representation learning
```

---

## Research Questions Framework

### 21. **Multi-Task Task Combination Analysis**

**Decision**: Systematic progression from single-task to multi-task (b2→b3→s1→s2→s3→s4→s5)
**Code Reference**: `src/pretrain/pretrain.py:43-52` - `ACTIVE_TASKS` definition

```
Our experimental design follows a systematic progression: single-task baselines (b2: node masking, b3: node contrastive) → dual-task combinations (s1: generative, s2: contrastive) → 4-task (s3) → 5-task (s4) → 6-task with domain adversarial (s5). This progression allows systematic analysis of how task combinations affect learning and identifies optimal multi-task configurations.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. How do different task combination strategies affect multi-task learning performance and task interference patterns?
2. What causes positive vs negative task interference in multi-task learning, and how can it be predicted?
3. Are there principled approaches for selecting optimal task combinations in multi-task scenarios?
4. How does the number of tasks in multi-task learning affect overall performance and individual task quality?
5. What theoretical frameworks exist for understanding task synergy and interference in representation learning?
6. How do different types of tasks (generative, contrastive, adversarial) interact when combined in multi-task learning?

Please provide:
- Systematic studies on multi-task learning with varying numbers and types of tasks
- Theoretical analysis of task interference and synergy in multi-task optimization
- Research on principled task combination selection strategies and methodologies
- Empirical studies showing how task combinations affect individual and overall performance
- Analysis of positive vs negative transfer in multi-task learning scenarios
- Studies on task compatibility and interference patterns across different objective types
- Research on optimal multi-task learning curricula and task scheduling strategies
- Evidence for systematic experimental design approaches in multi-task learning evaluation
```

---

### 22. **Task Affinity Analysis**

**Decision**: Analyze which pre-training tasks transfer best to which domains
**Code Reference**: Comprehensive logging in `src/pretrain/pretrain.py:158-162`

```
Our framework enables systematic analysis of task-domain affinity patterns by logging per-domain-per-task losses and transfer performance. We analyze which pre-training tasks (node masking, link prediction, contrastive learning, etc.) transfer most effectively to which downstream domains (molecular vs citation graphs) and task types (classification vs link prediction).

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. How should task affinity and transfer effectiveness be measured and quantified in multi-task learning?
2. What factors determine which source tasks transfer successfully to which target tasks and domains?
3. Are there theoretical frameworks for predicting transfer success based on task and domain characteristics?
4. How do task similarity measures correlate with actual transfer learning performance?
5. What patterns exist in cross-domain task transfer, and how do they relate to task and domain properties?
6. How can task affinity analysis inform optimal pre-training task selection for specific target domains?

Please provide:
- Studies on measuring and quantifying task affinity and transfer learning effectiveness
- Research on factors that determine successful task transfer across domains
- Theoretical frameworks for predicting transfer learning success based on task/domain similarity
- Empirical analysis of task affinity patterns in multi-task and transfer learning scenarios
- Studies on task similarity measures and their correlation with transfer performance
- Research on cross-domain transfer patterns and their relationship to task characteristics
- Analysis of how pre-training task selection affects downstream performance across domains
- Evidence for task affinity patterns in representation learning and self-supervised learning contexts
```

---

### 23. **Self-Supervised Learning Task Selection**

**Decision**: Selected specific SSL objectives (node masking, contrastive learning, link prediction, graph property prediction)
**Code Reference**: `src/pretrain/tasks.py` - comprehensive task implementations

```
We selected 6 specific self-supervised learning objectives for our multi-task pre-training framework: node feature masking (BERT-style), node/graph contrastive learning (SimCLR/InfoGraph-style), link prediction, graph property prediction, and domain adversarial training. This combination covers generative, contrastive, and predictive SSL paradigms across different granularities (node, edge, graph).

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. What are the main categories of self-supervised learning objectives for graph neural networks, and how do they compare?
2. Why are BERT-style masked reconstruction approaches effective for graph representation learning?
3. How do generative vs contrastive vs predictive SSL objectives complement each other in multi-task learning?
4. What empirical evidence exists for the effectiveness of our specific SSL task selection vs other possible combinations?
5. How do different SSL paradigms (GraphMAE, SimGRACE, MVGRL, etc.) compare to our chosen approaches?
6. What theoretical frameworks exist for selecting optimal combinations of SSL objectives?

Please provide:
- Comprehensive survey of self-supervised learning approaches for graph neural networks
- Comparative studies between different SSL paradigms (generative, contrastive, predictive)
- Empirical evidence for the effectiveness of BERT-style masking in graph domains
- Analysis of task complementarity in multi-task self-supervised learning
- Studies comparing our chosen SSL tasks with alternatives like GraphMAE, SimGRACE, etc.
- Theoretical foundations for SSL task selection and combination strategies
- Evidence for why our specific SSL combination is well-suited for cross-domain transfer
- Research on SSL task synergy and interference patterns in multi-task scenarios
```

---

### 24. **Relationship to Foundation Models and Large-Scale Pre-training**

**Decision**: Focus on systematic multi-task analysis rather than large-scale foundation model approach
**Code Reference**: Overall experimental design and dataset selection

```
Our work takes a systematic, controlled approach to multi-task pre-training analysis using carefully selected benchmark datasets, rather than pursuing large-scale foundation model pre-training on massive graph corpora. This design choice enables rigorous experimental control and statistical analysis of multi-task learning principles.

SPECIFIC RESEARCH QUESTIONS requiring citations:
1. How does our systematic multi-task pre-training approach relate to the current trend of large-scale graph foundation models?
2. What are the trade-offs between controlled, systematic analysis vs large-scale foundation model approaches in graph learning?
3. What evidence exists for the effectiveness of large-scale pre-training vs smaller-scale, systematic approaches in graph domains?
4. How do the principles discovered through systematic multi-task analysis apply to foundation model development?
5. What is the current state of graph foundation models and how do they relate to multi-task pre-training strategies?
6. What are the data scale requirements for effective graph representation learning, and how does our scale compare?

Please provide:
- Overview of current graph foundation models and large-scale pre-training approaches
- Comparative analysis of systematic vs large-scale approaches in representation learning
- Evidence for data scale requirements in effective graph pre-training
- Studies on how multi-task learning principles scale to foundation model settings
- Analysis of when controlled experiments vs large-scale pre-training are more appropriate
- Research on the relationship between pre-training data scale and transfer learning effectiveness
- Studies on how systematic multi-task insights inform foundation model development
- Evidence for the scientific value of systematic analysis vs purely empirical large-scale approaches
```

---

---

## Quick Reference: Core Citations Needed

**Essential Papers to Find:**
```
1. PCGrad: Yu, Tianhe, et al. "Gradient surgery for multi-task learning." NeurIPS 2020.
2. DANN: Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." JMLR 2016.
3. GIN: Xu, Keyulu, et al. "How powerful are graph neural networks?" ICLR 2019.
4. SimCLR: Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." ICML 2020.
5. InfoGraph: Sun, Fan-Yun, et al. "InfoGraph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization." ICLR 2020.
6. BERT: Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
```

## Usage Instructions

**Workflow for gathering citations:**

1. **Start with the context prompt** to establish your research framework
2. **Use individual prompts** for each specific design decision
3. **Follow the specific research questions** listed in each prompt
4. **Prioritize top-tier venues**: NeurIPS, ICML, ICLR, ACL, EMNLP, ICCV, CVPR
5. **Gather both theoretical and empirical evidence** for each decision
6. **Look for contradictory evidence** to address in limitations/discussion

**What each prompt will give you:**
- **Direct citations** with full bibliographic details
- **Specific findings** that support your design choices
- **Theoretical frameworks** explaining why your approach works
- **Contradictory evidence** you should acknowledge
- **Empirical validation** from similar contexts

**ACL Citation Format:**
```
Author, A. A., Author, B. B., and Author, C. C. (Year). Title of paper. In Proceedings of Conference Name (pp. X-Y). Publisher.
```

**Pro Tips:**
- Each prompt is designed to be **self-contained** with full context
- **Copy prompts directly** using the ``` format provided
- **Ask follow-up questions** if you need more specific citations
- **Request page numbers** and specific findings for direct quotes
- **Ask for recent surveys** if available for comprehensive coverage

This systematic approach ensures comprehensive literature support for your ACL paper with proper justification for every major design decision.

---

## **Summary Table: Design Decisions for Excel Reference**

| Decision Category | Decision Description | Prompt Reference |
|-------------------|---------------------|-----------------|
| **GNN Architecture** | Used 5-layer Graph Isomorphism Network (GIN) as backbone architecture. GIN was chosen for its theoretical expressiveness and empirical performance on graph-level tasks, with 5 layers providing optimal balance between representational capacity and avoiding over-smoothing phenomena in message passing. | Prompt #1 |
| **Hidden Dimensions** | Set hidden dimension to 256 across all GNN layers. This dimension size balances representational capacity with computational efficiency, allowing sufficient expressiveness for diverse graph domains while maintaining manageable memory requirements for multi-task training across 6 different objectives. | Prompt #2 |
| **Input Encoding** | Implemented domain-specific input encoders (Linear→BatchNorm1d→ReLU→Dropout) that map varying input dimensions (4D to 1433D) to standardized 256D embeddings. This approach handles heterogeneous feature spaces across domains while enabling effective cross-domain knowledge transfer through shared backbone representation space. | Prompt #3 |
| **Node Masking Rate** | Applied 15% node feature masking rate for reconstructive pre-training task. Following BERT-style masked language modeling principles adapted to graphs, we randomly mask 15% of node embeddings and train the model to reconstruct them using MSE loss, providing generative self-supervised learning signal. | Prompt #4 |
| **Temperature Scheduling** | Implemented progressive temperature scheduling from 0.5→0.05 for contrastive learning tasks (node and graph contrastive). This annealing strategy starts with softer similarity distributions and gradually sharpens them, following SimCLR principles to improve contrastive learning dynamics and representation quality. | Prompt #5 |
| **Hard Negative Mining** | Applied 30% hard negative ratio with minimum 8 hard negatives for contrastive and link prediction tasks. Hard negatives (highest similarity among all negatives) provide more challenging training signal than random negatives, improving representation quality by forcing the model to learn more discriminative features. | Prompt #6 |
| **Gradient Surgery** | Used PCGrad (Projecting Conflicting Gradients) to resolve gradient conflicts between the 6 multi-task objectives. When gradients from different tasks conflict (negative dot product), PCGrad projects them onto orthogonal directions, enabling cooperative multi-task optimization without task interference. | Prompt #7 |
| **Loss Balancing** | Implemented adaptive loss balancing with inverse magnitude weighting: w_i = (1/\|L_i\|) / Σ(1/\|L_j\|). Tasks with smaller loss magnitudes receive higher weights, automatically balancing the contribution of different objectives (reconstruction, contrastive, adversarial) that naturally operate at different scales. | Prompt #8 |
| **Domain Adversarial** | Included domain adversarial training (scheme s5) using gradient reversal layer to learn domain-invariant representations. The domain classifier attempts to predict source domain while the backbone learns to fool it through gradient reversal, encouraging features that transfer well across the 4 graph domains. | Prompt #9 |
| **Learning Rates** | Applied task-specific learning rates: 1e-5 for main tasks (masking, contrastive, graph property), 1e-6 for link prediction (stability), 5e-6 for domain adversarial. Different tasks have varying sensitivity and convergence characteristics, requiring differentiated learning rates for optimal multi-task performance. | Prompt #10 |
| **Batch Size** | Used batch size 32 for pre-training across all domains and tasks. This size balances gradient noise (sufficient for stable estimates) with memory constraints and computational efficiency, while being appropriate for the graph sizes and multi-task complexity in our framework. | Prompt #11 |
| **Early Stopping** | Implemented early stopping with patience = 50% of total epochs (25/50 for pre-training). This strategy prevents overfitting while allowing sufficient time for complex multi-task objectives to converge, balancing training time efficiency with performance optimization across diverse tasks. | Prompt #12 |
| **Dataset Selection** | Selected TU datasets (MUTAG, PROTEINS, NCI1, ENZYMES) and Planetoid datasets (Cora, CiteSeer) for comprehensive evaluation. This combination provides diverse graph types (molecular vs citation), sizes (188-4110 graphs), and feature dimensions (4D-1433D), enabling robust cross-domain transfer learning assessment. | Prompt #13 |
| **Domain Comparison** | Compared cross-domain (s4: trained on 4 domains) vs single-domain (b4: ENZYMES only) pre-training using identical 5-task combinations. This critical comparison directly tests whether domain diversity helps or hurts transfer learning, controlling for task complexity while varying domain exposure. | Prompt #14 |
| **Statistical Validation** | Conducted experiments with 3 random seeds (42, 84, 126) and applied proper statistical testing with Bonferroni correction and effect size reporting. This addresses reproducibility concerns and ensures robust statistical conclusions about transfer learning effectiveness across multiple experimental conditions. | Prompt #15 |
| **Fine-tuning Strategies** | Compared three fine-tuning approaches: from-scratch (b1), linear probing (freeze backbone), and full fine-tuning (train all parameters). This systematic comparison reveals when and why pre-training helps, and which fine-tuning strategy is optimal for different transfer scenarios. | Prompt #16 |
| **Evaluation Metrics** | Used comprehensive task-specific metrics: accuracy/F1/precision/recall for classification, AUC for link prediction, plus efficiency metrics (convergence epochs, training time). Multiple complementary metrics provide robust evaluation of both transfer learning effectiveness and computational efficiency. | Prompt #17 |
| **Graph Augmentations** | Applied node dropping, edge dropping, and attribute masking for contrastive learning positive pair generation. These structural and feature-based augmentations preserve essential graph semantics while providing sufficient variation for effective contrastive learning across different graph domains. | Prompt #18 |
| **Gradient Clipping** | Set maximum gradient norm to 0.5 to prevent gradient explosions in multi-task training. The combination of diverse objectives (reconstruction, contrastive, adversarial) can create unstable gradient dynamics, making gradient clipping essential for stable optimization across all tasks. | Prompt #19 |
| **Feature Normalization** | Applied StandardScaler normalization followed by [-3, 3] clipping for continuous features (PROTEINS, ENZYMES). This two-step process handles extreme outliers while maintaining normalized distributions, critical for stable cross-domain training with heterogeneous feature scales. | Prompt #20 |
| **Task Combination Strategy** | Designed systematic progression from single-task baselines (b2: node masking, b3: node contrastive) → dual-task combinations (s1: generative, s2: contrastive) → 4-task (s3) → 5-task (s4) → 6-task with domain adversarial (s5). This progression enables systematic analysis of how task combinations affect learning and identifies optimal multi-task configurations. | Prompt #21 |
| **Task Affinity Analysis** | Implemented comprehensive logging to analyze which pre-training tasks transfer most effectively to which downstream domains and task types. Through per-domain-per-task loss tracking, we systematically evaluate task-domain affinity patterns to understand which tasks (node masking, contrastive, etc.) provide the best transfer to specific domains (molecular vs citation graphs). | Prompt #22 |
| **SSL Task Selection** | Selected 6 specific self-supervised learning objectives covering generative (node masking), contrastive (node/graph contrastive), predictive (link prediction, graph properties), and adversarial (domain adversarial) paradigms. This combination spans different SSL approaches and granularities (node, edge, graph-level) to provide comprehensive representation learning across diverse graph domains. | Prompt #23 |
| **Foundation Model Approach** | Chose systematic, controlled multi-task analysis using benchmark datasets rather than large-scale foundation model pre-training on massive graph corpora. This design enables rigorous experimental control and statistical analysis of multi-task learning principles, providing scientific insights that inform foundation model development. | Prompt #24 |

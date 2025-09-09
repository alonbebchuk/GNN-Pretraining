# Comprehensive Analysis Plan for Multi-Task GNN Pre-training Results

## Overview

This document provides a detailed, step-by-step analysis plan for the multi-task, cross-domain pre-training results. The analysis will be conducted using **Jupyter notebooks** to enable interactive exploration, visualization, and iterative analysis of the experimental results. We have **324 total experiments** consisting of:

- **6 Fine-tuning Datasets:**
  1. `ENZYMES` (graph classification)
  2. `PTC_MR` (graph classification) 
  3. `Cora_NC` (node classification)
  4. `CiteSeer_NC` (node classification)
  5. `Cora_LP` (link prediction)
  6. `CiteSeer_LP` (link prediction)

- **9 Pre-training Schemes:**
  - `b1`: From-scratch baseline (no pre-training)
  - `b2`: Node feature masking only
  - `b3`: Node contrastive only
  - `s1`: Node feature masking + Link prediction
  - `s2`: Node contrastive + Graph contrastive
  - `s3`: Combined 4-task (s1 + s2)
  - `s4`: All 5 tasks (cross-domain)
  - `s5`: All 5 tasks + Domain adversarial
  - `b4`: All 5 tasks (single-domain: ENZYMES only)

- **2 Fine-tuning Strategies:**
  - `linear_probe`: Freeze backbone, train head only
  - `full_finetune`: Train all parameters

- **3 Random Seeds:** 42, 84, 126

**Total:** 6 × 9 × 2 × 3 = 324 experiments

## Dataset Information

The 6 fine-tuning datasets can be found in:
- **File:** `src/data/data_setup.py` (lines 52-59: `TASK_TYPES` dictionary)
- **File:** `summary.txt` (lines 178-187: "Downstream Evaluation Tasks")

### Task Type Groupings:
- **Graph Classification:** ENZYMES, PTC_MR
- **Node Classification:** Cora_NC, CiteSeer_NC  
- **Link Prediction:** Cora_LP, CiteSeer_LP

### Available Metrics:
For all tasks: `accuracy`, `f1`, `precision`, `recall`, `auc`, `loss`
- **Primary metric for classification:** `accuracy`
- **Primary metric for link prediction:** `auc`

---

## Analysis Structure and Best Practices

The analysis will be conducted using a combination of Python scripts and Jupyter notebooks in the `analysis/` directory, following coding best practices:

**Python Scripts** (for data processing, statistical analysis, and reproducible computations):
- Reliable, version-controlled, and easily testable
- Automated data processing with proper error handling
- Consistent results across different environments
- Easy to integrate into CI/CD pipelines

**Jupyter Notebooks** (for exploratory analysis, visualization, and reporting):
- Interactive exploration of results and patterns
- Rich documentation with markdown and inline plots
- Ideal for presenting findings and creating reports

### Analysis Structure:
1. **`analysis/data_collection.py`** - Data extraction, cleaning, and aggregation (Python script)
2. **`analysis/statistical_analysis.py`** - Core statistical computations and hypothesis testing (Python script)
3. **`analysis/02_research_questions_exploration.ipynb`** - Interactive exploration and visualization of RQ1-RQ4 results
4. **`analysis/03_task_specific_exploration.ipynb`** - Interactive deep dive analysis for different task types
5. **`analysis/04_summary_and_publication.ipynb`** - Efficiency analysis, master summary, and publication materials

**Python Scripts** will include:
- Robust error handling and logging
- Comprehensive docstrings and type hints
- Modular functions for reusability
- Configuration management
- Data validation and quality checks
- Automated report generation

**Jupyter Notebooks** will include:
- Clear markdown documentation of objectives
- Interactive visualizations for data exploration
- Narrative explanations of findings
- Summary tables and key insights
- Export functionality for figures and reports

---

## Analysis Steps

### Step 1: Data Collection and Aggregation

#### 1.1 Extract All Experimental Results from WandB

**Python Script:** `analysis/data_collection.py`

```
Task: Extract all fine-tuning experimental results from WandB

Requirements:
1. Connect to WandB project for fine-tuning experiments
2. Query all runs with the naming pattern: {domain}_{finetune_strategy}_{pretrained_scheme}_{seed}
3. For each run, extract the following final test metrics:
   - test/accuracy
   - test/f1
   - test/precision  
   - test/recall
   - test/auc
   - test/loss
   - test/convergence_epochs
   - test/training_time
   - test/total_parameters
   - test/trainable_parameters

4. Create a master DataFrame with columns:
   - domain_name (e.g., 'ENZYMES')
   - finetune_strategy ('linear_probe' or 'full_finetune')
   - pretrained_scheme ('b1', 'b2', 'b3', 's1', 's2', 's3', 's4', 's5', 'b4')
   - seed (42, 84, 126)
   - task_type ('graph_classification', 'node_classification', 'link_prediction')
   - All test metrics listed above

5. Save the raw results as: `analysis/results/raw_experimental_results.csv`

Expected output: CSV file with 324 rows (one per experiment) and all metrics
```

#### 1.2 Aggregate Results Across Seeds

**Python Script:** `analysis/data_collection.py` (continued function)

```
Task: Aggregate experimental results across the 3 random seeds

Requirements:
1. Load the raw experimental results from Step 1.1
2. Group results by combination: (domain_name, finetune_strategy, pretrained_scheme)
3. For each combination, compute across the 3 seeds:
   - mean (average)
   - std (standard deviation)
   - min (minimum value)
   - max (maximum value)
   - median
   - sem (standard error of mean)

4. Apply this aggregation to all metrics:
   - test/accuracy → accuracy_mean, accuracy_std, accuracy_min, accuracy_max, accuracy_median, accuracy_sem
   - test/f1 → f1_mean, f1_std, f1_min, f1_max, f1_median, f1_sem
   - test/precision → precision_mean, precision_std, precision_min, precision_max, precision_median, precision_sem
   - test/recall → recall_mean, recall_std, recall_min, recall_max, recall_median, recall_sem
   - test/auc → auc_mean, auc_std, auc_min, auc_max, auc_median, auc_sem
   - test/loss → loss_mean, loss_std, loss_min, loss_max, loss_median, loss_sem
   - test/convergence_epochs → convergence_epochs_mean, convergence_epochs_std, etc.
   - test/training_time → training_time_mean, training_time_std, etc.

5. Create aggregated DataFrame with columns:
   - domain_name
   - finetune_strategy  
   - pretrained_scheme
   - task_type
   - All aggregated metrics (mean, std, min, max, median, sem for each original metric)

6. Save as: `analysis/results/aggregated_results.csv`

Expected output: CSV file with 108 rows (6 domains × 9 schemes × 2 strategies) and aggregated statistics
```

---

### Step 2: Research Question Analysis

#### 2.1 RQ1: Pre-training Effectiveness Analysis

**Python Script:** `analysis/statistical_analysis.py` (core computations)  
**Notebook:** `analysis/02_research_questions_exploration.ipynb` (visualization and exploration)

```
Task: Analyze the effectiveness of pre-training compared to from-scratch training (RQ1)

Requirements:
1. Load aggregated results from Step 1.2
2. For each (domain, finetune_strategy) combination:
   a. Extract baseline performance (pretrained_scheme = 'b1')
   b. Extract all pre-trained scheme performances ('b2', 'b3', 's1', 's2', 's3', 's4', 's5', 'b4')
   
3. Calculate improvement metrics for each pre-trained scheme vs baseline:
   - Primary metric improvement: (pretrained_primary - baseline_primary) / baseline_primary * 100
   - Where primary metric is:
     * accuracy for graph/node classification
     * auc for link prediction
   
4. Statistical significance testing:
   - Perform paired t-tests between each pre-trained scheme and baseline
   - Apply Bonferroni correction for multiple comparisons (8 schemes × 6 domains × 2 strategies = 96 comparisons)
   - Calculate Cohen's d effect sizes
   - Report p-values, corrected p-values, and effect sizes

5. Create summary tables:
   - Table 1: Mean improvement (%) for each scheme across all domains/strategies
   - Table 2: Statistical significance results (p-values, effect sizes)
   - Table 3: Best performing scheme per domain/strategy combination

6. Generate visualizations:
   - Box plots showing primary metric distributions for each scheme vs baseline
   - Heatmap showing improvement percentages across domains and schemes
   - Bar plots with error bars and significance markers

7. Save outputs:
   - `analysis/results/rq1_improvement_analysis.csv`
   - `analysis/results/rq1_statistical_tests.csv`
   - `analysis/figures/rq1_effectiveness_boxplots.png`
   - `analysis/figures/rq1_improvement_heatmap.png`

Expected deliverable: Comprehensive analysis answering whether pre-training improves performance significantly
```

#### 2.2 RQ2: Task Combination Analysis

**Python Script:** `analysis/statistical_analysis.py` (continued functions)  
**Notebook:** `analysis/02_research_questions_exploration.ipynb` (continued)

```
Task: Analyze which task combinations are most effective for pre-training (RQ2)

Requirements:
1. Load aggregated results from Step 1.2
2. Group pre-training schemes by task combination logic:
   - Single-task: b2 (node_feat_mask), b3 (node_contrast)
   - Two-task: s1 (generative), s2 (contrastive)
   - Four-task: s3 (combined)
   - Five-task: s4 (full), b4 (single-domain)
   - Six-task: s5 (+ domain_adv)

3. Statistical comparison analysis:
   - Compare performance between progressive task combinations
   - Test: b2 vs s1 (adding link prediction to node feature masking)
   - Test: b3 vs s2 (adding graph contrastive to node contrastive)
   - Test: s1&s2 vs s3 (combining generative and contrastive)
   - Test: s3 vs s4 (adding graph property prediction)
   - Test: s4 vs s5 (adding domain adversarial)
   - Test: s4 vs b4 (cross-domain vs single-domain)

4. Task synergy analysis:
   - Calculate synergy scores: performance(multi-task) - max(performance(single-tasks))
   - Identify positive vs negative task interactions
   - Rank task combinations by effectiveness across domains

5. Domain-specific analysis:
   - Identify which task combinations work best for each domain
   - Analyze whether patterns are consistent across task types

6. Create outputs:
   - Task combination performance matrix
   - Statistical comparison results with effect sizes
   - Synergy score analysis
   - Domain-specific recommendations

7. Generate visualizations:
   - Progressive task combination performance plots
   - Task synergy heatmaps
   - Domain-specific task effectiveness charts

8. Save outputs:
   - `analysis/results/rq2_task_combination_analysis.csv`
   - `analysis/results/rq2_synergy_scores.csv`
   - `analysis/figures/rq2_task_combinations.png`
   - `analysis/figures/rq2_synergy_heatmap.png`

Expected deliverable: Evidence-based recommendations for optimal task combinations
```

#### 2.3 RQ3: Fine-tuning Strategy Comparison

**Python Script:** `analysis/statistical_analysis.py` (continued functions)  
**Notebook:** `analysis/02_research_questions_exploration.ipynb` (continued)

```
Task: Compare linear probing vs full fine-tuning strategies (RQ3)

Requirements:
1. Load aggregated results from Step 1.2
2. For each (domain, pretrained_scheme) combination:
   - Compare linear_probe vs full_finetune performance
   - Calculate performance difference and relative improvement
   - Analyze computational efficiency trade-offs

3. Performance analysis:
   - Primary metric comparison (accuracy/auc)
   - Secondary metrics analysis (f1, precision, recall)
   - Statistical significance testing between strategies

4. Efficiency analysis:
   - Training time comparison
   - Parameter efficiency: performance per trainable parameter
   - Convergence speed analysis (epochs to convergence)

5. Strategy effectiveness by scheme type:
   - Analyze which pre-training schemes benefit more from full fine-tuning
   - Identify cases where linear probing is sufficient
   - Determine scheme-strategy interaction effects

6. Cost-benefit analysis:
   - Performance improvement vs computational cost
   - Identify optimal strategy for different scenarios (time-constrained, performance-critical)

7. Create analysis outputs:
   - Strategy comparison matrix across all combinations
   - Efficiency trade-off analysis
   - Recommendations by scenario type

8. Generate visualizations:
   - Performance vs computational cost scatter plots
   - Strategy effectiveness by pre-training scheme
   - Time-to-convergence comparisons

9. Save outputs:
   - `analysis/results/rq3_strategy_comparison.csv`
   - `analysis/results/rq3_efficiency_analysis.csv`
   - `analysis/figures/rq3_performance_vs_cost.png`
   - `analysis/figures/rq3_strategy_effectiveness.png`

Expected deliverable: Clear guidelines for choosing fine-tuning strategies based on constraints and requirements
```

#### 2.4 RQ4: Domain-Task Affinity Analysis

**Python Script:** `analysis/statistical_analysis.py` (continued functions)  
**Notebook:** `analysis/02_research_questions_exploration.ipynb` (continued)

```
Task: Analyze which pre-training tasks show strongest affinity for specific downstream domains (RQ4)

Requirements:
1. Load aggregated results from Step 1.2
2. Create domain-task affinity matrix:
   - Rows: Pre-training schemes (b2, b3, s1, s2, s3, s4, s5, b4)
   - Columns: Downstream domains (ENZYMES, PTC_MR, Cora_NC, CiteSeer_NC, Cora_LP, CiteSeer_LP)
   - Values: Improvement over baseline (b1) for each combination

3. Cross-domain transfer analysis:
   - Compare b4 (ENZYMES-only) vs s4 (cross-domain) on ENZYMES fine-tuning
   - Analyze transfer from molecular domains (MUTAG, PROTEINS, NCI1, ENZYMES) to:
     * Citation networks (Cora, CiteSeer)
     * Different molecular tasks (PTC_MR)

4. Task type affinity patterns:
   - Graph classification: Which schemes work best for ENZYMES and PTC_MR?
   - Node classification: Which schemes transfer best to Cora_NC and CiteSeer_NC?
   - Link prediction: Which schemes are optimal for Cora_LP and CiteSeer_LP?

5. Specialization vs generalization analysis:
   - Identify schemes that work well across all domains (generalization)
   - Identify schemes that excel in specific domains (specialization)
   - Quantify trade-offs between broad applicability and peak performance

6. Statistical analysis:
   - Cluster analysis to group similar domains by pre-training effectiveness
   - Correlation analysis between domain characteristics and scheme effectiveness
   - ANOVA to test for significant domain-scheme interactions

7. Domain adaptation insights:
   - Analyze role of domain adversarial training (s5 vs s4)
   - Quantify benefits of cross-domain vs single-domain pre-training
   - Identify domain similarity patterns

8. Generate visualizations:
   - Domain-task affinity heatmap with clustering
   - Transfer learning effectiveness charts
   - Specialization vs generalization scatter plots

9. Save outputs:
   - `analysis/results/rq4_domain_affinity_matrix.csv`
   - `analysis/results/rq4_transfer_analysis.csv`
   - `analysis/figures/rq4_affinity_heatmap.png`
   - `analysis/figures/rq4_transfer_patterns.png`

Expected deliverable: Comprehensive understanding of which pre-training approaches work best for different types of downstream tasks
```

---

### Step 3: Task-Type Specific Analysis

#### 3.1 Graph Classification Analysis

**Notebook:** `analysis/03_task_specific_exploration.ipynb`

```
Task: Deep analysis of graph classification performance (ENZYMES, PTC_MR)

Requirements:
1. Extract results for ENZYMES and PTC_MR from aggregated data
2. Compare performance patterns between the two graph classification datasets
3. Analyze which pre-training schemes are most effective for graph-level tasks
4. Statistical comparison between datasets to identify transferable insights

Specific analyses:
- Best performing schemes for each dataset
- Consistency of scheme rankings between ENZYMES and PTC_MR
- Impact of dataset size (ENZYMES: 600 graphs, PTC_MR: 344 graphs)
- Multi-class (ENZYMES: 6 classes) vs binary (PTC_MR: 2 classes) performance patterns

Outputs:
- Graph classification specific performance tables
- Cross-dataset consistency analysis
- Recommendations for graph classification pre-training

Files: `analysis/results/graph_classification_analysis.csv`, `analysis/figures/graph_classification_comparison.png`
```

#### 3.2 Node Classification Analysis

**Notebook:** `analysis/03_task_specific_exploration.ipynb` (continued)

```
Task: Deep analysis of node classification performance (Cora_NC, CiteSeer_NC)

Requirements:
1. Extract results for Cora_NC and CiteSeer_NC from aggregated data
2. Analyze cross-domain transfer from molecular graphs to citation networks
3. Compare effectiveness of different pre-training approaches for node-level tasks
4. Investigate impact of feature dimensionality differences

Specific analyses:
- Molecular-to-citation transfer effectiveness
- Impact of feature dimension mismatch (molecular: 4-37 dim, citation: 1433-3703 dim)
- Node-level task synergy analysis
- Comparison between Cora and CiteSeer patterns

Outputs:
- Node classification specific performance analysis
- Cross-domain transfer quantification
- Feature adaptation insights

Files: `analysis/results/node_classification_analysis.csv`, `analysis/figures/node_classification_transfer.png`
```

#### 3.3 Link Prediction Analysis

**Notebook:** `analysis/03_task_specific_exploration.ipynb` (continued)

```
Task: Deep analysis of link prediction performance (Cora_LP, CiteSeer_LP)

Requirements:
1. Extract results for Cora_LP and CiteSeer_LP from aggregated data
2. Analyze effectiveness of link prediction pre-training (present in s1, s3, s4, s5, b4)
3. Compare performance between citation network datasets
4. Investigate structural learning transfer patterns

Specific analyses:
- Impact of link prediction as pre-training task
- Structural pattern transfer from molecular to citation networks
- Graph structure similarity analysis
- Comparison of AUC performance patterns

Outputs:
- Link prediction specific performance analysis
- Structural transfer insights
- Link prediction pre-training effectiveness

Files: `analysis/results/link_prediction_analysis.csv`, `analysis/figures/link_prediction_effectiveness.png`
```

---

### Step 4: Computational Efficiency Analysis

#### 4.1 Training Efficiency Analysis

**Notebook:** `analysis/04_summary_and_publication.ipynb`

```
Task: Analyze computational efficiency across all experimental configurations

Requirements:
1. Load training time and convergence data from aggregated results
2. Calculate efficiency metrics:
   - Performance per minute (primary_metric / training_time_minutes)
   - Performance per epoch (primary_metric / convergence_epochs)
   - Parameter efficiency (primary_metric / trainable_parameters)

3. Compare efficiency across:
   - Pre-training schemes (complex multi-task vs simple single-task)
   - Fine-tuning strategies (linear probe vs full fine-tuning)
   - Dataset sizes and types

4. Identify efficiency-performance trade-offs:
   - High-performance but slow approaches
   - Fast but lower-performance approaches
   - Optimal balance points

5. Cost-benefit analysis for different use cases:
   - Research scenarios (performance-critical)
   - Production scenarios (time-constrained)
   - Resource-constrained scenarios

Outputs:
- Comprehensive efficiency analysis
- Trade-off visualizations
- Scenario-specific recommendations

Files: `analysis/results/efficiency_analysis.csv`, `analysis/figures/efficiency_tradeoffs.png`
```

---

### Step 5: Comprehensive Summary and Recommendations

#### 5.1 Create Master Summary Report

**Notebook:** `analysis/04_summary_and_publication.ipynb` (continued)

```
Task: Generate comprehensive summary report with actionable recommendations

Requirements:
1. Synthesize findings from all previous analysis steps (RQ1-RQ4, task-specific, efficiency)
2. Create executive summary with key findings
3. Provide evidence-based recommendations for:
   - Best pre-training schemes for different scenarios
   - Optimal fine-tuning strategies by use case
   - Domain-specific best practices
   - Computational efficiency guidelines

4. Generate final summary tables:
   - Overall performance ranking across all metrics
   - Scheme-domain compatibility matrix
   - Computational efficiency rankings
   - Statistical significance summary

5. Create comprehensive visualizations:
   - Master performance heatmap
   - Recommendation decision tree
   - Efficiency-performance frontier plots

6. Write detailed conclusions addressing each research question with statistical evidence

Outputs:
- Executive summary report
- Comprehensive recommendation guide  
- Master visualization dashboard

Files: `analysis/results/master_summary_report.md`, `analysis/results/recommendations_guide.md`, `analysis/figures/master_dashboard.png`
```

#### 5.2 Generate Academic Paper Tables and Figures

**Notebook:** `analysis/04_summary_and_publication.ipynb` (continued)

```
Task: Create publication-ready tables and figures for academic paper

Requirements:
1. Generate main paper tables:
   - Table 1: Overall performance comparison with statistical significance
   - Table 2: Task combination effectiveness analysis
   - Table 3: Fine-tuning strategy comparison
   - Table 4: Computational efficiency summary

2. Generate main paper figures:
   - Figure 1: Pre-training effectiveness overview (box plots with significance)
   - Figure 2: Task synergy heatmap with clustering
   - Figure 3: Domain-task affinity matrix
   - Figure 4: Efficiency vs performance scatter plots

3. Generate supplementary materials:
   - Supplementary tables with complete statistical results
   - Supplementary figures with detailed breakdowns
   - Complete experimental details and parameters

4. Format according to academic standards:
   - LaTeX table formatting
   - High-resolution figures (300 DPI)
   - Proper statistical notation and significance marking
   - Complete captions and legends

Outputs:
- Publication-ready tables and figures
- LaTeX source files
- High-resolution image files

Files: `analysis/paper_materials/tables/`, `analysis/paper_materials/figures/`, `analysis/paper_materials/supplementary/`
```

---

## Expected Timeline and Dependencies

### Phase 1: Data Collection (Steps 1.1-1.2)
- **Time:** 1-2 days
- **Dependencies:** WandB access, experimental runs completed
- **Output:** Clean, aggregated dataset ready for analysis

### Phase 2: Core Analysis (Steps 2.1-2.4)  
- **Time:** 3-4 days
- **Dependencies:** Phase 1 completion
- **Output:** Complete analysis of all research questions

### Phase 3: Specialized Analysis (Steps 3.1-3.3, 4.1)
- **Time:** 2-3 days  
- **Dependencies:** Phase 2 completion
- **Output:** Task-specific insights and efficiency analysis

### Phase 4: Synthesis and Publication (Steps 5.1-5.2)
- **Time:** 2-3 days
- **Dependencies:** All previous phases
- **Output:** Publication-ready materials and comprehensive recommendations

**Total Estimated Time:** 8-12 days

---

## Quality Assurance Checklist

- [ ] All 324 experiments accounted for in analysis
- [ ] Statistical significance testing with multiple comparison correction
- [ ] Effect size calculations for practical significance
- [ ] Comprehensive error handling and data validation
- [ ] Reproducible analysis with version-controlled code
- [ ] Complete documentation of methods and assumptions
- [ ] Cross-validation of key findings
- [ ] Publication-ready visualizations and tables

---

## File Organization Structure

```
analysis/
├── data_collection.py
├── statistical_analysis.py
├── 02_research_questions_exploration.ipynb
├── 03_task_specific_exploration.ipynb
├── 04_summary_and_publication.ipynb
├── results/
│   ├── raw_experimental_results.csv
│   ├── aggregated_results.csv
│   ├── rq1_improvement_analysis.csv
│   ├── rq1_statistical_tests.csv
│   ├── rq2_task_combination_analysis.csv
│   ├── rq2_synergy_scores.csv
│   ├── rq3_strategy_comparison.csv
│   ├── rq3_efficiency_analysis.csv
│   ├── rq4_domain_affinity_matrix.csv
│   ├── rq4_transfer_analysis.csv
│   ├── graph_classification_analysis.csv
│   ├── node_classification_analysis.csv
│   ├── link_prediction_analysis.csv
│   ├── efficiency_analysis.csv
│   ├── master_summary_report.md
│   └── recommendations_guide.md
├── figures/
│   ├── rq1_effectiveness_boxplots.png
│   ├── rq1_improvement_heatmap.png
│   ├── rq2_task_combinations.png
│   ├── rq2_synergy_heatmap.png
│   ├── rq3_performance_vs_cost.png
│   ├── rq3_strategy_effectiveness.png
│   ├── rq4_affinity_heatmap.png
│   ├── rq4_transfer_patterns.png
│   ├── graph_classification_comparison.png
│   ├── node_classification_transfer.png
│   ├── link_prediction_effectiveness.png
│   ├── efficiency_tradeoffs.png
│   └── master_dashboard.png
└── paper_materials/
    ├── tables/
    ├── figures/
    └── supplementary/
```

This comprehensive analysis plan ensures systematic, rigorous evaluation of all experimental results with proper statistical validation and actionable insights for the research community.


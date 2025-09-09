# Summary Tables for GNN Pre-training Analysis

## Table 1: Overall Performance Summary by Pre-training Scheme

| Scheme | Description | Tasks | Avg Improvement (%) | Std Dev | Best Domain | Worst Domain |
|--------|-------------|-------|---------------------|---------|-------------|--------------|
| b1 | From-scratch baseline | 0 | 0.00 | 0.00 | - | - |
| b2 | Node feature masking | 1 | -0.39 | 6.15 | CiteSeer_LP (+4.72%) | ENZYMES (-5.26%) |
| b3 | Node contrastive | 1 | -4.53 | 11.21 | CiteSeer_LP (+7.17%) | CiteSeer_NC (-12.45%) |
| s1 | Node mask + Link pred | 2 | -2.67 | 16.89 | PTC_MR (+26.13%) | ENZYMES (-18.18%) |
| s2 | Node + Graph contrastive | 2 | -2.81 | 7.87 | CiteSeer_LP (+9.17%) | CiteSeer_NC (-12.33%) |
| s3 | Combined 4-task | 4 | -2.58 | 6.14 | CiteSeer_LP (+1.30%) | CiteSeer_NC (-7.20%) |
| s4 | All 5 tasks | 5 | -5.89 | 4.87 | CiteSeer_LP (+0.45%) | CiteSeer_NC (-12.37%) |
| s5 | All tasks + Domain adv | 6 | -5.85 | 7.95 | Cora_LP (-5.42%) | CiteSeer_NC (-8.43%) |
| b4 | All 5 tasks (ENZYMES only) | 5 | -4.04 | 4.68 | PTC_MR (0.00%) | Cora_NC (-7.82%) |

## Table 2: Domain-Specific Performance Analysis

| Domain | Type | Task | Features | Avg Improvement | Best Scheme | Success Rate |
|--------|------|------|----------|-----------------|-------------|--------------|
| ENZYMES | Molecular | Graph Class | 3 | -8.75% | b2 (-5.26%) | 0/8 |
| PTC_MR | Molecular | Graph Class | 18 | +3.26% | s1 (+26.13%) | 4/8 |
| Cora_NC | Citation | Node Class | 1433 | -4.55% | s1 (+0.65%) | 1/8 |
| CiteSeer_NC | Citation | Node Class | 3703 | -9.29% | b2 (-0.61%) | 0/8 |
| Cora_LP | Citation | Link Pred | 1433 | -2.36% | s2 (-0.29%) | 0/8 |
| CiteSeer_LP | Citation | Link Pred | 3703 | +2.30% | s2 (+9.17%) | 5/8 |

## Table 3: Task Contribution Analysis

| Pre-training Task | Schemes Using | Avg Performance | Contribution | Gradient Conflicts |
|-------------------|---------------|-----------------|--------------|-------------------|
| Node Feature Masking | 6/8 | -3.57% | +0.10% | Low |
| Link Prediction | 5/8 | -4.21% | -1.63% | High |
| Node Contrastive | 6/8 | -4.29% | -2.75% | Medium |
| Graph Contrastive | 5/8 | -4.24% | -1.71% | Medium |
| Graph Property Pred | 3/8 | -5.26% | -2.66% | Low |
| Domain Adversarial | 1/8 | -5.85% | -2.57% | Very High |

## Table 4: Fine-tuning Strategy Comparison

| Strategy | Avg Performance | Training Time | Parameters | Cost-Benefit Ratio |
|----------|-----------------|---------------|------------|-------------------|
| Linear Probing | 32.8% | 1.2 min | 10K | High |
| Full Fine-tuning | 34.5% | 12.5 min | 2.5M | Low |

### Performance by Domain Type:
- **Molecular → Molecular**: Linear probing retains 95% performance
- **Molecular → Citation**: Full fine-tuning necessary (only 75% retention)

## Table 5: Statistical Significance Summary

| Comparison | p-value | Effect Size (Cohen's d) | Interpretation |
|------------|---------|------------------------|----------------|
| Pre-trained vs Baseline | 0.023 | 0.42 | Small negative effect |
| Single vs Multi-task | 0.001 | 0.68 | Medium effect |
| Molecular vs Citation | <0.001 | 1.23 | Large effect |
| Linear vs Full FT | 0.084 | 0.31 | Small effect |

## Table 6: Computational Efficiency Analysis

| Scheme | Avg Epochs | Training Time (min) | Performance/Epoch | Efficiency Rank |
|--------|------------|-------------------|-------------------|-----------------|
| b2 | 87.3 | 3.38 | 0.11% | 1 |
| b3 | 56.7 | 3.52 | 0.16% | 2 |
| s1 | 89.7 | 3.93 | 0.09% | 6 |
| s2 | 118.0 | 3.71 | 0.08% | 7 |
| s3 | 109.7 | 4.07 | 0.08% | 8 |
| s4 | 65.0 | 3.28 | 0.14% | 3 |
| s5 | 29.7 | 2.94 | 0.30% | 4 |
| b4 | 43.3 | 3.25 | 0.21% | 5 |

## Table 7: Domain Similarity and Transfer Success Correlation

| Source → Target | Domain Similarity | Avg Transfer | Correlation |
|-----------------|------------------|--------------|-------------|
| Molecular → Molecular | 0.83 | -2.67% | r = 0.71 |
| Molecular → Citation | 0.12 | -5.47% | r = -0.68 |
| ENZYMES → Molecular | 0.91 | -3.11% | r = 0.52 |
| ENZYMES → Citation | 0.09 | -5.22% | r = -0.74 |

## Table 8: Recommendations by Use Case

| Use Case | Recommended Approach | Expected Performance | Time Investment |
|----------|---------------------|---------------------|-----------------|
| Quick Prototype | b2 + Linear Probing | Baseline -1% | 5 minutes |
| Same-Domain Transfer | b2/s1 + Full FT | Baseline +2% | 30 minutes |
| Cross-Domain Transfer | From-scratch (b1) | Baseline | 20 minutes |
| Research Benchmark | Multiple schemes | Varies | 2-3 hours |

## Table 9: Key Insights Statistical Support

| Finding | Statistical Evidence | Confidence |
|---------|---------------------|------------|
| Domain mismatch hurts transfer | p < 0.001, d = 1.23 | Very High |
| Multi-task degrades performance | p = 0.001, d = 0.68 | High |
| Feature dimensions matter | r = -0.72, p < 0.01 | High |
| Gradient surgery insufficient | 60-80% conflicts remain | High |
| Linear probing often sufficient | p = 0.084, 90% performance | Medium |

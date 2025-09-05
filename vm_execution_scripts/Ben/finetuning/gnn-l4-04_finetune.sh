#!/bin/bash
# Ben's VM4 (gnn-l4-04) - Finetuning Script
# Standby VM - No finetuning workload (balanced distribution)

set -e
echo "=== Ben VM4 (gnn-l4-04) - Finetuning Phase ==="
echo "This VM is not needed for finetuning (equal distribution achieved)"
echo "Start time: $(date)"

echo "Ben and Tim now have equal workload:"
echo "- Ben: 3 domains (CiteSeer_NC, Cora_LP, CiteSeer_LP) = 162 experiments"
echo "- Tim: 3 domains (ENZYMES, PTC_MR, Cora_NC) = 162 experiments"
echo ""
echo "This VM can be stopped to save costs, or kept as backup."
echo "Monitor progress at: https://wandb.ai/alon-bebchuk-tel-aviv-university/gnn-pretraining"

echo "=== Ben VM4 Standby Complete ==="
echo "End time: $(date)"

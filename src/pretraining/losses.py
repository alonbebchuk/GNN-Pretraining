# Loss Constants
UNCERTAINTY_LOSS_COEF = 0.5       # Weight for uncertainty regularization
LOGSIGMA_TO_SIGMA_SCALE = 0.5     # Scale factor for log(σ) → σ conversion

import torch
import torch.nn as nn
from typing import Dict, Tuple, List

from src.pretraining.tasks import TASK_ZERO_LOSS


def get_default_task_scale():
    """Get default task scaling factor."""
    return 1.0


class UncertaintyWeighter(nn.Module):
    """
    Computes uncertainty-weighted total loss across tasks with learnable log-sigma^2.

    For each task i (excluding domain adversarial):
        L_i^weighted = 0.5 * exp(-log_sigma_sq[i]) * L_i + 0.5 * log_sigma_sq[i]

    Domain adversarial loss is added with a negative scheduled weight -lambda:
        L_total = sum_i L_i^weighted - lambda * L_domain
    """

    def __init__(self, task_names: List[str]) -> None:
        super().__init__()
        self.weighted_task_names = [t for t in task_names if t != 'domain_adv']
        self.log_sigma_sq = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in self.weighted_task_names})
        self.include_domain_adv = 'domain_adv' in task_names

    def forward(self, raw_losses: Dict[str, torch.Tensor], lambda_val: float = TASK_ZERO_LOSS) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            raw_losses: Mapping from task name to its raw scalar loss tensor.
            lambda_val: GRL schedule value for domain adversarial term.

        Returns:
            total_loss: Scalar loss to backprop through
            components: Dict of per-task contributions (already weighted)
        """
        device = next(iter(raw_losses.values())).device
        total = torch.tensor(TASK_ZERO_LOSS, device=device)
        components = {}

        for t in self.weighted_task_names:
            ls = raw_losses[t]
            
            # CRITICAL FIX: Apply task-specific loss scaling BEFORE uncertainty weighting
            task_loss_scales = {
                'node_feat_mask': 0.1,
                'graph_prop': 0.3,
                'node_contrast': 1.0,
                'graph_contrast': 2.0,
                'link_pred': 1.5,
                'domain_adv': 1.0,
            }
            default_scale = 1.0
            task_scale = task_loss_scales.get(t, default_scale)
            ls_scaled = ls * task_scale
            
            # Apply uncertainty weighting to scaled loss
            ls_weighted = UNCERTAINTY_LOSS_COEF * torch.exp(-self.log_sigma_sq[t]) * ls_scaled + UNCERTAINTY_LOSS_COEF * self.log_sigma_sq[t]
            total = total + ls_weighted
            components[t] = ls_weighted

        if self.include_domain_adv:
            domain_scale = 1.0  # Default scale for domain adversarial
            domain_term = -float(lambda_val) * raw_losses['domain_adv'] * domain_scale
            total = total + domain_term
            components['domain_adv'] = domain_term

        return total, components

    def get_task_sigmas(self) -> Dict[str, float]:
        """Return current sigma values per weighted task as Python floats."""
        sigmas = {}
        for t, p in self.log_sigma_sq.items():
            sigmas[t] = float(torch.exp(LOGSIGMA_TO_SIGMA_SCALE * p).detach().cpu())
        return sigmas

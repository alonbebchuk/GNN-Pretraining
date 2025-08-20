import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from src.common import (
    UNCERTAINTY_LOSS_COEF, 
    LOGSIGMA_TO_SIGMA_SCALE, 
    TASK_LOSS_SCALES,
    DEFAULT_TASK_SCALE,
    TASK_ZERO_LOSS
)


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
            task_scale = TASK_LOSS_SCALES.get(t, DEFAULT_TASK_SCALE)
            ls_scaled = ls * task_scale
            
            # Apply uncertainty weighting to scaled loss
            ls_weighted = UNCERTAINTY_LOSS_COEF * torch.exp(-self.log_sigma_sq[t]) * ls_scaled + UNCERTAINTY_LOSS_COEF * self.log_sigma_sq[t]
            total = total + ls_weighted
            components[t] = ls_weighted

        if self.include_domain_adv:
            domain_scale = TASK_LOSS_SCALES.get('domain_adv', DEFAULT_TASK_SCALE)
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

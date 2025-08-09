import torch
import torch.nn as nn
from typing import Dict, Tuple


class UncertaintyWeighter(nn.Module):
    """
    Computes uncertainty-weighted total loss across tasks with learnable log-sigma^2.

    For each task i (excluding domain adversarial):
        L_i^weighted = 0.5 * exp(-log_sigma_sq[i]) * L_i + 0.5 * log_sigma_sq[i]

    Domain adversarial loss is added with a negative scheduled weight -lambda:
        L_total = sum_i L_i^weighted - lambda * L_domain
    """

    def __init__(self, task_names: list[str]):
        super().__init__()
        # Exclude domain adversarial from uncertainty weighting
        self.weighted_task_names = [t for t in task_names if t != 'domain_adv']
        # One scalar log_sigma_sq per weighted task
        self.log_sigma_sq = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros(())) for t in self.weighted_task_names}
        )

    def forward(
        self,
        raw_losses: Dict[str, torch.Tensor],
        lambda_val: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            raw_losses: Mapping from task name to its raw scalar loss tensor.
            lambda_val: GRL schedule value for domain adversarial term.

        Returns:
            total_loss: Scalar loss to backprop through
            components: Dict of per-task contributions (already weighted)
        """
        if len(raw_losses) == 0:
            # Create a zero scalar on CPU; caller can move as needed
            zero = torch.tensor(0.0, requires_grad=True)
            return zero, {}

        # Use any provided loss to pick a device
        any_loss = next(iter(raw_losses.values()))
        device = any_loss.device
        total = torch.tensor(0.0, device=device)
        components: Dict[str, torch.Tensor] = {}

        # Uncertainty-weighted tasks
        for t in self.weighted_task_names:
            if t not in raw_losses:
                continue
            ls = raw_losses[t]
            ls_weighted = 0.5 * torch.exp(-self.log_sigma_sq[t]) * ls + 0.5 * self.log_sigma_sq[t]
            total = total + ls_weighted
            components[t] = ls_weighted

        # Domain adversarial
        if 'domain_adv' in raw_losses:
            domain_term = -float(lambda_val) * raw_losses['domain_adv']
            total = total + domain_term
            components['domain_adv'] = domain_term

        return total, components

    def get_task_sigmas(self) -> Dict[str, float]:
        """Return current sigma values per weighted task as Python floats."""
        sigmas: Dict[str, float] = {}
        for t, p in self.log_sigma_sq.items():
            # sigma = exp(0.5 * log_sigma_sq)
            sigmas[t] = float(torch.exp(0.5 * p).detach().cpu())
        return sigmas


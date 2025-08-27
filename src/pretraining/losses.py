from typing import Dict, List, Tuple

import torch
import torch.nn as nn

UNCERTAINTY_LOSS_COEF = 0.5
LOGSIGMA_TO_SIGMA_SCALE = 0.5


class UncertaintyWeighter(nn.Module):
    def __init__(self, task_names: List[str]) -> None:
        super().__init__()
        self.log_sigma_sq = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(()))
            for t in task_names if t != 'domain_adv'
        })
        self.domain_adv = 'domain_adv' in task_names

    def forward(self, raw_losses: Dict[str, torch.Tensor], lambda_val: float = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = next(iter(raw_losses.values())).device
        total = torch.tensor(0.0, device=device)
        components = {}

        for t, log_sigma_sq in self.log_sigma_sq.items():
            ls = raw_losses[t]

            ls_weighted = UNCERTAINTY_LOSS_COEF * torch.exp(-log_sigma_sq) * ls + UNCERTAINTY_LOSS_COEF * log_sigma_sq

            total += ls_weighted
            components[t] = ls_weighted

        if self.domain_adv:
            domain_term = -(lambda_val * raw_losses['domain_adv'])

            total += domain_term
            components['domain_adv'] = domain_term

        return total, components

    def get_task_sigmas(self) -> Dict[str, float]:
        sigmas = {}
        for t, log_sigma_sq in self.log_sigma_sq.items():
            sigmas[t] = float(torch.exp(LOGSIGMA_TO_SIGMA_SCALE * log_sigma_sq).detach().cpu())
        return sigmas

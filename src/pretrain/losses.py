from typing import Dict, List

import torch
import torch.nn as nn


class UncertaintyWeighter(nn.Module):
    def __init__(self, task_names: List[str]) -> None:
        super().__init__()
        self.log_sigma_sq = nn.ParameterDict({
            task_name: nn.Parameter(torch.zeros(()))
            for task_name in task_names if task_name != 'domain_adv'
        })
        self.domain_adv = 'domain_adv' in task_names

    def forward(self, raw_losses: Dict[str, torch.Tensor], lambda_val: float) -> torch.Tensor:
        device = next(iter(raw_losses.values())).device
        total_weighted_loss = torch.tensor(0.0, device=device)

        for task_name, log_sigma_sq in self.log_sigma_sq.items():
            raw_loss = raw_losses[task_name]
            weighted_loss = 0.5 * (raw_loss * torch.exp(-log_sigma_sq) + log_sigma_sq)
            total_weighted_loss += weighted_loss

        if self.domain_adv:
            domain_adv_loss = -(lambda_val * raw_losses['domain_adv'])
            total_weighted_loss += domain_adv_loss

        return total_weighted_loss

    def get_task_sigmas(self) -> Dict[str, float]:
        sigmas = {}
        for task_name, log_sigma_sq in self.log_sigma_sq.items():
            sigmas[task_name] = float(torch.exp(0.5 * log_sigma_sq).detach().cpu())
        return sigmas

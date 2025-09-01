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

        if len(self.log_sigma_sq) == 1:
            for task_name in self.log_sigma_sq.keys():
                total_weighted_loss += raw_losses[task_name]
        else:
            for task_name, log_sigma_sq in self.log_sigma_sq.items():
                raw_loss = raw_losses[task_name]
                clamped_log_sigma_sq = torch.clamp(log_sigma_sq, min=-5.0, max=5.0)
                weighted_loss = 0.5 * (raw_loss * torch.exp(-clamped_log_sigma_sq) + clamped_log_sigma_sq)
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

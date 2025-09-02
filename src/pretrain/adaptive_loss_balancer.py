import torch
from typing import Dict

EPSILON = 1e-8
MIN_TOTAL_LOSS = 1e-6
WARMUP_STEPS = 100


class AdaptiveLossBalancer:
    def __init__(self):
        self.step_count = 0
        self.current_weights = {}

    def balance_losses(self, task_losses: Dict[str, torch.Tensor], domain_adv_lambda: float) -> torch.Tensor:
        if len(task_losses) == 1:
            return list(task_losses.values())[0]

        self.step_count += 1

        processed_losses = task_losses.copy()
        if 'domain_adv' in processed_losses:
            domain_adv_loss = -domain_adv_lambda * processed_losses['domain_adv']
            other_losses_sum = sum([loss for name, loss in processed_losses.items() if name != 'domain_adv'])
            processed_losses['domain_adv'] = torch.clamp(domain_adv_loss, min=-max(other_losses_sum * 0.5, 1.0))

        if self.step_count > WARMUP_STEPS:
            loss_values = [float(loss.detach().cpu()) for loss in processed_losses.values()]
            task_names = list(processed_losses.keys())

            weights = {}
            total_magnitude = sum(abs(v) for v in loss_values)
            for i, task_name in enumerate(task_names):
                if total_magnitude > 0:
                    weights[task_name] = (1.0 / (abs(loss_values[i]) + EPSILON))
                else:
                    weights[task_name] = 1.0

            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

            self.current_weights = weights.copy()
        else:
            weights = {task: 1.0 / len(processed_losses) for task in processed_losses.keys()}
            self.current_weights = weights.copy()

        weighted_losses = []
        for task_name, loss in processed_losses.items():
            weighted_loss = weights[task_name] * loss
            weighted_losses.append(weighted_loss)

        total_loss = torch.clamp(torch.stack(weighted_losses).sum(), min=MIN_TOTAL_LOSS)

        return total_loss
    
    def get_current_weights(self) -> Dict[str, float]:
        return self.current_weights

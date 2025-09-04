import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import random


class GradientSurgery:
    def __init__(self, device: torch.device):
        self.device = device

    def apply_gradient_surgery(self, model: nn.Module, task_losses: Dict[str, torch.Tensor], task_names: List[str]) -> Dict[str, float]:
        if len(task_losses) <= 1:
            return {}

        original_grads = self._get_gradients(model)

        task_gradients = {}
        for task_name, task_loss in task_losses.items():
            model.zero_grad(set_to_none=True)
            task_loss.backward(retain_graph=True)
            task_gradients[task_name] = self._get_gradients(model)

        modified_gradients, metrics = self._apply_pcgrad(task_gradients, task_names)

        self._set_gradients(model, modified_gradients)

        return metrics

    def _get_gradients(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone().to(self.device)
        return grads

    def _set_gradients(self, model: nn.Module, gradients: Dict[str, torch.Tensor]):
        for name, param in model.named_parameters():
            if name in gradients:
                param.grad = gradients[name].to(param.device)

    def _apply_pcgrad(self, task_gradients: Dict[str, Dict[str, torch.Tensor]], task_names: List[str]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        task_list = list(task_names)
        random.shuffle(task_list)

        modified_grads = {name: {} for name in task_list}
        total_conflicts = 0
        total_projections = 0

        for i, task_i in enumerate(task_list):
            modified_grads[task_i] = task_gradients[task_i].copy()

            for j in range(i):
                task_j = task_list[j]

                conflicts, projections = self._project_gradient(modified_grads[task_i], task_gradients[task_j], f"{task_i}_vs_{task_j}")

                total_conflicts += conflicts
                total_projections += projections

        final_gradients = {}
        for param_name in task_gradients[task_list[0]].keys():
            param_grads = []
            for task_name in task_list:
                if param_name in modified_grads[task_name]:
                    param_grads.append(modified_grads[task_name][param_name])

            if param_grads:
                final_gradients[param_name] = torch.stack(param_grads).mean(dim=0)

        conflict_metrics = {
            'gradient_surgery/total_conflicts': total_conflicts,
            'gradient_surgery/total_projections': total_projections,
            'gradient_surgery/conflict_ratio': total_conflicts / max(total_projections, 1)
        }

        return final_gradients, conflict_metrics

    def _project_gradient(self, grad_i: Dict[str, torch.Tensor], grad_j: Dict[str, torch.Tensor], pair_name: str) -> Tuple[int, int]:
        conflicts = 0
        total_projections = 0

        for param_name in grad_i.keys():
            if param_name not in grad_j:
                continue

            g_i = grad_i[param_name].flatten()
            g_j = grad_j[param_name].flatten()

            if g_i.norm() == 0 or g_j.norm() == 0:
                continue

            total_projections += 1

            dot_product = torch.dot(g_i, g_j)
            if dot_product < 0:
                conflicts += 1

                projection = (dot_product / (g_j.norm() ** 2)) * g_j
                g_i_projected = g_i - projection

                grad_i[param_name] = g_i_projected.reshape(grad_i[param_name].shape)

        return conflicts, total_projections

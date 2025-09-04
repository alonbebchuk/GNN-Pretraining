import torch
import torch.nn as nn
from typing import Dict, List

DEFAULT_LR = 1e-5
DEFAULT_WEIGHT_DECAY = 1e-5

TASK_SPECIFIC_LR = {
    'link_pred': 5e-7,
    'node_feat_mask': 1e-5,
    'node_contrast': 1e-5,
    'graph_contrast': 1e-5,
    'graph_prop': 1e-5,
    'domain_adv': 5e-6,
}


class TaskSpecificOptimizer:
    def __init__(self, model: nn.Module, active_tasks: List[str]):
        self.model = model
        self.task_configs = self._create_task_configs(active_tasks)

        self.param_groups = self._create_parameter_groups()

        self.optimizer = torch.optim.AdamW(self.param_groups)

    def _create_task_configs(self, active_tasks: List[str]) -> Dict[str, Dict]:
        task_configs = {}
        for task_name in active_tasks:
            task_configs[task_name] = {
                'lr': TASK_SPECIFIC_LR[task_name],
                'weight_decay': DEFAULT_WEIGHT_DECAY,
                'params': [f'heads.{task_name}']
            }
        return task_configs

    def _create_parameter_groups(self) -> List[Dict]:
        task_params = {}
        used_param_names = set()

        for task_name, config in self.task_configs.items():
            task_params[task_name] = []
            param_patterns = config['params']

            for name, param in self.model.named_parameters():
                if any(pattern in name for pattern in param_patterns):
                    task_params[task_name].append(param)
                    used_param_names.add(name)

        param_groups = []

        for task_name, params in task_params.items():
            if params:
                config = self.task_configs[task_name]
                param_groups.append({
                    'params': params,
                    'lr': config['lr'],
                    'weight_decay': config['weight_decay'],
                    'name': task_name
                })

        default_params = []
        for name, param in self.model.named_parameters():
            if name not in used_param_names:
                default_params.append(param)

        if default_params:
            param_groups.append({
                'params': default_params,
                'lr': DEFAULT_LR,
                'weight_decay': DEFAULT_WEIGHT_DECAY,
                'name': 'default'
            })

        return param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.optimizer.step()

import torch

GAMMA = 10.0
MAX_LAMBDA = 1.0


class GRLLambdaScheduler:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0

    def __call__(self) -> float:
        p = float(self.current_step) / float(self.total_steps)
        lambda_val = 2.0 / (1.0 + torch.exp(-GAMMA * p)) - 1.0
        return float(lambda_val * MAX_LAMBDA)

    def step(self):
        self.current_step += 1

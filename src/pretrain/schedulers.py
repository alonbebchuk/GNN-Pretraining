import math
from dataclasses import dataclass

GRL_GAMMA = 10.0
PRETRAIN_LR_MIN_FACTOR = 0.001
WARMUP_FRACTION = 0.15


@dataclass
class GRLLambdaScheduler:
    total_steps: int

    def __post_init__(self) -> None:
        self._current_step = 0

    def __call__(self) -> float:
        progress = self._current_step / float(self.total_steps)
        lambda_val = 2.0 / (1.0 + math.exp(-GRL_GAMMA * progress)) - 1.0
        return lambda_val

    def step(self) -> None:
        self._current_step += 1

    def state_dict(self) -> dict:
        return {'lambda_val': self()}


@dataclass
class CosineWithWarmup:
    total_steps: int

    def __post_init__(self) -> None:
        self._current_step = 0
        self._warmup_steps = int(self.total_steps * WARMUP_FRACTION)

    def __call__(self) -> float:
        if self._current_step < self._warmup_steps:
            warmup_progress = self._current_step / float(self._warmup_steps)
            return float(PRETRAIN_LR_MIN_FACTOR + (1.0 - PRETRAIN_LR_MIN_FACTOR) * warmup_progress)
        else:
            progress = (self._current_step - self._warmup_steps) / float(self.total_steps - self._warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(PRETRAIN_LR_MIN_FACTOR + (1.0 - PRETRAIN_LR_MIN_FACTOR) * cosine)

    def step(self) -> None:
        self._current_step += 1

    def state_dict(self) -> dict:
        return {'lr_multiplier': self()}

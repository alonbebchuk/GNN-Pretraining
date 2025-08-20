import math
from dataclasses import dataclass
from src.common import (
    GRL_GAMMA, 
    GRL_LAMBDA_MIN, 
    GRL_LAMBDA_MAX,
    PRETRAIN_LR_WARMUP_FRACTION,
    PRETRAIN_LR_MIN_FACTOR,
    SCHEDULER_PROGRESS_CLAMP_MIN,
    SCHEDULER_PROGRESS_CLAMP_MAX,
    GRL_CORE_NUMERATOR,
    COSINE_MULTIPLIER,
    COSINE_OFFSET
)


@dataclass
class GRLLambdaScheduler:
    """
    Scheduler for the Gradient Reversal Layer (GRL) lambda used in domain adversarial training.

    Uses the DANN schedule:
        p = current_step / total_steps
        lambda(p) = lambda_min + (lambda_max - lambda_min) * (2 / (1 + exp(-gamma * p)) - 1)

    The scheduler is callable: calling it returns the current lambda value.
    Advance progress via step() or update(progress).
    """

    total_steps: int
    gamma: float = GRL_GAMMA
    lambda_min: float = GRL_LAMBDA_MIN
    lambda_max: float = GRL_LAMBDA_MAX

    def __post_init__(self) -> None:
        self._current_step = 0

    def __call__(self) -> float:
        p = max(SCHEDULER_PROGRESS_CLAMP_MIN, min(SCHEDULER_PROGRESS_CLAMP_MAX, self._current_step / float(self.total_steps)))
        core = GRL_CORE_NUMERATOR / (SCHEDULER_PROGRESS_CLAMP_MAX + math.exp(-self.gamma * p)) - SCHEDULER_PROGRESS_CLAMP_MAX
        return float(self.lambda_min + (self.lambda_max - self.lambda_min) * core)

    def step(self, n: int = 1) -> None:
        self._current_step = min(self.total_steps, self._current_step + n)

    def update(self, progress: float) -> None:
        self._current_step = int(round(progress * self.total_steps))

    def reset(self) -> None:
        self._current_step = 0
    
    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            'total_steps': self.total_steps,
            'gamma': self.gamma,
            'lambda_min': self.lambda_min,
            'lambda_max': self.lambda_max,
            '_current_step': self._current_step,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.total_steps = state_dict['total_steps']
        self.gamma = state_dict['gamma']
        self.lambda_min = state_dict['lambda_min']
        self.lambda_max = state_dict['lambda_max']
        self._current_step = state_dict['_current_step']


@dataclass
class CosineWithWarmup:
    """
    Cosine annealing with linear warmup scheduler for scalar multipliers.

    Produces a multiplier in [lr_min_factor, 1.0] applied to base LR.
    """

    total_steps: int
    warmup_fraction: float = PRETRAIN_LR_WARMUP_FRACTION
    lr_min_factor: float = PRETRAIN_LR_MIN_FACTOR

    def __post_init__(self) -> None:
        self._current_step: int = 0
        self._warmup_steps: int = max(1, int(round(self.warmup_fraction * self.total_steps)))

    def __call__(self) -> float:
        s = min(self._current_step, self.total_steps)
        if s < self._warmup_steps:
            return float((s + 1) / float(self._warmup_steps))
        progress = (s - self._warmup_steps) / max(SCHEDULER_PROGRESS_CLAMP_MAX, float(self.total_steps - self._warmup_steps))
        cosine = COSINE_MULTIPLIER * (COSINE_OFFSET + math.cos(math.pi * progress))
        return float(self.lr_min_factor + (SCHEDULER_PROGRESS_CLAMP_MAX - self.lr_min_factor) * cosine)

    def step(self, n: int = 1) -> None:
        self._current_step = min(self.total_steps, self._current_step + n)

    def update(self, step: int) -> None:
        self._current_step = min(self.total_steps, step)

    def reset(self) -> None:
        self._current_step = 0
    
    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            'total_steps': self.total_steps,
            'warmup_fraction': self.warmup_fraction,
            'lr_min_factor': self.lr_min_factor,
            '_current_step': self._current_step,
            '_warmup_steps': self._warmup_steps,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.total_steps = state_dict['total_steps']
        self.warmup_fraction = state_dict['warmup_fraction']
        self.lr_min_factor = state_dict['lr_min_factor']
        self._current_step = state_dict['_current_step']
        self._warmup_steps = state_dict['_warmup_steps']

import math
from dataclasses import dataclass
from src.common import GRL_GAMMA, GRL_LAMBDA_MIN, GRL_LAMBDA_MAX
import math


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
        self._current_step: int = 0

    def __call__(self) -> float:
        p = max(0.0, min(1.0, self._current_step / float(self.total_steps)))
        core = 2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0  # in [0, 1]
        return float(self.lambda_min + (self.lambda_max - self.lambda_min) * core)

    def step(self, n: int = 1) -> None:
        """Advance the scheduler by n steps (e.g., after each optimizer step)."""
        self._current_step = min(self.total_steps, self._current_step + n)

    def update(self, progress: float) -> None:
        """
        Set progress directly.

        Args:
            progress: Value in [0, 1] indicating training progress.
        """
        self._current_step = int(round(progress * self.total_steps))

    def reset(self) -> None:
        self._current_step = 0


@dataclass
class CosineWithWarmup:
    """
    Cosine annealing with linear warmup scheduler for scalar multipliers.

    Produces a multiplier in [lr_min_factor, 1.0] applied to base LR.
    """

    total_steps: int
    warmup_fraction: float = 0.1
    lr_min_factor: float = 0.0

    def __post_init__(self) -> None:
        self._current_step: int = 0
        self._warmup_steps: int = max(1, int(round(self.warmup_fraction * self.total_steps)))

    def __call__(self) -> float:
        s = min(self._current_step, self.total_steps)
        if s < self._warmup_steps:
            # Linear warmup from near-zero to 1.0
            return float((s + 1) / float(self._warmup_steps))
        # Cosine decay to lr_min_factor
        progress = (s - self._warmup_steps) / max(1.0, float(self.total_steps - self._warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(self.lr_min_factor + (1.0 - self.lr_min_factor) * cosine)

    def step(self, n: int = 1) -> None:
        self._current_step = min(self.total_steps, self._current_step + n)

    def update(self, step: int) -> None:
        self._current_step = min(self.total_steps, step)

    def reset(self) -> None:
        self._current_step = 0

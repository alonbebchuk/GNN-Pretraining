import math
from dataclasses import dataclass


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
    gamma: float = 10.0
    lambda_min: float = 0.0
    lambda_max: float = 1.0

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

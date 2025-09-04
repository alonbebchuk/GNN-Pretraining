import math

FINAL_TEMP = 0.05
GAMMA = 10.0
INITIAL_TEMP = 0.5
MAX_LAMBDA = 0.01
START_ADVERSARIAL_EPOCH_FRACTION = 0.4


class TemperatureScheduler:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0

    def __call__(self) -> float:
        progress = min(1.0, self.current_step / self.total_steps)
        temp = INITIAL_TEMP * (FINAL_TEMP / INITIAL_TEMP) ** progress
        return float(temp)

    def step(self):
        self.current_step += 1


class GRLScheduler:
    def __init__(self, total_epochs: int, steps_per_epoch: int):
        self.start_steps = START_ADVERSARIAL_EPOCH_FRACTION * total_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.current_step = 0

    def __call__(self) -> float:
        if self.current_step < self.start_steps:
            return 0.0

        adversarial_steps = self.current_step - self.start_steps
        remaining_steps = self.total_steps - self.start_steps

        p = float(adversarial_steps) / float(remaining_steps)
        lambda_val = 2.0 / (1.0 + math.exp(-GAMMA * p)) - 1.0

        final_lambda = float(lambda_val * MAX_LAMBDA)

        return final_lambda

    def step(self):
        self.current_step += 1

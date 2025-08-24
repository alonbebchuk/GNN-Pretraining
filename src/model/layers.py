import torch
import torch.nn as nn
from typing import Any, Tuple


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = lambda_val
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        lambda_val = ctx.lambda_val
        return grad_output.neg() * lambda_val, None


class GradientReversalLayer(nn.Module):
    def __init__(self) -> None:
        super(GradientReversalLayer, self).__init__()

    def forward(self, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        return GradientReversalFn.apply(x, lambda_val)

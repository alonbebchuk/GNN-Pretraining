from typing import Any, Tuple

import torch
import torch.nn as nn


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg(), None


class GradientReversalLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x)

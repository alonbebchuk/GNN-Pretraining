import torch
import torch.nn as nn
from typing import Any, Tuple


class GradientReversalFn(torch.autograd.Function):
    """
    Gradient Reversal Function for domain-adversarial training.

    This function acts as an identity in the forward pass but reverses 
    and scales the gradient in the backward pass.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """
        Forward pass: identity function.

        Args:
            ctx: Context object to save information for backward pass
            x: Input tensor
            lambda_val: Scaling factor for gradient reversal

        Returns:
            Input tensor x unchanged
        """
        ctx.lambda_val = lambda_val
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: reverse and scale the gradient.

        Args:
            ctx: Context object containing saved information
            grad_output: Gradient flowing from the next layer

        Returns:
            Tuple of gradients: (reversed gradient for x, None for lambda_val)
        """
        lambda_val = ctx.lambda_val
        return grad_output.neg() * lambda_val, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer wrapper.

    This layer applies gradient reversal during backpropagation while
    acting as an identity during the forward pass. Used for domain-adversarial
    training to encourage domain-invariant features.
    """

    def __init__(self) -> None:
        """Initialize the Gradient Reversal Layer."""
        super(GradientReversalLayer, self).__init__()

    def forward(self, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """
        Forward pass through the gradient reversal layer.

        Args:
            x: Input tensor of any shape
            lambda_val: Scaling factor for gradient reversal (scalar)

        Returns:
            Input tensor x unchanged (identity in forward pass)
        """
        return GradientReversalFn.apply(x, lambda_val)

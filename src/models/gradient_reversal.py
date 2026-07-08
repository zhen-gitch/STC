"""Gradient Reversal Layer (GRL) for Stage B1 identity-adversarial training.

The GRL is the identity function in the forward pass and negates (and scales
by ``lambda``) the gradient in the backward pass.  It is used to train a
subject-id attacker head that, through reversed gradients, pushes the shared
representation ``z_dep`` away from encoding subject identity -- without
changing the default baseline behaviour when ``identity_adversarial`` is off.

This module is self-contained and has no dependency on the rest of the model,
so it can be unit-tested in isolation.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    """Autograd function: forward = identity, backward = -lambda * grad."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Thin ``nn.Module`` wrapper around :class:`_GradientReversalFunction`.

    Args:
        lambda_: Gradient reversal strength.  Stored as a plain float (not a
            parameter) so it is never learned and never appears in the
            optimizer state.  The Stage B1 sweep uses ``0.02 / 0.05 / 0.10 /
            0.20``.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambda_)

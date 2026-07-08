"""Unit tests for the Gradient Reversal Layer (Stage B1).

The GRL must be the identity in the forward pass and negate (scaled by
``lambda``) the gradient in the backward pass.  These properties are tested in
isolation from the rest of the model.
"""

import torch

from src.models.gradient_reversal import GradientReversalLayer


def test_grl_forward_is_identity():
    grl = GradientReversalLayer(lambda_=0.1)
    x = torch.randn(4, 8, requires_grad=True)
    y = grl(x)
    assert torch.allclose(y, x)


def test_grl_backward_negates_and_scales_gradient():
    for lam in (1.0, 0.5, 0.1, 2.0):
        grl = GradientReversalLayer(lambda_=lam)
        x = torch.randn(6, requires_grad=True)
        y = grl(x)
        # Use sum so the upstream gradient is all ones -> the GRL must pass
        # back -lambda for every element.
        y.sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, -lam))


def test_grl_lambda_not_a_parameter():
    grl = GradientReversalLayer(lambda_=0.05)
    assert len(list(grl.parameters())) == 0
    assert grl.lambda_ == 0.05

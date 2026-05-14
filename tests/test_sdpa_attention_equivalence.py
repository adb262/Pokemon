"""Tests for the SDPA-based SpatioTemporalEncoderBlock.

These tests verify:

1. Forward pass runs without error in all 3 configs (spatial/temporal/both).
2. Outputs are deterministic (same input -> same output).
3. Gradient flow works (backward produces non-zero gradients).

Run with:
    PYTHONPATH=src .venv/bin/python tests/test_sdpa_attention_equivalence.py
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers.spatio_temporal_encoder_block import SpatioTemporalEncoderBlock


def test_forward_and_determinism(
    name: str, use_spatial: bool, use_temporal: bool
) -> None:
    torch.manual_seed(0)
    d_model = 64
    num_heads = 4

    block = SpatioTemporalEncoderBlock(
        num_images_in_video=5,
        num_heads=num_heads,
        d_model=d_model,
        use_spatial_transformer=use_spatial,
        use_temporal_transformer=use_temporal,
    ).eval()

    x = torch.randn(2, 5, 16, d_model)

    with torch.no_grad():
        y1 = block(x)
        y2 = block(x)

    assert y1.shape == x.shape, f"Shape mismatch: {y1.shape} vs {x.shape}"
    assert torch.equal(y1, y2), f"{name}: outputs not deterministic"
    print(f"  {name}: forward OK, shape={y1.shape}, deterministic=True")


def test_gradient_flow(
    name: str, use_spatial: bool, use_temporal: bool
) -> None:
    torch.manual_seed(0)
    d_model = 64
    num_heads = 4

    block = SpatioTemporalEncoderBlock(
        num_images_in_video=5,
        num_heads=num_heads,
        d_model=d_model,
        use_spatial_transformer=use_spatial,
        use_temporal_transformer=use_temporal,
    ).train()

    x = torch.randn(2, 5, 16, d_model, requires_grad=True)
    y = block(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, f"{name}: no gradient on input"
    assert x.grad.abs().sum() > 0, f"{name}: zero gradient"

    params_with_grad = sum(
        1 for p in block.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_params = sum(1 for _ in block.parameters())
    print(f"  {name}: gradient flow OK ({params_with_grad}/{total_params} params with grads)")


CONFIGS = [
    ("spatial-only", True, False),
    ("temporal-only", False, True),
    ("combined", True, True),
]


if __name__ == "__main__":
    print("Forward + determinism tests:")
    for name, sp, tp in CONFIGS:
        test_forward_and_determinism(name, sp, tp)

    print("\nGradient flow tests:")
    for name, sp, tp in CONFIGS:
        test_gradient_flow(name, sp, tp)

    print("\nAll tests passed!")

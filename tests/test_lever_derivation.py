"""Tests for the lever derivation and its validation.

Two things are checked, and both matter because the derivation runs INSIDE every
kernel: if it silently returns the wrong columns, a whole dose-response sweep is
meaningless.

  1. A synthetic base/edited pair built with the published algebra
     (W[:,j] -= 6*(W[:,j].d_hat)*d_hat on a few columns) must be recovered exactly,
     with the implied strength ~6 and the columns parallel to one direction.
  2. A pair whose difference is NOT a single-direction removal must be REJECTED --
     the guard has to be seen firing, or it is a hope rather than a gate.
"""
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from lever import build_lever_from_checkpoints

OUT_FEATURES, IN_FEATURES = 8, 16
LAYERS = (0,)


def fake_model(weights):
    """A stand-in with the attribute path the derivation uses."""
    layers = []
    for index in sorted(weights):
        layer = types.SimpleNamespace()
        layer.mlp = types.SimpleNamespace(
            down_proj=types.SimpleNamespace(weight=torch.as_tensor(weights[index])))
        layers.append(layer)
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))


def published_edit(weight, columns, direction, strength=6.0):
    """Apply the published algebra: remove `strength` * the projection along d_hat."""
    edited = weight.copy()
    unit = direction / np.linalg.norm(direction)
    for column in columns:
        projection = float(edited[:, column] @ unit)
        edited[:, column] -= strength * projection * unit
    return edited


def test_recovers_the_published_edit():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(OUT_FEATURES, IN_FEATURES)).astype(np.float32)
    direction = rng.normal(size=OUT_FEATURES).astype(np.float32)
    columns = [2, 5, 9]
    edited = published_edit(base, columns, direction)

    spec, diagnostics = build_lever_from_checkpoints(
        fake_model({0: base}), fake_model({0: edited}), layers=LAYERS)

    indices, delta = spec[0]
    assert sorted(indices) == columns, f"wrong columns recovered: {indices}"
    assert delta.shape == (OUT_FEATURES, len(columns))
    assert np.abs(delta - (edited - base)[:, columns]).max() < 1e-6, "delta mismatch"
    assert diagnostics[0]["min_cosine"] > 0.999
    assert 5.9 < abs(diagnostics[0]["implied_strength"]) < 6.1, \
        f"implied strength {diagnostics[0]['implied_strength']} is not ~6"


def test_rejects_a_pair_that_is_not_a_single_direction_edit():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(OUT_FEATURES, IN_FEATURES)).astype(np.float32)
    edited = base.copy()
    edited[:, 3] += rng.normal(size=OUT_FEATURES).astype(np.float32) * 0.5
    edited[:, 7] += rng.normal(size=OUT_FEATURES).astype(np.float32) * 0.5

    try:
        build_lever_from_checkpoints(fake_model({0: base}), fake_model({0: edited}),
                                     layers=LAYERS)
    except AssertionError as error:
        assert "strength" in str(error) or "parallel" in str(error), \
            f"rejected for the wrong reason: {error}"
        return
    raise AssertionError("a noisy, non-rank-1 edit was accepted")


if __name__ == "__main__":
    test_recovers_the_published_edit()
    print("PASS: the published edit is recovered exactly (columns, delta, strength ~6)")
    test_rejects_a_pair_that_is_not_a_single_direction_edit()
    print("PASS: a non-rank-1 difference is rejected (the guard fires)")

"""Regression test: the glial gain must not change the activation dtype.
(The fp16-model/fp32-gain mismatch crashed two kernel runs before this test existed.)"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from glial import GlialConfig, GlialField

cfg = GlialConfig(read_layers=(1, 2), target_layers=(3, 4))
idx = {3: torch.arange(4), 4: torch.arange(4) + 10}
field = GlialField(cfg, d_model=8, inter=16, n_neurons={3: 4, 4: 4}, neuron_idx=idx)

for dtype in (torch.float16, torch.float32):
    field._gains = {3: torch.full((1, 4), 0.25, dtype=torch.float32)}
    act = torch.randn(1, 5, 16, dtype=dtype)
    out = field.gain_for(3, act)
    assert out.dtype == act.dtype, f"dtype changed: {act.dtype} -> {out.dtype}"
    assert torch.allclose(out[:, :, idx[3]].float(),
                          (act[:, :, idx[3]] * 1.25).float(), atol=1e-3), "gain not applied"
    assert torch.equal(out[:, :, 5], act[:, :, 5]), "non-target neurons changed"
print("PASS: gain preserves dtype and touches only target neurons")

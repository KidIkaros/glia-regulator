"""Regression test for the lever's sign convention.

The bug this prevents: D = W_edited - W_base already carries the sign of the edit (the
edit REMOVES the loop direction, so D points against d_hat). Subtracting D from the
layer output applies the loop direction TWICE -- an anti-fix that still "works" in the
sense of changing the model, so it is invisible to a smoke test. It cost us two kernel
launches to find.

Test: a synthetic down_proj with two edited columns. The lever's output must equal the
edited layer's output exactly, and must NOT equal the sign-flipped version.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from lever import DeltaLever

torch.manual_seed(0)
OUT_F, IN_F = 6, 10
idx = [1, 5]

down = torch.nn.Linear(IN_F, OUT_F, bias=False)
W = down.weight.detach().clone()                       # [OUT_F, IN_F]

# a plausible edit: remove a direction from two columns (D points AGAINST d_hat)
d = torch.randn(OUT_F)
d = d / d.norm()
D = torch.stack([-6.0 * float(W[:, j] @ d) * d for j in idx], dim=1)   # [OUT_F, n]
We = W.clone()
for k, j in enumerate(idx):
    We[:, j] = We[:, j] + D[:, k]

act = torch.randn(2, 3, IN_F)

# reference: the edited layer's output
ref = act @ We.T
# base output
base = act @ W.T

alpha_state = {"a": 1.0}
lv = DeltaLever(down, idx, D, lambda: alpha_state["a"])
got = down(act)                                        # lever applied (alpha=1)
lv.remove()

err = (got - ref).abs().max().item()
assert err < 1e-5, f"lever output != edited output (max err {err:.2e})"

# the sign-flipped variant must NOT match
flipped = base - (act[:, :, idx] @ D.T)
assert not torch.allclose(got, flipped, atol=1e-5), \
    "lever output equals the sign-flipped (anti-fix) version - sign convention broken"

# alpha=0 must be a no-op
alpha_state["a"] = 0.0
lv2 = DeltaLever(down, idx, D, lambda: alpha_state["a"])
got0 = down(act)
lv2.remove()
assert torch.allclose(got0, base, atol=1e-6), "alpha=0 must be a no-op"

print("PASS: lever reproduces the edited layer exactly, is a no-op at alpha=0, "
      "and does not match the sign-flipped anti-fix")

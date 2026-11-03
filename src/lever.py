"""Lever: a learned, context-dependent strength over a KNOWN, VERIFIED fix.

Design (ADR-0002). The static edit that fixes Qwen3.5-0.8B's reasoning loop is known
to us exactly: the authors' released checkpoint differs from the base checkpoint in
64 down_proj columns of L15 and 64 of L19. We extract that delta D[L] (shape
[1024, 64]) at runtime and apply

    out[t] += alpha(t) * ( act[t, idx] @ D[L][:, idx]^T )

where alpha(t) is a scalar produced per token by the glial's slow-state reader.

Properties:
  * alpha = 0  -> the base model (the loop).
  * alpha = 1  -> numerically the authors' edited model (the fix), so the lever's
    authority is exactly the verified one (closed-thinking 11% -> 83%).
  * alpha in (0,1) -> a context-dependent interpolation. The glial's ONLY learned
    parameter is WHEN to apply the fix, not what the fix is.

This replaces the earlier activation-gate lever, which the diagnostics showed to have
almost no authority (KL 0.003 at full attenuation) and no context signal
(reader output 99.8% identical across documents).
"""
import numpy as np
import torch


class DeltaLever:
    """Hooks one down_proj: pre-hook reads the selected neurons' activations,
    post-hook subtracts alpha * act_selected @ D^T from the output."""

    def __init__(self, down_proj, idx, delta, alpha_fn):
        """down_proj: nn.Linear (out=1024, inter=3584); idx: neuron ids; delta:
        [1024, n] numpy/tensor of the edited-minus-base weight columns; alpha_fn:
        () -> float."""
        self.idx = torch.as_tensor(list(idx), dtype=torch.long)
        self.delta = torch.as_tensor(np.asarray(delta), dtype=torch.float32)   # [1024, n]
        self.alpha_fn = alpha_fn
        self.a = None
        self._h1 = down_proj.register_forward_pre_hook(self._pre)
        self._h2 = down_proj.register_forward_hook(self._post)

    def _pre(self, module, args):
        act = args[0]                                  # [B,T,inter]
        dev = act.device
        self.idx = self.idx.to(dev)
        self.delta = self.delta.to(dev)
        self.a = act[:, :, self.idx].float()            # [B,T,n]
        return None

    def _post(self, module, args, output):
        if self.a is None:
            return None
        alpha = self.alpha_fn()
        if alpha == 0.0:
            return None
        out = output[0] if isinstance(output, tuple) else output
        # D = W_edited - W_base already carries the edit's SIGN (the edit REMOVES the
        # loop direction, so D is negative along d_hat). Therefore the correction is
        # ADDED, not subtracted. Subtracting it applies the loop direction twice.
        corr = torch.einsum("btn,on->bto", self.a, self.delta)     # [B,T,1024]
        corr = corr * alpha
        new = (out.float() + corr).to(out.dtype)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new

    def remove(self):
        self._h1.remove()
        self._h2.remove()


def extract_edit(base_down, edited_down, k=64, thresh=0.05, max_cols=256):
    """Find the columns the authors edited and return (idx, delta[1024,n], norms).

    Selection is threshold-based (|delta column| > thresh) rather than top-k, so the
    check self-verifies: if the authors touched more than k columns we keep them all
    (up to max_cols) and the logit-equivalence result tells us whether we captured
    the edit exactly. Unedited columns are bit-identical between the two bf16
    checkpoints (both are roundings of the same values), so the threshold is safe.
    """
    Wb = base_down.weight.detach().float().cpu().numpy()
    We = edited_down.weight.detach().float().cpu().numpy()
    D = We - Wb
    norms = np.linalg.norm(D, axis=0)
    changed = np.where(norms > thresh)[0]
    if len(changed) == 0:                      # fall back to top-k if the threshold is off
        changed = np.sort(np.argsort(-norms)[:k])
    elif len(changed) > max_cols:
        changed = np.sort(np.argsort(-norms)[:max_cols])
    idx = np.sort(changed)
    return idx, D[:, idx], norms

"""The loop direction as a cheap detector of "this context is looping".

Why. Training a context-reading regulator (alpha(t)) needs LABELS: which contexts are
looping and which are not. We have 18 labelled problems, which is nowhere near enough
to train a reader that generalises - and labelling by rollout is ruinously expensive
(one looping problem costs ~410 s of T4 time at the 8192-token cap, so 1000 labels
would be ~114 GPU-hours).

But the loop is not a mood: it has a MECHANISM. The published fix removes ONE direction
from the MLP output of layers 15 and 19 (64 columns each, all parallel to a single
unit direction d). So the component of the MLP output along d is a signature of the
loop state, and it is available after ONE forward pass - no generation at all.

This module measures that signature:

    loop_component(t) = <down_proj_out[t], d_hat>      (per layer, per position)

and summarises it per context. If the projection separates the known loopers from the
known healthy problems, then we have a cheap labeller: a few thousand contexts can be
labelled in minutes, which is what makes Stage B (training the reader) possible at all.

The measurement is deliberately in the SAME place as the lever's correction: the lever
adds alpha * (act @ D^T) to down_proj's output, so the quantity the lever acts on is
exactly the component along d_hat.
"""
import numpy as np
import torch


def loop_direction(delta):
    """Unit direction of the loop, from the lever's delta columns.

    All changed columns of the published edit are parallel to one direction (the
    build_lever_from_checkpoints validation asserts this), so the first column defines
    it. `delta` is [out_features, n_columns].
    """
    columns = torch.as_tensor(np.asarray(delta), dtype=torch.float32)
    direction = columns[:, 0] / (columns[:, 0].norm() + 1e-9)
    return direction


class LoopSignalProbe:
    """Records the projection of down_proj's output onto the loop direction.

    Used as a forward hook on `model.model.layers[layer].mlp.down_proj`.
    """

    def __init__(self, down_proj, direction):
        self.direction = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
        self.projections = []
        self._handle = down_proj.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        direction = self.direction.to(out.device)
        proj = out.float() @ direction            # [B, T]
        self.projections.append(proj.detach().cpu())
        return None

    def reset(self):
        self.projections = []

    def stack(self):
        """Concatenated per-position projections across the forward passes so far."""
        if not self.projections:
            return None
        return torch.cat(self.projections, dim=1)

    def remove(self):
        self._handle.remove()


def summarise(projections, tail=64):
    """Summarise a context's loop signature.

    `tail` positions are used for the recent-state statistics: a context that has
    *entered* the loop has a strong, sustained component along d; a context that merely
    mentions something loop-adjacent has a weak or transient one.
    """
    if projections is None:
        return None
    recent = projections[0, -tail:] if projections.shape[1] > tail else projections[0]
    values = recent.float().numpy()
    return dict(
        mean=float(values.mean()),
        mean_abs=float(np.abs(values).mean()),
        max_abs=float(np.abs(values).max()),
        min=float(values.min()),
        max=float(values.max()),
        last=float(values[-1]),
        n_positions=int(projections.shape[1]),
    )


def score(summary, direction_sign=1.0):
    """A single signed score: positive means "the loop direction is active"."""
    if summary is None:
        return float("nan")
    return float(direction_sign * summary["mean_abs"])


def separation(loopers, healthy):
    """How well does the score separate the two groups?

    Returns the AUC-like rank statistic (the probability that a random looper scores
    above a random healthy control) plus the two group means, so the result can be
    read as a separation, not just a number.
    """
    import itertools

    a = [score(s) for s in loopers]
    b = [score(s) for s in healthy]
    wins = ties = 0
    for x, y in itertools.product(a, b):
        if x > y:
            wins += 1
        elif x == y:
            ties += 1
    total = len(a) * len(b)
    auc = (wins + 0.5 * ties) / total if total else float("nan")
    return dict(auc=auc, n_loopers=len(a), n_healthy=len(b),
                mean_looper=float(np.mean(a)) if a else float("nan"),
                mean_healthy=float(np.mean(b)) if b else float("nan"))

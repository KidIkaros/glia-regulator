"""Glial regulator: a learned, context-reading modulator over selected MLP neurons.

Mechanism (ANGN-style per-neuron modulation + PGC-style slow state):
  - the glial field reads mean-pooled residuals at a few early/mid layers, keeps a
    slowly-decaying state across token chunks, and emits per-neuron gains;
  - a gain g_j scales neuron j's write by (1 + g_j) by scaling down_proj's INPUT
    activation, which is exactly a multiplicative modulation of that neuron's output;
  - gains are bounded (tanh * sigma) and initialised at 0, so an untrained glial field
    is bit-identical to the base model.

Causality: the glial field at layer L may only read layers < L (enforced in install()).
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class GlialConfig:
    read_layers: tuple = (4, 8, 12)          # residuals the field reads (all < min(target))
    target_layers: tuple = (15, 19)          # MLP layers whose neurons it modulates
    neurons_per_layer: int = 64              # 0 = all neurons
    hidden: int = 256                        # glial field width
    sigma: float = 0.5                       # max |gain|
    temp: float = 2.0                        # head temperature (anti-saturation)
    decay: float = 0.9                       # slow state per chunk
    chunk: int = 64                          # tokens per slow-state step
    behaviors: tuple = ("loop", "syco")


class GlialField(nn.Module):
    def __init__(self, cfg: GlialConfig, d_model: int, inter: int, n_neurons: dict,
                 neuron_idx: dict):
        super().__init__()
        self.cfg = cfg
        self.reader = nn.Sequential(
            nn.Linear(d_model * len(cfg.read_layers), cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden))
        self.state_proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.heads = nn.ModuleDict({
            b: nn.Linear(cfg.hidden, sum(n_neurons[l] for l in cfg.target_layers))
            for b in cfg.behaviors})
        self.n_neurons = n_neurons
        self.neuron_idx = neuron_idx  # {layer: LongTensor of selected neurons}
        for h in self.heads.values():
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)  # start as identity
        self.active_heads = set(cfg.behaviors)   # per-behavior gating (interference test)
        self._state = None
        self._buf = {}
        self._gains = {}
        self.last_gain_abs = 0.0

    # ---- glial dynamics -------------------------------------------------
    def reset_state(self):
        self._state = None
        self._buf.clear()
        self._gains.clear()

    def read(self, layer: int, act: torch.Tensor):
        """Capture a layer's mean-pooled residual over the current chunk."""
        self._buf[layer] = act.detach().float().mean(dim=1)  # [B, d]

    def step(self, chunk_len: int):
        """Advance the slow state and compute per-layer, per-neuron gains."""
        cfg = self.cfg
        if len(self._buf) < len(cfg.read_layers):
            return
        x = torch.cat([self._buf[l] for l in cfg.read_layers], dim=-1)   # [B, k*d]
        h = self.reader(x)
        self._state = h if self._state is None else cfg.decay * self._state + h
        z = self.state_proj(torch.tanh(self._state)) / (self.cfg.hidden ** 0.5)
        acc = {}
        for b in cfg.behaviors:
            if b not in self.active_heads:
                continue
            g = torch.tanh(self.heads[b](z) / cfg.temp) * cfg.sigma    # [B, sum n]
            off = 0
            for l in cfg.target_layers:
                n = self.n_neurons[l]
                gi = g[:, off:off + n]
                acc[l] = gi if l not in acc else acc[l] + gi
                off += n
        self._gains = acc
        self.last_gain_abs = float(torch.cat([v.flatten() for v in acc.values()]).abs().mean()) \
            if acc else 0.0
        self._buf.clear()

    def gain_for(self, layer: int, act: torch.Tensor) -> torch.Tensor:
        """[B, T, inter] -> same, with selected neurons' writes scaled by (1 + g)."""
        g = self._gains.get(layer)
        if g is None:
            return act
        idx = self.neuron_idx[layer]
        scale = (1.0 + g.unsqueeze(1)).to(act.dtype)    # [B, 1, n], dtype-matched
        act = act.clone()
        act[:, :, idx] = act[:, :, idx] * scale
        return act


def _resolve(model, suffix: str):
    """Find the unique module whose qualified name ends with `suffix`."""
    hits = [(n, m) for n, m in model.named_modules() if n.endswith(suffix)]
    if len(hits) != 1:
        raise RuntimeError(f"{suffix!r} matched {len(hits)} modules: {[h[0] for h in hits][:5]}")
    return hits[0]


def install(model, cfg: GlialConfig, neuron_idx: dict, inter: int, d_model: int):
    """Attach hooks: read residuals at read_layers, modulate down_proj inputs at targets.

    Modules are resolved by name suffix so the same code works whether the text stack
    is model.layers or model.language_model.layers.
    """
    n_neurons = {l: len(neuron_idx[l]) for l in cfg.target_layers}
    field = GlialField(cfg, d_model, inter, n_neurons, neuron_idx)
    try:
        field = field.to(next(model.parameters()).device)
    except StopIteration:
        pass
    handles = []
    for l in cfg.read_layers:
        layer = _resolve(model, f".layers.{l}")[1]

        def mk_read(L):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                field.read(L, out)
                if L == max(cfg.read_layers):
                    field.step(out.shape[1])
            return hook
        handles.append(layer.register_forward_hook(mk_read(l)))
    for l in cfg.target_layers:
        down = _resolve(model, f".layers.{l}.mlp.down_proj")[1]

        def mk_mod(L):
            def hook(module, args):
                return (field.gain_for(L, args[0]),) + tuple(args[1:])
            return hook
        handles.append(down.register_forward_pre_hook(mk_mod(l)))
    return field, handles


if __name__ == "__main__":
    # shape test on a dummy stack: no model, no GPU
    torch.manual_seed(0)
    cfg = GlialConfig(read_layers=(1, 2), target_layers=(3, 4), neurons_per_layer=8)
    idx = {3: torch.arange(8), 4: torch.arange(8) + 100}
    n = {3: 8, 4: 8}
    f = GlialField(cfg, d_model=16, inter=32, n_neurons=n, neuron_idx=idx)
    f.read(1, torch.randn(1, 64, 16)); f.read(2, torch.randn(1, 64, 16))
    f.step(64)
    a = torch.randn(1, 64, 32)
    b = f.gain_for(3, a)
    assert b.shape == a.shape
    assert torch.allclose(b[:, :, idx[3]], a[:, :, idx[3]]), "gain is exactly zero at init"
    assert torch.equal(b[:, :, 20], a[:, :, 20]), "non-target neurons untouched"
    print("glial self-test OK: shapes match, init is identity, targets isolated")

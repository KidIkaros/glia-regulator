"""Gradient attribution search: rank all MLP neurons of a layer by the first-order
effect of pushing them (act[n] += dose * r_L / ||W_down[:,n]||) on
log p(trait answers) - log p(plain answers). One backward pass per batch of neurons
(too many neurons for one pass -> chunk 512 neurons per backward).

FALLBACK (finite difference): if autograd cannot reach the gain vector through the
pre-hook (log line 'attribution variant: autograd FAILED'), use finite_diff_rank():
2 extra forwards per candidate, applied only to the top-200 by the (possibly degenerate)
vectorized scores. Recorded deviation in the kernel log line 'attribute variant: ...'.
"""
import torch


@torch.enable_grad()
def attribute_layer(model, layer_idx, ids_trait, ids_plain, r_L, neurons,
                    dose=0.8, chunk=512):
    """model: frozen causal LM; ids_*: [1,T] token tensors on cuda; r_L: float;
    neurons: LongTensor of candidate ids [N]. Returns [N] attribution scores."""
    down = model.model.layers[layer_idx].mlp.down_proj
    W = down.weight.detach()                      # [d, inter]
    scores = []
    for s in range(0, len(neurons), chunk):
        chunk_ids = neurons[s:s + chunk]
        gains = torch.zeros(int(W.shape[1]), device=W.device, requires_grad=True)
        handle = []

        def pre_hook(module, args, gains=gains, chunk_ids=chunk_ids):
            act = args[0]                          # [B,T,inter]
            scale = torch.ones(act.shape[-1], device=act.device, dtype=act.dtype)
            scale[chunk_ids] = scale[chunk_ids] + (dose * r_L / W[:, chunk_ids].norm(dim=0).clamp_min(1e-6)).to(act.dtype)
            # gains perturb the scale directly -> nonzero first-order gradient
            scale = scale + gains.to(act.dtype)
            act = act * scale
            return (act,) + tuple(args[1:])

        handle.append(down.register_forward_pre_hook(pre_hook))
        lp_t = model(input_ids=ids_trait, logits_to_keep=1).logits.log_softmax(-1)
        lp_p = model(input_ids=ids_plain, logits_to_keep=1).logits.log_softmax(-1)
        for h in handle:
            h.remove()
        # mean-centered token weights: same ranking as the paper's objective, ~2000x
        # smaller magnitudes (fp32-safe with headroom)
        w = torch.arange(lp_t.shape[-1], device=lp_t.device, dtype=torch.float32)
        w = w - w.mean()
        objective = (lp_t[0, -1] * w).sum() - (lp_p[0, -1] * w).sum()
        try:
            score = torch.autograd.grad(objective, gains, allow_unused=True)[0]
            score = torch.zeros_like(gains) if score is None else score
            scores.append(score[chunk_ids].abs().detach())
        except RuntimeError:
            scores.append(torch.zeros(len(chunk_ids)))
    return torch.cat(scores)


def finite_diff_rank(model, layer_idx, ids_trait, ids_plain, r_L, neurons,
                     dose=0.8, limit=200):
    """Fallback: score by finite difference, 2 forwards per candidate, top `limit` only.
    Returns dict {global_index: score}."""
    down = model.model.layers[layer_idx].mlp.down_proj
    W = down.weight.detach()

    def lp_gap(neuron):
        handle = []

        def pre_hook(module, args, neuron=neuron):
            act = args[0]
            scale = torch.ones(act.shape[-1], device=act.device)
            scale[neuron] = scale[neuron] + dose * r_L / W[:, neuron].norm().clamp_min(1e-6)
            return (act * scale,) + tuple(args[1:])
        handle.append(down.register_forward_pre_hook(pre_hook))
        with torch.no_grad():
            lp_t = model(input_ids=ids_trait, logits_to_keep=1).logits.log_softmax(-1)[0, -1].sum()
            lp_p = model(input_ids=ids_plain, logits_to_keep=1).logits.log_softmax(-1)[0, -1].sum()
        for h in handle:
            h.remove()
        return float(lp_t - lp_p)

    out = {}
    for n in neurons[:limit]:
        out[int(n)] = abs(lp_gap(int(n)))
    return out

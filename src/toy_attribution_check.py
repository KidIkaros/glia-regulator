"""Exact replica of the kernel's attribution construction, on a toy model, CPU fp16.
Verifies the gradient is nonzero and structured. If it degenerates here too, the bug
is in the construction, not the device."""
import torch
import torch.nn as nn

torch.manual_seed(0)
dev = "cpu"
dtype = torch.float16   # kernel dtype

d_model, inter, vocab = 1024, 3584, 2000
up = nn.Linear(d_model, inter, dtype=dtype, device=dev)
down = nn.Linear(inter, d_model, bias=False, dtype=dtype, device=dev)   # weight [d, inter]
lm = nn.Linear(d_model, vocab, dtype=dtype, device=dev)
W = down.weight.detach()

x0 = torch.randn(1, 24, d_model, dtype=dtype, device=dev)
ids_t = torch.randint(0, vocab, (1, 24), device=dev)
ids_p = torch.randint(0, vocab, (1, 24), device=dev)


def fwd(gains, chunk_ids, dose=0.8, r_L=24.0):
    handle = []

    def pre_hook(module, args):
        act = args[0].clone()
        push = (dose * r_L /
                W[:, chunk_ids].norm(dim=0).clamp_min(1e-6)).to(act.dtype)  # [c]
        g = gains.to(act.dtype)[chunk_ids]                                  # [c]
        act[0, :, chunk_ids] = act[0, :, chunk_ids] + push[None, None, :] + g[None, None, :]
        return (act,) + tuple(args[1:])

    handle.append(down.register_forward_pre_hook(pre_hook))
    with torch.no_grad():
        base_act = up(x0)                        # frozen upstream
    act = base_act
    logits = lm(down(act))                       # [1, T, vocab] (logits_to_keep=1 in kernel)
    for h in handle:
        h.remove()
    lp_t = torch.log_softmax(logits[0, -1].float(), -1)   # kernel: logits[0,-1] is [vocab]
    lp_p = torch.log_softmax(logits[0, -1].float(), -1)
    # kernel: objective = (lp_t[0,-1] - lp_p[0,-1]).sum()  -- NOTE: identical lp here;
    # use different picks to emulate trait-vs-plain
    return (lp_t * torch.arange(vocab, dtype=torch.float32)).sum() - \
        (lp_p * torch.arange(vocab, dtype=torch.float32).flip(0)).sum()


gains = torch.zeros(inter, dtype=torch.float32, device=dev, requires_grad=True)
obj = fwd(gains, torch.arange(512))
g = torch.autograd.grad(obj, gains, allow_unused=True)[0]
print("grad None:", g is None)
if g is not None:
    print("norm:", round(float(g.norm()), 6), "| nonzero:", int((g.abs() > 0).sum()), "/", g.numel())
    print("grad[:8]:", [round(float(v), 6) for v in g[:8]])
    print("VERDICT:", "CONSTRUCTION OK" if float(g.norm()) > 0 else "GRADIENT DEGENERATE")

"""Stage A2: give the 128 neurons and the loop direction their identity.

Phases (one per invocation - the Colab CLI aborts a cell whose reply misses its
deadline, and every phase is resumable):

  fit_r / fit_s  the R- and S-lens fits for the BASE model, via neuron-lens's own
                 02_fit.py (which already supports Qwen3.5's DeltaNet layers and
                 resumes from its saved state, so re-invoking continues it);
  read           read the loop direction d through both lenses, read each of the 128
                 neurons' writes through the R-lens, and report which subset agrees
                 with d;
  fork           paired rollouts (base, alpha=1.25) on the labelled problems, storing
                 the TEXT, the close position, and the 128 neurons' activations per
                 token - the Stage-B reader's candidate inputs.

Environment: PHASE, DOCS (fit), PROBLEM_CHUNK (fork), RESULTS_DIR.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import (block_triton_kernels, eos_token_ids, greedy_rollout, load_model,
                     load_pairs, prompt_ids)
from lever import build_lever_from_checkpoints

block_triton_kernels()

MODEL_ID = "Qwen/Qwen3.5-0.8B"
EDITED_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
PHASE = os.environ.get("PHASE", "fit_r")
DOCS = os.environ.get("DOCS", "200")
PROBLEM_CHUNK = int(os.environ.get("PROBLEM_CHUNK", "2"))
ALPHA_STAR = float(os.environ.get("ALPHA_STAR", "1.25"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/content/glia_out/identity")
NL_ROOT = "/content/neuron-lens"
started = time.time()


def log(*parts):
    print(f"[identity {PHASE} {round(time.time()-started)}s]", *parts, flush=True)


def find_pairs():
    for candidate in (os.path.join(HERE, "s1_pairs.json"),
                      os.path.join(os.path.dirname(HERE), "s1_pairs.json")):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("s1_pairs.json not found next to the driver or one level up")


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def fit(lens):
    """The lens fit via neuron-lens's own script; it saves periodically and resumes."""
    os.makedirs(f"{NL_ROOT}/data/corpus", exist_ok=True)
    if not os.path.exists(f"{NL_ROOT}/data/corpus/dclm.jsonl"):
        subprocess.run([sys.executable, "scripts/01_lens_corpus.py", "--docs", "4"],
                       cwd=NL_ROOT, check=True)
    proc = subprocess.run([sys.executable, "scripts/02_fit.py", "--lens", lens,
                           "--model", MODEL_ID, "--docs", DOCS, "--device", "cuda"],
                          cwd=NL_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-1500:])
        print(proc.stderr[-1500:])
        raise SystemExit(f"fit {lens} failed with {proc.returncode}")
    log(f"fit {lens}: {proc.stdout[-400:]}")


def extract_direction(base, edited, layers=(15, 19)):
    """The lever spec (columns + delta) from the checkpoints actually in memory."""
    spec, diagnostics = build_lever_from_checkpoints(base, edited)
    for layer, (indices, delta) in spec.items():
        columns = torch.as_tensor(np.asarray(delta), dtype=torch.float32)
        direction = columns[:, 0] / (columns[:, 0].norm() + 1e-9)
        spec[layer] = (indices, delta, direction)
    return spec, diagnostics


def read_directions():
    """Read d through both lenses, and each neuron's write through the R-lens."""
    from pathlib import Path

    from neuron_lens.lenses import load_fits
    from neuron_lens.readout import read

    base, tok = load_model(MODEL_ID, device="cuda", dtype=torch.float32)
    edited, _ = load_model(EDITED_ID, device="cuda", dtype=torch.float32)
    spec, diagnostics = extract_direction(base, edited)
    w_u = base.lm_head.weight.detach().float()

    results = {"diagnostics": {str(k): v for k, v in diagnostics.items()}, "directions": {},
               "neurons": {}}
    for lens in ("R", "S"):
        fit_path = Path(NL_ROOT) / "data" / "fits"
        candidates = sorted(fit_path.glob(f"{lens.lower()}-lens*{MODEL_ID.split('/')[-1]}*.pt"))
        if not candidates:
            candidates = sorted(fit_path.glob("*.pt"))
        if not candidates:
            raise SystemExit(f"no {lens}-lens fit found in {fit_path} - run PHASE=fit_{lens.lower()} first")
        metas, acc = load_fits([candidates[-1]], lens, MODEL_ID)
        log(f"loaded {lens}-lens fit {candidates[-1].name}; layer keys: {list(acc.layers)[:3]}...")
        for layer, (indices, delta, direction) in spec.items():
            key = f"model.layers.{layer}"
            jacobian = acc.mean("out")  # [n_layers, d, d] or dict - handled below
            if isinstance(jacobian, dict):
                j = jacobian[key]
            else:
                j = jacobian[acc.layers[key]]
            d_norm = (direction / direction.norm()).to(j.device).float()
            scores = read(d_norm.unsqueeze(0), j, w_u.to(j.device))
            top = torch.topk(scores[0] if scores.dim() > 1 else scores, k=20)
            entry = dict(
                top_promoted=[(int(i), float(v)) for v, i in zip(top.values, top.indices)],
                bottom_suppressed=[(int(i), float(v)) for v, i in
                                   zip(*torch.topk(-scores[0] if scores.dim() > 1 else -scores, k=20))],
            )
            results["directions"][f"L{layer}_{lens}"] = entry
            if lens == "R":
                # each neuron's write through the same lens, and its agreement with d
                cols = torch.as_tensor(np.asarray(delta), dtype=torch.float32).to(j.device)
                neuron_scores = read(cols.T.to(j.device), j, w_u.to(j.device))
                if neuron_scores.dim() == 3:
                    neuron_scores = neuron_scores
                top_per_neuron = neuron_scores.argmax(dim=-1)
                d_scores = scores[0] if scores.dim() > 1 else scores
                cosines = []
                for n in range(neuron_scores.shape[0]):
                    v = neuron_scores[n].float()
                    cosines.append(float(torch.nn.functional.cosine_similarity(
                        v, d_scores, dim=0)))
                results["neurons"][f"L{layer}"] = dict(
                    n_columns=len(indices),
                    read_cosine_with_direction_mean=float(np.mean(cosines)),
                    read_cosine_with_direction=[round(c, 4) for c in cosines],
                    top_token_per_neuron=[int(t) for t in top_per_neuron],
                )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "reads.json")
    json.dump(results, open(out, "w"), indent=1)
    log("wrote", out)
    print(json.dumps(results["directions"], indent=1)[:2000])


class ActivationProbe:
    """Captures the 128 target neurons' activations at every step of a rollout."""

    def __init__(self, down_proj, idx):
        self.idx = torch.as_tensor(list(idx), dtype=torch.long)
        self.steps = []
        self._h = down_proj.register_forward_pre_hook(self._pre)

    def _pre(self, module, args):
        act = args[0]
        self.steps.append(act[:, -1, self.idx].float().detach().cpu())
        return None

    def take(self):
        out = torch.cat(self.steps, dim=0) if self.steps else None
        self.steps = []
        return out

    def remove(self):
        self._h.remove()


def fork():
    """Paired rollouts with text, close position and per-token activations."""
    base, tok = load_model(MODEL_ID, device="cuda", dtype=torch.float32)
    edited, _ = load_model(EDITED_ID, device="cuda", dtype=torch.float32)
    spec, _ = extract_direction(base, edited)
    probes = {layer: ActivationProbe(base.model.layers[layer].mlp.down_proj, spec[layer][0])
              for layer in spec}
    switch = {"a": 0.0}
    from lever import DeltaLever
    levers = [DeltaLever(base.model.layers[layer].mlp.down_proj, spec[layer][0],
                         spec[layer][1], lambda: switch["a"]) for layer in spec]
    stop_ids = eos_token_ids(tok)
    pairs = load_pairs(find_pairs())

    done = set()
    out_path = os.path.join(RESULTS_DIR, "fork.jsonl")
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                done.add(json.loads(line)["key"])
    os.makedirs(RESULTS_DIR, exist_ok=True)

    written = 0
    for alpha in (0.0, ALPHA_STAR):
        switch["a"] = alpha
        for problem in pairs:
            key = f"a{alpha:g}_{problem['id']}"
            if key in done:
                continue
            if written >= PROBLEM_CHUNK:
                log("chunk budget reached - run PHASE=fork again to continue")
                return
            encoded = prompt_ids(tok, problem["problem"]).to("cuda")
            tokens, info = greedy_rollout(base, encoded, 8192, stop_ids, tokenizer=tok,
                                          presence_penalty=1.5)
            acts = {f"L{layer}": probe.take() for layer, probe in probes.items()}
            row = dict(key=key, alpha=alpha, id=problem["id"],
                       base_looped=as_bool(problem.get("base_looped", False)),
                       closed=info["closed"], close_position=info.get("close_position"),
                       chars=info["chars"], n=info["n"], seconds=info["seconds"],
                       text=tok.decode(tokens, skip_special_tokens=True)[-6000:],
                       activation_shape={k: list(v.shape) if v is not None else None
                                          for k, v in acts.items()})
            np.savez_compressed(os.path.join(RESULTS_DIR, f"{key}.npz"),
                                **{k: v.half().numpy() for k, v in acts.items() if v is not None})
            with open(out_path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            written += 1
            log(f"recorded {key}: closed={info['closed']} n={info['n']} "
                f"close@{info.get('close_position')}")
    for probe in probes.values():
        probe.remove()


if __name__ == "__main__":
    if PHASE in ("fit_r", "fit_s"):
        fit(PHASE.split("_")[1].upper())
    elif PHASE == "read":
        read_directions()
    elif PHASE == "fork":
        fork()
    else:
        raise SystemExit(f"unknown PHASE {PHASE}")

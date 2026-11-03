# Qwen3.5-0.8B SAE release — findings (decoderesearch/qwen-3.5-saes)

Researched 2026-11-03. All claims verified live against HF/Neuronpedia APIs.

## 1. Repo contents

Repo: https://huggingface.co/decoderesearch/qwen-3.5-saes (sha 3a67187, lastModified 2026-08-09, library_name `saelens`, ~11.28 GB usedStorage). File list from `https://huggingface.co/api/models/decoderesearch/qwen-3.5-saes`:

- `qwen-3.5-0.8b/btk-mat-layer-{5,11,17}-k-100/{cfg.json, runner_cfg.json, sae_weights.safetensors}` — INSTRUCT (chat) model
- `qwen-3.5-0.8b-base/btk-mat-layer-{5,11,17}-k-100/{cfg.json, runner_cfg.json, sae_weights.safetensors}` — BASE model
- `qwen-3.5-4b[-base]/btk-mat-layer-{3,15,27}-k-100/{cfg.json, sae_weights.safetensors, sparsity.safetensors}` (+ a top-level `runner_cfg.json` per 4b variant)

Layers confirmed: 5, 11, 17 for both 0.8B variants (README lists exactly these; cfgs for 5 and 17 have `hook_name: model.layers.5` / `model.layers.17`).

**Variant disambiguation (resolved from cfg.json metadata):**
- `qwen-3.5-0.8b/*` → `model_name: "Qwen/Qwen3.5-0.8B"` (the chat/instruct model), trained on `lmsys/lmsys-chat-1m` with `use_chat_formatting: true`.
- `qwen-3.5-0.8b-base/*` → `model_name: "Qwen/Qwen3.5-0.8B-Base"`, trained on `chanind/pile-uncopyrighted-qwen-3.5-1024-abbrv-2B` (Pile Uncopyrighted), `use_chat_formatting: false`.

So for the host model Qwen/Qwen3.5-0.8B, use the **`qwen-3.5-0.8b/`** directories.

Training (both variants): 1B tokens, context 1024, SAELens 6.42.0, seed 0, lr 1e-4 constant, matryoshka BatchTopK, wandb project `modal-saes`.

## 2. Architecture

`btk-mat-layer-N-k-100` = **Matryoshka BatchTopK** SAE (paper: https://arxiv.org/abs/2503.17547), `k=100` (training-time batch top-k), layer N. From `runner_cfg.json` (`saes.layer_N`): `d_in: 1024, d_sae: 65536, k: 100, matryoshka_widths: [2048, 16384, 65536], decoder_init_norm: 0.1, rescale_acts_by_decoder_norm: true, apply_b_dec_to_input: true`. So 65k latents with inner prefixes of 2k and 16k (README confirms: "width 65k latents, and 2 inner Matryoshka prefixes of 2k latents and 16k").

**Attachment point:** `cfg.json` metadata `hook_name: "model.layers.N"` on `AutoModelForCausalLM` — a forward hook on the decoder layer module captures its **output hidden state = residual stream AFTER the whole layer block (attn + MLP), i.e. resid_post**. Neuronpedia corroborates: feature JSON `hookName: "blocks.11.model.layers.11"`. Note for the host model: layer 11 is a full-attention layer (of 3,7,11,15,19,23); layer 17 is a Gated DeltaNet linear-attention layer.

**Export cfg says `"architecture": "jumprelu"`** — Matryoshka BatchTopK exports as a JumpReLU-style inference SAE with a per-feature `threshold` vector. Verified by reading `sae_weights.safetensors` (instruct L11) via HTTP Range: `threshold` is **constant 0.0491 across all 65,536 features** (the single BatchTopK global threshold), confirming: at inference you gate `pre_acts > threshold`, and `k=100` is irrelevant.

Verified tensor header of `sae_weights.safetensors` (537,399,696 bytes ≈ 537 MB, all F32):
```
W_enc     [1024, 65536]   (encoder: pre_acts = (x - b_dec) @ W_enc)
W_dec     [65536, 1024]   (decoder — one row per feature = its write direction)
b_enc     [65536]         (observed range [-0.492, 0.005]; NOTE: with apply_b_dec_to_input, b_enc is unused at inference)
b_dec     [1024]          (rms 0.031; subtract from input)
threshold [65536]         (all 0.0491 for instruct L11)
```
(4b SAEs additionally ship `sparsity.safetensors`; 0.8B SAEs do not.)

## 3. Load-and-use recipe (plain torch/numpy)

```python
import numpy as np
from safetensors.numpy import load_file  # or safetensors.torch.load_file

st = load_file("sae_weights.safetensors")       # keys: W_enc, W_dec, b_enc, b_dec, threshold
W_enc, W_dec = st["W_enc"], st["W_dec"]         # [1024,65536], [65536,1024]
b_dec, thr   = st["b_dec"], st["threshold"]     # [1024], [65536]

# 1) encode: acts at residual stream AFTER layer N (resid_post of model.layers.N)
pre = (x - b_dec) @ W_enc                       # x: [..., 1024]
f   = np.maximum(pre, 0) * (pre > thr)          # feature activations (sparse, ~k=100 nonzero)

# 2) Matryoshka prefixes = first slices of f / rows of W_dec:
#    f[:2048] (2k), f[:16384] (16k), f (65k)

# 3) reconstruction / read direction: x_hat = f @ W_dec + b_dec

# 4) effective write direction of feature f_i on the residual stream:
d = W_dec[f_i]                                  # [1024]; unit-normalize: d /= np.linalg.norm(d)

# 5) alignment with MLP down_proj columns (per-neuron attribution).
#    Host MLP: mlp.down_proj.weight has shape [1024, 3584] (d_model x d_intermediate);
#    neuron j's write direction onto the residual stream is down_proj.weight[:, j].
Dn = down_proj_weight                            # [1024, 3584]
cos = (d @ Dn) / (np.linalg.norm(Dn, axis=0) + 1e-8)   # [3584] cosine of feature-write vs each neuron's write
gain = d @ Dn                                    # unnormalized projection coefficient (write magnitude through neuron j)
# candidates = top |cos| (or |gain|) neurons; gating neuron j's post-activation scales its
# write Dn[:, j], so cos/gain measure how much feature f_i's write flows through neuron j.
```

SAELens path: `from sae_lens import SAE; sae = SAE.from_pretrained("decoderesearch/qwen-3.5-saes", "qwen-3.5-0.8b/btk-mat-layer-11-k-100")` (README).

Caveat: because the SAE sits on resid_post (after attention AND MLP), a feature's write direction may include attention writes, not only MLP-ones; down_proj alignment quantifies the MLP-shared fraction.

## 4. Neuronpedia

- Model `qwen3.5-0.8b` = Qwen/Qwen3.5-0.8B, `instruct: true`, PUBLIC. Only **one** SAE set is uploaded: `11-res-matryoshka-65k` (layer 11, instruct SAE). `GET /api/feature/qwen3.5-0.8b/5-res-matryoshka-65k/0` and `.../17-...` → **404**; `qwen3.5-0.8b-base` → 404.
- `GET https://www.neuronpedia.org/api/feature/qwen3.5-0.8b/11-res-matryoshka-65k/{index}` returns JSON with: `maxActApprox`, `frac_nonzero`, `neuron_alignment_{indices,values,l1}` (top-aligned MLP neurons, e.g. feature 0 → neurons 565/384/489, cos 0.22), `correlated_neurons_{indices,pearson,l1}`, `pos_str/pos_values`, `neg_str/neg_values` (top/bottom logit-effect tokens), activation/logits histograms, `hookName: "blocks.11.model.layers.11"`.
- `POST https://www.neuronpedia.org/api/explanation/search-model` body `{"modelId": "qwen3.5-0.8b", "query": "code"}` works; returns auto-explanations (`explanationModelName: "gemini-2.5-flash-lite"`, typeName `np_acts-logits-general`) + full feature objects.
- Practical implication: Neuronpedia gives you labels/alignment only for **instruct layer 11**; layers 5/17 must be interpreted from raw weights yourself.

## 5. License / usage

- README (https://huggingface.co/decoderesearch/qwen-3.5-saes/blob/main/README.md): usage snippet (SAELens `from_pretrained`), Matryoshka BatchTopK description, citation "Chanin, David and Lin, Johnny — Decode Research, 2026", acknowledgement of Modal compute. **No license declared**: no LICENSE file in the repo file list, no `license` field/tag in the HF API response → treat as unspecified/all-rights-reserved by default; usage code is the README snippet only (no inference code shipped).
- Runner configs also credit dataset `chanind/pile-uncopyrighted-qwen-3.5-1024-abbrv-2B` (base) and `lmsys/lmsys-chat-1m` (instruct) — check those dataset licenses for training-data provenance.

## URLs cited
- Repo API/file list: https://huggingface.co/api/models/decoderesearch/qwen-3.5-saes
- Instruct L11 cfg: https://huggingface.co/decoderesearch/qwen-3.5-saes/resolve/main/qwen-3.5-0.8b/btk-mat-layer-11-k-100/cfg.json
- Instruct L11 runner_cfg: https://huggingface.co/decoderesearch/qwen-3.5-saes/resolve/main/qwen-3.5-0.8b/btk-mat-layer-11-k-100/runner_cfg.json
- Base L11 cfg: https://huggingface.co/decoderesearch/qwen-3.5-saes/resolve/main/qwen-3.5-0.8b-base/btk-mat-layer-11-k-100/cfg.json
- Weights: https://huggingface.co/decoderesearch/qwen-3.5-saes/resolve/main/qwen-3.5-0.8b/btk-mat-layer-11-k-100/sae_weights.safetensors (header + b_dec/threshold tensors read via Range requests)
- README: https://huggingface.co/decoderesearch/qwen-3.5-saes/blob/main/README.md
- Matryoshka SAE paper: https://arxiv.org/abs/2503.17547
- Neuronpedia feature: https://www.neuronpedia.org/api/feature/qwen3.5-0.8b/11-res-matryoshka-65k/0
- Neuronpedia explanation search: https://www.neuronpedia.org/api/explanation/search-model

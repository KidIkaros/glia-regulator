# Qwen Neuron Zoo / neuronary.org — methodology findings

## Key sources
- neuronary about/method: https://neuronary.org/about (llms.txt, agents.md also exist)
- neuronary source + runner: https://github.com/lyramakesmusic/neuronary (`scripts/run.py`, `schema/recipe.schema.json`)
- neuron-explorer (the brute-force sweep that found most neurons): https://github.com/lyramakesmusic/neuron-explorer (research_log.md = the methodology diary)
- Validation/eval with controls + blind judge (Kaggle T4!): https://github.com/shayansanjideh/neuron-lab
- Attribution-based search + judge harness: https://github.com/justinrhowell/neuron-steering + write-up https://justinrhowell.github.io/neuron-steering/
- Community registry (Qwen3-8B full sweep): https://github.com/jfonsecarivera (site: https://qwen-neurons-site.vercel.app/), repo makefunstuff/qwen-neurons, ananymishra/qwen-personality-studio
- Data on HF: https://huggingface.co/datasets/howellx/qwen3.5-neuron-steering
- Twitter authors: @_lyraaaa_ (Lyra, neuron-explorer/neuronary author), @Sauers_ (random-neuron controls), @treetowntree (Qwen Neuron Zoo recipes: ALL CAPS L17N1991+0.8, Cookie L24N3102+0.8, Male L19N351+0.8), @vashinthewild (threads)

## (1) How candidate neurons were found
NOT SAEs, NOT logit-lens, NOT activation statistics. Two proven pipelines:

A. **Brute-force sweep (neuron-explorer, how most neuronary neurons were found).**
- Push EVERY MLP neuron of Qwen3.5-4B (294,912) up and down, one rollout set each, keep everything.
- Doses in "rel" units: push size as a fraction of the layer's median residual norm (−0.7/+0.5/+0.8, plus +3 on a free prompt for a "wreckage readout"). Decision recorded in research_log.md: rel-units beat abs_max units because the coherent window is ~0.5–0.8× residual norm for every neuron.
- Prompts: {who, snake, orange} × doses + {free} × +10 (early design), greedy.
- Ranking signal: "words distinctive to a unit's rollouts vs all other units' rollouts" (word in ≥2 rollouts, absent from baselines, rarity-weighted). Random directions' top word appears in ≤4/12 rollouts; real neurons 6–7.
- Key negative results from the log: firing-contrast and logit-lens ranking do NOT find these neurons (hostile neuron ranks ~19,000th / ~14,000th); top-activating "trait" neurons are word-inserters, not persona switches.

B. **Gradient attribution search (justinrhowell scripts/search.py) — the practical replication path.**
- Rank every neuron by first-order effect of a push on log p(trait answers) − log p(plain answers) — ONE backward pass over a contrastive prompt pair (hostile neuron ranked 42nd/193,536 by this, vs ~7,000th by "impact").
- Steer the top 60 for real on 4 prompts, judge outputs; best 3 go to full eval with random controls. ~7–10 min per trait on a 4B.

C. For objective behaviors (your case), even simpler: score each neuron's steered rollouts with a programmatic metric (ALL-CAPS ratio, language-ID of reply) instead of an LLM judge — the neuron-lab follow-up did exactly this with word counts (25/40 → 1/40 alien words) plus Fisher-exact p-values.

## (2) Exact push protocol
Site of injection: input of layer L's `mlp.down_proj` (the `silu(gate) * up` activations). Forward PRE-hook:

```
act[n] += dose × median(‖residual leaving layer L‖) / ‖W_down[:, n]‖
```

- **r_L**: median over tokens of ‖hidden state leaving layer L‖ measured on a FIXED CALIBRATION TEXT (a few paragraphs of ordinary text, embedded in `scripts/run.py`, same as the explorer). Qwen3.5-4B: 24.8 at L22 via the calibration text vs 25.4 over 2M sweep tokens (stable).
- **Token 0 skipped** in the median: it is an attention sink with a much larger norm (a mean or including it inflates the scale). Caveat from the explorer log: if you then skip position 0 at runtime too, position shifts — measure on positions ≥1.
- **Applied at EVERY token**: prefill AND generation, all positions. Rationale: later tokens read earlier ones through attention; steering only the newest token lets the prompt pull the model back.
- `add` mode adds to the activation; `mul` mode multiplies it (neuronary schema supports both). neuron-lab also tested `set` (clamp) — similar hit rates to add.
- Dose 1 = push as large as the whole residual stream. Coherent window ≈ 0.5–0.8 per single neuron; doses ADD across neurons in a recipe. Dose 1 on Qwen3.5-4B L14 N8949 adds ~23 to a neuron whose usual value is ~0.
- Doses are comparable across layers/neurons within one model only; neuron indices are valid only for the exact weights/revision (bf16 vs q4 differs slightly).

## (3) Evaluation protocol (neuron-lab + neuron-steering, replicable)
- **Pre-registration**: fix the prompt set BEFORE the run ("40 fresh prompts fixed before the run"); pre-registered hypotheses that fail are reported as failures.
- **Arms**: steered neuron | unsteered baseline | k random same-layer neurons at the SAME dose (3 seeds) | sign-flipped dose | dose ladder (0.25/0.5/1.0). Each preset's example on cards follows a fixed rule, not hand-picking.
- **Fixed decoding**: greedy, thinking off via chat template, 200 new tokens (neuron-lab: greedy + 3 samples at T=0.7, N=32).
- **Blind judging**: replies shuffled, stripped of setting labels, judged by a fresh LLM session (Claude headless, or any OpenAI-compatible model — howell used Qwen3.6-35B-A3B on vLLM, temp 0) scoring trait 0–10 and coherence 0–10, plus yes/no trait & broken labels. blind_key.json kept separately.
- **Specificity bar**: a neuron is "specific" when it beats its random control by ≥3 judge points (howell) / when trait hits are significant vs random controls (neuron-lab: paired exact Fisher tests, e.g. 25/40 vs 23-30/40 → p ≤ 2e-7). "Output changed" is NOT evidence — ±1 changes greedy output 83–98% of the time; random pushes break text (16/24) without producing the trait (0/24).
- Known reproducibility caveats: fp32-on-T4 vs bf16, prompt wording, and dose calibration shift counts (lyra's sarcasm neuron: 6/32 alone but works inside her 5-neuron mix); judge is inconsistent run-to-run (two identical replies relabeled); mid-layer focus L5–L26 of 32; avoid first/last ~5 layers (surface strings; tiny residual norm makes same dose break the model).
- Side-effect stats seen on neuronary: ppl change on an eval suite + language-mixing rate (the Korean hanja recipe warns ppl +2948%, 92% language mixing), held-out prompt counts with objective counts (17→3 hanja chars over 4 prompts).

## (4) Model coverage / API
261 recipes total (API, 2026-11-03): Qwen3.5-4B 247, Qwen3.8-27B 10, Qwen3-8B 1, **Qwen3.5-0.8B exactly 1** ("Loses reality dream state", L0N550+0.76, by aseelund, 2026-10-28, score 0, 0 reproductions — layer 0, which the site's own tips say to avoid), gemma-3-4b-it 1, LFM3-1.7B 1. So: essentially 4B/27B territory; 0.8B is unexplored — your replication would be novel.
- **API**: yes. Reads need NO token: GET https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes (limit/offset), /recipes/:id, /neurons/:id. Writes need a Twitter-verified token (account menu → API & agents). MCP endpoint: .../api/mcp. Tools: search_recipes, get_recipe, publish_recipe, get_neuron, publish_interpretation, publish_readouts, report_reproduction...
- Agent docs: https://neuronary.org/agents.md, llms.txt, Claude skill at https://neuronary.org/skill/neuronary/SKILL.md
- GitHub: https://github.com/lyramakesmusic/neuronary (CC0 recipes, schema/recipe.schema.json, scripts/run.py validates+runs any recipe)

## (5) Code to reuse
| Repo | What | Fit for your 0.8B-on-T4 goal |
|---|---|---|
| lyramakesmusic/neuron-explorer | brute-force sweep infra (vLLM steering fork RhizoNymph/vllm, rollout harvest, ranking by distinctive words) | heavy; sweep was on CUDA-13 pods, but the design scales down |
| justinrhowell/neuron-steering | ~100-line hook (steering/core.py), steer.py (controls built in), judge.py, **search.py (attribution search)**, MIT | BEST fit: search.py + judge.py + quickstart.ipynb; supports --model any HF name |
| shayansanjideh/neuron-lab | Kaggle-T4 notebooks (fp32, 2×T4), playground, batch gallery runner, blind-judge labels, Fisher-exact follow-up, `neuron_agent.ipynb` (autonomous hunter with MMLU/GSM8K sanity checks) | BEST fit for Kaggle: kaggle kernels push workflow documented |
| lyramakesmusic/neuronary | run.py runner, recipe schema, API client clients/neuronary.py (stdlib) | publish/lookup recipes |
| Neuronpedia (neuronpedia.org) | feature/neuron dashboards with steering | interpret what a found neuron reads |

## Pseudocode — full identification + validation pipeline for Qwen3.5-0.8B (24L × 3584)
```python
# --- Calibration (once) ---
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=bf16)
calib_ids = tokenize(FIXED_CALIBRATION_TEXT)          # few ordinary paragraphs
hidden = model(calib_ids, output_hidden_states=True).hidden_states  # 25 entries
r_L = [median(norm(h[1:], dim=-1)) for h in hidden]   # skip token 0 (attention sink)

# --- Stage 1: attribution search per trait (one backward pass) ---
trait_prompts, plain_prompts = contrastive_pair(trait)  # e.g. all-caps answers vs normal
for batch of neurons:                                    # or full layer at once
    logp_t = logprob(model, trait_prompts)
    g = autograd.grad(logp_t - logp(model, plain_prompts), mlp_in_activations)
    score[n] ≈ g[n] * W_down[:, n] · (something like mean activation shift)  # first-order push effect
top60 = argsort(score)[-60:]

# --- Stage 2: real steering screen of top60 (hooks, every token) ---
def make_hook(L, n, dose):
    amt = dose * r_L[L] / norm(model.model.layers[L].mlp.down_proj.weight[:, n].float())
    return lambda(mod, args): (args[0].clone().index_add_(-1, n, amt*ones),) + args[1:]

for n in top60:
    replies = generate(model, 4 screening prompts, hook=make_hook(L?, n, 0.8),
                       greedy, thinking_off, max_new=200)   # hook active during prefill+decode
    keep_top3 = programmatic_score(replies, trait)          # ALL-CAPS ratio / langdetect drift

# --- Stage 3: validation of finalists (pre-registered) ---
prompts = fix_before_run(40 diverse prompts)               # never look before arm definitions
for n, L, dose in finalists:
    arms = {steered(n), baseline, random_same_layer × 3 seeds, sign_flip, dose 0.4}
    metrics = [caps_ratio(r) for r in arms]                # objective metric instead of LLM judge
    for each random control: p = fisher_exact(tally(steered_hits, control_hits))
    specificity = steered_mean - max(control_mean) >= 3    # or p after Holm/Bonferroni over k neurons
report(neuron, dose, p, side_effects={perplexity_delta, language_mixing_rate})
# publish: neuronary POST /recipes (CC0) or GitHub PR with recipes/<id>.json
```
Budget note: neuron-lab ran the 4B in fp32 on two Kaggle T4s; the 0.8B is ~5× smaller, so the whole top-60 screen + validation should fit a single T4 per session, likely <1 h per trait.

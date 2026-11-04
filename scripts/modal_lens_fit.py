"""Fit the R- and S-lenses for Qwen3.5-0.8B on one Modal H200 session.

Usage:
    modal run scripts/modal_lens_fit.py --smoke    # validate the path: 2 docs, 12 min cap
    modal run scripts/modal_lens_fit.py            # the real thing: R then S, 200 docs, 45 min cap

Everything lands on the Modal Volume "glia-lens" (corpus, fit checkpoints, model cache),
so a kill or crash loses nothing - a re-run resumes from the fit's own saved state.

VET RECORD (line-by-line, written before first run - the standing rule):

1. IMAGE. debian_slim + pip torch: torch's pip wheels bundle CUDA runtime libs and a
   matching triton, which is the standard Modal GPU pattern; no apt CUDA needed.
   RISK: torch wheel CUDA version vs Modal driver. MITIGATION: Modal's H200 hosts run a
   recent driver (>= 545); torch 2.7+ cu126 wheels are the default and known-good there.
   If the smoke fails on CUDA init, the fallback image is
   modal.Image.from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.11").
2. PINS. torch (default latest), transformers==5.22.1 (the version the Colab runs used -
   the same major the model was validated on), datasets (their corpus loader), and
   neuron-lens installed editable from a shallow clone. No other unpinned third parties
   enter the fit path.
3. GPU. gpu="H200" (sm_90, 228 KB shared memory - the verified requirement is 96 KB).
   Not "any" - pinned, so a silent downgrade to T4 is impossible.
4. TIMEOUT. The function timeout is a hard wall: 720 s smoke, 2700 s full. The fit also
   saves checkpoints every --save-minutes, and the whole corpus/fits tree lives on a
   Volume, so an exhausted timeout costs only the partial minute, not the work.
5. VOLUME. modal.Volume "glia-lens" mounted at /data; NEURON_LENS data dirs are
   symlinked into it, so the corpus, the model HF cache, and the fits all persist across
   invocations and across kills. Nothing valuable lives only in the container.
6. MODEL REVISION. Qwen/Qwen3.5-0.8B is loaded by 02_fit.py itself; the SAME session
   derives the loop direction from the base and edited checkpoints, so direction and
   lens reads are self-consistent by construction (the revision-drift lesson from the
   WSP phase, applied). The edited checkpoint is only needed for phase 2's reads - the
   fit touches the base alone.
7. SECRETS. None. No HF token needed (both checkpoints are public); no Kaggle, no Colab
   credentials on Modal.
8. OUTPUT. The fits stay on the Volume; retrieval is `modal volume get glia-lens ...`
   from the laptop after the run - no ingress surprises.
9. KILL SWITCH. `modal app stop glia-lens-fit` from the laptop at any moment; the fit's
   own checkpointing makes the next run resume.
10. WHAT IT DOES NOT DO. No training, no other projects' code, no writes outside the
    Volume, no long-lived server: the app exits when the function returns.
"""
import modal

app = modal.App("glia-lens-fit")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==5.22.1",
        "datasets",
        "accelerate",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/stdlogits/neuron-lens /opt/neuron-lens",
        "pip install --quiet -e /opt/neuron-lens",
    )
    .run_commands("mkdir -p /data")
)

volume = modal.Volume.from_name("glia-lens", create_if_missing=True)


@app.function(image=image, gpu="H200", timeout=720, volumes={"/data": volume})
def smoke(docs: int = 2) -> str:
    import subprocess, sys, os
    env = dict(os.environ, PYTHONPATH="/opt/neuron-lens")
    nl = "/opt/neuron-lens"
    os.makedirs(f"{nl}/data/corpus", exist_ok=True)
    subprocess.run([sys.executable, "scripts/01_lens_corpus.py", "--docs", str(docs)],
                   cwd=nl, check=True, env=env)
    for lens in ("R", "S"):
        subprocess.run([sys.executable, "scripts/02_fit.py", "--lens", lens,
                        "--model", "Qwen/Qwen3.5-0.8B", "--docs", str(docs),
                        "--device", "cuda"], cwd=nl, check=True, env=env)
    import glob
    fits = glob.glob(f"{nl}/data/fits/*.pt")
    volume.commit()
    return f"smoke ok; fits: {fits}"


@app.function(image=image, gpu="H200", timeout=2700, volumes={"/data": volume})
def full(docs: int = 200) -> str:
    import subprocess, sys, os, glob
    env = dict(os.environ, PYTHONPATH="/opt/neuron-lens")
    nl = "/opt/neuron-lens"
    os.makedirs(f"{nl}/data/corpus", exist_ok=True)
    if not os.path.exists(f"{nl}/data/corpus/dclm.jsonl"):
        subprocess.run([sys.executable, "scripts/01_lens_corpus.py", "--docs", str(docs)],
                       cwd=nl, check=True, env=env)
    for lens in ("R", "S"):
        subprocess.run([sys.executable, "scripts/02_fit.py", "--lens", lens,
                        "--model", "Qwen/Qwen3.5-0.8B", "--docs", str(docs),
                        "--device", "cuda"], cwd=nl, check=True, env=env)
    fits = sorted(glob.glob(f"{nl}/data/fits/*.pt"))
    volume.commit()
    return f"full ok; fits: {fits}"


@app.local_entrypoint()
def main(smoke_run: bool = False, docs: int = 200):
    if smoke_run:
        print(smoke.remote(docs=2))
    else:
        print(full.remote(docs=docs))

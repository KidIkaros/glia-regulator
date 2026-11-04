#!/usr/bin/env bash
# Loop the chunked sweep on one Colab session.
#   run_sweep_session.sh <session> "<alphas>" <results_path> <invocations>
set -u
SESSION="$1"; ALPHAS="$2"; RESULTS="$3"; N_INVOCATIONS="${4:-12}"
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
for i in $(seq 1 "$N_INVOCATIONS"); do
  echo "=== invocation $i (alphas=$ALPHAS) ==="
  cat colab_sweep_run.py | timeout 1500 colab exec -s "$SESSION" --timeout 1300 \
      --env ALPHAS="$ALPHAS" --env N_PROBLEMS=8 --env ROLLOUT=8192 \
      --env PRESENCE_PENALTY=1.5 --env MAX_PROBLEMS_PER_RUN=2 \
      --env BALANCED=1 --env RESULTS_PATH="$RESULTS" 2>&1 | tail -14
  echo "--- invocation $i done at $(date -u +%H:%M) ---"
done

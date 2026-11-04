#!/usr/bin/env bash
# Block until both sweep loops have finished, then collect and grade the rows.
#
#   ./scripts/wait_for_sweep.sh
#
# The Colab sweeps run as two background loops (one per session, one alpha arm set
# each). This waits for both to exit, then pulls the per-(alpha, problem) rows back
# over the authenticated CLI and grades them against the pre-registered gates.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
while pgrep -f "run_sweep_session.sh" > /dev/null; do sleep 30; done
echo "both sweep loops finished at $(date -u +%H:%M)"
python3 scripts/collect_sweep.py a8d8a8 d3b480

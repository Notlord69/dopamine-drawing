#!/usr/bin/env bash
# The final experiment end to end: week 10 (connectome vs random network seed 0, C6, no-brain
# controls), then week 10b (three more random networks and the wiring verdict).
#
#   scripts/run_final.sh            needs a CUDA GPU; ~100 min (week 10) + ~110 min (week 10b) on a T4
#
# Same as opening notebooks/12_week10_final.ipynb and 13_week10b_shuffles.ipynb and running all
# cells. Results go to $FLYARM_DATA_DIR/week10/ and week10b/ (default <repo>/data). Executed copies
# of the notebooks are written to outputs/. On Kaggle, run the notebooks directly instead (see README).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/outputs"
cd "$REPO_ROOT/notebooks"
for nb in 12_week10_final 13_week10b_shuffles; do
  echo "== $nb"
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output-dir "$REPO_ROOT/outputs" "$nb.ipynb"
done

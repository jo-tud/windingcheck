#!/usr/bin/env bash
# Release check: the tools/windingcheck README quickstart from a fresh clone.
#   VILLA_SF=<villa>/spiral-fitting VILLA_PY=<python that imports villa> \
#     bash experiments/2026-10-13-windingcheck-v0.1-release/run.sh [WORK_DIR]
# WORK_DIR (default ./windingcheck-release-check) is deleted and recreated; claims.json alone is 4 GB.
# Long job (~40 min, ~27 GB peak): see README.md. It never touches the checkout it is started from.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${1:-windingcheck-release-check}"
PY312="${PY312:-python3.12}"
export VILLA_PY="${VILLA_PY:?set VILLA_PY to a python that can import villa (it needs torch)}"
export VILLA_SF="${VILLA_SF:?set VILLA_SF to villa/spiral-fitting}"
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=-1
unset WINDINGCHECK_DATA PYTHONPATH

rm -rf "$WORK"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"  # absolute: the script cd-s into the clone below
log() { echo "[$(date -Is)] $*" | tee -a "$WORK/run.log"; }
timed() {  # name, command...
  log "start $1: ${*:2}"
  /usr/bin/time -v -o "$WORK/time_$1.txt" "${@:2}" > "$WORK/out_$1.txt" 2> "$WORK/err_$1.txt"
  log "end $1"
}

log "clone $REPO"
git clone -q --branch "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" "$REPO" "$WORK/clone"
git -C "$WORK/clone" log --oneline -1 | tee "$WORK/commit.txt"
cd "$WORK/clone"

# --- README quickstart, verbatim apart from the absolute python path ---
"$PY312" -m venv .venv
. .venv/bin/activate
pip install -q -e "tools/windingcheck[tests,plot]" -c tools/windingcheck/constraints.txt 2>&1 | tail -3 | tee -a "$WORK/run.log"
pip freeze > "$WORK/pip_freeze.txt"
python -m pytest -q tools/windingcheck 2>&1 | tail -2 | tee "$WORK/pytest.txt"

timed fetch windingcheck fetch --crossings
W=data/spiral_datasets/PHercParis4/winding_inference
timed score_raw windingcheck score --heldout --crossings $W --json "$WORK/score_raw.json"
timed generate windingcheck generate --claims --crossings $W --out "$WORK/claims.json"
timed score_claims windingcheck score --heldout "$WORK/claims.json" --plot "$WORK/drift.png" --json "$WORK/score_claims.json"
timed every50 windingcheck generate --claims --crossings $W --out "$WORK/claims_every50.json" --every 50

log "villa loader"
"$VILLA_PY" -c "import sys, json; sys.path.insert(0, sys.argv[1])
from point_collection import load_point_collection as L
c = L(sys.argv[2]); print(json.dumps([len(c), sum(len(v['points']) for v in c.values())]))" \
  "$VILLA_SF" "$WORK/claims_every50.json" > "$WORK/villa.txt" 2> "$WORK/villa_err.txt" || true
git -C "$(dirname "$VILLA_SF")" rev-parse HEAD > "$WORK/villa_commit.txt"

log "json keys"
for f in claims claims_every50; do
  # The JSON is one 4 GB line: split it first so grep streams (grep -o on it exhausted 27 GB).
  tr ',{' '\n\n' < "$WORK/$f.json" | grep -o '"[a-z_]*":' | awk '!seen[$0]++' | sort > "$WORK/keys_$f.txt"
done
find "$WORK" -path "$WORK/clone" -prune -o \( -name '*.png' -o -name '*.jpg' -o -name '*.tif*' \) -print > "$WORK/images.txt"
# .venv excluded: pip install writes matplotlib/scipy package icons there; they are not run outputs.
find "$WORK/clone" -path "$WORK/clone/.venv" -prune -o -newer "$WORK/commit.txt" \( -name '*.png' -o -name '*.jpg' -o -name '*.tif*' \) -print >> "$WORK/images.txt"

log "check"
python "$WORK/clone/experiments/2026-10-13-windingcheck-v0.1-release/check.py" "$WORK"
cp "$WORK/results.json" "$HERE/results.json"
log "done"

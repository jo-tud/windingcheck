#!/usr/bin/env bash
# Full-volume check: fresh clone of this branch, fresh venv, tests, then `windingcheck generate --claims` on
# all of Paris 4's ray crossings (CPU, 12 threads, no CT), then check_full.py.
#   bash experiments/2026-10-13-claims-noct-generator/full_volume.sh [WORK_DIR]
# WORK_DIR defaults to ./noct-full-work. CHECK_PY is a python with numpy, scipy and the
# windingcheck dev install (default: python3 on PATH).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${1:-noct-full-work}"
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
DATA="$REPO/data/spiral_datasets"
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=-1

rm -rf "$WORK/clone" "$WORK/venv"
mkdir -p "$WORK"
git clone -q --branch "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" "$REPO" "$WORK/clone"
git -C "$WORK/clone" log --oneline -1 | tee "$WORK/commit.txt"
uv venv -q "$WORK/venv" --python 3.12
VIRTUAL_ENV="$WORK/venv" uv pip install -q -e "$WORK/clone/tools/windingcheck[tests]"
"$WORK/venv/bin/python" -m pytest -q "$WORK/clone/tools/windingcheck/tests" 2>&1 | tail -2 | tee "$WORK/pytest.txt"

run() {  # name, extra args
  /usr/bin/time -v -o "$WORK/time_$1.txt" "$WORK/venv/bin/windingcheck" --data "$DATA" generate --claims \
    --crossings "$DATA/PHercParis4/winding_inference" --workers 12 --out "$WORK/claims_$1.json" "${@:2}" \
    | tee "$WORK/out_$1.json"
}
run full
run every50 --every 50
"${CHECK_PY:-python3}" "$HERE/check_full.py" "$WORK"

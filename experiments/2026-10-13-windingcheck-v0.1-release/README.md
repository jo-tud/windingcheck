# windingcheck v0.1: does the quickstart reproduce the numbers from a fresh clone?

**Question.** Does the quickstart in `tools/windingcheck/README.md`, run from a fresh clone in a
fresh venv with the pinned versions, reproduce the numbers in that README exactly? The pipeline
is deterministic, so every count must match, not just come close.

## What `run.sh` does

`run.sh WORK_DIR` runs the quickstart from nothing:

1. `git clone` the current branch into `WORK_DIR/clone`; record the commit.
2. `python3.12 -m venv`, `pip install -e "tools/windingcheck[tests,plot]" -c constraints.txt`;
   record `pip freeze`; `pytest tools/windingcheck`.
3. `windingcheck fetch --crossings` into the clone's own `data/` (real download, one connection).
4. `windingcheck score --heldout --crossings …`
5. `windingcheck generate --claims …` (whole volume), then `windingcheck score --heldout` on it.
6. `windingcheck generate --claims --every 50 …`; load that file with villa's
   `point_collection.load_point_collection` (villa `47111d1`, villa's own venv).
7. `check.py` compares everything with the expected values and writes `results.json`.

Each command runs under `/usr/bin/time -v`. CPU only, 12 threads.

I wrote down the expected counts and limits before the run:

- install with the constraints file gives exactly the pinned numpy/scipy, and 11 tests pass;
- every downloaded crossing array matches the sha256 in its shard manifest;
- raw crossings, held out: 2,489 tests, adjacent exact 889 of 1,159;
- generator, whole volume: τ = 0.74, 8,268,376 chains, 20,064,412 points, ≤ 30 min, ≤ 32 GB;
- generated, held out: 1,060 tests, 972 exact; adjacent exact 659 of 721; 389 of 974 adjacent
  pairs covered;
- the `--every 50` file: 165,568 chains and 402,054 points, read with the same counts by villa;
- the outputs hold only point-collection keys (coordinates, levels, tags) and the run writes no
  image other than `drift.png`.

## Run

    VILLA_SF=<villa>/spiral-fitting VILLA_PY=<python that imports villa> \
      bash experiments/2026-10-13-windingcheck-v0.1-release/run.sh WORK_DIR

About 40 min, peaking at 27 GB of memory (scoring the 4 GB full-volume file). Give it a 32 GB
memory limit.

## Result (`results.json`)

**Everything matched. The held-out counts are identical to the README's.**

| Step | Measured |
|---|---|
| install and tests | 11 passed; numpy 2.5.4, scipy 1.18.1 |
| fetch | 42 of 42 sha256 match; 73 s |
| raw crossings, held out | 2,489 tests; adjacent exact 889/1,159 (76.70%) |
| generator, whole volume | τ 0.74; 8,268,376 chains; 20,064,412 points; 4.6 min; 9.0 GB peak |
| generated, held out | 1,060 tests, 972 exact; adjacent exact 659/721 (91.40%); covered 389/974 (39.94%) |
| villa loader | 165,568 chains / 402,054 points written and read by villa `47111d1` |
| output contents | 14 key names, all point-collection keys; no image other than `drift.png` |

Whole run: 38 min wall, 15.8 min CPU, 26.9 GB memory peak (the held-out score of the 4 GB file,
23.7 GB). The generator took 4.6 min here against 14.4 min in the README's cost table, which was
measured on a busier machine; the README keeps the slower figure.

**Harness fixes.** The first attempt ran every windingcheck step, then the harness itself failed:
`grep -o` on the one-line 4 GB `claims.json` ran out of memory. The image search also counted 25
matplotlib/scipy package icons that `pip install` wrote into `clone/.venv`. I fixed both in
`run.sh` (key names are now read with a streaming pipe; the image search skips `.venv`), left the
expected values unchanged and ran the whole check again; the table above is that second run.
Later `run.sh` lost its machine-specific paths: `WORK_DIR` defaults to a relative directory and
`VILLA_PY`/`VILLA_SF` must be set. No step and no recorded number changed.

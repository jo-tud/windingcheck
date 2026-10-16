# The v2 claim generator without CT, as a shipped command

**Question.** v2 (`../2026-10-12-claim-confidence-v2/`) is my best winding generator on human
time (0.70–1.40 h saved of the 3.5–7.0 h Paris 4 relative-chain budget at 3× fix cost), and its
CT features added nothing (AUC 0.7141 with CT, 0.7143 without). Can it ship as
`windingcheck generate --claims` with **no CT read at all**, on the whole volume, on CPU, without
losing its held-out numbers?

Dropping CT removes a ~163 GB read for the full Paris 4 volume, so the tool runs on a laptop
from the 170 MB crossing files alone.

## Method

- **Model.** Same claim rows as v2 (every even-chain claim on the region rays), the first 7 claim
  features only (`log_k`, `v1_mean`, `v1_min`, `log_s_min`, `log_s_max`, `edge_max`, `log_r`),
  same logistic fit (L2 = 1e-2). Saved as `models/paris4_claims_v2_noct.json`.
- **τ.** Same rule as v2: lowest τ on the grid 0.50…0.99 with even-chain adjacent pairs ≥ 95%
  exact and ≥ 100 adjacent tests. Even chains only.
- **Region run.** Same 17,060 region rays as v2; generate; re-score against the odd (held-out)
  human chains with `windingcheck`. This is the comparison against v2.
- **Full-volume run.** `windingcheck generate --claims` on all 22.7 M crossings, from a fresh
  `git clone` in a fresh venv, with `--data` pointing at the already-fetched crossing files. The
  generator works in blocks of rays, so memory stays bounded; the greedy rule is v2's. CPU,
  ≤ 12 threads. Its output, re-scored against the odd chains, must equal the region run, because
  only region rays can reach a click (match tolerance ≤ 26 voxels < 60-voxel region).

Before fitting the model I fixed what "without losing" means: held-out precision, coverage and
human time each within 0.5 points of v2 (v2: 90.86% adjacent exact, 91.33% all pairs, 7.02% of
human pairs, 40.97% of adjacent pairs, *f_ok* − 3·*f_bad* = 0.2002), the full volume in
≤ 30 min and ≤ 32 GB on CPU, and the output loading in villa.

## Run

    python experiments/2026-10-13-claims-noct-generator/run.py          # region run
    bash experiments/2026-10-13-claims-noct-generator/full_volume.sh    # fresh clone, full volume

## Result (`results_region.json`, `results_full.json`)

**v2 without CT ships as `windingcheck generate --claims`: the whole Paris 4 volume in 14.4 min
on CPU at 8.95 GB, no CT read. Precision and human time hold; coverage drops by 1.0 point, more
than the half point I allowed.**

| Held out (odd chains) | v2 with CT | **v2 without CT (shipped)** |
|---|---|---|
| adjacent pairs exact | 90.86% | **91.40%** |
| all pairs exact | 91.33% | **91.70%** |
| human pairs covered | 7.02% | 6.68% |
| adjacent pairs covered | 40.97% | **39.94%** (allowed down to 40.47%) |
| *f_ok* − 3·*f_bad* | 0.200 | **0.206 (0.72–1.45 h saved at 3× fix cost)** |

- **Why coverage dropped.** The no-CT model ranks held-out claims like v2 (AUC 0.7143), but on
  the even chains τ = 0.73 reached 94.97% adjacent exact, 0.03 points short of the 95% rule, so
  τ became 0.74. The stricter τ trades about 1 point of adjacent coverage for 0.5 points of
  precision. I did not re-pick τ.
- **Full volume:** all 22,706,484 crossings in 862 s wall, peak 8.95 GB, 107% CPU; 8,268,376
  chains, 20,064,412 points (4.0 GB JSON).
- **Full = region:** the full-volume output on the held-out chains gives 1,060 tests, 972 exact,
  the same as the region run.
- **Loads in the tools:** the `--every 50` file has 402,054 points; villa's loader reads 165,568
  collections / 402,054 points, `windingcheck` 402,054.
- **Clean checkout:** in the fresh clone and venv the tests pass (10) and the full-volume run is
  that clone's command.

**Human time** (net hours of the 3.5–7.0 h Paris 4 relative-chain budget, held-out adjacent
pairs; 35.1% of the 974 human adjacent pairs right, 4.8% wrong):

| Fix cost | Raw crossings | v1 | v2 (CT) | **v2 no CT, shipped** |
|---|---|---|---|---|
| 1× | 0.93–1.86 h | 0.79–1.59 h | 1.07–2.13 h | **1.06–2.12 h** |
| 3× | 0.03–0.07 h | 0.52–1.04 h | 0.70–1.40 h | **0.72–1.45 h** |
| 5× | −0.87 to −1.73 h | 0.25–0.50 h | 0.33–0.67 h | **0.38–0.77 h** |

## Notes

- `check_full.py` crashed on its first call on a bookkeeping bug (a tuple unpacking of the
  results dict) before it computed anything. I fixed it and re-ran it on the same outputs; the
  generator outputs were not re-made.
- `save_pcl` now streams one collection at a time; the old writer built the whole file in memory
  with a per-chain scan, which does not finish on 8 M chains. A test checks the bytes are
  identical to the old writer.
- The full-volume file is 4.0 GB; for VC3D use `--every N` or crop. Scoring the 4 GB file once
  peaked at 23.7 GB.
- One scroll only: Paris 4 is the only scroll with both public crossings and human winding
  labels.

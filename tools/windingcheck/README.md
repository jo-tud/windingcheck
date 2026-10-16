# windingcheck v0.1

Score automatic winding constraints against human winding annotations, and generate new
relative-winding constraints from the organisers' neural ray crossings. CPU only, no CT read.

Any winding source becomes **chains**: points with a relative sheet count. That covers VC3D
point-collection files (`relative_windings.json`, `same_windings.json`, anything a generator
writes in that format), the organisers' neural `winding_inference` ray crossings, and spiral
priors. The truth is the human relative-winding chains of PHerc. Paris 4, the only scroll with
public human winding labels.

## Quickstart (fresh clone)

Needs Python 3.12 or newer with `venv` (tested on 3.12, Linux x86_64) and `git`. About 170 MB
download over one connection, no GPU. The download took 73 s on one run and 4.7 min on another;
it depends on your line.

    git clone <this repository> && cd <repository>
    python3 -m venv .venv && . .venv/bin/activate
    pip install -e "tools/windingcheck[plot]" -c tools/windingcheck/constraints.txt

    windingcheck fetch --crossings          # public Paris 4 annotations + ray crossings, ~170 MB
    W=data/spiral_datasets/PHercParis4/winding_inference

    # 1. score the organisers' raw crossings on the held-out human chains
    windingcheck score --heldout --crossings $W

    # 2+3. generate relative-winding constraints for the whole scroll, score all of them
    #      held out, and write a file small enough to open in VC3D (every 50th ray).
    #      Peak 7.0 GB, about 6 min: fits a 16 GB machine.
    windingcheck generate --claims --crossings $W --out claims_every50.json --every 50 \
      --score --heldout --plot drift.png --json score.json

`--score` scores every generated chain in memory before `--every` thins the output, so the
score is the full-set score, not the score of the thinned file. Do not score
`claims_every50.json` instead: it gives 12 adjacent tests, not 721.

**Full run (needs about 24 GB).** To keep all 20 M points on disk as well, write the full file
and score it from disk. Same numbers, but `score` holds the whole 4.0 GB file in memory:

    windingcheck generate --claims --crossings $W --out claims.json
    windingcheck score --heldout claims.json --plot drift.png

Files land in `data/spiral_datasets/` inside the clone; set `WINDINGCHECK_DATA` or `--data` to
put them elsewhere. Run the tests with `pip install -e "tools/windingcheck[tests]" -c
tools/windingcheck/constraints.txt && pytest tools/windingcheck`.

## What you should see

`--heldout` scores against the odd-numbered human chains only. The shipped model was fitted on
the even-numbered chains and never saw the odd ones during fitting. The odd chains did shape the
design and the choice of version; see Limits.

| Step | Key output | Value |
|---|---|---|
| 1. raw crossings | `adjacent_exact` / `adjacent_tests`, `human_adjacent_covered` | **76.7%** of 1,159 adjacent tests, at 52.3% coverage |
| 2+3. `score.json` | `adjacent_exact` / `adjacent_tests`, `human_adjacent_covered` | **91.4%** of 721 adjacent tests, at 39.9% of 974 adjacent human pairs |
| 2+3. `score.json` | `exact` / `tests`, `human_pairs_covered` | 91.7% of 1,060 tests, at 6.7% of all human pairs |
| 2+3. printed summary | `tau`, `chains_generated`, `points_generated` (all chains, scored) | 0.74, 8,268,376 chains, 20,064,412 points |
| 2+3. printed summary | `chains`, `points` (written, every 50th ray) | 165,568 chains, 402,054 points (79 MB) |
| full run `generate --claims` | `chains`, `points` | 8,268,376 chains, 20,064,412 points (4.0 GB file) |
| full run `score --heldout claims.json` | as `score.json` | identical in every field except `seconds` |

The two precisions are on different test sets, so they are not a gain of 14.7 points: the
generator never changes a sheet count, it only cuts rays into shorter chains, and on its own 721
tests the raw crossings also score 91.3%. Compare at equal coverage instead (next section).

An *adjacent test* is a pair of neighbouring human clicks (one sheet apart) that both match
points of one generated chain. *Exact* means the generated sheet count equals the human one.

## What the generator adds: precision at equal coverage

The fair baseline is a **random cut**: drop the same 37.7% of ray steps the generator drops, at
random, with no model. Held out (odd chains), 974 human adjacent pairs; brackets are 95% CIs from
a bootstrap over human chains. Details: `experiments/2026-10-13-windingcheck-baselines/`.

| Source | Adjacent exact, per test | Per human pair | Coverage of adjacent pairs | Δw ≥ 2 exact |
|---|---|---|---|---|
| raw crossings | 76.7% of 1,159 | 75.4% of 509 | 52.3% | 74.9% of 1,330 |
| random cut, 10 seeds | 84.2% (seeds 82.2–86.1) | 81.7% | 39.2% (38.4–39.9) | 83.3% |
| **`generate --claims`** | **91.4% of 721 [87.1–94.9]** | **87.9% of 389 [82.1–92.7]** | **39.9%** | **92.3% of 339 [88.2–95.8]** |
| generator − random cut | **+7.2 points [+3.7, +11.2]** | **+6.2 [+3.1, +9.7]** | +0.7 [−1.5, +2.7] | **+9.0 [+3.7, +15.8]** |
| one step (every ray step its own chain) | 98.8% of 905 | 97.5% of 434 | 44.6% | 0 of 5 |

- About half of the 14.7 points between raw and 91.4% come from cutting at all; the model's
  choice of where to cut gives the other half.
- Per test counts are correlated: 721 tests sit on 389 human pairs. Quote the per-pair 87.9%
  with its CI next to the per-test 91.4%.
- **The one-step row is a blind spot of the metric, not a better source.** A 2-point chain can
  only be wrong by sign; a spurious crossing in the middle of a gap matches no click, so its wrong
  step is never tested. Splitting my own chains into single steps lifts them to 98.5% with no new
  information, and a spiral fitter would join the steps straight back into the raw rays. 56% of
  the generator's chains are 2-point, so its score carries some of this too.

**Cost** (measured on a 24-core server, CPU only, 2026-10-13/14):

| Command | Wall clock | Peak RAM |
|---|---|---|
| `score --heldout --crossings` | 7 s | 2.4 GB |
| `generate --claims --every 50 --score --heldout --plot` (step 2+3) | 6.3 min | 7.0 GB (run under a 14 GB limit, no swap) |
| `generate --claims` (all 22.7 M crossings) | 14.4 min | 9.0 GB |
| `generate --claims --every 50` | 8.1 min | 7.0 GB |
| `score --heldout claims.json` (4.0 GB file) | — | about 24 GB (the whole file is held in memory) |

**Human time.** Against the 3.5–7.0 h it would take to click Paris 4's relative-winding chains
by hand, the generated chains save **0.72–1.45 h** if fixing a wrong constraint costs 3× making
one. Random cuts at the same coverage save 0.37–0.74 h and raw crossings 0.03–0.07 h, so the
model itself is worth **about 0.35–0.71 h**. 35.1% of the held-out adjacent human pairs come out
right, 4.8% wrong. The formula counts only errors the scorer can see, which is why the one-step
rule "saves" 1.40–2.80 h. Details: `experiments/2026-10-13-claims-noct-generator/` and
`experiments/2026-10-13-windingcheck-baselines/`.

## Output format

`generate` writes a **VC3D point-collection JSON** (`vc_pointcollections_json_version` 1): the
format VC3D writes for point annotations, and the format of the organisers' own
`relative_windings.json`. Each generated chain is one collection; each point has `p` (x, y, z in
full-resolution scan voxels) and `wind_a` (its relative sheet count); the collection `tags` hold
`source`, `confidence` and `tau`.

This is a point-annotation format, not a surface (tifxyz) or a volume (OME-Zarr), because the
output is winding constraints, not a mesh. It is the input villa's spiral fitting takes as
`relative_windings.json` (`fit_spiral.py`, `input_use_pcl_relative`). I checked that villa's own
loader, `spiral-fitting/point_collection.load_point_collection` at villa commit `47111d1`,
reads the `--every 50` file with the same point count as written (165,568 collections, 402,054
points). I have not tested loading it in the VC3D GUI.

## How it works

**Matching.** A candidate point counts as on the same sheet as a human click when it is within
`tol_n` of the click along the human chain's direction (≈ the sheet normal) and within
`tol_t` = 25 voxels along the sheet. `tol_n` = 0.35 × the local sheet spacing, clipped to
[1.5, 4] voxels, so a point cannot match two neighbouring sheets. Each pair of points from one
candidate chain that lands on two clicks of one human chain is a **test**.

**Joining.** Before scoring, candidate chains that share a point (two points of different
chains closer than `--join-tol`, default 0.001 voxels) are joined into one chain, as a spiral
fitter would compose them. So cutting a chain into pieces that share their end points cannot
change its score. Without this, a 2-point chain can only fail on a reversed sign: a false crossing
in the middle of a gap matches no click, and the wrong step into it is never tested. Pieces whose
shared points disagree are left unjoined and counted (`join.conflict_chains`). `--no-join` gives
the scorer without joining.

**Numbers out.** Tests; share of human pairs covered; exact; within ±1; sign errors (counted the
wrong way); adjacent exact; all of these split by the true sheet count between the two points.

**Generator (`generate --claims`, v2, recommended).** Scores each *claim* (two crossings on one
ray and the sheet count between them) from features of the crossings alone: the sheet count, the
v1 step confidences along the claim, its shortest and longest step, its position on the ray, and
the radius. It walks each ray greedily: a chain grows while both the claim from its start and the
last step reach τ. No CT and no labels are needed at run time. Model:
`models/paris4_claims_v2_noct.json`.

`generate` without `--claims` is the earlier v1 step filter (held out: 89.9% exact against 75.7%
raw, at half the coverage), kept for comparison. `windingcheck prior` scores plain spiral-prior
baselines.

## Related work

Other people built winding-constraint generators and a scorer for them before me. Their numbers
use their own pairs and definitions, so they are not directly comparable with mine. I have not
scored their outputs with `windingcheck`, nor mine with their benchmark.

| Project | Author | What it does | Headline, as its authors report it |
|---|---|---|---|
| [winding-ruler](https://github.com/pscamillo/winding-ruler) | pscamillo | Measures winding evidence for villa's spiral fit: where human annotations matter, why generators fail | A calibrated estimator recovers human relative-winding labels at 93% on adjacent (Δw = 1) pairs of Paris 4, "yet three progressively better constraint generators all degrade the fit" |
| [winding-sync](https://github.com/abundantjoe/winding-sync) | Joseph Balmaceda | Relative winding constraints from raw CT (structure-tensor lamina counting), reconciled by L1 integer synchronisation | Per probe ~38% exact, ~90% within one winding; stays ≥ 95% accurate up to 15% injected gross error (synthetic); absolute counts not calibrated |
| [constraint-gauge](https://github.com/pscamillo/constraint-gauge) | pscamillo | Benchmark that scores any winding-constraint generator against Paris 4 human annotations (8,156 within-collection pairs, Δw 1–6) and against nine human-verified meshes, with criteria fixed before measurement | Exact agreement on Δw = 1 pairs, mean absolute residual, confidence calibration, coverage; winding-sync scores 0.050 exact against verified meshes on one slice, where its own metric says 0.670. All three figures: constraint-gauge `README.md` at commit `a72c423` ("How ground truth is built", "Internal agreement is not external accuracy") |

How mine differs. My `adjacent_exact` is also exact agreement on Δw = 1, but counted per *test*
(one human pair can be hit by several generated chains, about 1.85 on average), only on the
odd-numbered human chains, and only where a generated chain matches both clicks. constraint-gauge
is the closest prior work: it is a scorer of the same kind on the same scroll. winding-ruler's
finding is the reason the first limit below matters: higher constraint accuracy has not, so far,
meant a better spiral fit.

## Limits

- **No spiral-fit gain.** I fitted villa's `fit_spiral.py` four times on PHerc. Paris 4,
  z 8400-9400 (12,000 steps each): no constraints, `generate --claims`, a random cut at equal
  steps, and the raw crossings. On the 21 held-out odd chains the claims fit scored 6.9 % slips
  per wrap against 6.1 % without constraints (paired difference +2.1 points, 95 % CI -4.7 to
  +9.4): no gain. The raw crossings scored best (3.4 %). This repeats winding-ruler's finding that
  a more precise constraint set is not a better fit input. Treat `generate --claims` as a filter
  with a measured precision, not as a fit improvement.
- **The held-out chains were not fully held out.** The model's coefficients and τ were fitted on
  the even chains only. But I looked at the odd (held-out) chains in a follow-up analysis of v1
  (`experiments/2026-10-12-confident-crossings/posthoc.py`) that led to the v2 claim-level design,
  and their scores decided between v1, v2 and the no-CT version. So the held-out numbers are not
  from data that was never looked at. The models are logistic fits on 7 features and the versions
  differ by about 1 point, so I expect the bias to be small, but I have not measured it.
- **About 1.1% of tests have the wrong sign.** 12 of the 1,060 held-out tests of `generate
  --claims` count the sheets the wrong way round: the chain runs against the human winding
  direction, because `wind_a` follows the ray's order. A spiral fitter would read these as
  reversed constraints. v0.1 does not fix this.
- **Unknown training set of the crossing model.** The crossings come from villa's
  `winding_model_3d_large_11`. I do not know what it was trained on. If Paris 4's human windings
  were in its training data, the raw 76.7% baseline, and everything built on the crossings, is
  in-sample for that model, and a new scroll will score lower.
- **One scroll.** Only PHerc. Paris 4 has both public ray crossings and public human winding
  labels, so every accuracy number here is from that one scroll. The model was fitted on Paris 4
  (even chains). On any other scroll it is untested.
- **Dropping CT cost a little coverage.** The shipped no-CT model covers 39.94% of held-out
  adjacent human pairs against 40.97% for the version that reads CT. I had allowed half a point
  of loss before the run; it lost 1.0 point, because τ moved one grid step. Precision and human
  time did not get worse.
- **Dropping steps can still raise a score.** Joining stops a split from hiding errors, but a
  candidate that drops every other step (disjoint 2-point chains, no shared points) is still
  scored only on the steps it kept, and a wrong step next to a dropped one is not tested. Compare
  generators at equal coverage, not on precision alone.
- **Not calibrated.** The `confidence` tag ranks chains; it is not a probability.
- **60% of adjacent pairs are not covered.** Those still need a human or another source.
- **Are the human chains right where the generator counts an extra sheet?** I tried to settle
  this by eye on 60 CT crops (4.8 µm, one axial slice each). It did not work: on the 20 control
  crops, where ray and human agree, I read "no sheet in between" on only 12, and 26 of the 60
  crops were unreadable. So there is no answer yet.
- The full-volume output is 4.0 GB. Use `--every N` for VC3D.
- `windingcheck.ct` (CT sampling for the research version) is not used by any command and needs
  the separate `inkcheck` package.

## Licence, data and credits

Code: MIT (`LICENSE`). Data: CC BY-NC 4.0 and EduceLab-Scrolls terms, fetched at run time, never
shipped; read `DATA_NOTICE.md` for the required citation. Outputs contain coordinates and counts
only, no CT and no text. Credits: `CREDITS.md`.

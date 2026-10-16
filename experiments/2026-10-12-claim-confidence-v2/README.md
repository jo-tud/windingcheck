# Claim-level confidence for ray-crossing winding constraints (generator v2)

**Question.** v1 (`../2026-10-12-confident-crossings/`) scored each *step* of a ray and
multiplied steps into a pair confidence. It lifted the organisers' crossings from 75.7% to 89.9%
exact on held-out human chains, but its confidence was not calibrated, and the follow-up analysis
showed why: the errors are extra crossings that only show across several steps, correlated along
a ray, so the product rule is wrong. v2 asks two things:

1. Does a model of the **claim itself** (a pair of crossings on one ray, *k* steps apart) give a
   calibrated confidence and a better precision/coverage trade-off than v1?
2. Does **CT** at and between the crossings add information the crossings alone do not have?

## What I looked at first (training chains only)

All on the **even-numbered human chains** (the training half); the odd chains, which give every
number below, were not opened. Scripts are in `explore/`.

- The "one sheet too many" errors (ray counts 2 where the human counts 1, 158 even-chain cases):
  the two ray steps are about equal (median shorter/longer 0.87) and together span 1.92× the
  ray's median step. The human click gap there is wider too (median 17.8 vs 14.1 voxels).
- CT (level 2, 9.6 µm) at the middle crossing of those cases is as dense as at the human-clicked
  ends (median 115 vs 114). The span between the two clicks stays dense (median profile ~90)
  where agreeing pairs dip to ~65 between sheets. So the extra crossing sits in a **compressed,
  dense stretch**, not in an air gap.
- On even-chain claims with *k* = 2–7 (1,717 claims, 63.8% exact): AUC of the v1 product 0.78;
  CT gap contrast (min over steps) 0.55; highest valley CT 0.59; neighbour-ray agreement on the
  same claim 0.56 (25 vx) to 0.64 (40 vx). Adding neighbour agreement to v1 lowers the AUC to
  0.74: neighbouring rays make the same mistake. So pair-level neighbour agreement is not in v2.
- Rays that pass within 60 voxels of any of the 2,173 human clicks: 17,060 of 1.15 M (323 k
  crossings). Their CT is ~12 GB of level-2 chunks; the full crossing set would touch ~83 k
  chunks (~163 GB).

From this I expected the CT features to add little, but kept them in so the answer would be
measured on held-out data.

## Method

**Region.** All rays with at least one crossing within 60 voxels of any human click (only click
*positions* choose the region, not their levels). Only these rays can match a click, so scores
inside the region equal scores on the full set. v1's chains restricted to the region reproduce
v1's held-out numbers (89.86% exact, 6.63% of human pairs).

**Unit.** A *claim* = two crossings *a* < *b* on one ray, *k* = *b* − *a* steps, claimed Δw =
level(*b*) − level(*a*). Training rows: every claim whose two ends match two clicks of one
**even** human chain, label = claimed Δw equals the human Δw.

**Claim features** (10; from crossings and CT, no labels, no `seed_winding`):

| Feature | Definition |
|---|---|
| `log_k` | log *k* |
| `v1_mean`, `v1_min` | mean and minimum over the *k* steps of the v1 step logit (v1 model, fitted on even chains) |
| `log_s_min`, `log_s_max` | min and max over the steps of v1's `log_s_ray` |
| `edge_max` | max over the steps of v1's `edge` |
| `log_r` | mean over the steps of v1's `log_r` |
| `ct_contrast_min` | min over the steps of (min CT at the two crossings − min CT on the step interior) / min CT at the two crossings. CT at a crossing = max of trilinear samples at −1.5…+1.5 voxels along the ray; interior = 8 samples at 15–85% of the step |
| `ct_mid_mean` | mean CT at the *k* − 1 intermediate crossings (the ends' mean if *k* = 1) |
| `ct_end_mean` | mean CT at the two end crossings |

CT is level 2 of `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr`, the
crossings' reference frame.

**Model.** Logistic regression, L2 = 1e-2, standardised features, numpy/scipy only. Two fits on
the same rows: *full* (10 features) and *no-CT* (first 7).

**Generator.** Per ray, greedy from the first crossing: a chain starting at crossing *s* extends
to crossing *j* while P(claim *s*→*j*) ≥ τ and P(claim *j*−1→*j*) ≥ τ. When the extension fails
the chain closes at *j* − 1 (kept if ≥ 2 points) and a new chain starts at *j*. τ is the lowest
value on 0.50, 0.51, …, 0.99 with even-chain adjacent pairs ≥ 95% exact and ≥ 100 adjacent
tests, as for v1. Every held-out number is the generated chains re-scored with `windingcheck`
against the odd chains.

I fixed the method, the checks and their thresholds before computing any v2 feature on a
held-out chain.

## Run

    python experiments/2026-10-12-claim-confidence-v2/run.py    # CPU; streams ~12 GB of CT on a cold cache

## Result (`results.json`)

17,060 region rays, 322,989 of 22.7 M crossings. Training: 2,809 even-chain claims (75.8%
exact). Held out: 2,489 odd-chain claims (75.7% exact), 1,559 with *k* = 2–7 (63.2% exact).
τ = 0.73 (even-chain adjacent pairs 95.1% exact, 753 tests). Output: 109,997 chains, 302,488
points. 2,316 s on CPU, 4.4 GB peak, almost all of it reading CT (including a ~15 min network
stall).

**v2 is the better generator on human time, but it misses the precision, coverage and
calibration I aimed for, and CT adds nothing.**

- **Held out:** 90.9% exact on adjacent pairs and 91.3% on all pairs (I aimed for 93% and 90%),
  covering 7.0% of human pairs (aimed for 8%) and 41.0% of adjacent pairs.
- **Against v1:** more coverage at the same precision: 41.0% against 30.5% of adjacent pairs,
  7.0% against 6.6% of all pairs, at 90.9% against 90.5% adjacent exact.
- **Ranking:** AUC 0.714 on claims with *k* = 2–7 (v1: 0.70; I aimed for 0.75). `v1_mean` alone
  reaches 0.722.
- **Calibration:** ECE 0.054 on all claims and 0.097 on *k* = 2–7 (aimed for 0.05 and 0.08).
  Mid-range claims are over-confident: bins at p ≈ 0.78 are 64% exact.
- **CT:** the full model's AUC is 0.7141, the no-CT model's 0.7143. `ct_contrast_min` alone
  reaches 0.63, but it carries nothing the crossing geometry does not. This is why the shipped
  `generate --claims` reads no CT (`../2026-10-13-claims-noct-generator/`).
- **Spatial transfer:** train below the median z, test above: 91.0% adjacent exact; the reverse
  96.1%. Within one scroll, not a second scroll.
- **Loads in the tools:** `windingcheck` reads all 302,488 points. On the first run my script
  called villa's loader with an interpreter that had no torch, so the import failed; the same
  file under villa's own environment loads as 109,997 collections and 302,488 points.
  `run.py` now picks villa's interpreter (`villa_python()`).

**Human time** (net hours saved of the 3.5–7.0 h Paris 4 relative-chain budget, held-out
adjacent pairs; right / wrong share of the 974 human adjacent pairs):

| Fix cost | Raw crossings | v1 | v2 |
|---|---|---|---|
| 1× | 0.93–1.86 h | 0.79–1.59 h | **1.07–2.13 h** |
| 3× | 0.03–0.07 h | 0.52–1.04 h | **0.70–1.40 h** |
| 5× | −0.87 to −1.73 h | 0.25–0.50 h | **0.33–0.67 h** |

v2 is best at every fix cost; v1 lost to the raw crossings at 1×. v2: 35.7% of adjacent pairs
right, 5.2% wrong; raw: 39.4% right, 12.8% wrong.

## Limits

- The ceiling is the same for v1 and v2: τ picked at 95% on the even chains gives about 91% on
  the odd chains for both, and no label-free feature I have separates the rest.
- Where the ray counts one sheet more than the human chain, is the human chain wrong? I tried to
  settle this by eye on 60 axial CT crops at 4.8 µm and could not: on the 20 control crops,
  where ray and human agree, I read "no sheet in between" on only 12, and 26 of the 60 crops were
  unreadable. Full-resolution crops or several slices per crop would be needed.
- One scroll only: Paris 4 is the only scroll with both public crossings and human winding
  labels.

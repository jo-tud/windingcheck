# windingcheck baselines: what `generate --claims` adds at equal coverage

**Question.** `windingcheck generate --claims` never changes a sheet count: it only cuts the
organisers' ray crossings into shorter chains (`claims.py:172`, `_to_chains` writes the raw
levels). So "76.7% raw → 91.4% generated" compares two different test sets (1,159 against 721
adjacent tests). How much of the 91.4% comes from the model's choice of *where* to cut, and how
much from cutting at all?

I had a first estimate of this from scripts I did not keep. This experiment re-measures it with
committed code and adds confidence intervals.

## Method

All sources are scored with windingcheck's own `build_truth` / `score_chains` against the **odd
(held-out) human chains** of PHerc. Paris 4, on the same **17,060 region rays** as the shipped
numbers (rays with a crossing within 60 voxels of any human click,
`../2026-10-13-claims-noct-generator/`). Only region rays can reach a click.

| Source | What |
|---|---|
| `raw` | the organisers' crossings, one chain per ray |
| `generator` | the region rays' chains from the full-volume `claims.json` (`windingcheck generate --claims`, model `paris4_claims_v2_noct.json`, τ = 0.74), streamed from disk |
| `random_cut` (seeds 0–9) | each ray step kept independently with probability *f* = the generator's kept-step fraction; chains are the maximal runs of kept steps (≥ 2 points). No model. |
| `one_step` | every ray step as its own 2-point chain. On a 2-point chain the scorer can only catch a reversed sign, so this measures the metric's blind spot, not a better source. |
| `generator_one_step` | the generator's kept steps, each as its own 2-point chain (same information as `generator`). |
| `raw_on_generator_tests` | raw crossings restricted to the tests the generator emits (same click pair, same ray). |

Numbers per source: tests and exact; adjacent tests and adjacent exact (per test); human
adjacent pairs covered; **per-pair precision** (pairs whose tests are all exact / pairs covered);
net hours at 3× fix cost (the formula the tool README uses); **Δw ≥ 2** tests and exact; sign
errors.

**Confidence intervals.** 95% percentile bootstrap over **human chains** (the odd chains are
resampled with replacement, 2,000 replicates, seed 12345). Every source uses the same resamples,
so differences are paired. `random_cut` is pooled over its 10 seeds before any ratio is taken.

I fixed the method and what would count as a real model gain before the script first ran: the
random cut must reach the generator's coverage within 2 points, and the generator must beat it by
≥ 3 points per test with every confidence interval above zero.

## Run

    windingcheck generate --claims --crossings $W --out claims.json     # the tool's step 2
    python experiments/2026-10-13-windingcheck-baselines/run.py claims.json

CPU only, no CT read, 19 s, 0.67 GB peak: `claims.json` (4.0 GB) is streamed and only the region
rays' crossings are kept. Writes `results.json` here.

## Result (`results.json`)

**At equal coverage (39.9% against 39.2% of held-out adjacent human pairs), the model's choice
of where to cut is worth +7.2 points per test (95% CI +3.7 to +11.2) and +6.2 points per human
pair (+3.1 to +9.7) over random cuts that drop the same 37.7% of steps. About half of the 14.7
points from raw (76.7%) to 91.4% come from cutting at all.**

Held out (odd chains), 974 human adjacent pairs.

| Source | Adjacent exact (per test) | Per human pair | Adjacent coverage | Δw ≥ 2 exact | Hours saved, 3× |
|---|---|---|---|---|---|
| raw crossings | 76.7% of 1,159 [67.3–85.4] | 75.4% of 509 [67.2–82.5] | 52.3% | 74.9% of 1,330 | 0.03–0.07 |
| random cut, 10 seeds pooled | 84.2% of 6,532 [77.3–90.2] (seeds 82.2–86.1) | 81.7% [74.7–87.9] (seeds 80.0–82.9) | 39.2% (seeds 38.4–39.9) | 83.3% of 3,269 (seeds 79.5–88.0) | 0.37–0.74 |
| **`generate --claims`** | **91.4% of 721 [87.1–94.9]** | **87.9% of 389 [82.1–92.7]** | **39.9%** | **92.3% of 339 [88.2–95.8]** | 0.72–1.45 |
| generator − random cut | **+7.2 points [+3.7, +11.2]** | **+6.2 [+3.1, +9.7]** | +0.7 [−1.5, +2.7] | **+9.0 [+3.7, +15.8]** | |
| one step (every ray step its own chain) | 98.8% of 905 | 97.5% of 434 | 44.6% | 0 of 5 | 1.40–2.80 (artefact) |
| generator, split into single steps | 98.5% of 671 | 97.2% of 358 | 36.8% | 0 of 4 | 1.14–2.29 (artefact) |

- The run reproduces the shipped counts exactly (generator: 111,478 region chains, 721 adjacent
  tests with 659 exact; raw: 1,159 with 889; one step: 905 with 894).
- On the generator's own 721 adjacent tests the raw crossings also score 91.3% (same prediction
  on 720 of them): the generator never changes a count, it only chooses which ones to keep.
- Per-test numbers overstate certainty: 721 tests sit on 389 human pairs (1.85 per pair) on a
  few hundred human chains. The chain-bootstrap CI on 91.4% is 87.1–94.9%, about twice as wide
  as a naive binomial interval.
- Single-step chains score 98–99% because a 2-point chain can only be wrong by sign; a spurious
  crossing mid-gap matches no click and is never tested. The 1.40–2.80 h "saved" by the
  one-step rule is this blind spot, not a better source: splitting my own chains the same way
  lifts them to 98.5% with no new information. 56% of the generator's chains are 2-point, so
  its own number carries some of this too.
- On human time at 3× fix cost the model is worth about 0.35–0.71 h over random cuts
  (0.72–1.45 h against 0.37–0.74 h), not the 0.69–1.38 h over raw crossings.
- My first estimate, from 5 random-cut seeds, gave 84.2–86.0% for the random cut and about +6
  points for the model; with 10 seeds the random cut spans 82.2–86.1% (pooled 84.2%) and the
  model's edge is +7.2.
- Nothing here shows a spiral-fit gain; see Limits in the tool README.

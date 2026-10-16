# Confidence-scored winding constraints from ray crossings (generator v1)

**Question.** Can a label-free confidence score on the organisers' neural ray crossings pick out
the winding constraints that are right, so that a generator emits constraints a human does not
have to check?

**Starting point** (`../2026-10-12-winding-scorer/`). The organisers' crossings are 75.8% exact
on the 2,659 human pairs they reach (16.3% of all), 80.5% exact on adjacent pairs, and 91% of
their adjacent errors count one sheet too many. Nothing tells a user which constraints to trust.

## Method

Unit = one **step**: crossing *k* to crossing *k+1* on one ray (a claim "these two points are one
sheet apart"). A pair constraint on a ray is a run of steps; its confidence is the product of the
step confidences.

Step features, all from the crossings alone (no CT, no labels, no `seed_winding`, which comes
from a fitted spiral that may have seen the human labels):

| Feature | Definition |
|---|---|
| `log_s_ray` | log(step length / median step length on the same ray) |
| `log_s_ray²` | its square (a step can be too short *or* too long) |
| `nbr_agree` | neighbour rays: for every other ray with a crossing within 40 vx of crossing *k* (nearest one along this ray's direction, \|offset\| < half the step), take its next crossing along this ray's direction; agree if that gap is within 35% of this step. `nbr_agree` = agreeing / all such rays (0.5 if none) |
| `log_support` | log(1 + number of such neighbour rays) |
| `edge` | \|position on ray − 192\| / 192 (0 at the slab centre, 1 at the ends) |
| `oblique` | 1 − \|cos\| of the angle between the ray and the radial direction from the umbilicus |
| `log_r` | log(radius from the umbilicus, voxels) |

Model: logistic regression on standardised features (numpy/scipy only), fitted on step labels
from adjacent human pairs of the **even-numbered** human chains. The steps between two crossings
that match an adjacent human pair are all right if the ray counts exactly 1 sheet there, all
wrong otherwise. The odd-numbered chains are held out for every number below.

Generator: `windingcheck generate` scores every step of the full crossing set, cuts each ray at
steps below a threshold τ, and writes each remaining run of ≥ 2 crossings as one relative chain
in a VC3D point-collection file (chain confidence in the collection `tags`). τ is fixed on the
even chains only: the lowest τ on the grid 0.500, 0.505, …, 0.995 at which the even-chain adjacent
pairs whose steps all reach τ are ≥ 95% exact, with ≥ 100 such pairs left. Held-out numbers come
from re-scoring the generated chains with `windingcheck`.

**Human time.** Paris 4's relative chains cost 3.5–7.0 h of clicking
(`../2026-10-12-winding-human-baseline/`). On held-out chains, *f_ok* = share of human adjacent
pairs the output reproduces exactly, *f_bad* = share it gets wrong. A wrong constraint has to be
found and fixed, charged at 3 clicks: net hours saved = (*f_ok* − 3·*f_bad*) × (3.5 to 7.0 h).

I fixed the method, the checks and their thresholds before computing the first feature.

## Run

    python experiments/2026-10-12-confident-crossings/run.py       # ~5 min CPU
    python experiments/2026-10-12-confident-crossings/posthoc.py   # follow-up analysis, ~5 min CPU
    windingcheck generate --crossings data/spiral_datasets/PHercParis4/winding_inference \
        --out confident.json --every 50

## Result (`results.json`, `results_posthoc.json`)

CPU only, 5 min 3 s, 6.8 GB peak RAM. Model fitted on 1,398 step rows from 1,200 even-chain
adjacent pairs; τ = 0.795. The generator keeps 6.35 M of 22.7 M crossings in 1.62 M chains.

| Held out (odd chains) | Tests | Human pairs covered | Adjacent covered | Exact, all | Exact, adjacent | ±1 |
|---|---|---|---|---|---|---|
| organisers' crossings, unfiltered | 2,489 | 13.8% | 52.3% | 75.7% | 76.7% | 93.1% |
| **confidence-filtered, τ = 0.795** | 1,055 | 6.6% | 30.5% | **89.9%** | **90.5%** | 98.0% |

Errors on held-out adjacent pairs fall from 12.8% of human pairs to 3.9%; correct coverage falls
from 39.4% to 26.6%.

- **Human time:** 0.52–1.04 h saved at 3× fix cost, against 0.03–0.07 h for the unfiltered
  crossings. At 1× fix cost the unfiltered crossings save more (0.93–1.86 h against
  0.79–1.59 h); at 5× they cost time (−0.87 to −1.73 h) and the filtered ones still save
  0.25–0.50 h. Filtering pays only if fixing a wrong constraint costs more than about 1.5 clicks.
- **Short of what I aimed for.** I wanted ≥ 93% exact on adjacent pairs and ≥ 90% on all pairs
  (got 90.5% and 89.9%), ≥ 8% of human pairs covered (got 6.6%), and a calibrated confidence
  (expected calibration error 0.087 on adjacent pairs and 0.250 on pairs 2–7 sheets apart,
  against 0.05 and 0.08 wanted).
- **Spatial transfer:** refitting on clicks below the median z and testing above gives 92.1%
  adjacent exact; the reverse gives 97.7%. This is within one scroll, not a second scroll.
- **Loads in the tools:** the `--every 50` file (32,462 collections, 128,025 points) reads with
  the same counts in villa's `point_collection.load_point_collection` and in `windingcheck`. On
  the first run my script looked for villa in the wrong place and the loader never ran; after
  fixing the path, the same file loaded.
- **Two guesses were wrong.** Very short steps are not where the extra sheets are: only 39 of
  1,474 held-out step rows are under 0.6 × the ray median, wrong 49% of the time against a 40%
  base rate. And `nbr_agree` is the weakest feature, not the strongest (AUC 0.52, see below).

**The AUC of 0.875 is inflated.** It scores confidence on pairs that are adjacent *for the
human*. On that set a ray that counts two steps is wrong by construction, and a pair's
confidence is a product over its steps, so any model with a roughly constant step confidence
separates right from wrong by the step count alone: every single-feature model scored AUC
0.90–0.99. `posthoc.py` scores the generator's own claims instead (a pair the ray says is Δw = k
apart, wherever both ends match one human chain):

| Claimed Δw | Held-out tests | Exact | AUC of confidence | ECE |
|---|---|---|---|---|
| 1 | 894 | 99.4% | 0.33 (5 errors, not meaningful) | 0.216 (under-confident) |
| 2–7 | 1,540 | 64.0% | **0.70** | 0.140 |

A spurious crossing never matches a human click, so a one-step claim that matches two clicks is
almost always right; the neural model's errors show up only as multi-step pairs. On those the
step product ranks right above wrong (AUC 0.70) but is under-confident: claims at p ≈ 0.1 are
right 44% of the time. Errors are not independent per step, so the product rule is wrong. The
re-scored precision, coverage and human-time numbers above do not have this problem.

## What it means

- A label-free filter makes the organisers' crossings much more trustworthy: about 3.3× fewer
  wrong constraints on held-out adjacent pairs (12.8% → 3.9%), 76% → 90% exact, at about half
  the coverage. It runs on all 22.7 M crossings in 5 minutes on a CPU.
- It is not good enough to skip review: one in ten emitted constraints is still wrong, and the
  confidence is not calibrated.
- The step is the wrong unit. Errors are extra crossings that only show across several steps
  and are correlated along the ray. That led to the claim-level v2
  (`../2026-10-12-claim-confidence-v2/`).
- One scroll only: Paris 4 is the only scroll with public crossings and human winding labels
  (checked 2026-10-12).

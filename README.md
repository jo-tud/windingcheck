# windingcheck

Score automatic winding constraints against the public human winding annotations of
PHerc. Paris 4, and generate new relative-winding constraints from the organisers' neural ray
crossings. CPU only; only the optional `normals` command reads CT.

Relative-winding chains are the most expensive human input in the public spiral data: the
2,173 clicks behind Paris 4's chains took 3.5–7.0 h, out of an 8 h budget for a whole scroll.
Any tool that writes those constraints automatically needs a number that says how often it is
right. `windingcheck score` gives that number for any VC3D point-collection file, for the
organisers' `winding_inference` crossings and for plain spiral priors, in seconds on a laptop.
`windingcheck generate --claims` writes new constraints in the same format, ready for villa's
spiral fitting.

![where the organisers' crossings and two spiral priors disagree with the human chains](experiments/2026-10-12-winding-scorer/drift_map.png)

*Adjacent pairs of human clicks on Paris 4: grey where a method counts the sheets right, red
where it counts too many, blue too few. Left to right: a spiral prior with one sheet spacing,
the same prior with spacing by radius, the organisers' neural crossings. The two prior panels
show the held-out (odd) chains only; the crossings panel shows all chains, so it reads 81% here
against 76.7% held out.*

## What came out

On the held-out half of the human chains (the odd-numbered ones):

- The organisers' raw crossings are **76.7%** exact on adjacent sheet pairs, at 52.3% coverage.
- `generate --claims` keeps the crossings it trusts and is **91.4%** exact (87.9% per human
  pair), at 39.9% coverage. Cutting the rays at random to the same coverage gives 84.2%, so the
  gain over random cuts is **+7.2 points** (95% CI +3.7 to +11.2) with this scorer. About half of
  that gain comes from the scorer, not the model: see the tilt limit below.
- Matched along the sheet normal measured in the CT (`windingcheck normals`, then
  `score --normals`, new in v0.1.2), the same held-out chains read **91.2%** raw, **97.0%** for
  `generate --claims` (95.4% per human pair) and 93.8% for random cuts. The gain over random cuts
  is **+3.2 points** (95% CI +1.3 to +5.6), but the generator now covers 51.6% of the adjacent
  pairs and random cuts 49.0%, so it is no longer a gain at equal coverage. Every number re-makes
  from this repository: `experiments/2026-10-23-windingcheck-ct-normal/`.
- In human time that is **0.72–1.45 h saved** of the 3.5–7.0 h, if fixing a wrong constraint
  costs three times as much as making one. With this scorer about half of it comes from where
  the model cuts, the rest from cutting at all.
- The whole scroll (22.7 M crossings) takes 14.4 min and 9 GB on a CPU. The output loads in
  villa's point-collection loader with the same point count.

## Run it

    python3 -m venv .venv && . .venv/bin/activate
    pip install -e "tools/windingcheck[plot]" -c tools/windingcheck/constraints.txt
    windingcheck fetch --crossings
    windingcheck score --heldout --crossings data/spiral_datasets/PHercParis4/winding_inference

Python 3.12, about 170 MB of downloads. The full quickstart, the output format and how the
matching works are in [`tools/windingcheck/README.md`](tools/windingcheck/README.md).

## What does not work yet

- **No spiral-fit gain seen.** I fitted villa's `fit_spiral.py` on one slab of Paris 4 with and
  without the generated constraints and saw no gain. That test was too small to decide either
  way (one slab, 21 held-out chains, one fit per setting at 12,000 steps; villa's default is
  30,000, and its DT losses start at step 25,000, so they were off); its fit files are not in this
  repository. Treat `generate --claims` as a filter with a measured precision, not as a fit
  improvement.
- **One scroll.** Paris 4 is the only scroll with both public ray crossings and human winding
  labels, so every number here comes from it.
- **The default match misses tests on tilted sheets.** It matches a generated point to a human
  click along the direction of the human chain, not along the sheet normal (on the held-out clicks
  the median angle between the two is about 25°). On synthetic sheets where every claim is correct,
  it reads 50% exact at 30–45° tilt. On real Paris 4 sheets with claims known to be correct it
  reads 98.4% there and 95.5% above 45°; there its cost is mostly half as many tests, plus a wrong
  share of 1.5% instead of 0.14%. So the numbers above do not read low by the synthetic amount.
  When I re-score the same held-out chains along the sheet normal measured in the CT
  (`score --normals`), every number above rises, and the model's gain over random cuts shrinks
  from +7.2 to +3.2 points (95% CI +1.3 to +5.6), no longer at equal coverage. I checked the
  sheet-normal match on those real claims (points ≤ 24 voxels from a click, on human-verified
  patches): 99.9% exact, 99.6% at 30–45°, and it reads every planted ±1 slip as wrong
  (`experiments/2026-10-23-windingcheck-real-sheets/`). Details: Limits in the tool README.
- **60% of adjacent pairs are not covered**, and the confidence ranks chains but is not a
  probability.

The tool README lists the other limits.

## Layout

| Path | Content |
|---|---|
| `tools/windingcheck/` | the tool: code, tests, quickstart, credits |
| `experiments/` | one folder per experiment: question, how to run, result |

## Licence and data

Code: MIT, see `LICENSE`.

The data this code reads (human annotations and ray crossings of PHerc. Paris 4, CT chunks, and
for one experiment the human-verified patches) is published by
the Vesuvius Challenge under CC BY-NC 4.0 and is not redistributed here. The annotations and
crossings sit in the frame of the 2026 Paris 4 scan `20260411134726` (2.4 µm). Cite both datasets:

> S. Parsons, C. S. Parker, C. Chapman, M. Hayashida, W. B. Seales,
> *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using
> X-ray CT*, arXiv:2304.02084, 2023.

> G. Angelotti, S. Parsons, S. Johnson, E. R. Dal Prà, J. Rudolph, P. Tafforeau, A. Mirone,
> P. Henderson, H. Schilling, F. McDonald, D. Josey, Y. Nader, C. S. Parker, W. B. Seales,
> *Vesuvius Challenge – CT Scans of Herculaneum Papyri*, Vesuvius Challenge, 2026.

See <https://scrollprize.org/data> for the data agreement. **PHerc. Paris 4 text is reserved**
until its title prize is announced (open until 2027-06-25). This repository ships no scan data,
and nothing `windingcheck` writes can carry text: its outputs and figures are coordinates,
counts and scores only. See `tools/windingcheck/DATA_NOTICE.md`.

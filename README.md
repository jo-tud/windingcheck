# windingcheck

Score automatic winding constraints against the public human winding annotations of
PHerc. Paris 4, and generate new relative-winding constraints from the organisers' neural ray
crossings. CPU only, no CT read.

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
  model's own share of the gain is **+7.2 points** (95% CI +3.7 to +11.2).
- In human time that is **0.72–1.45 h saved** of the 3.5–7.0 h, if fixing a wrong constraint
  costs three times as much as making one; about half of it comes from the model, the rest from
  cutting at all.
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
  way (one slab, 21 held-out chains, one fit per setting); its fit files are not in this
  repository. Treat `generate --claims` as a filter with a measured precision, not as a fit
  improvement.
- **One scroll.** Paris 4 is the only scroll with both public ray crossings and human winding
  labels, so every number here comes from it.
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

The data this code reads (human annotations and ray crossings of PHerc. Paris 4) is published by
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

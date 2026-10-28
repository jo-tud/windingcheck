# The sheet-normal match on real sheets, with claims known to be correct

**Question.** `score --normals` matches a generated point to a human click along the sheet normal
measured in the CT (`../2026-10-23-windingcheck-ct-normal/`). Its own tests use synthetic sheets.
On real PHerc. Paris 4 sheets, which bend and touch and give a noisier CT normal, does it still put
a point that is known to sit on a click's sheet on that click? And does it catch a wrong claim?

## Method

**Truth.** The 127 odd (held-out) human chains of `relative_windings.json`, 1,101 clicks, exactly
as `windingcheck score --heldout`. Two matches: the **chain direction** (`build_truth(odd)`,
shipped since v0.1) and the **CT sheet normal** (`build_truth(odd, normals=…)`, normals from
`windingcheck normals --heldout --z-band 7000 17000`; 1,099 of the 1,101 clicks get one).
Tolerances (`tol_t` 25 voxels, the `tol_n` rule) are unchanged.

**Claims that are correct by construction.** The organisers' human-verified patches (tifxyz
surfaces in `spiral_datasets/PHercParis4/verified_patches/`) each follow one sheet.

- A held-out click is **on** a patch if its distance to the patch's triangles is ≤ 2 voxels.
- A (patch, human chain) pair whose linked clicks carry two different `wind_a` is dropped: the
  patch then crosses two sheets of that chain, or a click sits on the wrong patch.
- In each of 8 replicates (seed 1684), every linked click takes one of its patches and one point on
  it 4–24 voxels away, never the same point twice. The point gets **the click's own `wind_a`**.
  The points of one human chain and one replicate form one candidate chain.
- The point lies on the click's sheet because a human verified the patch. It carries the click's
  own level, so an error in the human label cancels: truth and claim carry the same label. Every
  claim is correct. A wrong test means the match put a point on a neighbouring click, or a patch
  was linked to a click on a touching neighbour sheet; both count against the match here.
- A second file has the same points. In 30% of the candidate chains (seed 2684), the levels after
  one random cut get a +1 or −1 slip. A test should read wrong exactly when its two points lie on
  opposite sides of a planted cut (the "oracle").

**Counts.** 807 of the 1,101 held-out clicks are on a patch, in 125 chains; 549 adjacent click
pairs have both clicks on a patch. 193 of 4,918 links are dropped by the two-labels rule. 944
candidate chains, 6,400 points, 299 chains with a planted slip. Median displacement 13.9 voxels
(10th–90th percentile 5.9–21.9).

**Metric.** Adjacent tests (two points of one candidate chain on two neighbouring clicks of one
human chain). The tilt of a test is the larger angle between the CT normal and the chain
direction at its two clicks. Brackets: 95% bootstrap over human chains (2,000 replicates, seed
12345).

**Criteria, fixed before the first score.** The CLI gives the same numbers as the library; at
least 50 tests in each of 30–45° and 45–90°; the sheet-normal match reads ≥ 0.95 exact overall
and in each of those two bins; on the slip file it reads ≥ 95% of the slip-spanning tests as
wrong and stays within 0.03 of the oracle, overall and at ≥ 30°. All are recorded in
`results.json` under `criteria`.

## Run

    pip install tifffile        # this experiment only; the tool itself needs numpy and scipy
    windingcheck normals --heldout --z-band 7000 17000 --out normals.json   # 1.42 GB of CT, cached
    python experiments/2026-10-23-windingcheck-real-sheets/run.py normals.json

`run.py` downloads the 3,610 patches named in `patches.json` (0.41 GB: `meta.json`, `x.tif`,
`y.tif`, `z.tif` of each) from
`https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/verified_patches/` into
`data/spiral_datasets/PHercParis4/verified_patches/`, and stops if any patch's sha256 differs
from `patches.json`. It then builds the two candidate files, scores them with both matches through
the library and through `windingcheck score --heldout [--normals]`, writes `results.json`, and
compares every number with `expected.json` (integers exactly, floats to 1e-9 relative). It prints
"0 differences from expected.json" and exits 0 when everything matches, and lists the differences
and exits 1 otherwise. The patch download takes about 6 min (16 at a time); after it, the run
takes about a minute and 1 GB.

The patch list holds every patch that links to a held-out click, taken from the 62,275 patches of
this folder that I had used as spiral-fit input (z about 7,000–18,000). The folder holds about
89,000 patches; ones outside my 62,275 were not tested.

## Result

All criteria pass. Adjacent exact (tests); the chain direction is the shipped default:

| Tilt | Sheet normal | Chain direction |
|---|---|---|
| 0–10° | 1.000 (77) | 1.000 (76) |
| 10–20° | 1.000 (596) | 0.990 (503) |
| 20–30° | 1.000 (1,125) | 0.989 (628) |
| 30–45° | **0.996 (1,090)** [0.990, 1.000] | 0.984 (450) |
| 45–90° | **0.998 (656)** [0.995, 1.000] | 0.955 (179) |
| all | **0.999 (3,544)** [0.997, 1.000] | 0.985 (1,836) |
| wrong tests, all (not only adjacent) | 25 of 29,582 (0.08%) | 471 of 15,368 (3.1%) |

On the slip file the sheet-normal match reads every slip-spanning test as wrong (193 of 193; 90 of
90 at ≥ 30°), and its exact share is within 0.003 of the oracle. The chain-direction match reads
96.9% of them wrong (91.7% of 12 at 45–90°). The CT normal lies within 7.4° (median; 90th
percentile 16.2°) of the patch's own normal at the linked clicks.

**What this says about the chain-direction numbers.** On synthetic sheets 12 voxels apart with
every claim correct, the chain-direction match reads 50% exact at 30–45° tilt. On these real
claims it reads 98.4% there and 95.5% at 45–90°. Its cost here is mostly fewer tests: about half
as many adjacent tests (1,836 against 3,544; 59% fewer at 30–45°, 73% fewer at 45–90°), and a
wrong share about 10 times higher (1.5% against 0.14% of adjacent tests). So the earlier
chain-direction numbers do not read low by the synthetic amount; on claims like these, the change
to the sheet normal mostly adds tests.

## Limits

- The candidate points are at most 24 voxels from a click and lie on human-verified patches. Points
  farther away, and sheet contacts the patches avoid, are not tested. The synthetic failure of the
  chain direction may need such a regime (far points, sheets packed closer).
- 294 of the 1,101 held-out clicks are on no patch and are not tested.
- 5 of the 3,544 adjacent sheet-normal tests are wrong, all at tilt ≥ 30°, all reading 2 sheets
  where the truth is 1; 3 of the 5 come from one click pair. I did not check whether these come
  from a wrong patch link.
- One scroll, as everything in this repository.

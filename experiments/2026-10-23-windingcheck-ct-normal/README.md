# The October numbers matched along the CT sheet normal

**Question.** windingcheck matches a generated point to a human click along the direction of the
human chain. Where the sheet is tilted against the chain, a point on the neighbouring sheet can
match, so a correct claim counts as wrong or is not tested. On synthetic spiral sheets with every
claim correct, that match reads 50% exact at 30–45° tilt (on real claims known to be correct,
98%: see Limits). What happens to the numbers of
`../2026-10-13-windingcheck-baselines/` (raw crossings, `generate --claims`, random cuts, and the
gain over random cuts) when the same held-out chains are matched along the sheet normal measured
in the CT?

## Method

Everything is as in `../2026-10-13-windingcheck-baselines/`, whose functions `run.py` imports
unchanged: the 127 odd (held-out) human chains of PHerc. Paris 4 (1,101 clicks, 974 adjacent
pairs), the 17,060 region rays, the generator's region chains from the full `generate --claims`
output, random cuts with seeds 0–9 at the generator's kept-step fraction, and the bootstrap over
human chains (2,000 replicates, seed 12345). Only the match direction changes:

- **chain direction**: `build_truth(odd)`, as shipped since v0.1;
- **sheet normal**: `build_truth(odd, normals=n)`, with `n` from `windingcheck normals --heldout
  --z-band 7000 17000` (structure tensor of the raw CT at level 2, 9.6 µm voxels; see "Matching
  along the CT sheet normal" in the tool README). 1,099 of the 1,101 clicks get a normal. The 2
  clicks without one (chain 92, z 17,252.6, above the z band) keep the chain direction; no test of
  any source touches them.

Tolerances (`tol_t` 25 voxels, the `tol_n` rule) are unchanged. The tilt of a test is the larger
angle between the CT normal and the chain direction at its two clicks.

## Run

    windingcheck fetch --crossings
    W=data/spiral_datasets/PHercParis4/winding_inference
    windingcheck generate --claims --crossings $W --out claims.json     # 14 min, 9 GB, 4.0 GB file
    windingcheck normals --heldout --z-band 7000 17000 --out normals.json   # 1.42 GB of CT, cached
    python experiments/2026-10-23-windingcheck-ct-normal/run.py claims.json normals.json

`run.py` writes `results.json` and compares every number in it with `expected.json` (integers
exactly, floats to 1e-9 relative). It prints "0 differences from expected.json" and exits 0 when
everything matches, and lists the differences and exits 1 otherwise. After the inputs exist it
takes about 15 s and 0.7 GB. The chain-direction half of `expected.json` equals the numbers of
`../2026-10-13-windingcheck-baselines/` (721 adjacent tests, 91.4%, gain +7.25 points), so the same
run also checks that the inputs are the ones the October numbers came from.

## Result

| Number | Chain direction | CT sheet normal |
|---|---|---|
| raw crossings, adjacent exact | 76.7% (889 of 1,159) | 91.2% (1,254 of 1,375) |
| raw coverage of adjacent pairs | 52.3% | 62.1% |
| `generate --claims`, adjacent exact per test | 91.4% (659 of 721) [87.1, 94.9] | 97.0% (903 of 931) [94.8, 98.8] |
| per human pair | 87.9% (342 of 389) [82.1, 92.7] | 95.4% (480 of 503) [92.3, 98.0] |
| generator coverage | 39.9% | 51.6% |
| random cuts pooled, adjacent exact | 84.2% (5,497 of 6,532) | 93.8% (7,804 of 8,320) |
| random-cut mean coverage | 39.2% | 49.0% |
| gain (generator − random cuts) | +7.25 points [+3.68, +11.19] | +3.19 points [+1.26, +5.63] |
| coverage gap (generator − random cuts) | +0.7 | +2.6 [+0.6, +4.5] |
| wrong-sign tests of `generate --claims` | 12 of 1,060 | 17 of 1,433 |

Brackets: 95% bootstrap over human chains. CT tilt at the held-out clicks: median 25.2°, 90th
percentile 50.3°.

Every source rises. The gain over random cuts stays above zero but shrinks to less than half,
and with a 2.6-point coverage gap it is no longer a comparison at equal coverage. Random cuts
carry more tests on steeply tilted sheets (45–90°: 14% of random-cut tests, 12% of generator
tests) and rose more there (54.5% → 87.3%; the generator 69.7% → 93.6%). Below 10° tilt both
matches agree within 2 points for every source, so the sheet normal does not lift numbers where
the chain-direction match was already right.

## Limits

- The sheet-normal match has a synthetic check (`tools/windingcheck/tests/test_normals.py` and
  `test_score.py`: tilted sheets, every claim correct, exact under the CT normal) and a real-sheet
  check (`../2026-10-23-windingcheck-real-sheets/`): on the held-out clicks that lie on a
  human-verified patch, with points moved 4–24 voxels along the patch so every claim is correct, it
  reads 99.9% exact (99.6% at 30–45°) and reads every planted ±1 slip as wrong. Points farther
  than 24 voxels from a click, and sheet contacts the patches avoid, are not tested; there it could
  read too high.
- On those real claims the chain-direction match reads 98.4% at 30–45° and 95.5% at 45–90°, not
  the synthetic 50%; its cost there is mostly fewer tests. So the rise in the table above is partly
  more tests, not only fewer wrong reads.
- One scroll, as everything in this repository.

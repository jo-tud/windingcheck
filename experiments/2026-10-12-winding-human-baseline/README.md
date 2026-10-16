# How many human hours are in the public winding annotations? (PHerc. Paris 4)

**Question.** A winding generator should be judged in human time saved. What did the human
winding annotations of the one scroll that has them actually cost?

**Data.** `spiral-input` PHercParis4 (`https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/`):
`relative_windings.json`, `same_windings.json`, `abs_winding.json`. VC3D point-collection format,
coordinates `[x, y, z]` in full-resolution voxels, one `creation_time` (ms) per clicked point.

**Method.** Sort all click times. A gap shorter than a cap counts as work, a longer gap as a break.
Three caps (2, 5, 10 min) give a range. CPU, < 1 s.

    python experiments/2026-10-12-winding-human-baseline/human_time.py

The figure is a **lower bound** on human input: it leaves out review, thinking before a
session's first click, and the ~27k verified surface patches in the same dataset.

## Result (`results.json`)

| File | Collections | Points | Days | Active hours (2 / 5 / 10 min cap) |
|---|---|---|---|---|
| relative_windings | 300 | 2,173 | 2026-06-16 → 06-19 | 3.5 / 6.0 / 7.0 |
| same_windings | 154 | 5,413 | 2026-06-09 → 06-17 | 0.5 / 0.6 / 0.6 |
| abs_winding | 6 | 59 | 2026-06-12 → 06-17 | 0.04 |
| **total** | | **7,645** | | **4.0 / 6.7 / 7.7** |

- A relative-winding collection is a radial chain: one click per sheet crossing on one slice
  (`wind_a` = 1, 2, 3 …). 254 chains carry windings; median span 5 windings, max 51; 16,269
  pairwise relative constraints implied.
- Human cost: **5.8–11.5 s per relative-winding click**.
- Adjacent-wrap spacing along those chains: median 21.2 voxels (p10 12.0, p90 40.4). The low tail
  is the tightly packed wraps, where counting sheets along a ray is hardest.

**What it means.** Relative windings are 28% of the clicks but 87–91% of the active time. The
winding annotations alone for one scroll took 4.0–7.7 h, against the Grand Prize cap of 8 h of
human input for the whole pipeline. The relative-winding chains are where automation saves the
most human time.

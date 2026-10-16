# Data notice

The MIT licence covers the code in this directory only. It does not cover any scroll data.

`windingcheck` ships **no scroll data**. `windingcheck fetch` downloads the public files at run
time from the Vesuvius Challenge data server (`https://dl.ash2txt.org/datasets/spiral_datasets/`):

- `relative_windings.json`, `same_windings.json`, `abs_winding.json`, `umbilicus.json`
  (the `spiral-input` dataset, 2026-07);
- with `--crossings`: the `winding_inference/` ray crossings (about 170 MB).

The only scroll used is **PHerc. Paris 4 (Scroll 1)**.

## Licence and terms

- Vesuvius Challenge data is licensed under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) unless a file says otherwise,
  and is subject to the data server terms at <https://dl.ash2txt.org/LICENSE.txt>. By downloading
  it you accept those terms yourself. They include: no redistribution without written approval
  from Vesuvius Challenge, and no publication of revealed hidden text without written consent.
- Scrolls 1–4 and Fragments 1–6 scanned at DLS before 2025, including PHerc. Paris 4, belong to
  the **EduceLab-Scrolls Dataset, © EduceLab / The University of Kentucky**. Vesuvius Challenge
  has permission to use them, with the extra citation terms below
  (<https://scrollprize.org/data#how-to-cite>).

## Required citation

If you publish or present results that use the data fetched by this tool:

- In any published abstract, cite **EduceLab-Scrolls** as the source of the data.
- In any published manuscript, reference: Parsons, S., Parker, C. S., Chapman, C., Hayashida, M.,
  & Seales, W. B. (2023). *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri
  using X-ray CT*. ArXiv [Cs.CV]. https://doi.org/10.48550/arXiv.2304.02084
- In the methods section, include language like: "Data used in the preparation of this article
  were obtained from the EduceLab-Scrolls dataset [above citation]."

The human winding annotations and the ray crossings were made by the Vesuvius Challenge team; see
`CREDITS.md`.

## What the outputs contain

Every output of `windingcheck` is geometry only: point coordinates in scan voxels, winding
counts, confidences and scores. It reads no CT intensities (the shipped `generate --claims` path
reads the crossing files and the umbilicus only), renders no images of the scan, and contains no
ink, letters or text. The optional `--plot` drift map draws coloured dots at annotation
positions on a blank background.

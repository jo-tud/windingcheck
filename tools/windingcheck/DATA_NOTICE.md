# Data notice

The MIT licence covers the code in this directory only. It does not cover any scroll data.

`windingcheck` ships **no scroll data**. `windingcheck fetch` downloads the public files at run
time from the Vesuvius Challenge data server (`https://dl.ash2txt.org/datasets/spiral_datasets/`):

- `relative_windings.json`, `same_windings.json`, `abs_winding.json`, `umbilicus.json`
  (the `spiral-input` dataset, 2026-07);
- with `--crossings`: the `winding_inference/` ray crossings (about 170 MB).

The only scroll used is **PHerc. Paris 4 (Scroll 1)**. The annotations and crossings are in the
frame of the 2026 Paris 4 scan `20260411134726` (2.4 µm); `windingcheck` reads no CT from it.

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
- **PHerc. Paris 4 text is reserved.** Its title prize is open until 2027-06-25, and the Vesuvius
  Challenge rules ask that no reading of new text from it be made public before the prize is
  announced. Nothing this tool writes can carry text (see below). Do not combine its outputs with
  renders of Paris 4 in anything you publish without the organisers' approval.

## Required citation

If you publish or present results that use the data fetched by this tool, cite the datasets of
the scans the annotations sit on:

- **EduceLab-Scrolls** (PHerc. Paris 4 and the other legacy DLS scans). In any published
  abstract, cite EduceLab-Scrolls as the source of the data. In any published manuscript,
  reference: Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., & Seales, W. B. (2023).
  *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT*.
  ArXiv [Cs.CV]. https://doi.org/10.48550/arXiv.2304.02084. In the methods section, include
  language like: "Data used in the preparation of this article were obtained from the
  EduceLab-Scrolls dataset [above citation]."
- **Vesuvius Challenge CT scans** (scans released directly by Vesuvius Challenge, which includes
  the 2026 Paris 4 scan `20260411134726`): Giorgio Angelotti, Stephen Parsons, Sean Johnson,
  Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, Paul Henderson,
  Hendrik Schilling, Forrest McDonald, David Josey, Youssef Nader, C. Seth Parker,
  W. Brent Seales. *Vesuvius Challenge – CT Scans of Herculaneum Papyri*. Vesuvius Challenge. 2026.

The human winding annotations and the ray crossings were made by the Vesuvius Challenge team; see
`CREDITS.md`.

## What the outputs contain

Every output of `windingcheck` is geometry only: point coordinates in scan voxels, winding
counts, confidences and scores. It reads no CT intensities (the shipped `generate --claims` path
reads the crossing files and the umbilicus only), renders no images of the scan, and contains no
ink, letters or text. The optional `--plot` drift map draws coloured dots at annotation
positions on a blank background.

# Credits

`windingcheck` is a scorer and a generator built on other people's data and formats. It would
not exist without them.

## Data

- **Human winding annotations of PHerc. Paris 4** (`relative_windings.json`, `same_windings.json`,
  `abs_winding.json`) and the fitted **umbilicus** (`umbilicus.json`): the `spiral-input` dataset
  (2026-07), made and curated by the **Vesuvius Challenge team**. The dataset page and its README
  do not name the individual annotators.
  <https://scrollprize.org/data_datasets#spiral-input-2026-07>,
  <https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/>
- **Neural ray crossings** (`winding_inference/`, 22.7 M crossings on 1.15 M rays): published by
  the **Vesuvius Challenge team** with the same dataset. The manifest records the villa
  `neural_tracing` winding model (`winding_model_3d_large_11`) and villa's
  `export_spiral_supervision` export (2026-08).
- **CT scans of PHerc. Paris 4**: the annotations and crossings above sit in the frame of the
  2026 Paris 4 scan `20260411134726` (2.4 µm; the crossings' manifest names its 2× downsampled
  copy), released by the Vesuvius Challenge (Angelotti et al., *Vesuvius Challenge – CT Scans of
  Herculaneum Papyri*, 2026). PHerc. Paris 4 is also part of the EduceLab-Scrolls Dataset,
  © EduceLab / The University of Kentucky (Parsons, Parker, Chapman, Hayashida & Seales, 2023).
  See `DATA_NOTICE.md` for both citations.

## Formats and tools

- **VC3D / Volume Cartographer** (ScrollPrize/villa, `volume-cartographer/`): the point-collection
  JSON format (`vc_pointcollections_json_version` 1) that `windingcheck` reads and writes.
  Originally Volume Cartographer by EduceLab (Seales, Parker, Parsons, Chapman); VC3D by
  Hendrik Schilling, Forrest McDonald, Philip Allgaier and the villa contributors.
  <https://github.com/ScrollPrize/villa>
- **villa spiral fitting** (`spiral-fitting/`, `point_collection.load_point_collection`,
  `fit_spiral.py`): the loader I test the output against, and the consumer of relative-winding
  annotations that this tool is meant to feed. Tested at villa commit `47111d1`.
- The crossing-file layout read by `windingcheck/crossings.py` follows villa's
  `export_spiral_supervision` output and its `manifest.json`.

## Prior work on the same problem

These projects generate or score winding constraints. I did not use their code, but my survey
and my limits draw on what they found. The README's *Related work* section compares their
metrics with mine.

- **winding-ruler** by pscamillo: winding evidence for the spiral fit; 93% recovery of human
  relative-winding labels on adjacent pairs, and the finding that three increasingly accurate
  constraint generators each degraded the fit. <https://github.com/pscamillo/winding-ruler>
- **winding-sync** by Joseph Balmaceda: relative winding constraints from raw CT, reconciled by L1
  integer synchronisation. <https://github.com/abundantjoe/winding-sync>
- **constraint-gauge** by pscamillo: a benchmark that scores winding-constraint generators against
  the Paris 4 human annotations and human-verified meshes, with criteria fixed before
  measurement. It is the closest prior work to `windingcheck score`.
  <https://github.com/pscamillo/constraint-gauge>

## Code

No third-party code is copied or adapted into `windingcheck`. The point-collection reader and
writer are my own implementation of the VC3D format. Runtime libraries: NumPy and SciPy
(BSD-3-Clause); Matplotlib for the optional plot (PSF-based Matplotlib licence).

## Author

Johannes Pfeffer (`jo-tud`).

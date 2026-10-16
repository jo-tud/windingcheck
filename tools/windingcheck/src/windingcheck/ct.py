"""Point sampling of the CT volume for crossing features, chunk by chunk.

Crossings are in the reference frame of the organisers' winding model: level 2 of the public
PHerc. Paris 4 2.4 um volume (9.6 um voxels; the manifest's ``ct_mask_filter`` maps reference
coordinates to its CT at scale 0.25 of level 0 of a ds2 copy, which is level 2 here).

Points are grouped by the 128^3 chunk their trilinear stencil touches, so a sparse point set (a
few samples along each ray) pulls only the chunks it needs. Chunks come through inkcheck's
``ZarrLevel``, which keeps the shared, size-capped disk cache.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PARIS4_VOLUME = ("s3://vesuvius-challenge-open-data/PHercParis4/volumes/"
                 "20260411134726-2.400um-0.2m-78keV-masked.zarr")
PARIS4_LEVEL = 2


def _zarr_level(url, level):
    # prefer the inkcheck of this checkout over an installed copy from another branch
    for parent in Path(__file__).resolve().parents:
        src = parent / "tools" / "inkcheck" / "src"
        if src.is_dir():
            if "inkcheck" not in sys.modules:
                sys.path.insert(0, str(src))
            break
    from inkcheck.volume import ZarrLevel
    return ZarrLevel(url, level)


class CTSampler:
    """Trilinear samples of one zarr level at (N, 3) zyx points in that level's voxels."""

    def __init__(self, url=PARIS4_VOLUME, level=PARIS4_LEVEL, workers=16):
        self.arr = _zarr_level(url, level)
        self.chunks = np.asarray(self.arr.meta.chunks)
        self.shape = np.asarray(self.arr.shape)
        self.workers = workers

    def stats(self):
        return self.arr.stats()

    def voxels(self, v: np.ndarray) -> np.ndarray:
        """Values at integer zyx voxels (M, 3); 0 outside the volume or in missing chunks."""
        v = np.asarray(v, np.int64)
        out = np.zeros(len(v), np.float32)
        inside = np.all((v >= 0) & (v < self.shape), axis=1)
        idx = np.flatnonzero(inside)
        if len(idx) == 0:
            return out
        c = v[idx] // self.chunks
        nc = (self.shape + self.chunks - 1) // self.chunks
        cid = (c[:, 0] * nc[1] + c[:, 1]) * nc[2] + c[:, 2]
        order = np.argsort(cid, kind="stable")
        cid_s = cid[order]
        starts = np.flatnonzero(np.r_[True, cid_s[1:] != cid_s[:-1]])
        ends = np.r_[starts[1:], len(cid_s)]

        def one(se):
            s, e = se
            sel = idx[order[s:e]]
            cz, cy, cx = (int(x) for x in v[sel[0]] // self.chunks)
            ch = self.arr._chunk(cz, cy, cx)
            if ch is None:
                return
            loc = v[sel] - np.array([cz, cy, cx]) * self.chunks
            out[sel] = ch[loc[:, 0], loc[:, 1], loc[:, 2]]

        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(one, zip(starts, ends)))
        return out

    def sample(self, zyx: np.ndarray) -> np.ndarray:
        """Trilinear samples at float zyx points (N, 3)."""
        p = np.asarray(zyx, np.float64)
        f = np.floor(p).astype(np.int64)
        w = p - f
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        vals = self.voxels((f[:, None, :] + corners[None]).reshape(-1, 3)).reshape(len(p), 8)
        wz = np.where(corners[:, 0] == 1, w[:, [0]], 1 - w[:, [0]])
        wy = np.where(corners[:, 1] == 1, w[:, [1]], 1 - w[:, [1]])
        wx = np.where(corners[:, 2] == 1, w[:, [2]], 1 - w[:, [2]])
        return (vals * wz * wy * wx).sum(1)

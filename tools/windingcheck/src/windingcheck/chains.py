"""Chains: the one shape every winding source is turned into.

A chain is an ordered set of points in full-resolution voxels (xyz, VC3D order) with a relative
winding level per point. Level differences inside one chain are the constraint; levels in two
different chains are not comparable. A same-winding collection is a chain whose levels are all 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chains:
    xyz: np.ndarray      # (N, 3) float64, voxels, x y z
    level: np.ndarray    # (N,) float64, relative winding inside the chain
    chain: np.ndarray    # (N,) int64, chain id, contiguous 0..C-1
    names: list[str]     # per chain id

    def __len__(self) -> int:
        return len(self.level)

    @property
    def n_chains(self) -> int:
        return len(self.names)

    def subset(self, chain_ids) -> "Chains":
        keep = np.isin(self.chain, np.asarray(list(chain_ids)))
        old = np.unique(self.chain[keep])
        remap = np.full(self.n_chains, -1, np.int64)
        remap[old] = np.arange(len(old))
        return Chains(self.xyz[keep], self.level[keep], remap[self.chain[keep]],
                      [self.names[i] for i in old])

    def sorted(self) -> "Chains":
        """Sort points by (chain, level) so chain neighbours are adjacent in memory."""
        order = np.lexsort((self.level, self.chain))
        return Chains(self.xyz[order], self.level[order], self.chain[order], self.names)


def from_arrays(xyz, level, chain, names=None) -> Chains:
    chain = np.asarray(chain)
    uniq, inv = np.unique(chain, return_inverse=True)
    names = names if names is not None else [str(u) for u in uniq]
    return Chains(np.asarray(xyz, np.float64).reshape(-1, 3), np.asarray(level, np.float64),
                  inv.astype(np.int64), list(names))


def load_pcl(path, *, same_winding: bool | None = None) -> Chains:
    """Read a VC3D point-collection JSON (``vc_pointcollections_json_version`` 1).

    A collection whose points carry ``wind_a`` is a relative chain. A collection without
    ``wind_a`` is read as a same-winding collection (all levels 0). ``same_winding=True`` forces
    all collections to level 0, as the spiral fitter does for ``same_windings.json``.
    """
    data = json.loads(Path(path).read_text())
    xyz, level, chain, names = [], [], [], []
    for cid, col in data["collections"].items():
        pts = list(col["points"].values())
        if not pts:
            continue
        wind = [p.get("wind_a") for p in pts]
        is_same = same_winding if same_winding is not None else all(w is None for w in wind)
        k = len(names)
        names.append(f"{col.get('name', cid)}#{cid}")
        for p, w in zip(pts, wind):
            xyz.append(p["p"])
            level.append(0.0 if is_same else (np.nan if w is None else float(w)))
            chain.append(k)
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    level = np.asarray(level, np.float64)
    chain = np.asarray(chain, np.int64)
    ok = np.isfinite(level) & np.isfinite(xyz).all(1)
    return Chains(xyz[ok], level[ok], chain[ok], names)


def save_pcl(chains: Chains, path, *, absolute: bool = False, color=(0.9, 0.5, 0.1),
             same_winding: bool = False, creation_time_ms: int = 0, tags=None) -> None:
    """Write chains as a VC3D point-collection JSON that VC3D and ``fit_spiral.py`` load.

    ``same_winding=True`` omits ``wind_a`` (the format of ``same_windings.json``).
    ``tags``: optional per-chain dict of strings, written as the collection's VC3D ``tags``.
    """
    order = np.argsort(chains.chain, kind="stable")
    bounds = np.searchsorted(chains.chain[order], np.arange(chains.n_chains + 1))
    pid = 1
    with open(path, "w") as f:      # streamed, one collection at a time: bounded memory
        f.write('{"vc_pointcollections_json_version": "1", "collections": {')
        for k, name in enumerate(chains.names):
            m = order[bounds[k]:bounds[k + 1]]
            pts = {}
            for p, lv in zip(chains.xyz[m].tolist(), chains.level[m].tolist()):
                pts[str(pid)] = {"p": p, "creation_time": int(creation_time_ms),
                                 "wind_a": None if same_winding else lv}
                pid += 1
            col = {"name": name, "color": list(color),
                   "metadata": {"winding_is_absolute": bool(absolute)}, "points": pts}
            if tags is not None:
                col["tags"] = {str(a): str(b) for a, b in tags[k].items()}
            f.write((", " if k else "") + json.dumps(str(k + 1)) + ": " + json.dumps(col))
        f.write("}}")

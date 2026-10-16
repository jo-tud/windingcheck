"""Archimedean spiral priors: winding from distance to the umbilicus, no image data.

These are the floor any generator has to beat. The sheet spacing is fitted on training chains.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .score import Truth


class Umbilicus:
    def __init__(self, path):
        cp = json.loads(Path(path).read_text())["control_points"]
        zyx = np.array(sorted((p["z"], p["y"], p["x"]) for p in cp), np.float64)
        self.z, self.y, self.x = zyx[:, 0], zyx[:, 1], zyx[:, 2]

    def polar(self, xyz):
        xyz = np.asarray(xyz, np.float64)
        cx = np.interp(xyz[:, 2], self.z, self.x)
        cy = np.interp(xyz[:, 2], self.z, self.y)
        dx, dy = xyz[:, 0] - cx, xyz[:, 1] - cy
        return np.hypot(dx, dy), np.arctan2(dy, dx)


def adjacent_spacing(truth: Truth, umb: Umbilicus, chain_ids):
    """Radial gap and mid radius of every adjacent click pair with Δw = 1 in the given chains."""
    ia, ib = truth.pairs(adjacent_only=True)
    m = np.isin(truth.chains.chain[ia], np.asarray(list(chain_ids)))
    m &= (truth.chains.level[ib] - truth.chains.level[ia]) == 1
    ra, _ = umb.polar(truth.chains.xyz[ia[m]])
    rb, _ = umb.polar(truth.chains.xyz[ib[m]])
    return rb - ra, 0.5 * (ra + rb)


class SpiralPrior:
    """Δw between two points = ∫ dr / s(r) + direction · Δθ / 2π."""

    def __init__(self, umb: Umbilicus, r_knots, s_knots, direction=0):
        self.umb = umb
        self.r_knots = np.asarray(r_knots, np.float64)
        self.s_knots = np.asarray(s_knots, np.float64)
        self.direction = direction
        grid = np.linspace(0.0, 20_000.0, 40_001)
        inv = 1.0 / np.interp(grid, self.r_knots, self.s_knots)
        self._grid = grid
        self._F = np.r_[0.0, np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(grid))]

    def winding(self, xyz):
        r, th = self.umb.polar(xyz)
        return np.interp(r, self._grid, self._F) + self.direction * th / (2 * np.pi)

    def delta(self, xyz_a, xyz_b):
        ra, ta = self.umb.polar(xyz_a)
        rb, tb = self.umb.polar(xyz_b)
        dth = (tb - ta + np.pi) % (2 * np.pi) - np.pi
        F = lambda r: np.interp(r, self._grid, self._F)
        return F(rb) - F(ra) + self.direction * dth / (2 * np.pi)

    def to_json(self):
        return {"r_knots": self.r_knots.round(2).tolist(), "s_knots": self.s_knots.round(3).tolist(),
                "direction": self.direction}


def fit_prior(truth: Truth, umb: Umbilicus, train_chain_ids, *, bins=1) -> SpiralPrior:
    gap, mid = adjacent_spacing(truth, umb, train_chain_ids)
    gap, mid = gap[gap > 0], mid[gap > 0]
    if bins <= 1:
        s = float(np.median(gap))
        prior = SpiralPrior(umb, [0.0, 1.0], [s, s])
    else:
        edges = np.quantile(mid, np.linspace(0, 1, bins + 1))
        r_k, s_k = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (mid >= lo) & (mid <= hi)
            r_k.append(float(np.median(mid[m])))
            s_k.append(float(np.median(gap[m])))
        prior = SpiralPrior(umb, r_k, s_k)
    # the angular term barely matters on radial chains; pick its sign on the training chains
    from .score import score_field
    best = None
    for d in (0, 1, -1):
        prior.direction = d
        ex = score_field(truth, prior.delta, chain_ids=train_chain_ids)["exact"]
        if best is None or ex > best[0]:
            best = (ex, d)
    prior.direction = best[1]
    return prior

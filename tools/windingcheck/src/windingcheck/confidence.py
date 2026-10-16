"""A label-free confidence for every step of a ray-crossing set, and a generator that keeps the
steps worth trusting.

A **step** is crossing ``k`` to crossing ``k + 1`` on one ray: the claim "these two points are
one sheet apart". Features come from the crossings alone (no CT, no labels, no ``seed_winding``):

* ``log_s_ray``: log(step length / median step length on the same ray), and its square;
* ``nbr_agree``, ``log_support``: do neighbouring rays see the next sheet at the same distance?
* ``edge``: distance from the slab centre along the ray (the phase is extrapolated at the ends);
* ``oblique``: angle between the ray and the radial direction from the umbilicus;
* ``log_r``: radius from the umbilicus.

A logistic model maps them to P(step is one true sheet). A pair constraint on a ray is a run of
steps; its confidence is the product over the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .chains import Chains
from .crossings import RayStore

FEATURES = ("log_s_ray", "log_s_ray2", "nbr_agree", "log_support", "edge", "oblique", "log_r")
NBR_RADIUS = 40.0     # voxels, neighbour crossings considered around crossing k
NBR_K = 64            # nearest crossings looked at per step
GAP_TOL = 0.35        # neighbour gap agrees if within this fraction of the step


def step_index(store: RayStore) -> np.ndarray:
    """First-crossing index of every step (crossing k with k + 1 on the same ray)."""
    return np.flatnonzero(store.ray[1:] == store.ray[:-1])


def _ray_median_step(store: RayStore, k: np.ndarray, s: np.ndarray) -> np.ndarray:
    ray = store.ray[k]
    order = np.lexsort((s, ray))
    r_sorted = ray[order]
    starts = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    med = s[order[starts + (counts - 1) // 2]]
    per_ray = np.zeros(len(store.ray_dir))
    per_ray[r_sorted[starts]] = med
    return per_ray[ray]


def _neighbour_agreement(store: RayStore, k: np.ndarray, s: np.ndarray, tree: cKDTree,
                         *, chunk=400_000, workers=-1):
    agree = np.zeros(len(k), np.float64)
    support = np.zeros(len(k), np.int64)
    n = len(store)
    for c0 in range(0, len(k), chunk):
        kk = k[c0:c0 + chunk]
        ss = s[c0:c0 + chunk]
        d = store.ray_dir[store.ray[kk]]
        _, nb = tree.query(store.xyz[kk], k=NBR_K, distance_upper_bound=NBR_RADIUS, workers=workers)
        q = np.repeat(np.arange(len(kk)), NBR_K)
        j = nb.ravel()
        ok = j < n
        q, j = q[ok], j[ok]
        ok = store.ray[j] != store.ray[kk[q]]
        q, j = q[ok], j[ok]
        u = np.einsum("ij,ij->i", store.xyz[j] - store.xyz[kk[q]], d[q])
        ok = np.abs(u) < 0.5 * ss[q]
        q, j, u = q[ok], j[ok], u[ok]
        # one crossing per (step, neighbour ray): the nearest along this ray's direction
        order = np.lexsort((np.abs(u), store.ray[j], q))
        q, j = q[order], j[order]
        first = np.r_[True, (q[1:] != q[:-1]) | (store.ray[j][1:] != store.ray[j][:-1])]
        q, j = q[first], j[first]
        # the neighbour's next sheet along this ray's direction: j + 1 or j - 1 on its own ray
        gap = np.full(len(j), np.inf)
        for nxt in (j + 1, j - 1):
            valid = (nxt >= 0) & (nxt < n)
            valid[valid] &= store.ray[nxt[valid]] == store.ray[j[valid]]
            g = np.full(len(j), np.inf)
            g[valid] = np.einsum("ij,ij->i", store.xyz[nxt[valid]] - store.xyz[j[valid]], d[q[valid]])
            g[g <= 0] = np.inf
            gap = np.minimum(gap, g)
        seen = np.isfinite(gap)
        q, gap = q[seen], gap[seen]
        hit = np.abs(gap - ss[q]) < GAP_TOL * ss[q]
        support[c0:c0 + chunk] = np.bincount(q, minlength=len(kk))
        agree[c0:c0 + chunk] = np.bincount(q, weights=hit.astype(np.float64), minlength=len(kk))
    frac = np.where(support > 0, agree / np.maximum(support, 1), 0.5)
    return frac, support


def step_features(store: RayStore, umb, k: np.ndarray | None = None, *, tree: cKDTree | None = None,
                  workers=-1) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix (len(k), len(FEATURES)) for the steps starting at crossings ``k``.

    The per-ray median step length uses every step of the ray, whatever ``k`` holds.
    """
    all_k = step_index(store)
    k = all_k if k is None else np.asarray(k, np.int64)
    s_all = np.linalg.norm(store.xyz[all_k + 1] - store.xyz[all_k], axis=1)
    med_all = _ray_median_step(store, all_k, s_all)
    pos = np.searchsorted(all_k, k)
    if not np.array_equal(all_k[np.minimum(pos, len(all_k) - 1)], k):
        raise ValueError("k holds a crossing that does not start a step")
    s, med = s_all[pos], med_all[pos]
    log_s = np.log(np.maximum(s, 1e-3) / np.maximum(med, 1e-3))
    if tree is None:
        tree = cKDTree(store.xyz)
    agree, support = _neighbour_agreement(store, k, s, tree, workers=workers)
    mid_t = 0.5 * (store.t[k].astype(np.float64) + store.t[k + 1])
    half = 0.5 * (store.ray_length - 1)
    edge = np.abs(mid_t - half) / half
    mid = 0.5 * (store.xyz[k] + store.xyz[k + 1])
    r, th = umb.polar(mid)
    radial = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)
    oblique = 1.0 - np.abs(np.einsum("ij,ij->i", radial, store.ray_dir[store.ray[k]]))
    X = np.stack([log_s, log_s ** 2, agree, np.log1p(support), edge, oblique,
                  np.log(np.maximum(r, 1.0))], 1)
    return X, s


@dataclass
class StepModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray       # intercept first
    features: tuple = FEATURES

    @classmethod
    def fit(cls, X, y, *, l2=1e-2, cols=None):
        cols = np.arange(X.shape[1]) if cols is None else np.asarray(cols)
        mean, std = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), ((X - mean) / std)[:, cols]]
        y = np.asarray(y, np.float64)

        def loss(w):
            z = Z @ w
            nll = np.sum(np.logaddexp(0, z) - y * z)
            p = 1 / (1 + np.exp(-z))
            g = Z.T @ (p - y)
            return nll + l2 * w[1:] @ w[1:], g + 2 * l2 * np.r_[0, w[1:]]

        w = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        coef = np.zeros(X.shape[1] + 1)
        coef[0] = w[0]
        coef[1 + cols] = w[1:]
        return cls(mean, std, coef)

    def prob(self, X):
        z = self.coef[0] + ((X - self.mean) / self.std) @ self.coef[1:]
        return 1 / (1 + np.exp(-z))

    def to_json(self):
        return {"features": list(self.features), "mean": self.mean.tolist(),
                "std": self.std.tolist(), "coef": self.coef.tolist()}

    @classmethod
    def from_json(cls, d):
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]), np.asarray(d["coef"]),
                   tuple(d["features"]))

    def save(self, path, **extra):
        Path(path).write_text(json.dumps(dict(self.to_json(), **extra), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_json(json.loads(Path(path).read_text()))


def pair_steps(ca, cb):
    """Steps (by first crossing) between two crossings of one ray, as (owner index, step k)."""
    lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
    n = hi - lo
    owner = np.repeat(np.arange(len(lo)), n)
    k = np.repeat(lo, n) + (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n))
    return owner, k


def generate(store: RayStore, step_p: np.ndarray, tau: float, *, min_points=2) -> tuple[Chains, np.ndarray]:
    """Cut every ray at steps with confidence < tau; each run of >= ``min_points`` crossings is
    one chain. Returns the chains and each chain's confidence (product of its steps)."""
    k = step_index(store)
    good = np.zeros(len(store), bool)        # good[i]: step i -> i + 1 kept
    good[k] = step_p >= tau
    new_run = np.r_[True, ~good[:-1]]
    run = np.cumsum(new_run) - 1
    counts = np.bincount(run)
    keep = counts[run] >= min_points
    uniq, inv = np.unique(run[keep], return_inverse=True)
    logp = np.zeros(len(store))
    logp[k] = np.log(np.maximum(step_p, 1e-12))
    in_run_step = good & np.r_[run[1:] == run[:-1], False]
    conf = np.exp(np.bincount(run, weights=np.where(in_run_step, logp, 0.0)))[uniq]
    start = np.flatnonzero(new_run)[uniq]
    names = [f"ray{r}@{lv}" for r, lv in zip(store.ray[start], store.level[start])]
    ch = Chains(store.xyz[keep], store.level[keep].astype(np.float64), inv.astype(np.int64), names)
    return ch, conf

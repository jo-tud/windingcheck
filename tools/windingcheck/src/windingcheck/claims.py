"""Claim-level confidence (v2): score a pair of crossings on one ray as a whole.

A **claim** is two crossings ``a < b`` of one ray, ``k = b - a`` steps apart, saying "these points
are ``level[b] - level[a]`` sheets apart". v1 scored steps and multiplied them; the neural
model's errors (an extra crossing in a compressed stretch) are correlated along the ray, so the
product was miscalibrated. v2 fits P(claim exact) on claims directly.

Features (``CLAIM_FEATURES``), all label-free:

* ``log_k``;
* ``v1_mean``, ``v1_min``: mean / min v1 step logit over the claim's steps;
* ``log_s_min``, ``log_s_max``: min / max step length relative to the ray median;
* ``edge_max``: how close the claim gets to a ray end;
* ``log_r``: radius from the umbilicus;
* ``ct_contrast_min``: weakest CT gap between consecutive crossings;
* ``ct_mid_mean``: CT at the intermediate crossings; ``ct_end_mean``: CT at the two ends.

The generator walks each ray: a chain starting at ``s`` extends to ``j`` while both the claim
``s -> j`` and the step ``j - 1 -> j`` reach ``tau``.
"""

from __future__ import annotations

import numpy as np

from .chains import Chains
from .confidence import FEATURES as STEP_FEATURES, StepModel, pair_steps, step_index, step_features
from .crossings import RayStore

CLAIM_FEATURES = ("log_k", "v1_mean", "v1_min", "log_s_min", "log_s_max", "edge_max", "log_r",
                  "ct_contrast_min", "ct_mid_mean", "ct_end_mean")
N_NO_CT = 7
CROSS_OFFSETS = (-1.5, -0.75, 0.0, 0.75, 1.5)   # voxels along the ray around a crossing
STEP_FRACS = np.linspace(0.15, 0.85, 8)          # step interior samples


def subset_rays(store: RayStore, rays) -> tuple[RayStore, np.ndarray]:
    """Crossings of the given rays only (ray ids unchanged). Returns the store and the original
    crossing index of each kept crossing."""
    keep = np.flatnonzero(np.isin(store.ray, np.asarray(rays)))
    return RayStore(store.xyz[keep], store.level[keep], store.t[keep], store.ray[keep],
                    store.ray_dir, store.ray_length), keep


def ct_per_crossing(store: RayStore, sampler, batch=400_000):
    """(crossing CT, step-interior minimum CT by step start index). CT at a crossing is the max
    of samples at ``CROSS_OFFSETS`` along the ray; the interior minimum is over ``STEP_FRACS``."""
    n = len(store)
    cross = np.zeros(n, np.float32)
    d = store.ray_dir[store.ray]
    for c0 in range(0, n, batch):
        sl = slice(c0, min(c0 + batch, n))
        p = np.concatenate([store.xyz[sl] + o * d[sl] for o in CROSS_OFFSETS])
        cross[sl] = sampler.sample(p[:, ::-1]).reshape(len(CROSS_OFFSETS), -1).max(0)
    k = step_index(store)
    valley = np.full(n, np.nan, np.float32)
    for c0 in range(0, len(k), batch):
        kk = k[c0:c0 + batch]
        a, b = store.xyz[kk], store.xyz[kk + 1]
        p = (a[:, None, :] + STEP_FRACS[None, :, None] * (b - a)[:, None, :]).reshape(-1, 3)
        valley[kk] = sampler.sample(p[:, ::-1]).reshape(len(kk), len(STEP_FRACS)).min(1)
    return cross, valley


class StepTable:
    """Per-step quantities, indexed by the step's first crossing (length = number of crossings).

    ``X`` (v1 step features of ``step_index(store)``) can be passed in, e.g. computed on a larger
    crossing set so that neighbour features see rays outside ``store``."""

    def __init__(self, store: RayStore, umb, step_model: StepModel, ct_cross=None, ct_valley=None,
                 *, tree=None, X=None):
        self.n = len(store)
        self.k = step_index(store)
        if X is None:
            X, _ = step_features(store, umb, self.k, tree=tree)
        self.X = X
        self.set_step_model(step_model)
        f = {name: i for i, name in enumerate(STEP_FEATURES)}
        self.log_s = self._full(X[:, f["log_s_ray"]])
        self.edge = self._full(X[:, f["edge"]])
        self.log_r = self._full(X[:, f["log_r"]])
        if ct_cross is not None:
            cc = np.asarray(ct_cross, np.float64)
            lo = np.minimum(cc[self.k], cc[self.k + 1])
            self.contrast = self._full((lo - ct_valley[self.k]) / np.maximum(lo, 1.0))
            self.ct = cc
        else:
            self.contrast = self.ct = None

    def _full(self, v):
        out = np.full(self.n, np.nan)
        out[self.k] = v
        return out

    def set_step_model(self, step_model: StepModel):
        p = np.clip(step_model.prob(self.X), 1e-9, 1 - 1e-9)
        self.v1 = self._full(np.log(p / (1 - p)))


def claim_features(tab: StepTable, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Feature matrix (len(a), len(CLAIM_FEATURES)) for claims a -> b on one ray (any order)."""
    lo, hi = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
    m = len(lo)
    owner, ks = pair_steps(lo, hi)
    k = (hi - lo).astype(np.float64)

    def red(v, how):
        out = np.full(m, np.inf if how == "min" else -np.inf)
        (np.minimum if how == "min" else np.maximum).at(out, owner, v[ks])
        return out

    def mean(v):
        return np.bincount(owner, weights=v[ks], minlength=m) / k

    cols = [np.log(k), mean(tab.v1), red(tab.v1, "min"), red(tab.log_s, "min"),
            red(tab.log_s, "max"), red(tab.edge, "max"), mean(tab.log_r)]
    if tab.ct is not None:
        end = 0.5 * (tab.ct[lo] + tab.ct[hi])
        inner = ks != lo[owner]            # steps that start at an intermediate crossing
        mid_sum = np.bincount(owner[inner], weights=tab.ct[ks[inner]], minlength=m)
        mid = np.where(k > 1, mid_sum / np.maximum(k - 1, 1), end)
        cols += [red(tab.contrast, "min"), mid, end]
    return np.stack(cols, 1)


def all_ray_claims(store: RayStore):
    """Every claim (a < b) on every ray, grouped by ray. Returns (a, b, ray_start, ray_n)."""
    r = store.ray
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    n = np.diff(np.r_[starts, len(r)])
    A, B = _ray_claims(starts, n)
    return A, B, starts, n


def _greedy(a, b, p, starts, n, tau, *, min_points=2):
    """v2's greedy walk. Returns (crossing index, chain id from 0, chain confidence)."""
    off = 0
    keep_idx, chain_id, conf = [], [], []
    cid = 0
    for s, cnt in zip(starts, n):
        m = cnt * (cnt - 1) // 2
        o = off
        off += m
        if cnt < 2:
            continue
        P = np.zeros((cnt, cnt))
        P[a[o:o + m] - s, b[o:o + m] - s] = p[o:o + m]
        st = 0
        j = 1
        while st < cnt - 1:
            while j < cnt and P[st, j] >= tau and P[j - 1, j] >= tau:
                j += 1
            last = j - 1
            if last - st + 1 >= min_points:
                keep_idx.append(np.arange(st, last + 1) + s)
                chain_id.append(np.full(last - st + 1, cid))
                conf.append(P[st, last])
                cid += 1
            st = last + 1          # the next chain starts at the crossing that failed
            j = st + 1
    if not keep_idx:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    return np.concatenate(keep_idx), np.concatenate(chain_id), np.asarray(conf)


def _to_chains(store: RayStore, idx, ch):
    if len(idx) == 0:
        return Chains(np.empty((0, 3)), np.empty(0), np.empty(0, np.int64), [])
    first = idx[np.r_[True, ch[1:] != ch[:-1]]]
    names = [f"ray{r}@{lv}" for r, lv in zip(store.ray[first], store.level[first])]
    return Chains(store.xyz[idx], store.level[idx].astype(np.float64), ch.astype(np.int64), names)


def generate_claims(store: RayStore, a, b, p, starts, n, tau, *, min_points=2):
    """Greedy chains: extend while P(s -> j) >= tau and P(j-1 -> j) >= tau.

    ``a, b, p`` are all claims of :func:`all_ray_claims` with their confidences. Returns the
    chains and each chain's end-to-end confidence."""
    idx, ch, conf = _greedy(a, b, p, starts, n, tau, min_points=min_points)
    return _to_chains(store, idx, ch), conf


def _ray_claims(starts, n):
    A, B = [], []
    for s, cnt in zip(starts, n):
        ia, ib = np.triu_indices(cnt, 1)
        A.append(ia + s)
        B.append(ib + s)
    if not A:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(A), np.concatenate(B)


def generate_blocked(store: RayStore, tab: StepTable, model: StepModel, tau, *, block_rays=20_000,
                     min_points=2, progress=None):
    """v2's generator on any number of rays with bounded memory: rays are taken in blocks, each
    block's claims are scored with ``model`` (whose feature count picks the CT or no-CT set) and
    walked with the same greedy rule. Same output as :func:`generate_claims` on all claims."""
    r = store.ray
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    n = np.diff(np.r_[starts, len(r)])
    nf = len(model.mean)
    I, C, P = [], [], []
    base = 0
    for b0 in range(0, len(starts), block_rays):
        st, nn = starts[b0:b0 + block_rays], n[b0:b0 + block_rays]
        A, B = _ray_claims(st, nn)
        p = model.prob(claim_features(tab, A, B)[:, :nf]) if len(A) else np.empty(0)
        idx, ch, conf = _greedy(A, B, p, st, nn, tau, min_points=min_points)
        I.append(idx)
        C.append(ch + base)
        P.append(conf)
        base += len(conf)
        if progress:
            progress(b0 + len(st), len(starts))
    idx, ch = np.concatenate(I), np.concatenate(C)
    return _to_chains(store, idx, ch), np.concatenate(P)

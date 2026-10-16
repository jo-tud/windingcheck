"""Score winding sources against human relative-winding chains.

Truth is a set of human chains (``relative_windings.json``): clicks along a ray, one per sheet,
``wind_a`` counting sheets. Only differences inside one chain are known.

Two ways to score:

* ``score_chains``: the candidate is itself chains (a VC3D point-collection file, neural ray
  crossings). Candidate points are matched to clicks on the same sheet, and every pair of
  candidate points of one candidate chain that lands on two clicks of one human chain is a test.
* ``score_field``: the candidate is a function that gives a continuous winding at any point
  (a spiral prior, a fitted spiral). It is evaluated at every click, so coverage is complete.

Before candidate chains are scored, chains that share a point are joined (``join_shared``), as a
spiral fitter would compose them. Cutting a chain into pieces that share their end points then
cannot change its score: on a 2-point chain the scorer can only catch a reversed sign, so scoring
the pieces one by one would hide every wrong step whose far end matches no click.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .chains import Chains

DW_BINS = ((1, 1), (2, 3), (4, 7), (8, 10_000))


@dataclass
class Truth:
    chains: Chains            # sorted by (chain, level)
    normal: np.ndarray        # (N, 3) unit chain direction at each click (~ sheet normal), nan if unknown
    spacing: np.ndarray       # (N,) distance to the nearest neighbouring click along the normal
    tol_n: np.ndarray         # (N,) normal matching tolerance
    tol_t: float              # tangential matching tolerance

    def pairs(self, adjacent_only=False):
        """All within-chain click pairs (i < j by level) as index arrays."""
        c = self.chains.chain
        starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
        ends = np.r_[starts[1:], len(c)]
        ii, jj = [], []
        for s, e in zip(starts, ends):
            n = e - s
            if n < 2:
                continue
            if adjacent_only:
                a = np.arange(s, e - 1)
                ii.append(a)
                jj.append(a + 1)
            else:
                a, b = np.triu_indices(n, 1)
                ii.append(a + s)
                jj.append(b + s)
        if not ii:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return np.concatenate(ii), np.concatenate(jj)


def build_truth(chains: Chains, *, tol_t=25.0, tol_frac=0.35, tol_min=1.5, tol_max=4.0) -> Truth:
    ch = chains.sorted()
    n = len(ch)
    normal = np.full((n, 3), np.nan)
    spacing = np.full(n, np.nan)
    c = ch.chain
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        p = ch.xyz[s:e]
        prev = np.r_[[p[1] - p[0]], p[1:] - p[:-1]]
        nxt = np.r_[p[1:] - p[:-1], [p[-1] - p[-2]]]
        d = prev + nxt
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        normal[s:e] = d
        gap_prev = np.abs(np.einsum("ij,ij->i", prev, d))
        gap_next = np.abs(np.einsum("ij,ij->i", nxt, d))
        gap_prev[0] = np.inf
        gap_next[-1] = np.inf
        spacing[s:e] = np.minimum(gap_prev, gap_next)
    tol_n = np.clip(tol_frac * spacing, tol_min, tol_max)
    tol_n[~np.isfinite(spacing)] = np.nan
    return Truth(ch, normal, spacing, tol_n, float(tol_t))


def match(truth: Truth, cand: Chains):
    """Same-sheet matches between clicks and candidate points.

    Returns arrays (truth_idx, cand_idx, normal_dist). For each (click, candidate chain) only the
    candidate point nearest along the normal is kept. A candidate point that matches two clicks of
    one human chain is ambiguous and dropped.
    """
    ok = np.isfinite(truth.tol_n)
    tidx = np.flatnonzero(ok)
    if len(cand) == 0 or len(tidx) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    tree = cKDTree(cand.xyz)
    radius = np.sqrt(truth.tol_t ** 2 + truth.tol_n[tidx] ** 2)
    hits = tree.query_ball_point(truth.chains.xyz[tidx], radius)
    n_hits = np.fromiter((len(h) for h in hits), np.int64, len(hits))
    if n_hits.sum() == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    ti = np.repeat(tidx, n_hits)
    cj = np.fromiter((j for h in hits for j in h), np.int64, int(n_hits.sum()))
    d = cand.xyz[cj] - truth.chains.xyz[ti]
    dn = np.abs(np.einsum("ij,ij->i", d, truth.normal[ti]))
    dt = np.sqrt(np.maximum((d * d).sum(1) - dn * dn, 0.0))
    keep = (dn < truth.tol_n[ti]) & (dt < truth.tol_t)
    ti, cj, dn = ti[keep], cj[keep], dn[keep]
    # drop candidate points that match more than one click of the same human chain
    key = cj * (truth.chains.n_chains + 1) + truth.chains.chain[ti]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    keep = cnt[inv] == 1
    ti, cj, dn = ti[keep], cj[keep], dn[keep]
    # one candidate point per (click, candidate chain): the nearest along the normal
    key = ti * (cand.n_chains + 1) + cand.chain[cj]
    order = np.lexsort((dn, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    sel = order[first]
    return ti[sel], cj[sel], dn[sel]


def join_shared(cand: Chains, tol: float = 1e-3):
    """Join candidate chains that share a point (two points of different chains closer than
    ``tol`` voxels).

    The levels of each joined chain are shifted so every shared point has one level, the shared
    duplicates are dropped, and the pieces become one chain. A component whose links disagree (a
    closed loop of shared points with a non-zero level sum) is left unjoined and counted in
    ``conflict_chains``. Returns (joined chains, index into ``cand`` of every kept point, summary).
    """
    src = np.arange(len(cand))
    info = {"tol": float(tol), "chains_in": int(cand.n_chains), "links": 0,
            "chains_out": int(cand.n_chains), "conflict_chains": 0, "points_merged": 0}
    if len(cand) < 2 or not tol > 0:
        return cand, src, info
    pairs = cKDTree(cand.xyz).query_pairs(tol, output_type="ndarray")
    pa, pb = pairs[:, 0], pairs[:, 1]
    ca, cb = cand.chain[pa], cand.chain[pb]
    cross = ca != cb
    pa, pb, ca, cb = pa[cross], pb[cross], ca[cross], cb[cross]
    info["links"] = int(len(pa))
    if len(pa) == 0:
        return cand, src, info
    # weighted union-find over the linked chains: off[c] - off[root] is kept along the tree;
    # a link says off[cb] - off[ca] = level[pa] - level[pb]
    nodes, inv = np.unique(np.r_[ca, cb], return_inverse=True)
    la, lb = inv[: len(ca)].tolist(), inv[len(ca):].tolist()
    d = (cand.level[pa] - cand.level[pb]).tolist()
    parent = list(range(len(nodes)))
    w = [0.0] * len(nodes)

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        acc = 0.0
        for y in reversed(path):     # compress: every node on the path points at the root
            acc += w[y]
            w[y] = acc
            parent[y] = x
        return x

    bad_roots = set()
    for a, b, dd in zip(la, lb, d):
        ra, rb = find(a), find(b)
        if ra == rb:
            if abs((w[b] - w[a]) - dd) > 1e-6:
                bad_roots.add(ra)
            continue
        parent[rb] = ra
        w[rb] = dd + w[a] - w[b]
        if rb in bad_roots:
            bad_roots.add(ra)
    root = np.fromiter((find(x) for x in range(len(nodes))), np.int64, len(nodes))
    off_local = np.asarray(w, np.float64)
    off_local[root == np.arange(len(nodes))] = 0.0
    bad = np.isin(root, np.fromiter(bad_roots, np.int64, len(bad_roots)))
    lab = np.arange(cand.n_chains)
    off = np.zeros(cand.n_chains)
    lab[nodes[~bad]] = nodes[root[~bad]]
    off[nodes[~bad]] = off_local[~bad]
    info["conflict_chains"] = int(bad.sum())
    # a link is merged away when both ends ended up in one chain at one level
    level = cand.level + off[cand.chain]
    same = (lab[ca] == lab[cb]) & (np.abs(level[pa] - level[pb]) < 1e-6)
    keep = np.ones(len(cand), bool)
    keep[pb[same]] = False
    info["points_merged"] = int((~keep).sum())
    uniq, newc = np.unique(lab[cand.chain[keep]], return_inverse=True)
    out = Chains(cand.xyz[keep], level[keep], newc.astype(np.int64), [cand.names[u] for u in uniq])
    info["chains_out"] = int(out.n_chains)
    return out, src[keep], info


def _metrics(true_dw, pred_dw):
    true_dw = np.asarray(true_dw, np.float64)
    pred_dw = np.asarray(pred_dw, np.float64)
    err = pred_dw - true_dw
    out = {"tests": int(len(err))}
    if len(err) == 0:
        return out
    out.update({
        "exact": float(np.mean(err == 0)),
        "within_1": float(np.mean(np.abs(err) <= 1)),
        "sign_errors": float(np.mean(pred_dw * true_dw < 0)),
        "mean_abs_error": float(np.mean(np.abs(err))),
    })
    by = {}
    for lo, hi in DW_BINS:
        m = (np.abs(true_dw) >= lo) & (np.abs(true_dw) <= hi)
        if m.any():
            by[f"{lo}-{hi}" if hi < 10_000 else f">={lo}"] = {
                "tests": int(m.sum()), "exact": float(np.mean(err[m] == 0)),
                "within_1": float(np.mean(np.abs(err[m]) <= 1))}
    m = true_dw == 0
    if m.any():
        by["0"] = {"tests": int(m.sum()), "exact": float(np.mean(err[m] == 0)),
                   "within_1": float(np.mean(np.abs(err[m]) <= 1))}
    out["by_true_dw"] = by
    return out


def score_chains(truth: Truth, cand: Chains, *, join=True, join_tol=1e-3, return_tests=False):
    """Score candidate chains. Δw of every candidate chain is level(b) - level(a).

    ``join``: first join chains that share a point (``join_shared``); ``join=False`` scores every
    chain on its own, as v0.1 did. Returned test indices ``ca``/``cb`` always index ``cand``."""
    src = np.arange(len(cand))
    jinfo = None
    if join:
        cand, src, jinfo = join_shared(cand, join_tol)
    ti, cj, _ = match(truth, cand)
    tc = truth.chains.chain[ti]
    cc = cand.chain[cj]
    order = np.lexsort((ti, cc, tc))
    ti, cj, tc, cc = ti[order], cj[order], tc[order], cc[order]
    grp = np.r_[True, (tc[1:] != tc[:-1]) | (cc[1:] != cc[:-1])]
    starts = np.flatnonzero(grp)
    ends = np.r_[starts[1:], len(ti)]
    A, B = [], []
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        a, b = np.triu_indices(e - s, 1)
        A.append(a + s)
        B.append(b + s)
    if A:
        A, B = np.concatenate(A), np.concatenate(B)
    else:
        A = B = np.empty(0, np.int64)
    ta, tb = ti[A], ti[B]
    true_dw = truth.chains.level[tb] - truth.chains.level[ta]
    pred_dw = cand.level[cj[B]] - cand.level[cj[A]]
    res = _metrics(true_dw, pred_dw)
    res.update(_coverage(truth, ta, tb))
    res["matched_clicks"] = int(len(np.unique(ti)))
    res["clicks"] = int(len(truth.chains))
    adj = (tb == ta + 1) & (truth.chains.chain[ta] == truth.chains.chain[tb])
    res["adjacent_exact"] = float(np.mean(pred_dw[adj] == true_dw[adj])) if adj.any() else None
    res["adjacent_tests"] = int(adj.sum())
    if jinfo is not None:
        res["join"] = jinfo
    if return_tests:
        return res, {"a": ta, "b": tb, "true_dw": true_dw, "pred_dw": pred_dw,
                     "ca": src[cj[A]], "cb": src[cj[B]]}
    return res


def _coverage(truth: Truth, ta, tb):
    all_i, all_j = truth.pairs()
    adj_i, adj_j = truth.pairs(adjacent_only=True)
    n = len(truth.chains) + 1
    covered = np.unique(np.minimum(ta, tb) * n + np.maximum(ta, tb))
    return {
        "human_pairs": int(len(all_i)),
        "human_pairs_covered": float(np.isin(all_i * n + all_j, covered).mean()) if len(all_i) else 0.0,
        "human_adjacent_pairs": int(len(adj_i)),
        "human_adjacent_covered": float(np.isin(adj_i * n + adj_j, covered).mean()) if len(adj_i) else 0.0,
    }


def score_field(truth: Truth, delta_fn, *, chain_ids=None, return_tests=False):
    """Score a winding field. ``delta_fn(xyz_a, xyz_b)`` returns predicted Δw (float) per pair;
    it is rounded to the nearest integer."""
    ia, ib = truth.pairs()
    if chain_ids is not None:
        m = np.isin(truth.chains.chain[ia], np.asarray(list(chain_ids)))
        ia, ib = ia[m], ib[m]
    true_dw = truth.chains.level[ib] - truth.chains.level[ia]
    pred_dw = np.rint(delta_fn(truth.chains.xyz[ia], truth.chains.xyz[ib]))
    res = _metrics(true_dw, pred_dw)
    if return_tests:
        return res, {"a": ia, "b": ib, "true_dw": true_dw, "pred_dw": pred_dw}
    return res


def loop_consistency(truth: Truth, same: Chains):
    """Check human labels against each other.

    A same-winding collection S that matches a click on chain A and a click on chain B says those
    two clicks are on one sheet. Two such collections S, S' linking the same chains A and B close
    a loop: the sheets between them, counted on A, must equal the count on B. Returns the loop
    holonomies (count on A minus count on B; 0 = consistent) and the number of (collection,
    chain) groups that matched more than one distinct ``wind_a`` (a direct contradiction).
    """
    ti, cj, _ = match(truth, same)
    tc = truth.chains.chain[ti]
    sc = same.chain[cj]
    w = truth.chains.level[ti]
    # per (collection, chain): the set of matched wind_a
    link = {}
    multi = 0
    for key in np.unique(sc * (truth.chains.n_chains + 1) + tc):
        s, c = divmod(int(key), truth.chains.n_chains + 1)
        vals = np.unique(w[(sc == s) & (tc == c)])
        if len(vals) > 1:
            multi += 1
            continue
        link.setdefault(c, {})[s] = vals[0]
    groups = sum(len(v) for v in link.values()) + multi
    # loops: chains A < B, collections S < S' that both link A and B
    hol = []
    chains_ = sorted(link)
    for ai, A in enumerate(chains_):
        for B in chains_[ai + 1:]:
            common = sorted(set(link[A]) & set(link[B]))
            for x in range(len(common)):
                for y in range(x + 1, len(common)):
                    S, S_ = common[x], common[y]
                    hol.append((link[A][S_] - link[A][S]) - (link[B][S_] - link[B][S]))
    return {"groups": int(groups), "groups_multi_valued": int(multi),
            "loops": int(len(hol)),
            "loops_consistent": float(np.mean(np.asarray(hol) == 0)) if hol else None,
            "holonomy_hist": {str(int(k)): int(v) for k, v in
                              zip(*np.unique(np.asarray(hol, np.int64), return_counts=True))} if hol else {}}

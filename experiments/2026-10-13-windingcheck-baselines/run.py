"""windingcheck baselines at equal coverage, per-pair precision, Δw >= 2, chain bootstrap.

    python experiments/2026-10-13-windingcheck-baselines/run.py CLAIMS_JSON [--crossings DIR]

CLAIMS_JSON is the full-volume output of ``windingcheck generate --claims``. It is streamed; only
the region rays' chains are kept. CPU only, no CT read, < 1 GB RAM. The thresholds in
the checks below were fixed before this script first ran.
"""
import argparse
import json
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "windingcheck" / "src"))

from windingcheck import data  # noqa: E402
from windingcheck.chains import Chains, load_pcl  # noqa: E402
from windingcheck.score import build_truth, score_chains  # noqa: E402

REGION_RADIUS = 60.0          # as ../2026-10-13-claims-noct-generator (v2's REGION_RADIUS)
SEEDS = range(10)
N_BOOT = 2000
BOOT_SEED = 12345
FIX_COST = 3
HUMAN_HOURS = (3.5, 7.0)


# ---------------------------------------------------------------- region crossings, shard by shard
def region_crossings(root, clicks_xyz):
    """Crossings of every ray with a crossing within REGION_RADIUS of a click, in file order.
    Global ray ids as in ``load_ray_store``. Returns xyz, level, ray (global)."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    tree = cKDTree(clicks_xyz)
    XYZ, LV, RAY = [], [], []
    base = 0
    for i, shard in enumerate(manifest["shards"]):
        d = root / shard.get("name", f"shard_{i}")
        origin = np.load(d / "ray_origin_zyx.npy")
        step = np.load(d / "ray_step_zyx.npy")
        ti = np.load(d / "crossing_t.npy")
        offsets = np.load(d / "crossing_offsets.npy").astype(np.int64)
        n_rays = len(origin)
        r = np.repeat(np.arange(n_rays, dtype=np.int64), np.diff(offsets[: n_rays + 1]))
        xyz = (origin[r].astype(np.float64) + ti[:, None].astype(np.float64) * step[r])[:, ::-1]
        dd, _ = tree.query(xyz, distance_upper_bound=REGION_RADIUS)
        keep = np.isin(r, np.unique(r[np.isfinite(dd)]))
        XYZ.append(np.ascontiguousarray(xyz[keep]))
        LV.append(np.load(d / "crossing_level.npy").astype(np.int64)[keep])
        RAY.append(r[keep] + base)
        base += n_rays
    return np.concatenate(XYZ), np.concatenate(LV), np.concatenate(RAY)


# ---------------------------------------------------------------- stream the generator's output
MARK = '{"name": "ray'


def stream_region_chains(path, region_rays, chunk=1 << 26):
    """Yield (ray, levels, xyz) for every collection of ``path`` whose name is ``ray<r>@...`` with
    ``r`` in ``region_rays`` (bool array by global ray id). Only those are JSON-decoded."""
    dec = json.JSONDecoder()
    buf = ""
    with open(path) as f:
        eof = False
        while not eof:
            more = f.read(chunk)
            eof = not more
            buf += more
            starts = [m.start() for m in re.finditer(re.escape(MARK), buf)]
            last = len(starts) if eof else len(starts) - 1     # the last one may be incomplete
            for s in starts[:max(last, 0)]:
                at = buf.index("@", s)
                r = int(buf[s + len(MARK):at])
                if r < len(region_rays) and region_rays[r]:
                    col, _ = dec.raw_decode(buf, s)
                    pts = list(col["points"].values())
                    yield (r, np.array([p["wind_a"] for p in pts], np.float64),
                           np.array([p["p"] for p in pts], np.float64))
            buf = buf[starts[last]:] if (not eof and starts) else ("" if starts else buf[-len(MARK):])


# ---------------------------------------------------------------- chain builders
def runs(link):
    """Chains from a step mask: link[i] means crossing i and i + 1 are in one chain. Returns
    (crossing index, chain id) for the maximal runs (>= 2 points)."""
    prev = np.r_[False, link[:-1]]
    on = link | prev
    start = link & ~prev
    idx = np.flatnonzero(on)
    ch = np.cumsum(start)[idx] - 1
    return idx, ch


def as_chains(xyz, level, idx, ch):
    return Chains(xyz[idx], level[idx].astype(np.float64), ch.astype(np.int64),
                  [str(i) for i in range(int(ch.max()) + 1 if len(ch) else 0)])


def one_step(k):
    """Every step k -> k + 1 as its own 2-point chain."""
    return np.stack([k, k + 1], 1).ravel(), np.repeat(np.arange(len(k)), 2)


# ---------------------------------------------------------------- scoring
def per_chain(truth, res, t):
    """Per human chain counts for the bootstrap, plus the totals."""
    n = truth.chains.n_chains
    hc = truth.chains.chain
    a, b, tdw, pdw = t["a"], t["b"], t["true_dw"], t["pred_dw"]
    ok = pdw == tdw
    adj = (b == a + 1) & (hc[a] == hc[b])
    dw2 = np.abs(tdw) >= 2
    pair_a = np.unique(a[adj])                         # one key per covered adjacent pair
    bad_a = np.unique(a[adj & ~ok])
    pair_ok = ~np.isin(pair_a, bad_a)
    bc = lambda m, ids: np.bincount(hc[ids[m]] if m is not None else hc[ids], minlength=n).astype(np.float64)
    C = {
        "tests": bc(None, a), "exact": bc(ok, a),
        "adj_tests": bc(adj, a), "adj_exact": bc(adj & ok, a),
        "dw2_tests": bc(dw2, a), "dw2_exact": bc(dw2 & ok, a),
        "sign": bc(pdw * tdw < 0, a),
        "pairs_cov": bc(None, pair_a), "pairs_ok": bc(pair_ok, pair_a),
        "pairs_adj": np.bincount(hc[truth.pairs(adjacent_only=True)[0]], minlength=n).astype(np.float64),
    }
    return C


def summary(C):
    s = {k: float(v.sum()) for k, v in C.items()}
    r = lambda x, y: (s[x] / s[y]) if s[y] else None
    hours = lambda: [round((s["pairs_ok"] - FIX_COST * (s["pairs_cov"] - s["pairs_ok"])) / s["pairs_adj"] * h, 3)
                     for h in HUMAN_HOURS]
    return {
        "tests": int(s["tests"]), "exact": r("exact", "tests"),
        "adjacent_tests": int(s["adj_tests"]), "adjacent_exact": r("adj_exact", "adj_tests"),
        "adjacent_pairs": int(s["pairs_adj"]), "adjacent_pairs_covered": int(s["pairs_cov"]),
        "adjacent_coverage": r("pairs_cov", "pairs_adj"),
        "pair_precision": r("pairs_ok", "pairs_cov"),
        "f_ok": r("pairs_ok", "pairs_adj"), "f_bad": (s["pairs_cov"] - s["pairs_ok"]) / s["pairs_adj"],
        "hours_saved_3x": hours(),
        "dw2_tests": int(s["dw2_tests"]), "dw2_exact": r("dw2_exact", "dw2_tests"),
        "sign_errors": r("sign", "tests"),
        "tests_per_covered_pair": r("adj_tests", "pairs_cov"),
    }


RATIOS = {"adjacent_exact": ("adj_exact", "adj_tests"), "pair_precision": ("pairs_ok", "pairs_cov"),
          "dw2_exact": ("dw2_exact", "dw2_tests"), "exact": ("exact", "tests"),
          "adjacent_coverage": ("pairs_cov", "pairs_adj")}


def boot_ratios(C, W):
    return {k: (W @ C[x]) / np.maximum(W @ C[y], 1e-12) for k, (x, y) in RATIOS.items()}


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("claims", help="full-volume output of windingcheck generate --claims")
    ap.add_argument("--crossings", default=None, help="winding_inference dir (default: fetched data)")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    d = data.fetch("PHercParis4")
    wi = Path(args.crossings) if args.crossings else d / "winding_inference"
    human = load_pcl(d / "relative_windings.json")
    ids = np.arange(human.n_chains)
    truth = build_truth(human.subset(ids[ids % 2 == 1]))
    R = {"claims_file": str(Path(args.claims).name), "heldout_human_chains": int(truth.chains.n_chains)}

    # ---- region crossings (local index = position in file order)
    xyz, level, gray = region_crossings(wi, human.xyz)
    rays, ray_local = np.unique(gray, return_inverse=True)
    n = len(level)
    ray_start = np.flatnonzero(np.r_[True, ray_local[1:] != ray_local[:-1]])
    assert np.array_equal(ray_local[ray_start], np.arange(len(rays)))
    k = np.flatnonzero(ray_local[1:] == ray_local[:-1])                # steps
    assert np.all(level[k + 1] - level[k] == 1)
    R["region"] = {"rays": int(len(rays)), "crossings": int(n), "steps": int(len(k))}
    print(json.dumps(R["region"]), flush=True)

    # ---- the generator's chains on the region rays, mapped back to crossings
    is_region = np.zeros(int(rays.max()) + 1, bool)
    is_region[rays] = True
    loc = np.full(len(is_region), -1, np.int64)
    loc[rays] = np.arange(len(rays))
    gi, gc, glen = [], [], []
    for c, (r, lv, p) in enumerate(stream_region_chains(args.claims, is_region)):
        s = ray_start[loc[r]]
        i = s + (lv.astype(np.int64) - level[s])
        assert np.all((i >= s) & (i < n)) and np.all(ray_local[i] == loc[r]), r
        assert np.allclose(xyz[i], p, atol=1e-6), r
        gi.append(i)
        gc.append(np.full(len(i), c))
        glen.append(len(i))
    gi, gc, glen = np.concatenate(gi), np.concatenate(gc), np.asarray(glen)
    R["generator_chains"] = {"chains": int(len(glen)), "points": int(len(gi)),
                             "two_point_share": float(np.mean(glen == 2)),
                             "seconds_streamed": round(time.time() - t0, 1)}
    print(json.dumps(R["generator_chains"]), flush=True)

    kept = np.zeros(n, bool)
    same = (gc[1:] == gc[:-1]) & (gi[1:] == gi[:-1] + 1)
    kept[gi[:-1][same]] = True
    frac = float(kept[k].mean())
    R["kept_step_fraction"] = frac

    # ---- sources
    sources = {
        "raw": (np.arange(n), ray_local),
        "generator": (gi, gc),
        "one_step": one_step(k),
        "generator_one_step": one_step(np.flatnonzero(kept)),
    }
    for sd in SEEDS:
        link = np.zeros(n, bool)
        link[k] = np.random.default_rng(sd).random(len(k)) < frac
        sources[f"random_cut_{sd}"] = runs(link)

    S, PC, tests_of = {}, {}, {}
    for name, (idx, ch) in sources.items():
        res, t = score_chains(truth, as_chains(xyz, level, idx, ch), return_tests=True)
        PC[name] = per_chain(truth, res, t)
        S[name] = summary(PC[name])
        S[name]["chains"] = int(ch.max() + 1) if len(ch) else 0
        S[name]["human_pairs_covered"] = res["human_pairs_covered"]
        tests_of[name] = (t, idx)
        print(name, json.dumps({x: S[name][x] for x in ("adjacent_tests", "adjacent_exact", "adjacent_coverage",
                                                         "pair_precision", "dw2_tests", "dw2_exact")}), flush=True)

    rc = [f"random_cut_{sd}" for sd in SEEDS]
    PC["random_cut_pooled"] = {key: sum(PC[s][key] for s in rc) for key in PC[rc[0]]}
    S["random_cut_pooled"] = summary(PC["random_cut_pooled"])
    S["random_cut_range"] = {x: [min(S[s][x] for s in rc), max(S[s][x] for s in rc)]
                             for x in ("adjacent_exact", "adjacent_coverage", "pair_precision", "dw2_exact",
                                       "exact", "human_pairs_covered", "hours_saved_3x")}
    S["random_cut_range"]["adjacent_coverage_mean"] = float(np.mean([S[s]["adjacent_coverage"] for s in rc]))

    # ---- raw on the generator's own tests (same click pair, same ray)
    nc = len(truth.chains)
    tg, ig = tests_of["generator"]
    tr, _ = tests_of["raw"]
    key_g = (tg["a"] * nc + tg["b"]) * len(rays) + ray_local[ig[tg["ca"]]]
    key_r = (tr["a"] * nc + tr["b"]) * len(rays) + ray_local[tr["ca"]]
    hc = truth.chains.chain
    adj_g = (tg["b"] == tg["a"] + 1) & (hc[tg["a"]] == hc[tg["b"]])
    adj_r = (tr["b"] == tr["a"] + 1) & (hc[tr["a"]] == hc[tr["b"]])
    sel = adj_r & np.isin(key_r, key_g[adj_g])
    pg = dict(zip(key_g[adj_g].tolist(), tg["pred_dw"][adj_g].tolist()))
    S["raw_on_generator_tests"] = {
        "adjacent_tests": int(sel.sum()),
        "adjacent_exact": float(np.mean(tr["pred_dw"][sel] == tr["true_dw"][sel])),
        "same_prediction_as_generator": int(sum(pg[kk] == pp for kk, pp in
                                                zip(key_r[sel].tolist(), tr["pred_dw"][sel].tolist()))),
    }

    # ---- chain bootstrap, paired
    rng = np.random.default_rng(BOOT_SEED)
    nh = truth.chains.n_chains
    W = np.stack([np.bincount(rng.integers(0, nh, nh), minlength=nh) for _ in range(N_BOOT)]).astype(np.float64)
    B = {name: boot_ratios(PC[name], W) for name in ("raw", "generator", "one_step", "random_cut_pooled")}
    CI = {name: {x: ci(v) for x, v in B[name].items()} for name in B}
    delta = {x: {"point": S["generator"][x] - S["random_cut_pooled"][x],
                 "ci95": ci(B["generator"][x] - B["random_cut_pooled"][x])}
             for x in ("adjacent_exact", "pair_precision", "dw2_exact", "exact", "adjacent_coverage")}
    R["sources"], R["ci95_chain_bootstrap"], R["generator_minus_random_cut"] = S, CI, delta

    # ---- checks
    g, raw, os_ = S["generator"], S["raw"], S["one_step"]
    exact_n = lambda s: int(round(s["adjacent_exact"] * s["adjacent_tests"]))
    crit = {}
    crit["reproduction"] = {"pass": (R["generator_chains"]["chains"] == 111_478 and g["adjacent_tests"] == 721
                           and exact_n(g) == 659 and raw["adjacent_tests"] == 1159 and exact_n(raw) == 889
                           and os_["adjacent_tests"] == 905 and exact_n(os_) == 894),
                  "measured": {"generator_chains": R["generator_chains"]["chains"],
                               "generator": [g["adjacent_tests"], exact_n(g)],
                               "raw": [raw["adjacent_tests"], exact_n(raw)],
                               "one_step": [os_["adjacent_tests"], exact_n(os_)]}}
    cov_mean = S["random_cut_range"]["adjacent_coverage_mean"]
    crit["equal_coverage"] = {"pass": abs(cov_mean - g["adjacent_coverage"]) <= 0.02,
                  "measured": {"random_cut_mean": cov_mean, "generator": g["adjacent_coverage"]}}
    crit["model_edge_per_test"] = {"pass": delta["adjacent_exact"]["point"] >= 0.03 and delta["adjacent_exact"]["ci95"][0] > 0,
                  "measured": delta["adjacent_exact"]}
    crit["model_edge_per_pair"] = {"pass": delta["pair_precision"]["ci95"][0] > 0, "measured": delta["pair_precision"]}
    crit["model_edge_dw2"] = {"pass": delta["dw2_exact"]["ci95"][0] > 0, "measured": delta["dw2_exact"]}
    rog = S["raw_on_generator_tests"]
    crit["never_changes_a_count"] = {"pass": abs(rog["adjacent_exact"] - g["adjacent_exact"]) <= 0.01,
                  "measured": {"raw_on_generator_tests": rog, "generator": g["adjacent_exact"]}}
    R["criteria"] = crit
    R["peak_rss_gb"] = round(rss_gb(), 2)
    R["seconds_total"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(R, indent=2))
    print(json.dumps({"criteria": {c: v["pass"] for c, v in crit.items()}, "delta": delta,
                      "peak_rss_gb": R["peak_rss_gb"], "seconds": R["seconds_total"]}, indent=2))


if __name__ == "__main__":
    main()

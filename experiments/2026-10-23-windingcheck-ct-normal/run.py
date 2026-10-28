"""The October numbers re-scored along the CT sheet normal (windingcheck v0.1.2).

    windingcheck fetch --crossings
    windingcheck generate --claims --crossings data/spiral_datasets/PHercParis4/winding_inference \\
        --out claims.json                                         # 14 min, 9 GB peak, 4.0 GB file
    windingcheck normals --heldout --z-band 7000 17000 --out normals.json   # reads 1.42 GB of CT
    python experiments/2026-10-23-windingcheck-ct-normal/run.py claims.json normals.json

Scores the held-out human chains (1,101 clicks, 974 adjacent pairs) twice: matched along the human
chain direction (the default, the v0.1 numbers) and along the CT sheet normal (``build_truth(normals=)``).
Sources, chain builders, summaries and the bootstrap are those of
``../2026-10-13-windingcheck-baselines/run.py``, imported unchanged. Writes results.json and compares
every number with expected.json (integers exactly, floats to 1e-9 relative); exit 1 on any difference.
CPU only; about 1 GB RAM and 20 s after the inputs exist.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "windingcheck" / "src"))

_spec = importlib.util.spec_from_file_location("f040", HERE.parent / "2026-10-13-windingcheck-baselines" / "run.py")
f040 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f040)

from windingcheck import data  # noqa: E402
from windingcheck.chains import load_pcl  # noqa: E402
from windingcheck.normals import for_chains  # noqa: E402
from windingcheck.score import build_truth, score_chains  # noqa: E402

BINS = [0, 10, 20, 30, 45, 90.0001]
BIN_NAMES = ["0-10", "10-20", "20-30", "30-45", "45-90"]


def ct_normals(odd, path):
    """One row per held-out click in input order; NaN where the file has no normal (exact xyz match)."""
    n = for_chains(odd, path)
    ok = np.isfinite(n).all(1)
    miss = np.flatnonzero(~ok)
    return n, {"odd_clicks": int(len(odd)), "with_ct_normal": int(ok.sum()), "missing": int(len(miss)),
               "missing_rows": miss.tolist(), "missing_chain": odd.chain[miss].tolist(),
               "missing_z": odd.xyz[miss, 2].round(1).tolist()}


def tilt_deg(old, ct):
    """Per sorted click: angle between the chain direction (old) and the CT normal; NaN without CT normal."""
    c = np.abs(np.einsum("ij,ij->i", old.normal, ct.normal))
    t = np.degrees(np.arccos(np.clip(c, 0, 1)))
    same = np.all(np.isclose(old.normal, ct.normal, atol=1e-12) | ~np.isfinite(old.normal), axis=1)
    return t, same


# ---------------------------------------------------------------- sources (built once, without the truth)
def build_sources(claims, crossings=None):
    d = data.fetch("PHercParis4")
    human = load_pcl(d / "relative_windings.json")
    ids = np.arange(human.n_chains)
    odd = human.subset(ids[ids % 2 == 1])
    xyz, level, gray = f040.region_crossings(Path(crossings) if crossings else d / "winding_inference", human.xyz)
    rays, ray_local = np.unique(gray, return_inverse=True)
    n = len(level)
    ray_start = np.flatnonzero(np.r_[True, ray_local[1:] != ray_local[:-1]])
    k = np.flatnonzero(ray_local[1:] == ray_local[:-1])
    assert np.all(level[k + 1] - level[k] == 1)
    is_region = np.zeros(int(rays.max()) + 1, bool)
    is_region[rays] = True
    loc = np.full(len(is_region), -1, np.int64)
    loc[rays] = np.arange(len(rays))
    gi, gc = [], []
    for c, (r, lv, p) in enumerate(f040.stream_region_chains(claims, is_region)):
        s = ray_start[loc[r]]
        i = s + (lv.astype(np.int64) - level[s])
        assert np.all((i >= s) & (i < n)) and np.all(ray_local[i] == loc[r]), r
        assert np.allclose(xyz[i], p, atol=1e-6), r
        gi.append(i)
        gc.append(np.full(len(i), c))
    gi, gc = np.concatenate(gi), np.concatenate(gc)
    kept = np.zeros(n, bool)
    same = (gc[1:] == gc[:-1]) & (gi[1:] == gi[:-1] + 1)
    kept[gi[:-1][same]] = True
    frac = float(kept[k].mean())
    src = {"raw": (np.arange(n), ray_local), "generator": (gi, gc)}
    for sd in f040.SEEDS:
        link = np.zeros(n, bool)
        link[k] = np.random.default_rng(sd).random(len(k)) < frac
        src[f"random_cut_{sd}"] = f040.runs(link)
    meta = {"region_rays": int(len(rays)), "crossings": int(n), "generator_chains": int(gc.max() + 1),
            "kept_step_fraction": frac}
    return odd, xyz, level, src, meta


# ---------------------------------------------------------------- one truth
def score_all(truth, xyz, level, src, no_normal_sorted, tilt):
    rc = [f"random_cut_{sd}" for sd in f040.SEEDS]
    S, PC, extra = {}, {}, {}
    hc = truth.chains.chain
    for name, (idx, ch) in src.items():
        res, t = score_chains(truth, f040.as_chains(xyz, level, idx, ch), return_tests=True)
        PC[name] = f040.per_chain(truth, res, t)
        S[name] = f040.summary(PC[name])
        a, b = t["a"], t["b"]
        ok = t["pred_dw"] == t["true_dw"]
        adj = (b == a + 1) & (hc[a] == hc[b])
        tt = np.fmax(tilt[a], tilt[b])                       # larger of the two clicks
        nn = no_normal_sorted[a] | no_normal_sorted[b]
        by = {}
        for lo, hi, nm in zip(BINS[:-1], BINS[1:], BIN_NAMES):
            m = adj & ~nn & (tt >= lo) & (tt < hi)
            by[nm] = [int(m.sum()), float(ok[m].mean()) if m.any() else None]
        m = adj & nn
        by["no_ct_normal"] = [int(m.sum()), float(ok[m].mean()) if m.any() else None]
        extra[name] = {"wrong_sign": int((t["pred_dw"] * t["true_dw"] < 0).sum()),
                       "not_exact": int((~ok).sum()),
                       "adjacent_exact_n": int((adj & ok).sum()),
                       "pairs_all_exact": int(PC[name]["pairs_ok"].sum()),
                       "tests_touching_no_normal_clicks": int(nn.sum()),
                       "adjacent_tests_touching_no_normal_clicks": int((adj & nn).sum()),
                       "adjacent_by_tilt": by}
    PC["random_cut_pooled"] = {key: sum(PC[s][key] for s in rc) for key in PC[rc[0]]}
    S["random_cut_pooled"] = f040.summary(PC["random_cut_pooled"])
    extra["random_cut_pooled"] = {
        "adjacent_exact_n": sum(extra[s]["adjacent_exact_n"] for s in rc),
        "wrong_sign": sum(extra[s]["wrong_sign"] for s in rc),
        "adjacent_by_tilt": {nm: [sum(extra[s]["adjacent_by_tilt"][nm][0] for s in rc),
                                  (sum(extra[s]["adjacent_by_tilt"][nm][0] * (extra[s]["adjacent_by_tilt"][nm][1] or 0)
                                       for s in rc) / max(sum(extra[s]["adjacent_by_tilt"][nm][0] for s in rc), 1))]
                             for nm in BIN_NAMES + ["no_ct_normal"]}}
    cov = [S[s]["adjacent_coverage"] for s in rc]
    S["random_cut_range"] = {"adjacent_exact": [min(S[s]["adjacent_exact"] for s in rc),
                                                max(S[s]["adjacent_exact"] for s in rc)],
                             "adjacent_coverage": [min(cov), max(cov)],
                             "adjacent_coverage_mean": float(np.mean(cov))}
    rng = np.random.default_rng(f040.BOOT_SEED)
    nh = truth.chains.n_chains
    W = np.stack([np.bincount(rng.integers(0, nh, nh), minlength=nh) for _ in range(f040.N_BOOT)]).astype(np.float64)
    B = {name: f040.boot_ratios(PC[name], W) for name in ("raw", "generator", "random_cut_pooled")}
    CI = {name: {x: f040.ci(v) for x, v in B[name].items()} for name in B}
    delta = {x: {"point": S["generator"][x] - S["random_cut_pooled"][x],
                 "ci95": f040.ci(B["generator"][x] - B["random_cut_pooled"][x])}
             for x in ("adjacent_exact", "pair_precision", "dw2_exact", "exact", "adjacent_coverage")}
    g = S["generator"]
    bars = {"B1_equal_coverage": abs(S["random_cut_range"]["adjacent_coverage_mean"] - g["adjacent_coverage"]) <= 0.02,
            "B2_gain": delta["adjacent_exact"]["point"] >= 0.03 and delta["adjacent_exact"]["ci95"][0] > 0,
            "gain_ci_above_0": delta["adjacent_exact"]["ci95"][0] > 0}
    keep = ("raw", "generator", "random_cut_pooled", "random_cut_range")
    return {"sources": {k: S[k] for k in keep}, "counts": extra, "ci95_chain_bootstrap": CI,
            "generator_minus_random_cut": delta, "bars": bars,
            "random_cut_seeds": {s: S[s] for s in rc}}


def compare(got, want, path="", out=None):
    """Paths where got and want differ: integers, strings, bools and None exactly, floats to 1e-9 relative."""
    out = [] if out is None else out
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            out.append(f"{path}: keys {sorted(got) if isinstance(got, dict) else type(got)} != {sorted(want)}")
            return out
        for k in want:
            compare(got[k], want[k], f"{path}/{k}", out)
    elif isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            out.append(f"{path}: {got} != {want}")
            return out
        for i, (g, w) in enumerate(zip(got, want)):
            compare(g, w, f"{path}[{i}]", out)
    elif isinstance(want, float) and not isinstance(want, bool):
        if not (isinstance(got, (int, float)) and
                (got == want or (np.isnan(got) and np.isnan(want)) or abs(got - want) <= 1e-9 * abs(want))):
            out.append(f"{path}: {got} != {want}")
    elif got != want or type(got) is not type(want):
        out.append(f"{path}: {got!r} != {want!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("claims", help="full-volume output of windingcheck generate --claims")
    ap.add_argument("normals", help="output of windingcheck normals --heldout --z-band 7000 17000")
    ap.add_argument("--crossings", default=None, help="winding_inference dir (default: fetched data)")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()
    t0 = time.time()
    odd, xyz, level, src, meta = build_sources(args.claims, args.crossings)
    n_ct, nc = ct_normals(odd, args.normals)
    old = build_truth(odd)
    ct = build_truth(odd, normals=n_ct)
    tilt, same = tilt_deg(old, ct)
    order = np.lexsort((odd.level, odd.chain))                 # sorted row -> input row (as build_truth)
    no_normal_sorted = ~np.isfinite(n_ct[order]).all(1)
    assert np.all(same[no_normal_sorted]), "clicks without a normal must keep the chain direction"
    tilt[no_normal_sorted] = np.nan
    R = {"meta": meta, "normal_coverage": nc,
         "tilt_deg": {"median": float(np.nanmedian(tilt)), "p90": float(np.nanpercentile(tilt, 90))},
         "tol_n_median": {"old": float(np.nanmedian(old.tol_n)), "ct": float(np.nanmedian(ct.tol_n))},
         "chain_direction": score_all(old, xyz, level, src, no_normal_sorted, tilt),
         "sheet_normal": score_all(ct, xyz, level, src, no_normal_sorted, tilt)}
    R = json.loads(json.dumps(R))                               # tuples -> lists, as read back from disk
    for s in ("raw", "generator", "random_cut_pooled"):
        a, b = R["chain_direction"]["sources"][s], R["sheet_normal"]["sources"][s]
        print(f"{s:18s} adjacent exact {100 * a['adjacent_exact']:.1f}% -> {100 * b['adjacent_exact']:.1f}%, "
              f"coverage {100 * a['adjacent_coverage']:.1f}% -> {100 * b['adjacent_coverage']:.1f}%")
    for k in ("chain_direction", "sheet_normal"):
        g = R[k]["generator_minus_random_cut"]["adjacent_exact"]
        print(f"gain over random cuts ({k}): {100 * g['point']:+.2f} points "
              f"[{100 * g['ci95'][0]:+.2f}, {100 * g['ci95'][1]:+.2f}]")
    diff = compare(R, json.loads((HERE / "expected.json").read_text()))
    R["seconds"] = round(time.time() - t0, 1)
    R["peak_rss_gb"] = round(f040.rss_gb(), 2)
    R["differences_from_expected"] = diff
    Path(args.out).write_text(json.dumps(R, indent=1))
    print(f"{len(diff)} differences from expected.json" + ("" if not diff else ":\n  " + "\n  ".join(diff[:20])))
    print("wrote", args.out, R["seconds"], "s", R["peak_rss_gb"], "GB")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())

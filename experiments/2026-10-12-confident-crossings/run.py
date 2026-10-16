"""First generator: confidence-scored ray crossings, with its checks, on PHerc. Paris 4.

    python experiments/2026-10-12-confident-crossings/run.py      (CPU only, ~10 min)

The thresholds in the checks below were fixed before this script first ran.
"""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _villa_spiral_fitting():
    """external/ is git-ignored, so a git worktree has none: fall back to the main checkout."""
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=HERE, capture_output=True, text=True).stdout.strip()
    for root in (REPO, Path(common).parent if common else REPO):
        if (root / "external" / "villa" / "spiral-fitting").is_dir():
            return root / "external" / "villa" / "spiral-fitting"
    return REPO / "external" / "villa" / "spiral-fitting"


VILLA_SF = _villa_spiral_fitting()
FIG = HERE / "figures_paris4"     # Paris 4 derived: not published before 2027-06-25
sys.path.insert(0, str(REPO / "tools" / "windingcheck" / "src"))

from windingcheck import data  # noqa: E402
from windingcheck.chains import load_pcl, save_pcl  # noqa: E402
from windingcheck.confidence import FEATURES, StepModel, generate, pair_steps, step_features, step_index  # noqa: E402
from windingcheck.crossings import load_ray_store  # noqa: E402
from windingcheck.prior import Umbilicus  # noqa: E402
from windingcheck.score import build_truth, score_chains  # noqa: E402

TAU_GRID = np.round(np.arange(0.5, 0.9951, 0.005), 3)
FIX_COST = 3
HUMAN_HOURS = (3.5, 7.0)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    r = rankdata(score)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def ece(p, y, bins=10):
    if len(p) == 0:
        return None, []
    order = np.argsort(p)
    out, tot = [], 0.0
    for idx in np.array_split(order, bins):
        if len(idx) == 0:
            continue
        mp, my = float(p[idx].mean()), float(np.mean(y[idx]))
        tot += len(idx) * abs(mp - my)
        out.append({"n": int(len(idx)), "mean_p": mp, "exact": my})
    return tot / len(p), out


def test_conf(tests, step_p_of_k):
    """Per test: product of its steps' confidences, and the minimum."""
    owner, k = pair_steps(tests["ca"], tests["cb"])
    p = step_p_of_k(k)
    n = len(tests["ca"])
    prod = np.exp(np.bincount(owner, weights=np.log(np.maximum(p, 1e-12)), minlength=n))
    mn = np.ones(n)
    np.minimum.at(mn, owner, p)
    return prod, mn


def step_rows(tests, mask):
    """Training rows: steps between the two crossings of each adjacent test in ``mask``."""
    idx = np.flatnonzero(mask)
    sub = {"ca": tests["ca"][idx], "cb": tests["cb"][idx]}
    owner, k = pair_steps(sub["ca"], sub["cb"])
    ok = (tests["pred_dw"][idx] == tests["true_dw"][idx])[owner]
    return k, ok


def choose_tau(min_conf, exact, min_tests=100):
    for tau in TAU_GRID:
        m = min_conf >= tau
        if m.sum() >= min_tests and exact[m].mean() >= 0.95:
            return float(tau)
    return None


def adjacent_outcome(truth, tests):
    """Per human adjacent pair: covered, ok (all tests exact), bad (any test wrong)."""
    ia, ib = truth.pairs(adjacent_only=True)
    n = len(truth.chains) + 1
    key_h = ia * n + ib
    adj = (tests["b"] == tests["a"] + 1) & (truth.chains.chain[tests["a"]] == truth.chains.chain[tests["b"]])
    key_t = tests["a"][adj] * n + tests["b"][adj]
    wrong_t = (tests["pred_dw"] != tests["true_dw"])[adj]
    covered = np.isin(key_h, key_t)
    bad = np.isin(key_h, key_t[wrong_t])
    return {"pairs": int(len(key_h)), "covered": float(covered.mean()),
            "f_ok": float((covered & ~bad).mean()), "f_bad": float(bad.mean())}


def net_hours(o, fix=FIX_COST):
    f = o["f_ok"] - fix * o["f_bad"]
    return [round(f * h, 3) for h in HUMAN_HOURS]


def main():
    t_all = time.time()
    FIG.mkdir(exist_ok=True)
    d = data.fetch("PHercParis4", crossings=True)
    human = load_pcl(d / "relative_windings.json")
    truth = build_truth(human)
    umb = Umbilicus(d / "umbilicus.json")
    ids = np.arange(truth.chains.n_chains)
    even, odd = ids[ids % 2 == 0], ids[ids % 2 == 1]
    R, crit = {}, {}

    # ---- features for every step of the full crossing set (the cost clock starts here)
    t0 = time.time()
    store = load_ray_store(d / "winding_inference")
    tree = cKDTree(store.xyz)
    all_k = step_index(store)
    X, s = step_features(store, umb, all_k, tree=tree)
    del tree
    t_feat = time.time() - t0
    pos_of = lambda k: np.searchsorted(all_k, k)

    # ---- raw crossings against all human chains: tests carry the crossing indices
    raw, tests = score_chains(truth, store.chains(), return_tests=True)
    chain_a = truth.chains.chain[tests["a"]]
    adj = (tests["b"] == tests["a"] + 1) & (chain_a == truth.chains.chain[tests["b"]])
    adj1 = adj & (tests["true_dw"] == 1)
    exact = tests["pred_dw"] == tests["true_dw"]
    is_even = np.isin(chain_a, even)
    is_odd = ~is_even
    R["raw_crossings_all_chains"] = {k: raw[k] for k in ("tests", "exact", "human_pairs_covered")}

    # ---- fit on even chains
    k_tr, y_tr = step_rows(tests, adj1 & is_even)
    model = StepModel.fit(X[pos_of(k_tr)], y_tr)
    t1 = time.time()
    step_p = model.prob(X)
    p_of = lambda k: step_p[pos_of(k)]
    prod, mn = test_conf(tests, p_of)
    R["train"] = {"step_rows": int(len(k_tr)), "step_rows_right": float(y_tr.mean()),
                  "adjacent_tests": int((adj1 & is_even).sum())}
    R["model"] = model.to_json()

    # ---- discrimination and calibration on held-out (odd) tests
    ho_adj = adj1 & is_odd
    g1 = auc(prod[ho_adj], exact[ho_adj])
    e_adj, rel_adj = ece(prod[ho_adj], exact[ho_adj])
    mid = is_odd & (np.abs(tests["true_dw"]) >= 2) & (np.abs(tests["true_dw"]) <= 7)
    e_mid, rel_mid = ece(prod[mid], exact[mid])
    R["discrimination_calibration"] = {"heldout_adjacent_tests": int(ho_adj.sum()), "heldout_adjacent_exact_raw": float(exact[ho_adj].mean()),
                  "auc": g1, "ece_adjacent": e_adj, "reliability_adjacent": rel_adj,
                  "heldout_dw2_7_tests": int(mid.sum()), "ece_dw2_7": e_mid, "reliability_dw2_7": rel_mid}
    crit["discrimination"] = {"pass": g1 is not None and g1 >= 0.75, "measured": g1}
    crit["calibration"] = {"pass": e_adj is not None and e_mid is not None and e_adj <= 0.05 and e_mid <= 0.08,
                  "measured": {"ece_adjacent": e_adj, "ece_dw2_7": e_mid}}

    # ---- single-feature models and the short-step hypothesis
    single = {}
    for name, cols in (("log_s_ray (+square)", [0, 1]), ("nbr_agree", [2]), ("log_support", [3]),
                       ("edge", [4]), ("oblique", [5]), ("log_r", [6])):
        m1 = StepModel.fit(X[pos_of(k_tr)], y_tr, cols=cols)
        pp1 = m1.prob(X)
        pr1, _ = test_conf(tests, lambda k: pp1[pos_of(k)])
        single[name] = auc(pr1[ho_adj], exact[ho_adj])
    R["single_feature_heldout_auc"] = single
    k_ho, y_ho = step_rows(tests, ho_adj)
    short = X[pos_of(k_ho), 0] < np.log(0.6)
    base_err = float(1 - y_ho.mean())
    short_err = float(1 - y_ho[short].mean()) if short.any() else None
    R["short_steps"] = {"heldout_step_rows": int(len(k_ho)), "base_error": base_err,
                           "short_rows": int(short.sum()), "short_error": short_err}

    # ---- tau on even chains, generate, re-score held out
    tr_adj = adj1 & is_even
    tau = choose_tau(mn[tr_adj], exact[tr_adj])
    R["tau"] = tau
    gen, conf = generate(store, step_p, tau if tau is not None else 1.0)
    t_gen = time.time() - t1
    R["generated"] = {"chains": gen.n_chains, "points": len(gen), "crossings_in": len(store),
                      "chain_confidence_pct": np.percentile(conf, [5, 25, 50, 75, 95]).round(4).tolist()}
    truth_odd = build_truth(human.subset(odd))
    truth_even = build_truth(human.subset(even))
    g_odd, gt_odd = score_chains(truth_odd, gen, return_tests=True)
    b_odd, bt_odd = score_chains(truth_odd, store.chains(), return_tests=True)
    g_even, _ = score_chains(truth_even, gen, return_tests=True)
    for name, res, tt, tr in (("generated_heldout", g_odd, gt_odd, truth_odd), ("raw_heldout", b_odd, bt_odd, truth_odd)):
        a_ = (tt["b"] == tt["a"] + 1) & (tr.chains.chain[tt["a"]] == tr.chains.chain[tt["b"]])
        res["adjacent_exact"] = float(np.mean(tt["pred_dw"][a_] == tt["true_dw"][a_])) if a_.any() else None
        res["adjacent_tests"] = int(a_.sum())
        res["adjacent_outcome"] = adjacent_outcome(tr, tt)
        R[name] = res
    R["generated_train_even"] = {k: g_even[k] for k in ("tests", "exact", "human_pairs_covered", "human_adjacent_covered")}
    G, B = R["generated_heldout"], R["raw_heldout"]
    crit["precision"] = {"pass": (G.get("adjacent_exact") or 0) >= 0.93 and G.get("exact", 0) >= 0.90,
                  "measured": {"adjacent_exact": G.get("adjacent_exact"), "exact": G.get("exact")}}
    crit["coverage"] = {"pass": G["human_pairs_covered"] >= 0.08 and G["human_adjacent_covered"] >= 0.30,
                  "measured": {"pairs_covered": G["human_pairs_covered"], "adjacent_covered": G["human_adjacent_covered"]}}

    # ---- net human hours
    hours = {}
    for fix in (1, 3, 5):
        hours[f"fix_{fix}x"] = {"generated": net_hours(G["adjacent_outcome"], fix),
                                "raw": net_hours(B["adjacent_outcome"], fix)}
    R["net_hours_saved"] = hours
    crit["net_human_time"] = {"pass": hours["fix_3x"]["generated"][0] > hours["fix_3x"]["raw"][0],
                  "measured": hours["fix_3x"]}

    # ---- spatial transfer between the two z halves of the human clicks
    z_med = float(np.median(truth.chains.xyz[:, 2]))
    za = truth.chains.xyz[tests["a"], 2]
    g5 = {"z_split": z_med}
    for name, lo_train in (("train_low_test_high", True), ("train_high_test_low", False)):
        tr = adj1 & ((za < z_med) if lo_train else (za >= z_med))
        te = adj1 & ((za >= z_med) if lo_train else (za < z_med))
        kk, yy = step_rows(tests, tr)
        m5 = StepModel.fit(X[pos_of(kk)], yy)
        pp5 = m5.prob(X)
        _, mn5 = test_conf(tests, lambda k: pp5[pos_of(k)])
        t5 = choose_tau(mn5[tr], exact[tr])
        keep = te & (mn5 >= (t5 if t5 is not None else 1.0))
        g5[name] = {"tau": t5, "test_adjacent": int(te.sum()), "kept": int(keep.sum()),
                    "kept_exact": float(exact[keep].mean()) if keep.any() else None,
                    "raw_exact": float(exact[te].mean())}
    R["spatial_transfer"] = g5
    crit["spatial_transfer"] = {"pass": all((g5[n]["kept_exact"] or 0) >= 0.90 for n in ("train_low_test_high", "train_high_test_low")),
                  "measured": {n: g5[n]["kept_exact"] for n in ("train_low_test_high", "train_high_test_low")}}

    secs = t_feat + t_gen
    crit["cost"] = {"pass": secs <= 3600 and rss_gb() <= 32,
                  "measured": {"seconds_features": round(t_feat, 1), "seconds_generate": round(t_gen, 1),
                               "peak_rss_gb": round(rss_gb(), 2)}}

    # ---- write a thinned file (every 50th ray) and load it with villa and windingcheck
    ray_of_chain = np.array([int(n.split("@")[0][3:]) for n in gen.names])
    sel = np.flatnonzero(ray_of_chain % 50 == 0)
    thin = gen.subset(sel)
    out = HERE / "confident_windings_every50.json"     # git-ignored, regenerated by this script
    save_pcl(thin, out, color=(0.16, 0.47, 0.84),
             tags=[{"source": "windingcheck-confident", "confidence": f"{conf[i]:.4f}", "tau": f"{tau}"} for i in sel])
    back = load_pcl(out)
    villa = subprocess.run([sys.executable, "-c",
                            "import sys, json; sys.path.insert(0, sys.argv[1]);"
                            "from point_collection import load_point_collection as L;"
                            "c = L(sys.argv[2]); print(json.dumps([len(c), sum(len(v['points']) for v in c.values())]))",
                            str(VILLA_SF), str(out)], capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    try:
        v_cols, v_pts = json.loads(villa.stdout.strip().splitlines()[-1])
    except Exception:
        v_cols = v_pts = None
    R["thinned_file"] = {"path": out.name, "bytes": out.stat().st_size, "chains": thin.n_chains, "points": len(thin),
                    "windingcheck_points": len(back), "villa_collections": v_cols, "villa_points": v_pts,
                    "villa_stderr_tail": villa.stderr[-400:]}
    crit["loads_in_tools"] = {"pass": v_pts == len(thin) and len(back) == len(thin),
                  "measured": {"written": len(thin), "villa": v_pts, "windingcheck": len(back)}}

    # ---- save the model for the CLI
    model.save(REPO / "tools" / "windingcheck" / "src" / "windingcheck" / "models" / "paris4_even_chains.json",
               tau=tau, fitted_on="PHerc. Paris 4 relative_windings.json, even-numbered chains, adjacent pairs",
               experiment="experiments/2026-10-12-confident-crossings")

    R["predictions"] = {
        "short steps (< 0.6 x ray median) wrong at >= 3x base rate": (short_err or 0) >= 3 * base_err,
        "nbr_agree strongest single feature": max(single, key=lambda k: single[k] or 0) == "nbr_agree",
    }
    R["criteria"] = crit
    R["seconds_total"] = round(time.time() - t_all, 1)
    (HERE / "results.json").write_text(json.dumps(R, indent=2))
    figures(R, prod, exact, ho_adj, mn, tests, is_odd, adj1, B)
    print(json.dumps({"criteria": crit, "predictions": R["predictions"], "tau": tau}, indent=2))


def figures(R, prod, exact, ho_adj, mn, tests, is_odd, adj1, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    # reliability, held-out adjacent pairs
    rel = R["discrimination_calibration"]["reliability_adjacent"]
    ax[0].plot([0, 1], [0, 1], color=muted, lw=1, ls="--")
    ax[0].plot([b["mean_p"] for b in rel], [b["exact"] for b in rel], color=blue, lw=2, marker="o", ms=6)
    ax[0].set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="predicted confidence (10 equal-count bins)",
              ylabel="share exact vs human", title="Held-out adjacent pairs: is the confidence honest?")
    ax[0].text(0.03, 0.93, f"ECE {R['discrimination_calibration']['ece_adjacent']:.3f}, AUC {R['discrimination_calibration']['auc']:.3f}", color=ink,
               transform=ax[0].transAxes)
    # precision vs coverage as tau moves (shortcut), held-out adjacent pairs
    m = adj1 & is_odd
    xs, ys = [], []
    for tau in TAU_GRID:
        keep = m & (mn >= tau)
        if keep.sum() >= 20:
            xs.append(keep.sum() / m.sum())
            ys.append(exact[keep].mean())
    ax[1].plot(xs, ys, color=blue, lw=2, label="confidence-filtered crossings")
    ax[1].plot([1.0], [exact[m].mean()], marker="o", ms=8, color=orange, ls="none",
               label="organisers' crossings, unfiltered")
    if R["tau"] is not None:
        keep = m & (mn >= R["tau"])
        ax[1].plot([keep.sum() / m.sum()], [exact[keep].mean()], marker="o", ms=8, color=blue, ls="none")
        ax[1].annotate(f"τ = {R['tau']} (fixed on even chains)", (keep.sum() / m.sum(), exact[keep].mean()),
                       textcoords="offset points", xytext=(8, -14), color=ink)
    ax[1].set(xlim=(0, 1.05), ylim=(0.7, 1.0), xlabel="share of crossing-covered adjacent pairs kept",
              ylabel="share exact vs human", title="Held-out: precision bought with coverage")
    ax[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=130)


if __name__ == "__main__":
    main()

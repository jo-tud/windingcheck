"""Generator v2: claim-level confidence with CT features, with its checks, on PHerc. Paris 4.

    python experiments/2026-10-12-claim-confidence-v2/run.py      (CPU only; streams ~12 GB CT cold)

The thresholds in the checks below were fixed before this script first ran.
"""
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE / "figures_paris4"     # Paris 4 derived: not published before 2027-06-25
sys.path.insert(0, str(REPO / "tools" / "windingcheck" / "src"))

_spec = importlib.util.spec_from_file_location("v1run", REPO / "experiments" / "2026-10-12-confident-crossings" / "run.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

from windingcheck import data  # noqa: E402
from windingcheck.chains import load_pcl, save_pcl  # noqa: E402
from windingcheck.claims import (CLAIM_FEATURES, N_NO_CT, StepTable, all_ray_claims,  # noqa: E402
                                 claim_features, ct_per_crossing, generate_claims, subset_rays)
from windingcheck.confidence import StepModel, generate, step_features, step_index  # noqa: E402
from windingcheck.crossings import load_ray_store  # noqa: E402
from windingcheck.ct import CTSampler  # noqa: E402
from windingcheck.prior import Umbilicus  # noqa: E402
from windingcheck.score import build_truth, score_chains  # noqa: E402

TAU_GRID = np.round(np.arange(0.50, 0.9951, 0.01), 2)
REGION_RADIUS = 60.0
auc, ece, adjacent_outcome, net_hours, step_rows = v1.auc, v1.ece, v1.adjacent_outcome, v1.net_hours, v1.step_rows


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def adj_mask(truth, t):
    ch = truth.chains.chain
    return (t["b"] == t["a"] + 1) & (ch[t["a"]] == ch[t["b"]])


def rescore(truth, chains):
    res, t = score_chains(truth, chains, return_tests=True)
    a = adj_mask(truth, t)
    res["adjacent_exact"] = float(np.mean(t["pred_dw"][a] == t["true_dw"][a])) if a.any() else None
    res["adjacent_tests"] = int(a.sum())
    res["adjacent_outcome"] = adjacent_outcome(truth, t)
    return res


def villa_python():
    """An interpreter that can import villa's point_collection (it needs torch). The first
    run used sys.executable (a venv without torch), so the villa load check failed there; see README."""
    venv = v1.VILLA_SF.parent / "vesuvius" / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def fit_claims(X, y, cols=None):
    return StepModel.fit(X, y, cols=cols)


def main():
    t_all = time.time()
    FIG.mkdir(exist_ok=True)
    d = data.fetch("PHercParis4", crossings=True)
    human = load_pcl(d / "relative_windings.json")
    truth_all = build_truth(human)
    umb = Umbilicus(d / "umbilicus.json")
    ids = np.arange(truth_all.chains.n_chains)
    even, odd = ids[ids % 2 == 0], ids[ids % 2 == 1]
    truth_even, truth_odd = build_truth(human.subset(even)), build_truth(human.subset(odd))
    R, crit = {}, {}

    # ---- region: rays with a crossing within REGION_RADIUS of any click (positions only)
    t0 = time.time()
    full = load_ray_store(d / "winding_inference")
    dd, _ = cKDTree(human.xyz).query(full.xyz, distance_upper_bound=REGION_RADIUS)
    rays = np.unique(full.ray[np.isfinite(dd)])
    store, keep = subset_rays(full, rays)
    k_reg = step_index(store)
    tree = cKDTree(full.xyz)
    X_step = step_features(full, umb, keep[k_reg], tree=tree)[0]
    del tree
    t_step = time.time() - t0
    R["region"] = {"rays": int(len(rays)), "crossings": int(len(store)), "steps": int(len(k_reg)),
                   "of_crossings": int(len(full))}
    print("region", R["region"], round(t_step, 1), "s", flush=True)

    # ---- v1 on the region: must reproduce the v1 held-out numbers
    v1_model = StepModel.load(REPO / "tools" / "windingcheck" / "src" / "windingcheck" / "models" / "paris4_even_chains.json")
    v1_tau = json.loads((REPO / "tools" / "windingcheck" / "src" / "windingcheck" / "models" / "paris4_even_chains.json").read_text())["tau"]
    v1_chains, _ = generate(store, v1_model.prob(X_step), v1_tau)
    R["v1_region_heldout"] = rescore(truth_odd, v1_chains)
    R["raw_region_heldout"] = rescore(truth_odd, store.chains())
    print("v1 region held-out", R["v1_region_heldout"]["exact"], R["v1_region_heldout"]["human_pairs_covered"], flush=True)

    # ---- CT at every region crossing and step interior
    t0 = time.time()
    ct = CTSampler(workers=32)
    ct_cross, ct_valley = ct_per_crossing(store, ct)
    t_ct = time.time() - t0
    R["ct"] = {"seconds": round(t_ct, 1), **ct.stats()}
    print("ct", R["ct"], flush=True)

    tab = StepTable(store, umb, v1_model, ct_cross, ct_valley, X=X_step)

    # ---- training claims: every even-chain test on the raw region crossings
    _, te = score_chains(truth_even, store.chains(), return_tests=True)
    _, to = score_chains(truth_odd, store.chains(), return_tests=True)
    Xe, ye = claim_features(tab, te["ca"], te["cb"]), te["pred_dw"] == te["true_dw"]
    Xo, yo = claim_features(tab, to["ca"], to["cb"]), to["pred_dw"] == to["true_dw"]
    ke, ko = np.abs(te["cb"] - te["ca"]), np.abs(to["cb"] - to["ca"])
    full_m = fit_claims(Xe, ye)
    noct_m = fit_claims(Xe, ye, cols=np.arange(N_NO_CT))
    po, po_noct = full_m.prob(Xo), noct_m.prob(Xo)
    R["train"] = {"claims": int(len(ye)), "exact": float(ye.mean())}
    R["model"] = dict(full_m.to_json(), features=list(CLAIM_FEATURES))

    # ---- discrimination, calibration and CT gain on held-out claims
    mid = (ko >= 2) & (ko <= 7)
    a_full, a_noct = auc(po[mid], yo[mid]), auc(po_noct[mid], yo[mid])
    e_all, rel_all = ece(po, yo.astype(float))
    e_mid, rel_mid = ece(po[mid], yo[mid].astype(float))
    R["heldout_claims"] = {"claims": int(len(yo)), "exact": float(yo.mean()), "k2_7": int(mid.sum()),
                           "k2_7_exact": float(yo[mid].mean()), "auc_full": a_full, "auc_no_ct": a_noct,
                           "ece_all": e_all, "ece_k2_7": e_mid, "reliability_all": rel_all,
                           "reliability_k2_7": rel_mid}
    single = {}
    for i, name in enumerate(CLAIM_FEATURES):
        single[name] = auc(fit_claims(Xe, ye, cols=[i]).prob(Xo)[mid], yo[mid])
    R["single_feature_heldout_auc_k2_7"] = single
    crit["discrimination"] = {"pass": a_full is not None and a_full >= 0.75, "measured": a_full}
    crit["calibration"] = {"pass": e_all <= 0.05 and e_mid <= 0.08, "measured": {"ece_all": e_all, "ece_k2_7": e_mid}}
    crit["ct_adds_information"] = {"pass": a_full - a_noct >= 0.02, "measured": {"auc_full": a_full, "auc_no_ct": a_noct,
                                                                 "gain": a_full - a_noct}}
    print("discrimination, calibration, CT gain", crit["discrimination"], crit["calibration"], crit["ct_adds_information"], flush=True)

    # ---- all claims on all region rays, tau on even chains, generate, re-score held out
    t0 = time.time()
    A, B, starts, n = all_ray_claims(store)
    pa = np.empty(len(A))
    for c0 in range(0, len(A), 2_000_000):
        pa[c0:c0 + 2_000_000] = full_m.prob(claim_features(tab, A[c0:c0 + 2_000_000], B[c0:c0 + 2_000_000]))
    t_claims = time.time() - t0
    R["all_claims"] = {"claims": int(len(A)), "seconds": round(t_claims, 1)}
    sweep, tau = [], None
    t0 = time.time()
    for tt in TAU_GRID:
        g, _ = generate_claims(store, A, B, pa, starts, n, tt)
        r = rescore(truth_even, g)
        sweep.append({"tau": float(tt), "adjacent_exact": r["adjacent_exact"], "adjacent_tests": r["adjacent_tests"],
                      "exact": r.get("exact"), "pairs_covered": r["human_pairs_covered"]})
        if tau is None and r["adjacent_tests"] >= 100 and (r["adjacent_exact"] or 0) >= 0.95:
            tau = float(tt)
            break
    t_sweep = time.time() - t0
    R["tau_sweep_even"] = sweep
    R["tau"] = tau
    gen, conf = generate_claims(store, A, B, pa, starts, n, tau if tau is not None else 1.01)
    R["generated"] = {"chains": gen.n_chains, "points": len(gen), "crossings_in": len(store)}
    G = rescore(truth_odd, gen)
    R["generated_heldout"] = G
    v1_res = R["v1_region_heldout"]
    crit["precision"] = {"pass": (G.get("adjacent_exact") or 0) >= 0.93 and G.get("exact", 0) >= 0.90,
                  "measured": {"adjacent_exact": G.get("adjacent_exact"), "exact": G.get("exact")}}
    crit["coverage"] = {"pass": G["human_pairs_covered"] >= 0.08 and G["human_adjacent_covered"] >= 0.30,
                  "measured": {"pairs_covered": G["human_pairs_covered"], "adjacent_covered": G["human_adjacent_covered"]}}
    hours = {f"fix_{f}x": {"v2": net_hours(G["adjacent_outcome"], f), "v1": net_hours(v1_res["adjacent_outcome"], f),
                           "raw": net_hours(R["raw_region_heldout"]["adjacent_outcome"], f)} for f in (1, 3, 5)}
    R["net_hours_saved"] = hours
    crit["beats_v1_on_human_time"] = {"pass": hours["fix_3x"]["v2"][0] > hours["fix_3x"]["v1"][0], "measured": hours["fix_3x"]}
    print("precision, coverage, human time", crit["precision"], crit["coverage"], crit["beats_v1_on_human_time"], flush=True)

    # ---- spatial transfer: z split, step model and claim model refit per half, shortcut on tests
    _, ta = score_chains(truth_all, store.chains(), return_tests=True)
    za = truth_all.chains.xyz[ta["a"], 2]
    z_med = float(np.median(truth_all.chains.xyz[:, 2]))
    adj_all = adj_mask(truth_all, ta) & (ta["true_dw"] == 1)
    exact_all = ta["pred_dw"] == ta["true_dw"]
    g7 = {"z_split": z_med}
    pos_of = lambda k: np.searchsorted(k_reg, k)
    for name, low in (("train_low_test_high", True), ("train_high_test_low", False)):
        tr = (za < z_med) if low else (za >= z_med)
        kk, yy = step_rows(ta, adj_all & tr)
        sm = StepModel.fit(X_step[pos_of(kk)], yy)
        tab.set_step_model(sm)
        Xa = claim_features(tab, ta["ca"], ta["cb"])
        cm = fit_claims(Xa[tr], exact_all[tr])
        p7 = cm.prob(Xa)
        t7 = None
        for tt in TAU_GRID:
            m = adj_all & tr & (p7 >= tt)
            if m.sum() >= 100 and exact_all[m].mean() >= 0.95:
                t7 = float(tt)
                break
        kept = adj_all & ~tr & (p7 >= (t7 if t7 is not None else 1.01))
        g7[name] = {"tau": t7, "test_adjacent": int((adj_all & ~tr).sum()), "kept": int(kept.sum()),
                    "kept_exact": float(exact_all[kept].mean()) if kept.any() else None,
                    "raw_exact": float(exact_all[adj_all & ~tr].mean())}
    tab.set_step_model(v1_model)
    R["spatial_transfer"] = g7
    crit["spatial_transfer"] = {"pass": all((g7[k]["kept_exact"] or 0) >= 0.90 for k in ("train_low_test_high", "train_high_test_low")),
                  "measured": {k: g7[k]["kept_exact"] for k in ("train_low_test_high", "train_high_test_low")}}

    # ---- cost
    secs = t_step + t_ct + t_claims + t_sweep
    chunks_region = len(np.unique((store.xyz[:, ::-1] // 128).astype(np.int64), axis=0))
    chunks_full = len(np.unique((full.xyz[:, ::-1] // 128).astype(np.int64), axis=0))
    R["cost"] = {"seconds_step_features": round(t_step, 1), "seconds_ct": round(t_ct, 1),
                 "seconds_claims": round(t_claims, 1), "seconds_tau_sweep": round(t_sweep, 1),
                 "peak_rss_gb": round(rss_gb(), 2), "chunks_at_crossings_region": chunks_region,
                 "chunks_at_crossings_full": chunks_full,
                 "full_volume_ct_gb_estimate": round(chunks_full * 2 / 1024, 1)}
    crit["cost"] = {"pass": secs <= 3600 and rss_gb() <= 32, "measured": R["cost"]}

    # ---- write and load in villa and windingcheck
    out = HERE / "claim_windings_region.json"     # git-ignored, regenerated by this script
    save_pcl(gen, out, color=(0.10, 0.62, 0.45),
             tags=[{"source": "windingcheck-claims-v2", "confidence": f"{c:.4f}", "tau": f"{tau}"} for c in conf])
    back = load_pcl(out)
    villa = subprocess.run([villa_python(), "-c",
                            "import sys, json; sys.path.insert(0, sys.argv[1]);"
                            "from point_collection import load_point_collection as L;"
                            "c = L(sys.argv[2]); print(json.dumps([len(c), sum(len(v['points']) for v in c.values())]))",
                            str(v1.VILLA_SF), str(out)], capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    try:
        v_cols, v_pts = json.loads(villa.stdout.strip().splitlines()[-1])
    except Exception:
        v_cols = v_pts = None
    R["thinned_file"] = {"path": out.name, "bytes": out.stat().st_size, "chains": gen.n_chains, "points": len(gen),
                    "windingcheck_points": len(back), "villa_collections": v_cols, "villa_points": v_pts,
                    "villa_stderr_tail": villa.stderr[-400:]}
    crit["loads_in_tools"] = {"pass": v_pts == len(gen) and len(back) == len(gen),
                  "measured": {"written": len(gen), "villa": v_pts, "windingcheck": len(back)}}

    full_m.save(REPO / "tools" / "windingcheck" / "src" / "windingcheck" / "models" / "paris4_claims_v2.json",
                tau=tau, features=list(CLAIM_FEATURES),
                fitted_on="PHerc. Paris 4 relative_windings.json, even-numbered chains, all claims",
                step_model="paris4_even_chains.json",
                experiment="experiments/2026-10-12-claim-confidence-v2")
    R["predictions"] = {"CT adds < 0.02 AUC": not crit["ct_adds_information"]["pass"], "calibration passes": crit["calibration"]["pass"],
                        "v2 coverage > v1 coverage": G["human_pairs_covered"] > v1_res["human_pairs_covered"]}
    R["criteria"] = crit
    R["seconds_total"] = round(time.time() - t_all, 1)
    (HERE / "results.json").write_text(json.dumps(R, indent=2))
    np.savez(HERE / "heldout_claims.npz", po=po, yo=yo, ko=ko)
    print(json.dumps({"criteria": crit, "predictions": R["predictions"], "tau": tau}, indent=2))


if __name__ == "__main__":
    main()

"""v2 claim model without CT. Region run: fit, tau on even chains, parity checks.

    python experiments/2026-10-13-claims-noct-generator/run.py      (CPU only, no CT read)

The thresholds in the checks below were fixed before this script first ran. The full-volume
checks are in full_volume.sh, which runs after this script has saved the model.
"""
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V2_DIR = REPO / "experiments" / "2026-10-12-claim-confidence-v2"
MODELS = REPO / "tools" / "windingcheck" / "src" / "windingcheck" / "models"
sys.path.insert(0, str(REPO / "tools" / "windingcheck" / "src"))

_spec = importlib.util.spec_from_file_location("v2run", V2_DIR / "run.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

from windingcheck import data  # noqa: E402
from windingcheck.chains import load_pcl  # noqa: E402
from windingcheck.claims import (CLAIM_FEATURES, N_NO_CT, StepTable, all_ray_claims,  # noqa: E402
                                 claim_features, generate_blocked, generate_claims, subset_rays)
from windingcheck.confidence import StepModel, step_features, step_index  # noqa: E402
from windingcheck.crossings import load_ray_store  # noqa: E402
from windingcheck.prior import Umbilicus  # noqa: E402
from windingcheck.score import build_truth, score_chains  # noqa: E402

WORKERS = 12


def main():
    t_all = time.time()
    ref = json.loads((V2_DIR / "results.json").read_text())["generated_heldout"]
    d = data.fetch("PHercParis4", crossings=True)
    human = load_pcl(d / "relative_windings.json")
    umb = Umbilicus(d / "umbilicus.json")
    ids = np.arange(human.n_chains)
    truth_even, truth_odd = build_truth(human.subset(ids[ids % 2 == 0])), build_truth(human.subset(ids[ids % 2 == 1]))
    R, crit = {}, {}

    # ---- same region as v2
    full = load_ray_store(d / "winding_inference")
    dd, _ = cKDTree(human.xyz).query(full.xyz, distance_upper_bound=v2.REGION_RADIUS)
    rays = np.unique(full.ray[np.isfinite(dd)])
    store, keep = subset_rays(full, rays)
    X_step = step_features(full, umb, keep[step_index(store)], tree=cKDTree(full.xyz), workers=WORKERS)[0]
    del full
    R["region"] = {"rays": int(len(rays)), "crossings": int(len(store))}
    step_model = StepModel.load(MODELS / "paris4_even_chains.json")
    tab = StepTable(store, umb, step_model, X=X_step)

    # ---- no-CT claim model on v2's training rows (even-chain claims on raw region crossings)
    _, te = score_chains(truth_even, store.chains(), return_tests=True)
    _, to = score_chains(truth_odd, store.chains(), return_tests=True)
    Xe, ye = claim_features(tab, te["ca"], te["cb"]), te["pred_dw"] == te["true_dw"]
    Xo, yo = claim_features(tab, to["ca"], to["cb"]), to["pred_dw"] == to["true_dw"]
    assert Xe.shape[1] == N_NO_CT
    model = StepModel.fit(Xe, ye)
    model.features = CLAIM_FEATURES[:N_NO_CT]
    ko = np.abs(to["cb"] - to["ca"])
    mid = (ko >= 2) & (ko <= 7)
    po = model.prob(Xo)
    R["train"] = {"claims": int(len(ye)), "exact": float(ye.mean())}
    R["heldout_claims"] = {"claims": int(len(yo)), "auc_k2_7": v2.auc(po[mid], yo[mid]),
                           "ece_all": v2.ece(po, yo.astype(float))[0],
                           "v2_auc_no_ct_reference": json.loads((V2_DIR / "results.json").read_text())["heldout_claims"]["auc_no_ct"]}

    # ---- tau on even chains, same rule and grid as v2
    A, B, starts, n = all_ray_claims(store)
    pa = model.prob(claim_features(tab, A, B))
    sweep, tau = [], None
    for tt in v2.TAU_GRID:
        g, _ = generate_claims(store, A, B, pa, starts, n, tt)
        r = v2.rescore(truth_even, g)
        sweep.append({"tau": float(tt), "adjacent_exact": r["adjacent_exact"], "adjacent_tests": r["adjacent_tests"]})
        if r["adjacent_tests"] >= 100 and (r["adjacent_exact"] or 0) >= 0.95:
            tau = float(tt)
            break
    R["tau_sweep_even"], R["tau"] = sweep, tau
    gen, conf = generate_claims(store, A, B, pa, starts, n, tau if tau is not None else 1.01)
    gb, cb = generate_blocked(store, tab, model, tau if tau is not None else 1.01, block_rays=1000)
    R["blocked_equals_all_claims"] = bool(np.array_equal(gen.xyz, gb.xyz) and np.array_equal(gen.chain, gb.chain)
                                          and np.allclose(conf, cb))
    R["generated"] = {"chains": gen.n_chains, "points": len(gen)}
    G = v2.rescore(truth_odd, gen)
    R["generated_heldout"] = G
    R["exact_count"] = int(round(G["exact"] * G["tests"]))
    net = lambda o: o["f_ok"] - 3 * o["f_bad"]
    R["v2_reference"] = {k: ref[k] for k in ("adjacent_exact", "exact", "human_pairs_covered", "human_adjacent_covered")}
    R["v2_reference"]["net_3x"] = net(ref["adjacent_outcome"])
    R["net_hours_saved"] = {f"fix_{f}x": v2.net_hours(G["adjacent_outcome"], f) for f in (1, 3, 5)}

    crit["precision_parity"] = {"pass": G["adjacent_exact"] >= 0.9036 and G["exact"] >= 0.9083,
                  "measured": {"adjacent_exact": G["adjacent_exact"], "exact": G["exact"]}}
    crit["coverage_parity"] = {"pass": G["human_pairs_covered"] >= 0.0652 and G["human_adjacent_covered"] >= 0.4047,
                  "measured": {"pairs_covered": G["human_pairs_covered"], "adjacent_covered": G["human_adjacent_covered"]}}
    crit["human_time_parity"] = {"pass": net(G["adjacent_outcome"]) >= 0.195,
                  "measured": {"f_ok_minus_3f_bad": net(G["adjacent_outcome"]), "hours_3x": R["net_hours_saved"]["fix_3x"]}}

    model.save(MODELS / "paris4_claims_v2_noct.json", tau=tau,
               fitted_on="PHerc. Paris 4 relative_windings.json, even-numbered chains, all claims",
               step_model="paris4_even_chains.json",
               experiment="experiments/2026-10-13-claims-noct-generator")
    R["criteria"] = crit
    R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    R["seconds_total"] = round(time.time() - t_all, 1)
    (HERE / "results_region.json").write_text(json.dumps(R, indent=2))
    print(json.dumps({"criteria": crit, "tau": tau, "blocked_equals_all_claims": R["blocked_equals_all_claims"],
                      "heldout_claims": R["heldout_claims"], "seconds": R["seconds_total"]}, indent=2))


if __name__ == "__main__":
    main()

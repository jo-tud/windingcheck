"""Follow-up analysis, written after run.py's first result (not one of run.py's checks).

Why: run.py's discrimination and calibration checks, and its single-feature comparison, score
confidence on *human*-adjacent pairs. On that set a ray that counts two
steps is wrong by construction, and a pair's confidence is a product over its steps, so any
near-constant model separates right from wrong by the step count alone (single-feature AUCs
0.90-0.99 beat the full model's 0.875). That inflates the AUC and makes the
single-feature comparison unreadable.

The clean unit is the generator's own claim: a pair of crossings the ray says are Δw = k apart,
scored against the human Δw wherever both crossings match clicks of one human chain. Nothing is
conditioned on the truth. Same model (fitted on even chains by run.py), held-out odd chains.

    python experiments/2026-10-12-confident-crossings/posthoc.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "windingcheck" / "src"))
sys.path.insert(0, str(HERE))

from run import FEATURES, auc, ece, test_conf  # noqa: E402
from windingcheck import data  # noqa: E402
from windingcheck.chains import load_pcl  # noqa: E402
from windingcheck.confidence import StepModel, step_features, step_index  # noqa: E402
from windingcheck.crossings import load_ray_store  # noqa: E402
from windingcheck.prior import Umbilicus  # noqa: E402
from windingcheck.score import build_truth, score_chains  # noqa: E402


def main():
    d = data.fetch("PHercParis4", crossings=True)
    truth = build_truth(load_pcl(d / "relative_windings.json"))
    umb = Umbilicus(d / "umbilicus.json")
    store = load_ray_store(d / "winding_inference")
    all_k = step_index(store)
    X, _ = step_features(store, umb, all_k, tree=cKDTree(store.xyz))
    pos = lambda k: np.searchsorted(all_k, k)
    model = StepModel.load(REPO / "tools/windingcheck/src/windingcheck/models/paris4_even_chains.json")
    p = model.prob(X)
    _, tests = score_chains(truth, store.chains(), return_tests=True)
    odd = truth.chains.chain[tests["a"]] % 2 == 1
    exact = tests["pred_dw"] == tests["true_dw"]
    prod, _ = test_conf(tests, lambda k: p[pos(k)])
    R = {}

    # claimed one-sheet steps (pred_dw == 1), held out
    one = odd & (tests["pred_dw"] == 1)
    e1, rel1 = ece(prod[one], exact[one])
    R["claimed_dw1_heldout"] = {"tests": int(one.sum()), "exact_raw": float(exact[one].mean()),
                                "auc": auc(prod[one], exact[one]), "ece": e1, "reliability": rel1}
    # raw single features on the same unit, direction-free (no model)
    k1 = np.minimum(tests["ca"], tests["cb"])[one]
    feat = {}
    for i, f in enumerate(FEATURES):
        a = auc(X[pos(k1), i], exact[one])
        feat[f] = None if a is None else max(a, 1 - a)
    R["claimed_dw1_single_feature_auc"] = feat
    # claimed multi-sheet pairs, product of steps
    multi = odd & (tests["pred_dw"] >= 2) & (tests["pred_dw"] <= 7)
    em, relm = ece(prod[multi], exact[multi])
    R["claimed_dw2_7_heldout"] = {"tests": int(multi.sum()), "exact_raw": float(exact[multi].mean()),
                                  "auc": auc(prod[multi], exact[multi]), "ece": em, "reliability": relm}
    # where do the claimed one-sheet steps go wrong? human says 2+ (missed sheet) or <= 0
    wrong = one & ~exact
    R["claimed_dw1_wrong_true_dw"] = {str(int(k)): int(v) for k, v in
                                      zip(*np.unique(tests["true_dw"][wrong], return_counts=True))}
    (HERE / "results_posthoc.json").write_text(json.dumps(R, indent=2))
    print(json.dumps({k: (v if "reliab" not in json.dumps(v)[:0] else v) for k, v in R.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()

"""Checks of the windingcheck scorer and three baselines on PHerc. Paris 4.

    python experiments/2026-10-12-winding-scorer/run.py     (CPU, ~minutes; fetches ~170 MB once)
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "windingcheck" / "src"))

from windingcheck import data  # noqa: E402
from windingcheck.chains import Chains, load_pcl  # noqa: E402
from windingcheck.crossings import load_crossings  # noqa: E402
from windingcheck.plot import drift_map  # noqa: E402
from windingcheck.prior import Umbilicus, fit_prior  # noqa: E402
from windingcheck.score import build_truth, loop_consistency, score_chains, score_field  # noqa: E402


def adjacent_exact(truth, tests):
    a, b = tests["a"], tests["b"]
    adj = (b == a + 1) & (truth.chains.chain[a] == truth.chains.chain[b])
    return float(np.mean(tests["pred_dw"][adj] == tests["true_dw"][adj])), int(adj.sum())


def exact_where(tests, mask):
    return float(np.mean(tests["pred_dw"][mask] == tests["true_dw"][mask])) if mask.any() else None


def main():
    t_all = time.time()
    d = data.fetch("PHercParis4", crossings=True)
    truth = build_truth(load_pcl(d / "relative_windings.json"))
    umb = Umbilicus(d / "umbilicus.json")
    c = truth.chains
    R = {"truth": {"chains": c.n_chains, "clicks": len(c), "pairs": int(len(truth.pairs()[0])),
                   "adjacent_pairs": int(len(truth.pairs(adjacent_only=True)[0])),
                   "tol_n_median": float(np.nanmedian(truth.tol_n)),
                   "clicks_without_normal": int(np.isnan(truth.tol_n).sum())}}
    crit = {}

    # the human chains scored against themselves
    r1 = score_chains(truth, c)
    R["self_score"] = r1
    crit["self_score"] = {"pass": r1["exact"] == 1.0 and r1["human_pairs_covered"] == 1.0,
                  "measured": {"exact": r1["exact"], "covered": r1["human_pairs_covered"]}}

    # control: levels shuffled within each chain
    rng = np.random.default_rng(0)
    lv = c.level.copy()
    for k in range(c.n_chains):
        m = np.flatnonzero(c.chain == k)
        lv[m] = rng.permutation(lv[m])
    r2, t2 = score_chains(truth, Chains(c.xyz, lv, c.chain, c.names), return_tests=True)
    adj2, nadj2 = adjacent_exact(truth, t2)
    R["shuffled_control"] = dict(r2, adjacent_exact=adj2, adjacent_tests=nadj2)
    crit["shuffled_control"] = {"pass": adj2 < 0.30, "measured": {"adjacent_exact": adj2}}

    # control: a missed sheet after the middle click of every chain with >= 4 clicks
    lv = c.level.copy()
    straddle = total = 0
    for k in range(c.n_chains):
        m = np.flatnonzero(c.chain == k)
        n = len(m)
        total += n * (n - 1) // 2
        if n >= 4 and np.isfinite(truth.tol_n[m]).all():
            mid = n // 2
            lv[m[mid + 1:]] += 1
            straddle += (mid + 1) * (n - mid - 1)
    expected = 1 - straddle / total
    r3 = score_chains(truth, Chains(c.xyz, lv, c.chain, c.names))
    R["missed_sheet_control"] = dict(r3, expected_exact=expected)
    crit["missed_sheet_control"] = {"pass": abs(r3["exact"] - expected) <= 0.001 and r3["sign_errors"] == 0,
                  "measured": {"exact": r3["exact"], "expected": expected, "sign_errors": r3["sign_errors"]}}

    # human label consistency: loop closure between same-winding collections and relative chains
    same = load_pcl(d / "same_windings.json", same_winding=True)
    r6 = loop_consistency(truth, same)
    R["human_loop_consistency"] = r6
    crit["human_loop_consistency"] = ({"pass": r6["loops_consistent"] >= 0.95, "measured": r6["loops_consistent"]}
                  if r6["loops"] >= 30 else {"pass": None, "measured": "not testable", "loops": r6["loops"]})

    # spiral priors, fitted on even chains, scored on odd chains
    ids = np.arange(c.n_chains)
    train, test = ids[ids % 2 == 0], ids[ids % 2 == 1]
    tests = {}
    for key, bins in (("prior_global", 1), ("prior_radius_binned", 10)):
        prior = fit_prior(truth, umb, train, bins=bins)
        r, t = score_field(truth, prior.delta, chain_ids=test, return_tests=True)
        R[key] = dict(r, prior=prior.to_json())
        tests[key] = t

    # organisers' neural crossings, full set (also the cost check)
    t0 = time.time()
    cross = load_crossings(d / "winding_inference")
    t_load = time.time() - t0
    r3b, t3b = score_chains(truth, cross, return_tests=True)
    secs = time.time() - t0
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    R["neural_crossings"] = dict(r3b, crossings=len(cross), rays=cross.n_chains,
                                    seconds_load=round(t_load, 1), seconds_total=round(secs, 1))
    tests["neural_crossings"] = t3b
    crit["cost"] = {"pass": secs < 600 and rss_gb < 16, "measured": {"seconds": round(secs, 1), "peak_rss_gb": round(rss_gb, 2)}}
    del cross

    # Same-pair comparison for the crossings prediction: odd chains, pairs the crossings cover, |Δw| >= 4
    b1 = tests["prior_global"]
    n1 = len(c) + 1
    key1 = b1["a"] * n1 + b1["b"]
    key3 = t3b["a"] * n1 + t3b["b"]
    on_odd = np.isin(c.chain[t3b["a"]], test)
    far3 = on_odd & (np.abs(t3b["true_dw"]) >= 4)
    far1 = np.isin(key1, key3[far3])
    R["crossings_vs_prior_same_pairs_dw_ge_4"] = {"crossings_tests": int(far3.sum()), "crossings_exact": exact_where(t3b, far3),
                                        "prior_pairs": int(far1.sum()), "prior_exact": exact_where(b1, far1)}
    # adjacent-pair exact for all three methods
    for k, t in tests.items():
        R[k]["adjacent_exact"], R[k]["adjacent_tests"] = adjacent_exact(truth, t)

    # Predictions (not pass/fail)
    glob, binned, cross = R["prior_global"], R["prior_radius_binned"], R["neural_crossings"]
    cmp = R["crossings_vs_prior_same_pairs_dw_ge_4"]
    R["predictions"] = {
        "global prior exact < 50% on all pairs": glob["exact"] < 0.5,
        "global prior exact >= 70% on adjacent pairs": glob["adjacent_exact"] >= 0.7,
        "radius-binned prior better than global prior on all pairs": binned["exact"] > glob["exact"],
        "crossings cover < 50% of human pairs": cross["human_pairs_covered"] < 0.5,
        "crossings exact > global prior for |dw| >= 4 (same pairs)": (cmp["crossings_exact"] or 0) > (cmp["prior_exact"] or 0),
    }
    R["criteria"] = crit
    R["seconds_total"] = round(time.time() - t_all, 1)
    (HERE / "results.json").write_text(json.dumps(R, indent=2))

    drift_map(truth, {"Spiral prior, one spacing": tests["prior_global"],
                      "Spiral prior, spacing by radius": tests["prior_radius_binned"],
                      "Organisers' neural crossings": t3b},
              HERE / "drift_map.png", umb=umb,
              title="PHerc. Paris 4: adjacent human sheet pairs, grey = right, red = too many sheets, blue = too few")
    print(json.dumps({"criteria": crit, "predictions": R["predictions"]}, indent=2))


if __name__ == "__main__":
    main()

"""Release checks from the outputs of run.sh.   python check.py WORK_DIR  ->  WORK_DIR/results.json"""
import hashlib
import json
import re
import sys
from pathlib import Path

ALLOWED_KEYS = {"vc_pointcollections_json_version", "collections", "name", "color", "metadata",
                "winding_is_absolute", "points", "p", "creation_time", "wind_a", "tags", "source",
                "confidence", "tau"}


def timing(path):
    t = Path(path).read_text()
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t).group(1).split(":")
    return {"wall_s": round(sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall))), 1),
            "peak_rss_gb": round(int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) / 1e6, 2),
            "exit": int(re.search(r"Exit status: (\d+)", t).group(1))}


def last_json(path):
    return json.loads(Path(path).read_text().strip().splitlines()[-1])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(work):
    work = Path(work)
    clone = work / "clone"
    R = {"commit": (work / "commit.txt").read_text().strip(),
         "pytest": (work / "pytest.txt").read_text().strip(),
         "times": {n: timing(work / f"time_{n}.txt")
                   for n in ("fetch", "score_raw", "generate", "score_claims", "every50")}}
    crit = {}

    pins = dict(l.split("==") for l in (clone / "tools/windingcheck/constraints.txt").read_text().splitlines()
                if "==" in l and not l.startswith("#"))
    frozen = dict(l.split("==") for l in (work / "pip_freeze.txt").read_text().splitlines() if "==" in l)
    pin_ok = all(frozen.get(k) == pins[k] for k in ("numpy", "scipy"))
    m = re.search(r"(\d+) passed", R["pytest"])
    crit["install_and_tests"] = {"pass": pin_ok and m is not None and int(m.group(1)) == 11 and "failed" not in R["pytest"],
                  "measured": {"pytest": R["pytest"], "numpy": frozen.get("numpy"), "scipy": frozen.get("scipy")}}

    wi = clone / "data/spiral_datasets/PHercParis4/winding_inference"
    man = json.loads((wi / "manifest.json").read_text())
    bad, n = [], 0
    for sh in man["shards"]:
        for a in sh["arrays"].values():
            n += 1
            if sha256(wi / sh["name"] / a["file"]) != a["sha256"]:
                bad.append(f'{sh["name"]}/{a["file"]}')
    crit["fetch_checksums"] = {"pass": R["times"]["fetch"]["exit"] == 0 and not bad and n > 0,
                  "measured": {"files_checked": n, "mismatches": bad, **R["times"]["fetch"]}}

    raw = json.loads((work / "score_raw.json").read_text())
    got = {"tests": raw["tests"], "adjacent_tests": raw["adjacent_tests"],
           "adjacent_exact_count": round(raw["adjacent_exact"] * raw["adjacent_tests"])}
    crit["raw_crossings_heldout"] = {"pass": got == {"tests": 2489, "adjacent_tests": 1159, "adjacent_exact_count": 889},
                  "measured": dict(got, adjacent_exact=raw["adjacent_exact"], **R["times"]["score_raw"])}

    gen = last_json(work / "out_generate.txt")
    T = R["times"]["generate"]
    crit["generator_whole_volume"] = {"pass": gen["tau"] == 0.74 and gen["chains"] == 8_268_376 and gen["points"] == 20_064_412
                  and T["wall_s"] <= 1800 and T["peak_rss_gb"] <= 32,
                  "measured": dict(gen, **T, bytes=(work / "claims.json").stat().st_size)}

    sc = json.loads((work / "score_claims.json").read_text())
    got = {"tests": sc["tests"], "exact_count": round(sc["exact"] * sc["tests"]),
           "adjacent_tests": sc["adjacent_tests"],
           "adjacent_exact_count": round(sc["adjacent_exact"] * sc["adjacent_tests"]),
           "adjacent_covered_count": round(sc["human_adjacent_covered"] * sc["human_adjacent_pairs"]),
           "human_adjacent_pairs": sc["human_adjacent_pairs"]}
    want = {"tests": 1060, "exact_count": 972, "adjacent_tests": 721, "adjacent_exact_count": 659,
            "adjacent_covered_count": 389, "human_adjacent_pairs": 974}
    crit["generated_heldout"] = {"pass": got == want,
                  "measured": dict(got, adjacent_exact=sc["adjacent_exact"],
                                   human_adjacent_covered=sc["human_adjacent_covered"], **R["times"]["score_claims"])}

    e50 = last_json(work / "out_every50.txt")
    try:
        v_cols, v_pts = last_json(work / "villa.txt")
    except Exception:
        v_cols = v_pts = None
    crit["loads_in_villa"] = {"pass": e50["chains"] == 165_568 and e50["points"] == 402_054
                  and v_cols == 165_568 and v_pts == 402_054,
                  "measured": {"written": e50, **R["times"]["every50"], "villa_collections": v_cols,
                               "villa_points": v_pts, "villa_commit": (work / "villa_commit.txt").read_text().strip(),
                               "villa_stderr_tail": (work / "villa_err.txt").read_text()[-300:]}}

    keys = {f: sorted(k.strip('":') for k in (work / f"keys_{f}.txt").read_text().split())
            for f in ("claims", "claims_every50")}
    extra = sorted(set().union(*map(set, keys.values())) - ALLOWED_KEYS)
    images = [l for l in (work / "images.txt").read_text().splitlines() if l and not l.endswith("/drift.png")]
    crit["geometry_only"] = {"pass": not extra and not images, "measured": {"keys": keys, "unexpected_keys": extra,
                                                                 "other_images": images}}
    R["criteria"] = crit
    R["all_pass"] = all(c["pass"] for c in crit.values())
    (work / "results.json").write_text(json.dumps(R, indent=2))
    print(json.dumps({k: v["pass"] for k, v in crit.items()}))


if __name__ == "__main__":
    main(sys.argv[1])

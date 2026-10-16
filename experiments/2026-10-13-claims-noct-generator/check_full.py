"""Full-volume checks from the outputs of full_volume.sh.   python check_full.py WORK_DIR"""
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V2_DIR = REPO / "experiments" / "2026-10-12-claim-confidence-v2"
sys.path.insert(0, str(REPO / "tools" / "windingcheck" / "src"))
_spec = importlib.util.spec_from_file_location("v2run", V2_DIR / "run.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

from windingcheck import data  # noqa: E402
from windingcheck.chains import load_pcl  # noqa: E402
from windingcheck.score import build_truth  # noqa: E402


def timing(path):
    t = Path(path).read_text()
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t).group(1).split(":")
    sec = sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall)))
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) / 1e6
    cpu = re.search(r"Percent of CPU this job got: (\d+)%", t).group(1)
    return {"wall_s": round(sec, 1), "peak_rss_gb": round(rss, 2), "cpu_percent": int(cpu)}


def main(work):
    work = Path(work)
    R = {"commit": (work / "commit.txt").read_text().strip(), "pytest": (work / "pytest.txt").read_text().strip()}
    crit = {}
    region = json.loads((HERE / "results_region.json").read_text())
    for name in ("full", "every50"):
        R[name] = dict(json.loads((work / f"out_{name}.json").read_text().strip().splitlines()[-1]),
                       **timing(work / f"time_{name}.txt"), bytes=(work / f"claims_{name}.json").stat().st_size)
    F = R["full"]
    crit["full_volume_on_cpu"] = {"pass": F["wall_s"] <= 1800 and F["peak_rss_gb"] <= 32 and F["crossings_in"] > 22_000_000,
                  "measured": F}

    d = data.fetch("PHercParis4")
    human = load_pcl(d / "relative_windings.json")
    ids = np.arange(human.n_chains)
    truth_odd = build_truth(human.subset(ids[ids % 2 == 1]))
    full = load_pcl(work / "claims_full.json")
    G = v2.rescore(truth_odd, full)
    got = {"tests": G["tests"], "exact_count": int(round(G["exact"] * G["tests"])), "points": len(full)}
    want = {"tests": region["generated_heldout"]["tests"], "exact_count": region["exact_count"]}
    R["full_heldout"] = G
    crit["full_equals_region"] = {"pass": got["tests"] == want["tests"] and got["exact_count"] == want["exact_count"],
                  "measured": {"full": got, "region": want}}
    del full

    out = work / "claims_every50.json"
    back = load_pcl(out)
    villa = subprocess.run([v2.villa_python(), "-c",
                            "import sys, json; sys.path.insert(0, sys.argv[1]);"
                            "from point_collection import load_point_collection as L;"
                            "c = L(sys.argv[2]); print(json.dumps([len(c), sum(len(v['points']) for v in c.values())]))",
                            str(v2.v1.VILLA_SF), str(out)], capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    try:
        v_cols, v_pts = json.loads(villa.stdout.strip().splitlines()[-1])
    except Exception:
        v_cols = v_pts = None
    written = R["every50"]["points"]
    crit["loads_in_tools"] = {"pass": v_pts == written and len(back) == written,
                  "measured": {"points_written": written, "windingcheck_points": len(back),
                               "villa_collections": v_cols, "villa_points": v_pts,
                               "villa_python": v2.villa_python(), "villa_stderr_tail": villa.stderr[-300:]}}
    crit["clean_checkout"] = {"pass": "passed" in R["pytest"] and "failed" not in R["pytest"], "measured": R["pytest"]}
    R["criteria"] = crit
    (HERE / "results_full.json").write_text(json.dumps(R, indent=2))
    print(json.dumps(crit, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])

"""How much human time went into the public PHerc. Paris 4 winding annotations?

Reads the three VC3D point-collection files of the `spiral-input` dataset and estimates active
annotation time from the per-point `creation_time` stamps (ms since epoch). Gaps longer than a cap
count as breaks, so the figure is a lower bound on human input: it excludes thinking before the
first click of a session, review, and the ~27k verified surface patches.

    python human_time.py [data_dir]      # default: data/spiral_datasets/PHercParis4
"""
import json, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

URL = "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/"
FILES = ["relative_windings.json", "same_windings.json", "abs_winding.json"]
CAPS_S = [120, 300, 600]
root = Path(__file__).resolve().parents[2]
data = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/spiral_datasets/PHercParis4"
data.mkdir(parents=True, exist_ok=True)


def day(t):
    return datetime.fromtimestamp(t, timezone.utc).date().isoformat()


def active_hours(t):
    g = np.diff(np.sort(t))
    return {f"cap_{c}s": round(float(g[g < c].sum()) / 3600, 2) for c in CAPS_S}


out, all_t = {}, []
for f in FILES:
    p = data / f
    if not p.exists():
        urllib.request.urlretrieve(URL + f, p)
    cols = json.load(open(p))["collections"]
    pts = [q for c in cols.values() for q in c["points"].values()]
    t = np.array([q["creation_time"] for q in pts], float) / 1000
    all_t.append(t)
    rec = {"collections": len(cols), "points": len(pts), "first_day": day(t.min()),
           "last_day": day(t.max()), "active_hours": active_hours(t)}
    if f == "relative_windings.json":
        # Each collection is a radial chain of clicks, one per sheet crossing (wind_a = 1, 2, 3...).
        spans, steps, pairs = [], [], 0
        for c in cols.values():
            q = sorted((p_["wind_a"], p_["p"]) for p_ in c["points"].values() if p_.get("wind_a") is not None)
            if len(q) < 2:
                continue
            w = np.array([a for a, _ in q]); xyz = np.array([b for _, b in q])
            spans.append(w.max() - w.min()); pairs += len(q) * (len(q) - 1) // 2
            d = np.diff(w)
            steps += list(np.linalg.norm(np.diff(xyz, axis=0), axis=1)[d == 1])
        rec.update({"chains_with_windings": len(spans), "median_span_windings": float(np.median(spans)),
                    "max_span_windings": float(max(spans)), "implied_pairwise_constraints": pairs,
                    "adjacent_wrap_spacing_vox": {"p10": round(float(np.percentile(steps, 10)), 1),
                                                  "median": round(float(np.median(steps)), 1),
                                                  "p90": round(float(np.percentile(steps, 90)), 1)}})
    out[f] = rec

t = np.concatenate(all_t)
rel = out["relative_windings.json"]
out["total"] = {"points": int(t.size), "active_hours": active_hours(t)}
out["seconds_per_relative_point"] = {k: round(v * 3600 / rel["points"], 1) for k, v in rel["active_hours"].items()}
print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).with_name("results.json"), "w"), indent=1)

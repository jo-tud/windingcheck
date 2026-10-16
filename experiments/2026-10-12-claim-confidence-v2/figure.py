"""Figure for the v2 write-up: held-out calibration and adjacent-pair outcomes (raw, v1, v2).

    python figure.py      (reads results.json; matplotlib)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#d9d8d4", "#0b0b0b", "#52514e"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
rel = R["heldout_claims"]["reliability_all"]
a1.plot([0, 1], [0, 1], color=GRAY, lw=1.5, zorder=1)
a1.plot([b["mean_p"] for b in rel], [b["exact"] for b in rel], color=BLUE, lw=2, marker="o", ms=8,
        markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=2)
a1.set_xlim(0, 1)
a1.set_ylim(0, 1)
a1.set_xlabel("v2 claim confidence (bin mean)", color=INK2)
a1.set_ylabel("share exact vs human chains", color=INK2)
a1.set_title(f"Calibration, held-out claims (ECE {R['heldout_claims']['ece_all']:.3f}; bar 0.05)",
             color=INK, fontsize=10, loc="left")

rows = [("organisers' crossings, raw", R["raw_region_heldout"]), ("v1 step filter", R["v1_region_heldout"]),
        ("v2 claim model", R["generated_heldout"])]
for i, (name, r) in enumerate(rows):
    o = r["adjacent_outcome"]
    ok, bad = o["f_ok"], o["f_bad"]
    none = 1 - ok - bad
    y = len(rows) - 1 - i
    a2.barh(y, ok, color=BLUE, height=0.55, edgecolor="#fcfcfb", linewidth=2)
    a2.barh(y, bad, left=ok, color=ORANGE, height=0.55, edgecolor="#fcfcfb", linewidth=2)
    a2.barh(y, none, left=ok + bad, color=GRAY, height=0.55, edgecolor="#fcfcfb", linewidth=2)
    a2.text(ok / 2, y, f"{ok:.0%}", ha="center", va="center", color="white", fontsize=9)
    a2.text(ok + bad + 0.01, y, f"{bad:.1%} wrong", ha="left", va="center", color=INK, fontsize=9)
a2.set_yticks(range(len(rows)))
a2.set_yticklabels([n for n, _ in rows][::-1], color=INK)
a2.set_xlim(0, 1)
a2.set_xlabel("share of the 974 held-out human adjacent pairs", color=INK2)
a2.set_title("Adjacent pairs: right (blue), wrong (orange), not covered (gray)", color=INK, fontsize=10, loc="left")
for ax in (a1, a2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
fig.tight_layout()
out = HERE / "figures_paris4" / "v2_calibration_and_outcomes.png"
fig.savefig(out, dpi=110)
print(out)

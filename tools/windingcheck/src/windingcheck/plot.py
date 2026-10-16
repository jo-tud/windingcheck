"""Pictures of where a winding source is wrong: every tested human pair drawn on the scroll."""

from __future__ import annotations

import numpy as np

from .score import Truth


def drift_map(truth: Truth, tests_by_method: dict, path, *, title="", umb=None):
    """One column per method. Top: x-y view, bottom: radius-z view (if ``umb``).
    Each adjacent human pair that was tested is a dot at its midpoint, coloured by the predicted
    minus the true sheet count (0 grey, positive red = counted too many sheets, negative blue)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 2 if umb is not None else 1
    n = len(tests_by_method)
    fig, axes = plt.subplots(rows, n, figsize=(5.0 * n, 4.6 * rows), squeeze=False, constrained_layout=True)
    xyz = truth.chains.xyz
    cmap = plt.get_cmap("RdBu_r")
    for k, (name, t) in enumerate(tests_by_method.items()):
        a, b = t["a"], t["b"]
        adj = np.abs(t["true_dw"]) == 1
        err = (t["pred_dw"] - t["true_dw"])[adj]
        mid = 0.5 * (xyz[a[adj]] + xyz[b[adj]])
        ok = err == 0
        views = [(mid[:, 0], mid[:, 1], "x (voxels)", "y (voxels)")]
        if umb is not None:
            r, _ = umb.polar(mid)
            views.append((r, mid[:, 2], "radius from umbilicus (voxels)", "z (voxels)"))
        for row, (u, v, xl, yl) in enumerate(views):
            ax = axes[row, k]
            ax.scatter(xyz[:, 0] if row == 0 else umb.polar(xyz)[0], xyz[:, 1] if row == 0 else xyz[:, 2],
                       s=1, c="#dddddd", lw=0)
            ax.scatter(u[ok], v[ok], s=3, c="#8a8a8a", lw=0)
            sc = ax.scatter(u[~ok], v[~ok], s=7, c=np.clip(err[~ok], -3, 3), cmap=cmap, vmin=-3, vmax=3, lw=0)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            if row == 0:
                ax.set_aspect("equal")
                exact = ok.mean() if len(ok) else float("nan")
                ax.set_title(f"{name}\nadjacent pairs tested {len(ok)}, exact {exact:.0%}", fontsize=9)
        fig.colorbar(sc, ax=axes[:, k].tolist(), shrink=0.6, label="predicted − human sheets")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

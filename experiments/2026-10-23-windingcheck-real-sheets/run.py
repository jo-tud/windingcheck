"""Real-sheet check of the CT sheet-normal match (windingcheck v0.1.2), PHerc. Paris 4.

    pip install tifffile                                   # this experiment only, to read the patches
    windingcheck normals --heldout --z-band 7000 17000 --out normals.json   # reads 1.42 GB of CT, cached
    python experiments/2026-10-23-windingcheck-real-sheets/run.py normals.json   # downloads 0.41 GB of patches

Every held-out click (odd human chain) that lies within 2 voxels of a human-verified patch is moved along that
patch to a point 4-24 voxels away and keeps the click's own wind_a. So every claim is correct, and every wrong test
is a matcher error (or a wrong patch link). A second file adds a +-1 slip at one cut in 30 % of the candidate chains.
Both files are scored with the chain-direction match (shipped since v0.1) and the CT sheet-normal match
(``build_truth(normals=)``), and through the CLI (``windingcheck score --heldout [--normals]``).
Writes results.json and compares every number with expected.json (integers exactly, floats to 1e-9 relative);
exit 1 on any difference. CPU only; about 1 GB RAM and 1 min after the inputs exist.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WC_SRC = REPO / "tools" / "windingcheck" / "src"
sys.path.insert(0, str(WC_SRC))
from windingcheck import data  # noqa: E402
from windingcheck.chains import from_arrays, load_pcl, save_pcl  # noqa: E402
from windingcheck.normals import for_chains  # noqa: E402
from windingcheck.score import build_truth, score_chains  # noqa: E402

PATCH_URL = f"{data.BASE}/PHercParis4/verified_patches"
PATCH_FILES = ("meta.json", "x.tif", "y.tif", "z.tif")

D_LINK = 2.0            # voxels: a click is "on" a patch
NODE_CUT = 35.0         # voxels: patch nodes considered around a click (grid step 20)
R_LO, R_HI = 4.0, 24.0  # voxels: displacement along the patch (inside tol_t 25)
N_SAMPLE = 3000         # area-weighted samples on the local patch per (click, patch) link
N_KEEP = 300            # samples kept per link (in R_LO..R_HI)
K_REP = 8               # replicates (candidate chains per human chain)
SEED = 1684
P_PLANT = 0.30          # share of candidate chains that get one +-1 slip
SEED_PLANT = 1684 + 1000
N_MISSING = 2           # held-out clicks without a CT normal (chain 92, z 17,252.6, above the z band)
BINS = [0, 10, 20, 30, 45, 90.0001]
BIN_NAMES = ["0-10", "10-20", "20-30", "30-45", "45-90"]
N_BOOT, BOOT_SEED = 2000, 12345


def log(s):
    print(s, flush=True)


# ---------------------------------------------------------------- inputs
def heldout():
    human = load_pcl(data.fetch("PHercParis4") / "relative_windings.json")
    ids = np.arange(human.n_chains)
    return human.subset(ids[ids % 2 == 1])


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_one(dest, nm, digest):
    """Download one patch's files (cached); True if its sha256 digest matches: the sha256 of its four files'
    sha256 hex digests, joined by newlines."""
    sums = []
    for f in PATCH_FILES:
        p = dest / nm / f
        if not (p.exists() and p.stat().st_size > 0):
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".part")
            for attempt in range(5):
                try:
                    urllib.request.urlretrieve(f"{PATCH_URL}/{nm}/{f}", tmp)
                    tmp.rename(p)
                    break
                except OSError:
                    if attempt == 4:
                        raise
        sums.append(sha256(p))
    return hashlib.sha256("\n".join(sums).encode()).hexdigest() == digest


def fetch_patches(dest, workers=16):
    """Download the patches named in patches.json (cached, `workers` at a time) and check every sha256 digest."""
    lst = json.loads((HERE / "patches.json").read_text())["patches"]
    names, t0 = sorted(lst), time.time()
    bad = []
    with ThreadPoolExecutor(workers) as ex:
        for i, (nm, ok) in enumerate(zip(names, ex.map(lambda n: fetch_one(dest, n, lst[n]), names))):
            if not ok:
                bad.append(nm)
            if (i + 1) % 500 == 0:
                log(f"patches: {i + 1} of {len(lst)} ({time.time() - t0:.0f} s)")
    if bad:
        sys.exit(f"sha256 mismatch in {len(bad)} patches, e.g. {bad[:3]}; the public patches changed")
    return names


def patch_grid(d):
    g = np.stack([tifffile.imread(d / f"{c}.tif").astype(np.float64) for c in "xyz"], -1)
    v = np.all(np.isfinite(g), -1) & ~np.all(g == -1, -1)
    return g, v


def point_tri_dist(p, a, b, c):
    """Distance from points p to triangles (a, b, c), all (n, 3), and the nearest points (Ericson 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum('ij,ij->i', ab, ap), np.einsum('ij,ij->i', ac, ap)
    bp = p - b
    d3, d4 = np.einsum('ij,ij->i', ab, bp), np.einsum('ij,ij->i', ac, bp)
    cp = p - c
    d5, d6 = np.einsum('ij,ij->i', ab, cp), np.einsum('ij,ij->i', ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    with np.errstate(divide='ignore', invalid='ignore'):
        v = vb / den
        w = vc / den
        q = a + ab * v[:, None] + ac * w[:, None]
        r = np.select([(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (d6 >= 0) & (d5 <= d6)], [0, 1, 2], 9)
        tab = d1 / (d1 - d3)
        tac = d2 / (d2 - d6)
        tbc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        e_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    q = np.where((r == 9)[:, None] & e_bc[:, None], b + (c - b) * tbc[:, None], q)
    q = np.where((r == 9)[:, None] & e_ac[:, None], a + ac * tac[:, None], q)
    q = np.where((r == 9)[:, None] & e_ab[:, None], a + ab * tab[:, None], q)
    q = np.where((r == 0)[:, None], a, q)
    q = np.where((r == 1)[:, None], b, q)
    q = np.where((r == 2)[:, None], c, q)
    d = np.linalg.norm(p - q, axis=1)
    d[~np.isfinite(d)] = np.inf
    return d, q


def local_triangles(g, v, foot, cut):
    """Triangles of grid g (quads with 4 valid nodes, split on the (0,0)-(1,1) diagonal) whose 3 nodes lie within
    `cut` of `foot`."""
    near = v & (np.linalg.norm(np.where(v[..., None], g, 1e9) - foot, axis=-1) < cut)
    a, b = near[:-1, :-1], near[:-1, 1:]
    c, d = near[1:, 1:], near[1:, :-1]
    i, j = np.nonzero(a & b & c & d)
    A, B, C, D = g[i, j], g[i, j + 1], g[i + 1, j + 1], g[i + 1, j]
    return np.concatenate([A, A]), np.concatenate([B, C]), np.concatenate([C, D])


# ---------------------------------------------------------------- candidates (no score)
def build(odd, patch_dir, names, work):
    """Link held-out clicks to patches, sample known-correct candidate points, plant slips. Writes the two
    candidate files to `work` and returns the counts and the arrays the scoring needs."""
    X = odd.xyz
    lo_all, hi_all = X.min(0) - NODE_CUT, X.max(0) + NODE_CUT
    links, samples = [], {}
    rng = np.random.default_rng(SEED)
    for pi, nm in enumerate(names):
        (x0, y0, z0), (x1, y1, z1) = json.loads((patch_dir / nm / "meta.json").read_text())["bbox"]
        lo, hi = np.array([x0, y0, z0]) - D_LINK, np.array([x1, y1, z1]) + D_LINK
        if np.any(hi < lo_all) or np.any(lo > hi_all):
            continue
        inbox = np.flatnonzero(np.all((X >= lo) & (X <= hi), 1))
        if not len(inbox):
            continue
        g, v = patch_grid(patch_dir / nm)
        P = g[v]
        if not len(P):
            continue
        dn, _ = cKDTree(P).query(X[inbox], distance_upper_bound=NODE_CUT)
        for ci in inbox[np.isfinite(dn)]:
            A, B, C = local_triangles(g, v, X[ci], NODE_CUT)
            if not len(A):
                continue
            dd, q = point_tri_dist(np.repeat(X[ci][None], len(A), 0), A, B, C)
            k = int(np.argmin(dd))
            if dd[k] > D_LINK:
                continue
            foot = q[k]
            tn = np.cross(B[k] - A[k], C[k] - A[k])
            tn = tn / (np.linalg.norm(tn) + 1e-12)          # the patch's own normal at the foot (diagnostic)
            area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
            if area.sum() <= 0:
                continue
            t = rng.choice(len(A), N_SAMPLE, p=area / area.sum())
            u, w = rng.random(N_SAMPLE), rng.random(N_SAMPLE)
            flip = u + w > 1
            u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
            S = A[t] + (B[t] - A[t]) * u[:, None] + (C[t] - A[t]) * w[:, None]
            r = np.linalg.norm(S - foot, axis=1)
            S = S[(r >= R_LO) & (r < R_HI)]
            if not len(S):
                continue
            samples[(int(ci), pi)] = S[:N_KEEP]
            links.append((int(ci), pi, float(dd[k]), *foot.tolist(), *tn.tolist()))
    L = np.array(links, np.float64).reshape(-1, 9)
    ci, pi = L[:, 0].astype(np.int64), L[:, 1].astype(np.int64)
    # conflict rule: a (patch, human chain) with two different wind_a among its linked clicks is dropped
    key = pi * (odd.n_chains + 1) + odd.chain[ci]
    bad = np.zeros(len(L), bool)
    for k in np.unique(key):
        m = key == k
        if len(np.unique(odd.level[ci[m]])) > 1:
            bad |= m
    L, ci, pi = L[~bad], ci[~bad], pi[~bad]
    # per replicate, each linked click -> one patch (uniform among its links) -> one unused sample point near a
    # uniform target radius (a reused point would join two candidate chains in score_chains)
    rng = np.random.default_rng(SEED + 1)
    uc = np.unique(ci)
    rows, used = [], {}
    for rep in range(K_REP):
        for c in uc:
            opts = pi[ci == c]
            p = int(opts[rng.integers(len(opts))])
            S = samples[(int(c), p)]
            foot = L[(ci == c) & (pi == p)][0, 3:6]
            r = np.linalg.norm(S - foot, axis=1)
            u = used.setdefault((int(c), p), set())
            d = np.abs(r - rng.uniform(R_LO, R_HI))
            d[list(u)] = np.inf
            j = int(np.argmin(d))
            assert np.isfinite(d[j]), (c, p)
            u.add(j)
            rows.append((int(c), rep, p, *S[j].tolist(), float(r[j])))
    R = np.array(rows, np.float64)
    assert len(cKDTree(R[:, 3:6]).query_pairs(1e-3)) == 0, "coincident candidate points"
    click, rep = R[:, 0].astype(np.int64), R[:, 1].astype(np.int64)
    cand_chain = odd.chain[click] * K_REP + rep
    lvl = odd.level[click]
    cc, cnt = np.unique(cand_chain, return_counts=True)
    keep = np.isin(cand_chain, cc[cnt >= 2])
    R, click, cand_chain, lvl = R[keep], click[keep], cand_chain[keep], lvl[keep]
    o = np.lexsort((lvl, cand_chain))
    R, click, cand_chain, lvl = R[o], click[o], cand_chain[o], lvl[o]
    clean = from_arrays(R[:, 3:6], lvl, cand_chain, names=[f"h{c // K_REP}_r{c % K_REP}" for c in np.unique(cand_chain)])
    # planted: one +-1 slip at a uniform cut in a seeded share of candidate chains
    rng = np.random.default_rng(SEED_PLANT)
    pos = np.zeros(len(lvl), np.int64)
    cut = np.full(clean.n_chains, -1, np.int64)
    plvl = lvl.copy()
    starts = np.flatnonzero(np.r_[True, clean.chain[1:] != clean.chain[:-1]])
    ends = np.r_[starts[1:], len(lvl)]
    for k, (s, e) in enumerate(zip(starts, ends)):
        pos[s:e] = np.arange(e - s)
        if rng.random() < P_PLANT:
            cut[k] = int(rng.integers(1, e - s))             # slip between positions cut-1 and cut
            plvl[s + cut[k]:e] += int(rng.choice([-1, 1]))
    planted = from_arrays(clean.xyz, plvl, clean.chain, names=clean.names)
    work.mkdir(parents=True, exist_ok=True)
    save_pcl(clean, work / "cand_clean.json")
    save_pcl(planted, work / "cand_planted.json")
    adj_linked = int(sum(1 for k in range(len(odd) - 1)
                         if odd.chain[k] == odd.chain[k + 1] and k in set(uc) and k + 1 in set(uc)))
    B = {"links_before_conflict": int(len(bad)), "links_dropped_conflict": int(bad.sum()), "links": int(len(L)),
         "heldout_clicks": int(len(odd)), "clicks_linked": int(len(uc)),
         "chains_with_linked_click": int(len(np.unique(odd.chain[uc]))),
         "adjacent_click_pairs_both_linked": adj_linked,
         "link_dist_median": float(np.median(L[:, 2])),
         "candidate_chains": int(clean.n_chains), "candidate_points": int(len(clean)),
         "planted_chains": int((cut >= 0).sum()), "radius_median": float(np.median(R[:, 6])),
         "radius_p10_p90": [float(np.percentile(R[:, 6], 10)), float(np.percentile(R[:, 6], 90))],
         "params": {"D_LINK": D_LINK, "R": [R_LO, R_HI], "K_REP": K_REP, "SEED": SEED, "P_PLANT": P_PLANT}}
    return B, {"rows": R, "pos": pos, "cut": cut, "chain": clean.chain, "level": lvl, "planted_level": plvl,
               "links": L}


# ---------------------------------------------------------------- score
def tilt_deg(old, ct):
    c = np.abs(np.einsum("ij,ij->i", old.normal, ct.normal))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


def arm(truth, cand, tilt, no_normal, planted_span=None):
    """Adjacent tests by tilt bin. planted_span(ca, cb) -> bool per test: the test spans a planted slip."""
    res, t = score_chains(truth, cand, return_tests=True)
    assert res["join"]["links"] == 0, res["join"]        # no join: candidate chains stay as built
    a, b = t["a"], t["b"]
    hc = truth.chains.chain
    ok = t["pred_dw"] == t["true_dw"]
    adj = (b == a + 1) & (hc[a] == hc[b])
    nn = no_normal[a] | no_normal[b]
    tt = np.fmax(tilt[a], tilt[b])
    span = planted_span(t["ca"], t["cb"]) if planted_span else np.zeros(len(a), bool)
    out = {"tests": res["tests"], "exact": res.get("exact"), "adjacent_tests": res["adjacent_tests"],
           "adjacent_exact": res["adjacent_exact"], "human_adjacent_covered": res["human_adjacent_covered"],
           "matched_clicks": res["matched_clicks"], "wrong_tests": int((~ok).sum())}
    sel = {"all": adj & ~nn, ">=30": adj & ~nn & (tt >= 30)}
    for lo, hi, nm in zip(BINS[:-1], BINS[1:], BIN_NAMES):
        sel[nm] = adj & ~nn & (tt >= lo) & (tt < hi)
    sel["no_ct_normal"] = adj & nn
    rng = np.random.default_rng(BOOT_SEED)
    nh = truth.chains.n_chains
    W = np.stack([np.bincount(rng.integers(0, nh, nh), minlength=nh) for _ in range(N_BOOT)]).astype(np.float64)
    by = {}
    for nm, m in sel.items():
        e = {"tests": int(m.sum())}
        if m.any():
            e["exact"] = float(ok[m].mean())
            n_c = np.bincount(hc[a[m]], minlength=nh).astype(np.float64)
            k_c = np.bincount(hc[a[m]], weights=ok[m].astype(np.float64), minlength=nh)
            with np.errstate(invalid="ignore", divide="ignore"):
                bs = (W @ k_c) / (W @ n_c)
            e["ci95"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
            if planted_span is not None:
                sp = m & span
                e["span_tests"] = int(sp.sum())
                e["detect"] = float((~ok[sp]).mean()) if sp.any() else None
                e["oracle_exact"] = float((~span[m]).mean())
                e["nonspan_exact"] = float(ok[m & ~span].mean()) if (m & ~span).any() else None
        by[nm] = e
    out["adjacent_by_tilt"] = by
    return out, t


def wrong_diag(T, odd, tilt, Z):
    """Every wrong CT adjacent test on the clean file: tilt, displacement radius of both candidate points, and the
    chain-direction match's reading of the same point pair."""
    t, s = T["ct"], T["shipped"]
    a, b = t["a"], t["b"]
    hc = build_truth(odd).chains.chain
    bad = np.flatnonzero((b == a + 1) & (hc[a] == hc[b]) & (t["pred_dw"] != t["true_dw"]))
    sh = {(int(x), int(y)): bool(p == q) for x, y, p, q in zip(s["ca"], s["cb"], s["pred_dw"], s["true_dw"])}
    rad = Z["rows"][:, 6]
    out = []
    for i in bad:
        ca, cb = int(t["ca"][i]), int(t["cb"][i])
        out.append({"human_chain": int(hc[a[i]]), "tilt": float(np.fmax(tilt[a[i]], tilt[b[i]])),
                    "radius": [float(rad[ca]), float(rad[cb])], "pred_dw": int(t["pred_dw"][i]),
                    "true_dw": int(t["true_dw"][i]), "shipped_same_pair": sh.get((ca, cb), "not matched")})
    return out


def cli_check(R, work, normals):
    """The CLI (`windingcheck score --heldout [--normals]`) gives the same headline numbers as the library."""
    out = {}
    env = dict(os.environ, PYTHONPATH=str(WC_SRC))
    for tn, extra in (("shipped", []), ("ct", ["--normals", str(normals)])):
        for cn in ("clean", "planted"):
            js = work / f"cli_{tn}_{cn}.json"
            subprocess.run([sys.executable, "-m", "windingcheck.cli", "score", str(work / f"cand_{cn}.json"),
                            "--heldout", *extra, "--json", str(js)], env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            c = json.loads(js.read_text())
            mine = R[tn][cn]
            keys = ("tests", "exact", "adjacent_tests", "adjacent_exact", "human_adjacent_covered")
            out[f"{tn}_{cn}"] = {"equal": all(c[k] == mine[k] for k in keys), "cli": {k: c[k] for k in keys}}
    out["all_equal"] = all(v["equal"] for v in out.values())
    return out


def criteria(R):
    """The pass/fail criteria of the README, in order. A bin with fewer tests than its minimum fails."""
    C = {}
    ctc, ctp = R["ct"]["clean"]["adjacent_by_tilt"], R["ct"]["planted"]["adjacent_by_tilt"]
    shc = R["shipped"]["clean"]["adjacent_by_tilt"]
    N_MIN, N_MIN_SPAN = 50, 30
    C["cli_equal"] = R["cli"]["all_equal"]
    C["tilted_tests"] = {b: ctc[b]["tests"] for b in ("30-45", "45-90", ">=30", "all")}
    C["tilted_tests_pass"] = all(ctc[b]["tests"] >= N_MIN for b in ("30-45", "45-90"))
    C["correct_claims_exact"] = {b: ctc[b].get("exact") for b in ("all", "30-45", "45-90", ">=30")}
    C["correct_claims_pass"] = C["tilted_tests_pass"] and all((ctc[b].get("exact") or 0) >= 0.95 for b in ("all", "30-45", "45-90"))
    C["planted_slips"] = {b: {"detect": ctp[b].get("detect"), "span_tests": ctp[b].get("span_tests"),
                   "measured": ctp[b].get("exact"), "oracle": ctp[b].get("oracle_exact")}
               for b in ("all", ">=30")}
    C["planted_slips_pass"] = all((ctp[b].get("span_tests") or 0) >= N_MIN_SPAN and (ctp[b].get("detect") or 0) >= 0.95
                       and abs(ctp[b]["exact"] - ctp[b]["oracle_exact"]) <= 0.03 for b in ("all", ">=30"))
    C["chain_direction_tilted_exact"] = {b: shc[b].get("exact") for b in ("30-45", "45-90")}
    C["chain_direction_below_0.90"] = any((shc[b].get("exact") is not None and shc[b]["tests"] >= N_MIN and shc[b]["exact"] < 0.90)
                              for b in ("30-45", "45-90"))
    C["verdict"] = "PASS" if (C["cli_equal"] and C["correct_claims_pass"] and C["planted_slips_pass"]) else "FAIL"
    return C


def compare(got, exp, path=""):
    """Differences between two JSON trees: integers and strings exactly, floats to 1e-9 relative."""
    if isinstance(exp, dict):
        if not isinstance(got, dict):
            return [f"{path}: not a dict"]
        out = [f"{path}/{k}: missing" for k in exp if k not in got]
        out += [f"{path}/{k}: extra" for k in got if k not in exp]
        for k in exp:
            if k in got:
                out += compare(got[k], exp[k], f"{path}/{k}")
        return out
    if isinstance(exp, list):
        if not isinstance(got, list) or len(got) != len(exp):
            return [f"{path}: length {len(got) if isinstance(got, list) else '-'} != {len(exp)}"]
        return [d for i, (g, e) in enumerate(zip(got, exp)) for d in compare(g, e, f"{path}[{i}]")]
    if isinstance(exp, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
        return [] if abs(got - exp) <= 1e-9 * max(1.0, abs(exp)) else [f"{path}: {got!r} != {exp!r}"]
    return [] if got == exp and type(got) is type(exp) else [f"{path}: {got!r} != {exp!r}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("normals", type=Path, help="output of `windingcheck normals --heldout --z-band 7000 17000`")
    ap.add_argument("--work", type=Path, default=None, help="folder for the candidate files "
                    "(default: <data root>/PHercParis4/real_sheets)")
    a = ap.parse_args()
    t0 = time.time()
    root = data.default_root() / "PHercParis4"
    work = a.work or root / "real_sheets"
    names = fetch_patches(root / "verified_patches")
    odd = heldout()
    B, Z = build(odd, root / "verified_patches", names, work)
    log(f"built {B['candidate_chains']} candidate chains, {B['candidate_points']} points ({time.time() - t0:.0f} s)")

    n_ct = for_chains(odd, a.normals)
    n_ok = int(np.isfinite(n_ct).all(1).sum())
    assert n_ok == len(odd) - N_MISSING, n_ok
    old = build_truth(odd)
    ct = build_truth(odd, normals=n_ct)
    order = np.lexsort((odd.level, odd.chain))
    no_normal = ~np.isfinite(n_ct[order]).all(1)
    tilt = tilt_deg(old, ct)
    tilt[no_normal] = np.nan
    clean, planted = load_pcl(work / "cand_clean.json"), load_pcl(work / "cand_planted.json")
    assert np.array_equal(clean.xyz, Z["rows"][:, 3:6]) and np.array_equal(clean.level, Z["level"])
    assert np.array_equal(planted.level, Z["planted_level"]) and np.array_equal(clean.chain, Z["chain"])
    pos, cut, chain = Z["pos"], Z["cut"], Z["chain"]

    def span(ca, cb):
        k = chain[ca]
        assert np.all(chain[cb] == k)
        c = cut[k]
        return (c >= 0) & ((pos[ca] < c) != (pos[cb] < c))

    R = {"build": B, "tilt_deg": {"median": float(np.nanmedian(tilt)), "p90": float(np.nanpercentile(tilt, 90))}}
    T = {}
    for tn, tr in (("shipped", old), ("ct", ct)):
        R[tn] = {}
        R[tn]["clean"], T[tn] = arm(tr, clean, tilt, no_normal)
        R[tn]["planted"], _ = arm(tr, planted, tilt, no_normal, span)
    # CT normal vs the verified patch's own normal at the linked clicks
    Lk = Z["links"]
    lc = Lk[:, 0].astype(np.int64)
    okn = np.isfinite(n_ct[lc]).all(1)
    ang = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", n_ct[lc][okn], Lk[okn, 6:9])), 0, 1)))
    R["diag_ct_vs_patch_normal_deg"] = {"links": int(okn.sum()), "median": float(np.median(ang)),
                                        "p90": float(np.percentile(ang, 90)), "share_gt_20": float(np.mean(ang > 20))}
    R["diag_wrong_ct_adjacent"] = wrong_diag(T, odd, tilt, Z)
    R["cli"] = cli_check(R, work, a.normals)
    R["criteria"] = criteria(R)
    (HERE / "results.json").write_text(json.dumps(R, indent=1) + "\n")
    log(json.dumps(R["criteria"], indent=1))
    log(f"done in {time.time() - t0:.0f} s")
    diffs = compare(R, json.loads((HERE / "expected.json").read_text()))
    if diffs:
        log(f"{len(diffs)} differences from expected.json:")
        for d in diffs[:50]:
            log("  " + d)
        sys.exit(1)
    log("0 differences from expected.json")


if __name__ == "__main__":
    main()

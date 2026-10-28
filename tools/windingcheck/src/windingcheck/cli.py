"""windingcheck command line.

    windingcheck fetch [--scroll PHercParis4] [--crossings]
    windingcheck score CANDIDATE.json [--scroll PHercParis4] [--same-winding] [--plot out.png]
    windingcheck score --crossings DIR [--scroll PHercParis4] [--heldout]
    windingcheck prior [--scroll PHercParis4]
    windingcheck normals --heldout --out normals.json      (CT sheet normal per human click, reads CT)
    windingcheck score ... --normals normals.json          (match along the CT sheet normal)
    windingcheck generate --crossings DIR --out confident.json [--every 50] [--tau T] [--model M.json]
    windingcheck generate --claims --crossings DIR --out claims.json [--every 50]     (v2, no CT)
    windingcheck generate --claims --crossings DIR --out f.json --every 50 --score --heldout [--plot p.png]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import data
from .chains import load_pcl
from .crossings import load_crossings
from .prior import Umbilicus, fit_prior
from .score import build_truth, score_chains, score_field


def _truth(args):
    d = data.fetch(args.scroll, args.data)
    human = load_pcl(d / "relative_windings.json")
    if getattr(args, "heldout", False):
        ids = np.arange(human.n_chains)
        human = human.subset(ids[ids % 2 == 1])
    nrm = None
    if getattr(args, "normals", None):
        from .normals import for_chains
        nrm = for_chains(human, args.normals)
        print(f"normals: {int(np.isfinite(nrm).all(1).sum())} of {len(nrm)} clicks have a CT normal; "
              "the rest keep the chain direction", file=sys.stderr)
    return d, build_truth(human, tol_t=args.tol_t, normals=nrm)


def main(argv=None):
    p = argparse.ArgumentParser(prog="windingcheck", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None, help="spiral_datasets root (default: <repo>/data/spiral_datasets)")
    p.add_argument("--scroll", default="PHercParis4")
    p.add_argument("--tol-t", type=float, default=25.0, help="tangential match tolerance, voxels")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--crossings", action="store_true", help="also fetch winding_inference (~170 MB)")
    s = sub.add_parser("score")
    s.add_argument("candidate", nargs="?", help="VC3D point-collection JSON")
    s.add_argument("--crossings", help="winding_inference directory instead of a JSON")
    s.add_argument("--same-winding", action="store_true", help="treat every collection as one sheet")
    s.add_argument("--plot", help="write a drift map PNG")
    s.add_argument("--json", help="write the scores here")
    s.add_argument("--no-join", action="store_true",
                   help="score every chain on its own (v0.1); default joins chains that share a point")
    s.add_argument("--join-tol", type=float, default=1e-3,
                   help="two points of different chains closer than this (voxels) are one shared point")
    s.add_argument("--heldout", action="store_true",
                   help="score against the odd-numbered human chains only (the shipped models never saw them)")
    s.add_argument("--normals", help="match along these per-click sheet normals (`windingcheck normals` output) "
                                     "instead of the human chain direction")
    sub.add_parser("prior")
    nm = sub.add_parser("normals", help="CT structure-tensor sheet normal at each human click (reads the CT)")
    nm.add_argument("--out", required=True, help="JSON to write")
    nm.add_argument("--heldout", action="store_true", help="odd-numbered human chains only (1,101 clicks)")
    nm.add_argument("--cache", default=None,
                    help="CT chunk cache (default: $WINDINGCHECK_CT_CACHE or data/ct_cache/PHercParis4_L2 in the clone)")
    nm.add_argument("--z-band", nargs=2, type=float, metavar=("Z0", "Z1"), default=None,
                    help="only clicks with Z0 <= z < Z1 get a normal (the README numbers use 7000 17000)")
    nm.add_argument("--threads", type=int, default=8, help="parallel chunk downloads")
    g = sub.add_parser("generate", help="confidence-filtered relative chains from ray crossings")
    g.add_argument("--crossings", required=True, help="winding_inference directory")
    g.add_argument("--out", required=True, help="VC3D point-collection JSON to write")
    g.add_argument("--claims", action="store_true",
                   help="v2: score whole claims (pairs of crossings) instead of single steps; no CT needed")
    g.add_argument("--model", default=None, help="model JSON (default: fitted on Paris 4)")
    g.add_argument("--tau", type=float, default=None, help="confidence threshold (default: the model's)")
    g.add_argument("--every", type=int, default=1, help="keep chains from every N-th ray (file size)")
    g.add_argument("--workers", type=int, default=12, help="threads for the neighbour search")
    g.add_argument("--score", action="store_true",
                   help="also score every generated chain (before --every) in memory: the same numbers "
                        "as `score` on the full file, without reading it back (fits in 16 GB)")
    g.add_argument("--heldout", action="store_true", help="with --score: odd-numbered human chains only")
    g.add_argument("--no-join", action="store_true", help="with --score: score every chain on its own (v0.1)")
    g.add_argument("--join-tol", type=float, default=1e-3, help="with --score: shared-point tolerance, voxels")
    g.add_argument("--plot", help="with --score: write a drift map PNG")
    g.add_argument("--json", help="with --score: write the scores here")
    g.add_argument("--normals", help="with --score: per-click sheet normals (`windingcheck normals` output)")
    args = p.parse_args(argv)

    if args.cmd == "generate":
        return _generate(args)

    if args.cmd == "normals":
        return _normals(args)
    if args.cmd == "fetch":
        print(data.fetch(args.scroll, args.data, crossings=args.crossings))
        return 0
    if args.cmd == "score":
        if args.crossings:
            cand = load_crossings(args.crossings)
        elif args.candidate:
            cand = load_pcl(args.candidate, same_winding=True if args.same_winding else None)
        else:
            p.error("give a candidate JSON or --crossings DIR")
        res = _score(args, cand)
    else:
        d, truth = _truth(args)
        umb = Umbilicus(d / "umbilicus.json")
        ids = np.arange(truth.chains.n_chains)
        res = {}
        for name, bins in (("global_spacing", 1), ("radius_binned_spacing", 10)):
            prior = fit_prior(truth, umb, ids[ids % 2 == 0], bins=bins)
            res[name] = score_field(truth, prior.delta, chain_ids=ids[ids % 2 == 1])
    out = json.dumps(res, indent=2)
    print(out)
    if getattr(args, "json", None):
        open(args.json, "w").write(out)
    return 0


def _score(args, cand):
    t0 = time.time()
    d, truth = _truth(args)
    res, tests = score_chains(truth, cand, join=not args.no_join, join_tol=args.join_tol,
                              return_tests=True)
    res["seconds"] = round(time.time() - t0, 1)
    if args.plot:
        from .plot import drift_map
        drift_map(truth, {"candidate": tests}, args.plot, umb=Umbilicus(d / "umbilicus.json"))
    return res


def _normals(args):
    from . import normals as N
    t0 = time.time()
    d = data.fetch(args.scroll, args.data)
    if args.scroll != "PHercParis4":
        raise SystemExit("normals: only PHercParis4 has a known CT frame")
    human = load_pcl(d / "relative_windings.json")
    if args.heldout:
        ids = np.arange(human.n_chains)
        human = human.subset(ids[ids % 2 == 1])
    ct = N.HttpU8Zarr(cache=args.cache, threads=args.threads)
    ct.check_meta()
    xyz = human.xyz
    inb = np.ones(len(xyz), bool)
    if args.z_band:
        inb = (xyz[:, 2] >= args.z_band[0]) & (xyz[:, 2] < args.z_band[1])
    chunks = ct.prefetch([N.click_box(p, N.HALF + 1) for p in xyz[inb]])
    n = np.full((len(xyz), 3), np.nan)
    coh, zs = np.full(len(xyz), np.nan), np.full(len(xyz), np.nan)
    n[inb], coh[inb], zs[inb] = N.ct_normals(xyz[inb], ct.box)
    meta = {"volume": N.PARIS4_L2_URL, "recipe": {"sigma_gradient": N.SIG_G, "sigma_tensor": N.SIG_I,
            "box_half": N.HALF, "zero_share_max": N.ZERO_SHARE}, "heldout": bool(args.heldout),
            "z_band": args.z_band}
    N.save(args.out, xyz, n, coh, zs, meta)
    print(json.dumps({"out": args.out, "clicks": int(len(xyz)),
                      "with_normal": int(np.isfinite(n).all(1).sum()), "chunks": chunks,
                      "seconds": round(time.time() - t0, 1)}))
    return 0


def _generate(args):
    from importlib.resources import files
    from scipy.spatial import cKDTree
    from .chains import save_pcl
    from .confidence import StepModel, generate, step_features, step_index
    from .crossings import load_ray_store

    t0 = time.time()
    models = files("windingcheck") / "models"
    default = "paris4_claims_v2_noct.json" if args.claims else "paris4_even_chains.json"
    meta = json.loads(open(args.model or models / default).read())
    model = StepModel.from_json(meta)
    tau = args.tau if args.tau is not None else meta["tau"]
    d = data.fetch(args.scroll, args.data)
    store = load_ray_store(args.crossings)
    umb = Umbilicus(d / "umbilicus.json")
    X, _ = step_features(store, umb, step_index(store), tree=cKDTree(store.xyz), workers=args.workers)
    if args.claims:
        from .claims import StepTable, generate_blocked
        step_model = StepModel.from_json(json.loads(open(models / meta["step_model"]).read()))
        tab = StepTable(store, umb, step_model, X=X)
        del X
        gen, conf = generate_blocked(store, tab, model, tau)
        del tab
        source = "windingcheck-claims-v2"
    else:
        gen, conf = generate(store, model.prob(X), tau)
        source = "windingcheck-confident"
    ray = np.array([int(n.split("@")[0][3:]) for n in gen.names])
    sel = np.flatnonzero(ray % args.every == 0)
    out = gen.subset(sel)
    save_pcl(out, args.out, color=(0.16, 0.47, 0.84),
             tags=[{"source": source, "confidence": f"{conf[i]:.4f}", "tau": f"{tau}"}
                   for i in sel])
    summary = {"out": args.out, "tau": tau, "chains": int(len(sel)), "points": len(out),
               "chains_generated": int(gen.n_chains), "points_generated": len(gen), "crossings_in": len(store),
               "seconds": round(time.time() - t0, 1)}
    if args.score:
        del store, out
        res = _score(args, gen)
        summary["score"] = res
        if args.json:
            open(args.json, "w").write(json.dumps(res, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CT sheet normals at the human clicks, for the sheet-normal match (``build_truth(normals=...)``).

The scorer matches a candidate point to a human click along a normal. By default that is the human
chain's direction; on a sheet tilted against the chain this lets a point of the neighbouring sheet
match (see Limits in the README). ``windingcheck normals`` measures the sheet normal in the CT
instead: the structure tensor of the raw CT around each click, the eigenvector of the largest
eigenvalue.

The CT is read at run time from the public PHerc. Paris 4 2.4 um scan (level 2, 9.6 um voxels, the
frame of the human annotations): an uncompressed u8 zarr v2 with 128^3 chunks over HTTPS, so no
extra package is needed. Each chunk is 2 MiB and is kept in a local cache; nothing from the CT is
shipped and the output holds unit vectors and two numbers per click, no CT values.

The recipe (fixed before any score under it existed): gradient sigma 1.0 vx (Gaussian
derivative), tensor smoothing sigma 3.0 vx, box +-18 vx around the click, trilinear at the click.
A click whose box is more than 20 % raw zeros (outside the scroll mask) gets no normal (NaN row) and
keeps the chain direction.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PARIS4_L2_URL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/"
                 "20260411134726-2.400um-0.2m-78keV-masked.zarr/2")
PARIS4_L2_SHAPE = (18946, 8174, 8174)          # zyx
CH = 128
SIG_G = 1.0
SIG_I = 3.0
HALF = 18                                      # >= 4 sig_i + 4 sig_g + 2
ZERO_SHARE = 0.2
FETCH_TRIES = 24


class HttpU8Zarr:
    """One level of an uncompressed u8 zarr v2 (128^3 chunks) over HTTPS, with a local chunk cache.
    Missing chunks (403/404) read as zeros, as do reads outside the volume."""

    def __init__(self, url=PARIS4_L2_URL, shape=PARIS4_L2_SHAPE, cache=None, threads=8):
        self.url, self.shape, self.threads = url.rstrip("/"), tuple(shape), threads
        if cache is None:
            from .data import default_root
            cache = os.environ.get("WINDINGCHECK_CT_CACHE") or default_root().parent / "ct_cache" / "PHercParis4_L2"
        self.cache = Path(cache)

    def check_meta(self):
        with urllib.request.urlopen(self.url + "/.zarray", timeout=60) as r:
            m = json.loads(r.read())
        if (m.get("dtype") != "|u1" or m.get("compressor") is not None or m.get("chunks") != [CH] * 3
                or tuple(m.get("shape", ())) != self.shape or m.get("filters")):
            raise RuntimeError(f"unexpected zarr layout at {self.url}: {m}")

    def _file(self, key):
        return self.cache / "/".join(map(str, key))

    def _fetch(self, key):
        f = self._file(key)
        if f.exists() or f.with_suffix(".absent").exists():
            return
        f.parent.mkdir(parents=True, exist_ok=True)
        err = None
        for attempt in range(FETCH_TRIES):
            try:
                with urllib.request.urlopen(self.url + "/" + "/".join(map(str, key)), timeout=60) as r:
                    b = r.read()
                if len(b) != CH ** 3:
                    raise IOError(f"chunk {key}: {len(b)} bytes")
                tmp = f.with_suffix(".part")
                tmp.write_bytes(b)
                tmp.rename(f)
                return
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    f.with_suffix(".absent").touch()
                    return
                err = e
            except Exception as e:  # noqa: BLE001
                err = e
            time.sleep(min(60, 2 ** (attempt + 1)) * random.uniform(0.75, 1.25))
        raise RuntimeError(f"fetch failed {key}: {err}")

    def _chunk(self, key):
        f = self._file(key)
        if f.exists():
            return np.fromfile(f, np.uint8).reshape(CH, CH, CH)
        return np.zeros((CH, CH, CH), np.uint8)

    def _keys(self, lo, hi):
        k0 = np.maximum(lo, 0) // CH
        k1 = (np.minimum(hi, self.shape) - 1) // CH + 1
        return [(a, b, c) for a in range(k0[0], k1[0]) for b in range(k0[1], k1[1])
                for c in range(k0[2], k1[2])]

    def prefetch(self, boxes):
        """Fetch every chunk of the given zyx boxes [(lo, hi), ...]; returns the number of chunks."""
        keys = sorted({k for lo, hi in boxes for k in self._keys(np.asarray(lo), np.asarray(hi))})
        with ThreadPoolExecutor(self.threads) as ex:
            list(ex.map(self._fetch, keys))
        return len(keys)

    def box(self, lo, hi):
        """Integer zyx box [lo, hi)."""
        lo, hi = np.asarray(lo, int), np.asarray(hi, int)
        keys = self._keys(lo, hi)
        for k in keys:
            self._fetch(k)
        out = np.zeros(tuple(hi - lo), np.uint8)
        for k in keys:
            o = np.array(k) * CH
            a = np.maximum(lo, o)
            b = np.minimum(hi, o + CH)
            if np.any(b <= a):
                continue
            out[tuple(slice(x - y, z - y) for x, y, z in zip(a, lo, b))] = \
                self._chunk(k)[tuple(slice(x - y, z - y) for x, y, z in zip(a, o, b))]
        return out


def click_box(p, half=HALF):
    """zyx box [lo, hi) around the click p (xyz, voxels)."""
    c = np.floor(p).astype(int)
    return (c - half)[::-1], (c + half + 1)[::-1]


def st_normal(raw, p_in_box, sig_g=SIG_G, sig_i=SIG_I):
    """Structure-tensor normal of a u8 zyx box at a point (xyz, box voxels). Returns (n_xyz, coherence)."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    b = raw.astype(np.float32)
    g = [gaussian_filter(b, sig_g, order=tuple(int(i == ax) for i in range(3))) for ax in range(3)]  # d/dz,y,x
    g = g[::-1]                                                                                       # x, y, z
    q = np.asarray(p_in_box, float)[::-1].reshape(3, 1)
    J = np.empty((3, 3))
    for i in range(3):
        for j in range(i, 3):
            J[i, j] = J[j, i] = map_coordinates(gaussian_filter(g[i] * g[j], sig_i), q, order=1)[0]
    w, v = np.linalg.eigh(J)
    coh = float((w[2] - w[1]) / (w[2] + w[1])) if w[2] + w[1] > 0 else 0.0
    return v[:, 2], coh


def ct_normals(xyz, box_fn, half=HALF, zero_share=ZERO_SHARE):
    """Normals at clicks xyz (N, 3). box_fn(lo_zyx, hi_zyx) -> u8 array. Returns (n, coherence,
    zero_share), n NaN where the box has more than ``zero_share`` raw zeros."""
    xyz = np.asarray(xyz, np.float64)
    n = np.full((len(xyz), 3), np.nan)
    coh = np.full(len(xyz), np.nan)
    zs = np.full(len(xyz), np.nan)
    for i, p in enumerate(xyz):
        lo, hi = click_box(p, half)
        raw = box_fn(lo, hi)
        zs[i] = float(np.mean(raw == 0))
        if zs[i] > zero_share:
            continue
        n[i], coh[i] = st_normal(raw, p - lo[::-1])
    return n, coh, zs


def save(path, xyz, n, coh, zs, meta=None):
    """JSON: one row per click; floats as repr, so a reload is bit-exact; NaN as null."""
    def f(a):
        return [None if not np.isfinite(x) else float(x) for x in a]
    rows = [{"xyz": [float(v) for v in p], "n": None if not np.isfinite(q).all() else f(q),
             "coherence": f([c])[0], "zero_share": f([z])[0]} for p, q, c, z in zip(xyz, n, coh, zs)]
    Path(path).write_text(json.dumps({"meta": meta or {}, "clicks": rows}))


def load(path):
    """(xyz, n) arrays from a ``save`` file; n NaN where the click has no normal."""
    rows = json.loads(Path(path).read_text())["clicks"]
    xyz = np.array([r["xyz"] for r in rows], np.float64).reshape(-1, 3)
    n = np.array([r["n"] if r["n"] is not None else [np.nan] * 3 for r in rows], np.float64).reshape(-1, 3)
    return xyz, n


def for_chains(chains, path, tol=1e-6):
    """One normal per click of ``chains`` (input row order), by exact position (distance < tol voxels)
    in a ``save`` file; NaN rows where the file has no normal for the click."""
    from scipy.spatial import cKDTree
    X, N = load(path)
    out = np.full((len(chains.xyz), 3), np.nan)
    if not len(X):
        return out
    dd, ii = cKDTree(X).query(chains.xyz)
    hit = dd < tol
    out[hit] = N[ii[hit]]
    return out

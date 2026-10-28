import numpy as np

from windingcheck import normals as N
from windingcheck.chains import from_arrays
from windingcheck.score import build_truth, score_chains

PITCH = 12.0


def _sheets(normal, shape=(64, 96, 96)):
    """u8 zyx volume of parallel bright sheets PITCH voxels apart along ``normal`` (xyz)."""
    z, y, x = np.indices(shape, dtype=np.float64)
    s = x * normal[0] + y * normal[1] + z * normal[2]
    d = (s + PITCH / 2) % PITCH - PITCH / 2
    return (40 + 180 * np.exp(-0.5 * (d / 1.5) ** 2)).astype(np.uint8)


def _box_fn(vol):
    def f(lo, hi):
        out = np.zeros(tuple(np.asarray(hi) - np.asarray(lo)), np.uint8)
        a, b = np.maximum(lo, 0), np.minimum(hi, vol.shape)
        out[tuple(slice(p - q, r - q) for p, q, r in zip(a, lo, b))] = vol[tuple(slice(p, r) for p, r in zip(a, b))]
        return out
    return f


def test_structure_tensor_finds_tilted_sheet_normal():
    t = np.deg2rad(40.0)
    true = np.array([np.cos(t), np.sin(t), 0.0])
    vol = _sheets(true)
    pts = np.array([[48.3, 40.7, 31.2], [30.0, 55.5, 25.0], [60.9, 47.1, 36.4]])
    n, coh, zs = N.ct_normals(pts, _box_fn(vol))
    ang = np.degrees(np.arccos(np.clip(np.abs(n @ true), 0, 1)))
    assert np.all(ang < 2.0), ang
    assert np.all(coh > 0.5) and np.all(zs == 0)


def test_mostly_empty_box_gets_no_normal():
    vol = np.zeros((64, 64, 64), np.uint8)
    vol[:, :, :8] = 100
    n, coh, zs = N.ct_normals(np.array([[32.0, 32.0, 32.0]]), _box_fn(vol))
    assert not np.isfinite(n).any() and zs[0] > N.ZERO_SHARE


def test_ct_normals_fix_tilted_match_end_to_end():
    """Sheets at x = PITCH * k, human chains tilted 40 deg in the xy plane, every candidate claim correct:
    the chain-direction match reads low, the structure-tensor normal from the CT reads exact."""
    a = np.deg2rad(40.0)
    vol = _sheets(np.array([1.0, 0, 0]), shape=(48, 480, 120))
    xyz, lvl, ch = [], [], []
    for c in range(4):
        for k in range(1, 8):
            xyz.append([PITCH * k + 0.3, 40.0 + 100.0 * c + PITCH * k * np.tan(a), 24.0])
            lvl.append(k)
            ch.append(c)
    human = from_arrays(xyz, lvl, ch)
    cx, cl, cc = [], [], []
    for r, y in enumerate(np.arange(20.0, 470.0, 2.0)):
        for k in range(1, 8):
            cx.append([PITCH * k + 0.3, y, 24.0])
            cl.append(k)
            cc.append(r)
    cand = from_arrays(cx, cl, cc)
    n, _, _ = N.ct_normals(human.xyz, _box_fn(vol))
    assert np.isfinite(n).all()
    old = score_chains(build_truth(human), cand)
    new = score_chains(build_truth(human, normals=n), cand)
    assert old["adjacent_exact"] < 0.9
    assert new["adjacent_exact"] == 1.0 and new["exact"] == 1.0


def test_save_load_round_trip_is_exact_and_maps_by_position(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.random((20, 3)) * 1000
    n = rng.normal(size=(20, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n[5] = np.nan
    coh, zs = rng.random(20), rng.random(20) * 0.1
    coh[5], zs[5] = np.nan, 0.5
    p = tmp_path / "normals.json"
    N.save(p, xyz, n, coh, zs, {"test": True})
    X, M = N.load(p)
    assert np.array_equal(X, xyz) and np.array_equal(M, n, equal_nan=True)
    perm = rng.permutation(20)[:15]
    extra = np.array([[-5.0, -5.0, -5.0]])
    chains = from_arrays(np.r_[xyz[perm], extra], np.arange(16), np.zeros(16, int))
    got = N.for_chains(chains, p)
    assert np.array_equal(got[:15], n[perm], equal_nan=True)
    assert not np.isfinite(got[15]).any()


def test_http_reader_assembles_boxes_from_cached_chunks(tmp_path):
    shape = (300, 200, 260)
    vol = (np.arange(np.prod(shape), dtype=np.int64) % 251).astype(np.uint8).reshape(shape)
    for kz in range(3):
        for ky in range(2):
            for kx in range(3):
                if (kz, ky, kx) == (1, 1, 0):
                    (tmp_path / "1" / "1").mkdir(parents=True, exist_ok=True)
                    (tmp_path / "1" / "1" / "0.absent").touch()      # a chunk the server does not have
                    continue
                c = np.zeros((N.CH,) * 3, np.uint8)
                part = vol[kz * N.CH:(kz + 1) * N.CH, ky * N.CH:(ky + 1) * N.CH, kx * N.CH:(kx + 1) * N.CH]
                c[tuple(slice(0, s) for s in part.shape)] = part
                f = tmp_path / str(kz) / str(ky) / str(kx)
                f.parent.mkdir(parents=True, exist_ok=True)
                c.tofile(f)
    ct = N.HttpU8Zarr(url="http://invalid.example", shape=shape, cache=tmp_path)
    want = np.zeros((60, 60, 60), np.uint8)
    lo = np.array([100, 110, -10])
    want[:, :, 10:] = vol[100:160, 110:170, 0:50]
    want[28:, 18:, 10:] = 0                                                         # the absent chunk
    got = ct.box(lo, lo + 60)
    assert np.array_equal(got[:, :, :10], np.zeros((60, 60, 10), np.uint8))     # outside the volume
    assert np.array_equal(got, want)

import numpy as np

from windingcheck.claims import N_NO_CT, StepTable, all_ray_claims, claim_features, generate_blocked, generate_claims
from windingcheck.confidence import FEATURES, StepModel
from windingcheck.crossings import RayStore


class _Umb:
    def polar(self, xyz):
        return np.hypot(xyz[:, 0], xyz[:, 1]), np.arctan2(xyz[:, 1], xyz[:, 0])


def _store(n_rays=7, seed=0):
    rng = np.random.default_rng(seed)
    xyz, level, t, ray = [], [], [], []
    for r in range(n_rays):
        n = rng.integers(1, 9)
        steps = rng.uniform(5, 15, n)
        for k in range(n):
            xyz.append([200 + steps[:k + 1].sum(), 30.0 * r, 0])
            level.append(k)
            t.append(10.0 * k)
            ray.append(r)
    dirs = np.tile([1.0, 0, 0], (n_rays, 1))
    return RayStore(np.array(xyz, float), np.array(level), np.array(t, np.float32), np.array(ray), dirs, 385)


def _models(rng):
    step = StepModel(np.zeros(len(FEATURES)), np.ones(len(FEATURES)), rng.normal(size=len(FEATURES) + 1))
    claim = StepModel(np.zeros(N_NO_CT), np.ones(N_NO_CT), rng.normal(size=N_NO_CT + 1))
    return step, claim


def test_blocked_generator_matches_all_claims():
    st = _store()
    rng = np.random.default_rng(1)
    step, claim = _models(rng)
    tab = StepTable(st, _Umb(), step)
    A, B, starts, n = all_ray_claims(st)
    p = claim.prob(claim_features(tab, A, B))
    for tau in (0.2, 0.5, 0.8):
        ref, rc = generate_claims(st, A, B, p, starts, n, tau)
        got, gc = generate_blocked(st, tab, claim, tau, block_rays=2)
        assert np.array_equal(ref.xyz, got.xyz) and np.array_equal(ref.chain, got.chain)
        assert ref.names == got.names and np.allclose(rc, gc)


def test_no_ct_model_ignores_ct_columns():
    st = _store()
    rng = np.random.default_rng(2)
    step, claim = _models(rng)
    plain = StepTable(st, _Umb(), step)
    with_ct = StepTable(st, _Umb(), step, rng.uniform(50, 150, len(st)), rng.uniform(20, 80, len(st)))
    a, b = generate_blocked(st, plain, claim, 0.5), generate_blocked(st, with_ct, claim, 0.5)
    assert np.array_equal(a[0].xyz, b[0].xyz) and np.allclose(a[1], b[1])

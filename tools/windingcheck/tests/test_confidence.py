import numpy as np

from windingcheck.confidence import StepModel, generate, pair_steps, step_index
from windingcheck.crossings import RayStore


def _store():
    # two rays of 5 crossings along x, 10 voxels apart
    xyz, level, t, ray = [], [], [], []
    for r in range(2):
        for k in range(5):
            xyz.append([100 + 10 * k, 50 * r, 0])
            level.append(k)
            t.append(10.0 * k)
            ray.append(r)
    return RayStore(np.array(xyz, float), np.array(level), np.array(t, np.float32), np.array(ray),
                    np.array([[1.0, 0, 0], [1.0, 0, 0]]), 385)


def test_step_index_skips_ray_ends():
    assert step_index(_store()).tolist() == [0, 1, 2, 3, 5, 6, 7, 8]


def test_pair_steps():
    owner, k = pair_steps(np.array([0, 7]), np.array([2, 5]))
    assert owner.tolist() == [0, 0, 1, 1] and k.tolist() == [0, 1, 5, 6]


def test_generate_cuts_at_low_steps():
    st = _store()
    p = np.array([0.9, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    ch, conf = generate(st, p, 0.5)
    # ray 0 splits into [0,1] and [2,3,4]; ray 1 stays whole
    assert ch.n_chains == 3
    assert np.allclose(sorted(conf), sorted([0.9, 0.81, 0.9 ** 4]))


def test_model_learns_a_separable_feature():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 7))
    y = X[:, 2] > 0
    m = StepModel.fit(X, y)
    assert np.mean((m.prob(X) > 0.5) == y) > 0.95

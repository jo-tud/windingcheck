import numpy as np

from windingcheck.chains import from_arrays, save_pcl, load_pcl
from windingcheck.score import build_truth, score_chains


def _spiral_chains(n_chains=6, n=10, gap=20.0):
    xyz, lvl, ch = [], [], []
    for c in range(n_chains):
        ang = 2 * np.pi * c / n_chains
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        for k in range(n):
            xyz.append([500, 500, 100 * c] + (200 + gap * k) * d)
            lvl.append(k + 1)
            ch.append(c)
    return from_arrays(xyz, lvl, ch)


def test_self_score_is_perfect():
    t = build_truth(_spiral_chains())
    r = score_chains(t, t.chains)
    assert r["exact"] == 1.0 and r["human_pairs_covered"] == 1.0


def test_missed_sheet_is_caught():
    c = _spiral_chains()
    t = build_truth(c)
    lv = c.level.copy()
    lv[c.level > 5] += 1
    r = score_chains(t, from_arrays(c.xyz, lv, c.chain))
    # pairs straddling the break (a <= 5 < b): 5 * 5 of 45 per chain
    assert abs(r["exact"] - 20 / 45) < 1e-9 and r["sign_errors"] == 0


def test_pcl_roundtrip(tmp_path):
    c = _spiral_chains()
    save_pcl(c, tmp_path / "w.json")
    back = load_pcl(tmp_path / "w.json")
    assert np.allclose(back.xyz, c.xyz) and np.allclose(back.level, c.level)


def test_save_pcl_streams_the_same_bytes(tmp_path):
    import json
    from windingcheck.chains import Chains, load_pcl, save_pcl
    rng = np.random.default_rng(0)
    ch = np.array([1, 0, 1, 2, 0, 2, 2])
    c = Chains(rng.normal(size=(7, 3)) * 100, np.arange(7.0), ch, ["a", "b", "c"])
    tags = [{"confidence": f"{i / 10}"} for i in range(3)]
    save_pcl(c, tmp_path / "s.json", tags=tags)
    # the earlier in-memory writer, kept here as the reference format
    cols, pid = {}, 1
    for k, name in enumerate(c.names):
        m = c.chain == k
        pts = {}
        for p, lv in zip(c.xyz[m], c.level[m]):
            pts[str(pid)] = {"p": [float(v) for v in p], "creation_time": 0, "wind_a": float(lv)}
            pid += 1
        cols[str(k + 1)] = {"name": name, "color": [0.9, 0.5, 0.1],
                            "metadata": {"winding_is_absolute": False}, "points": pts, "tags": tags[k]}
    ref = json.dumps({"vc_pointcollections_json_version": "1", "collections": cols})
    assert (tmp_path / "s.json").read_text() == ref
    back = load_pcl(tmp_path / "s.json")
    assert len(back) == 7 and back.n_chains == 3


def test_adjacent_exact_counts_neighbouring_clicks_only():
    c = _spiral_chains()
    t = build_truth(c)
    lv = c.level.copy()
    lv[c.level > 5] += 1
    r = score_chains(t, from_arrays(c.xyz, lv, c.chain))
    # 9 adjacent pairs per chain, one of them (5 -> 6) straddles the break
    assert r["adjacent_tests"] == 6 * 9 and abs(r["adjacent_exact"] - 8 / 9) < 1e-9


def _with_spurious_crossing(c, chain=0, after_level=5):
    """Insert a false crossing mid-gap on one chain: every later level of that chain shifts by 1."""
    m = (c.chain == chain) & (c.level == after_level)
    nxt = (c.chain == chain) & (c.level == after_level + 1)
    mid = (c.xyz[m] + c.xyz[nxt]) / 2
    lv = c.level.copy()
    lv[(c.chain == chain) & (c.level > after_level)] += 1
    return from_arrays(np.r_[c.xyz, mid], np.r_[lv, after_level + 1], np.r_[c.chain, chain])


def _split_into_steps(c):
    """Every step of every chain as its own 2-point chain; neighbouring steps share a point."""
    c = c.sorted()
    same = c.chain[1:] == c.chain[:-1]
    i = np.flatnonzero(same)
    idx = np.stack([i, i + 1], 1).ravel()
    return from_arrays(c.xyz[idx], c.level[idx], np.repeat(np.arange(len(i)), 2))


def test_splitting_into_steps_cannot_raise_the_score():
    c = _spiral_chains()
    t = build_truth(c)
    cand = _with_spurious_crossing(_with_spurious_crossing(c, 0, 5), 3, 2)
    whole = score_chains(t, cand)
    steps = _split_into_steps(cand)
    split = score_chains(t, steps)
    assert whole["exact"] < 1.0
    # joined, the steps are the same constraint as the whole chains: the same tests, the same score
    for k in ("tests", "exact", "within_1", "sign_errors", "adjacent_tests", "adjacent_exact",
              "human_pairs_covered", "human_adjacent_covered"):
        assert split[k] == whole[k], k
    assert split["join"]["chains_out"] == cand.n_chains and split["join"]["conflict_chains"] == 0
    # unjoined (v0.1), the false step is invisible: the step into the mid-gap point matches no click
    v01 = score_chains(t, steps, join=False)
    assert v01["exact"] == 1.0 > whole["exact"]


def test_split_with_shuffled_levels_and_order_joins_back():
    # pieces with arbitrary level offsets and in random order still join to the original chain
    c = _spiral_chains()
    t = build_truth(c)
    cand = _with_spurious_crossing(c, 2, 4)
    steps = _split_into_steps(cand)
    rng = np.random.default_rng(1)
    lv = steps.level + rng.integers(-50, 50, steps.n_chains)[steps.chain]
    perm = rng.permutation(len(steps))
    shuffled = from_arrays(steps.xyz[perm], lv[perm], steps.chain[perm])
    a, b = score_chains(t, cand), score_chains(t, shuffled)
    assert (a["tests"], a["exact"]) == (b["tests"], b["exact"])


def test_conflicting_pieces_are_not_joined():
    # two pieces that share both end points but disagree on the step count between them
    xyz = np.array([[0, 0, 0], [10, 0, 0], [0, 0, 0], [5, 0, 0], [10, 0, 0]], float)
    cand = from_arrays(xyz, [0, 1, 0, 1, 2], [0, 0, 1, 1, 1])
    from windingcheck.score import join_shared
    out, src, info = join_shared(cand)
    assert info["conflict_chains"] == 2 and out.n_chains == 2 and len(out) == 5
    assert np.array_equal(src, np.arange(5))


def test_join_keeps_test_indices_on_the_input():
    c = _spiral_chains()
    t = build_truth(c)
    steps = _split_into_steps(c)
    _, tests = score_chains(t, steps, return_tests=True)
    # the pieces keep their original levels, so the indices into the input give the same Δw
    assert np.array_equal(steps.level[tests["cb"]] - steps.level[tests["ca"]], tests["pred_dw"])
    assert np.allclose(steps.xyz[tests["ca"]], t.chains.xyz[tests["a"]])


def test_in_memory_score_equals_file_score(tmp_path):
    # generate --score scores the chains before writing; the numbers must equal `score` on the file
    c = _spiral_chains()
    t = build_truth(c)
    rng = np.random.default_rng(1)
    lv = c.level + (rng.random(len(c)) < 0.2)
    cand = from_arrays(c.xyz + rng.normal(scale=0.3, size=c.xyz.shape), lv, c.chain)
    save_pcl(cand, tmp_path / "g.json")
    assert score_chains(t, cand) == score_chains(t, load_pcl(tmp_path / "g.json"))

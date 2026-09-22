"""Solver correctness tests, including brute-force cross-checks."""

from __future__ import annotations

import random

import pytest

from app.solver import solve_mask_assignment

_SEED_750_EXPECTED = [
    0, 0, 0, 1, 0, 2, 0, 1, 1, 1, 0, 0, 2, 0, 1, 1, 0, 1, 0, 2, 2, 1,
    0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2, 0, 2, 0, 0, 1, 2, 1, 2, 2,
    2, 1, 1, 0,
]


def _seed_750_instance():
    """48 fragments, 88 conflicts from seed 750 at p=0.075."""
    rng = random.Random(750)
    fragments = list(range(48))
    conflicts = []
    for a in range(48):
        for b in range(a + 1, 48):
            if rng.random() < 0.075:
                conflicts.append([a, b])
    assert len(conflicts) == 88
    return fragments, conflicts


def brute_force(fragments, conflict_edges, stitch_edges):
    """Enumerate all canonical colorings; returns (best_cost, solutions)."""
    order = sorted(fragments)
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    conf = [[] for _ in range(n)]
    for a, b in conflict_edges:
        conf[pos[a]].append(pos[b])
        conf[pos[b]].append(pos[a])
    st = [[] for _ in range(n)]
    for a, b, w in stitch_edges:
        st[pos[a]].append((pos[b], w))
        st[pos[b]].append((pos[a], w))

    best = None
    sols = []
    colors = [0] * n

    def rec(i, max_color, cost):
        nonlocal best, sols
        if best is not None and cost > best:
            return
        if i == n:
            if best is None or cost < best:
                best = cost
                sols = [tuple(colors)]
            elif cost == best:
                sols.append(tuple(colors))
            return
        for k in range(min(max_color + 1, 2) + 1):  # canonical growth only
            if any(colors[j] == k for j in conf[i] if j < i):
                continue
            extra = sum(w for j, w in st[i] if j < i and colors[j] != k)
            colors[i] = k
            rec(i + 1, max(max_color, k), cost + extra)

    rec(0, -1, 0)
    return best, sols


def check_against_brute_force(fragments, conflicts, stitches):
    result = solve_mask_assignment(fragments, conflicts, stitches)
    best, sols = brute_force(fragments, conflicts, stitches)
    order = sorted(fragments)

    if best is None:
        assert result["status"] == "infeasible"
        return

    assert result["status"] == "optimal"
    assert result["objective"] == best

    seq = tuple(result["assignment"][str(f)] for f in order)
    assert seq == min(sols), "not the lexicographically smallest canonical optimum"
    assert result["unique"] == (len(sols) == 1)

    pos = {f: i for i, f in enumerate(order)}
    expected_cut = sorted(
        (tuple(sorted((a, b))), w) for a, b, w in stitches if seq[pos[a]] != seq[pos[b]]
    )
    got_cut = sorted(
        (tuple(sorted(edge["pair"])), edge["weight"]) for edge in result["cut_stitches"]
    )
    assert got_cut == expected_cut
    assert sum(w for _pair, w in got_cut) == best

    if len(sols) > 1:
        witness = result["witness"]
        assert witness is not None
        wseq = tuple(witness["assignment"][str(f)] for f in order)
        assert wseq in sols
        assert wseq != seq
    else:
        assert result["witness"] is None


def test_unique_optimum_with_weighted_stitches():
    fragments = [1, 2, 3, 4]
    conflicts = [[1, 2], [2, 3], [1, 3]]
    stitches = [[3, 4, 5], [1, 4, 1]]
    result = solve_mask_assignment(fragments, conflicts, stitches)
    assert result["status"] == "optimal"
    assert result["objective"] == 1
    assert result["unique"] is True
    assert result["assignment"] == {"1": 0, "2": 1, "3": 2, "4": 2}
    assert result["cut_stitches"] == [{"pair": [1, 4], "weight": 1}]
    assert result["witness"] is None


def test_multiple_optima_returns_lexicographically_smallest_and_witness():
    result = solve_mask_assignment([1, 2, 3, 4], [], [])
    assert result["status"] == "optimal"
    assert result["objective"] == 0
    assert result["unique"] is False
    assert result["assignment"] == {"1": 0, "2": 0, "3": 0, "4": 0}
    witness = result["witness"]
    assert witness is not None
    assert witness["assignment"] != result["assignment"]
    # The witness must itself be canonical (first occurrences 0,1,2 in order).
    seq = [witness["assignment"][str(f)] for f in (1, 2, 3, 4)]
    next_color = 0
    for color in seq:
        assert color <= next_color
        next_color = max(next_color, color + 1)


def test_infeasible_k4():
    conflicts = [[a, b] for a in range(1, 5) for b in range(a + 1, 5)]
    result = solve_mask_assignment([1, 2, 3, 4], conflicts, [])
    assert result["status"] == "infeasible"


def test_non_contiguous_fragment_ids_are_normalized_by_ascending_id():
    fragments = [40, 7, 13, 21]
    conflicts = [[7, 13], [13, 21]]
    stitches = [[7, 40, 2]]
    check_against_brute_force(fragments, conflicts, stitches)


@pytest.mark.parametrize("seed", range(40))
def test_random_instances_match_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randint(4, 8)
    fragments = sorted(rng.sample(range(1, 60), n))
    pairs = [
        (fragments[i], fragments[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    rng.shuffle(pairs)
    conflicts, stitches = [], []
    for a, b in pairs:
        roll = rng.random()
        if roll < 0.25:
            conflicts.append([a, b])
        elif roll < 0.5:
            stitches.append([a, b, rng.randint(1, 9)])
    check_against_brute_force(fragments, conflicts, stitches)


@pytest.mark.parametrize("seed", range(20))
def test_random_instances_with_huge_weights_match_brute_force(seed):
    # Weights beyond the exact-MILP domain force the pure integer engine;
    # every legal positive integer weight must compare exactly.
    rng = random.Random(2000 + seed)
    n = rng.randint(4, 9)
    fragments = sorted(rng.sample(range(1, 60), n))
    pairs = [
        (fragments[i], fragments[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    rng.shuffle(pairs)
    conflicts, stitches = [], []
    for a, b in pairs:
        roll = rng.random()
        if roll < 0.25:
            conflicts.append([a, b])
        elif roll < 0.5:
            stitches.append([a, b, rng.choice([10**9 + 1, 10**12 + 7, 10**30 + 3])])
    check_against_brute_force(fragments, conflicts, stitches)


def test_seed750_huge_weights_return_zero_optimum():
    # Regression: legal large integer weights pushed the instance onto the
    # exact search, whose fixed id-order enumeration exhausted its budget and
    # surfaced as HTTP 500.  Both stitch endpoints can keep the same mask, so
    # the canonical optimum is 0 regardless of the weight magnitude.
    fragments, conflicts = _seed_750_instance()
    stitches = [[0, 1, 1_000_000_001], [27, 30, 1_000_000_002]]
    assert all(tuple(sorted((a, b))) != (0, 1) for a, b in conflicts)
    assert all(tuple(sorted((a, b))) != (27, 30) for a, b in conflicts)

    result = solve_mask_assignment(fragments, conflicts, stitches)
    assert result["status"] == "optimal"
    assert result["objective"] == 0
    assert result["unique"] is False
    assert result["cut_stitches"] == []
    seq = [result["assignment"][str(i)] for i in fragments]
    assert seq == _SEED_750_EXPECTED
    # A second, equally optimal canonical witness must be returned.
    witness = result["witness"]
    assert witness is not None
    wseq = [witness["assignment"][str(i)] for i in fragments]
    assert wseq != seq
    assert witness["cut_stitches"] == []
    assert all(wseq[a] != wseq[b] for a, b in conflicts)
    assert wseq[0] == wseq[1] and wseq[27] == wseq[30]


def test_seed750_small_weight_control_matches_huge_weights():
    # Same fragments and conflict structure, small weights: the answer must
    # be identical (zero is attainable either way).
    fragments, conflicts = _seed_750_instance()
    small = solve_mask_assignment(fragments, conflicts, [[0, 1, 1], [27, 30, 2]])
    huge = solve_mask_assignment(
        fragments, conflicts, [[0, 1, 1_000_000_001], [27, 30, 1_000_000_002]]
    )
    assert small["status"] == huge["status"] == "optimal"
    assert small["objective"] == huge["objective"] == 0
    assert small["assignment"] == huge["assignment"]
    assert [small["assignment"][str(i)] for i in fragments] == _SEED_750_EXPECTED

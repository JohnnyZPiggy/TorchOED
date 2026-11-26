from collections import Counter
from itertools import combinations, product

import pytest
import torch

from torchoed import orthogonal_array, plackett_burman


def balanced(a, levels):
    assert a.dtype == torch.int64
    rows = a.tolist()
    n = len(rows)
    for c, s in enumerate(levels):
        assert Counter(row[c] for row in rows) == {v: n // s for v in range(s)}
    for c, d in combinations(range(len(levels)), 2):
        assert Counter((row[c], row[d]) for row in rows) == {
            pair: n // (levels[c] * levels[d])
            for pair in product(range(levels[c]), range(levels[d]))}


def test_exact_strength_two():
    for levels, n in [((2,) * 3, 4), ((2,) * 7, 8), ((3,) * 4, 9),
                      ((4,) * 5, 16), ((2, 4, 2, 4) + (2,) * 7, 16),
                      ((8, 2, 2), 64), ((9, 3, 3, 9), 81),
                      ((5,) * 6, 25), ((7,) * 8, 49)]:
        a = orthogonal_array(levels)
        assert a.shape == (n, len(levels))
        balanced(a, levels)


def test_two_level_branch():
    assert torch.equal(orthogonal_array((2,) * 11),
                       ((plackett_burman(12)[:, :11] + 1) / 2).to(torch.int64))
    for f, n in [(15, 16), (48, 64)]:
        a = orthogonal_array((2,) * f)
        assert a.shape == (n, f)
        balanced(a, (2,) * f)


def test_trust_boundary():
    for levels in [(2, 3), (6,), (2, 4, 8), (1, 2)]:
        with pytest.raises(ValueError):
            orthogonal_array(levels)
    with pytest.raises(ValueError):
        orthogonal_array((2,) * 40, max_elements=1000)


def test_binary_covering_array():
    from torchoed import covering_array
    for k, n in zip((2, 3, 4, 10, 11, 35, 36), (4, 4, 5, 6, 7, 8, 9)):
        a = covering_array(k, 2)
        assert a.shape == (n, k) and a.dtype == torch.int64
        for cols in combinations(range(k), 2):
            assert {tuple(row[j] for j in cols) for row in a.tolist()} == set(product(range(2), repeat=2))


def test_index_one_covering_array():
    from torchoed import covering_array
    for q, t, k in ((3, 2, 4), (4, 2, 5), (5, 3, 6), (2, 3, 3), (9, 3, 4)):
        a = covering_array(k, q, strength=t)
        assert a.shape == (q ** t, k) and a.dtype == torch.int64
        for cols in combinations(range(k), t):
            assert Counter(tuple(row[j] for j in cols) for row in a.tolist()) == dict.fromkeys(product(range(q), repeat=t), 1)


def test_covering_scope():
    from torchoed import covering_array
    for args, kw in [((4, 2), {"strength": 4}), ((2, 3), {"strength": 3}), ((3, True), {}), ((10, 2), {"max_elements": 59})]:
        with pytest.raises(ValueError):
            covering_array(*args, **kw)


def test_general_covering_coverage_sizes():
    from torchoed import covering_array
    sizes = {(2, 3, 5): 13, (2, 4, 10): 28, (3, 2, 4): 8, (3, 2, 10): 16}
    for t, v, k in [(2, 3, 5), (2, 4, 10), (2, 6, 4), (2, 6, 13),
                    (3, 2, 4), (3, 2, 10), (3, 3, 6), (3, 6, 5)]:
        a = covering_array(k, v, strength=t)
        assert a.dtype == torch.int64 and a.shape[1] == k
        if (t, v, k) in sizes:
            assert len(a) == sizes[t, v, k]
        rows = a.tolist()
        for cols in combinations(range(k), t):
            assert {tuple(row[j] for j in cols) for row in rows} == set(product(range(v), repeat=t))


def test_general_covering_rows_needed():
    from torchoed import covering_array
    rows = covering_array(13, 3).tolist()
    for i in range(len(rows)):
        remaining = rows[:i] + rows[i + 1:]
        assert any(len({tuple(row[j] for j in cols) for row in remaining}) < 9
                   for cols in combinations(range(13), 2))


def test_general_covering_scope():
    from torchoed import covering_array
    assert torch.equal(covering_array(5, 3), covering_array(5, 3))
    for args, kw in [((9, 7), {}), ((101, 3), {}), ((5, 3), {"strength": 4})]:
        with pytest.raises(ValueError):
            covering_array(*args, **kw)

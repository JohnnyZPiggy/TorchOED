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

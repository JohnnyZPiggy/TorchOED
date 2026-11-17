from fractions import Fraction
from itertools import combinations
from math import prod

import pytest
import torch

from torchoed import design_diagnostics, ff2n, fold_over, fractional_factorial, word_length_pattern


CASES = (
    (3, ((0, 1), (0, 2), (1, 2), (0, 1, 2)), (0, 0, 7, 7, 0, 0, 1), 3),
    (3, ((0, 1), (0, 2)), (0, 0, 2, 1, 0), 3),
    (4, ((0, 1, 2), (1, 2, 3)), (0, 0, 0, 3, 0, 0), 4),
    (4, ((0, 1, 2, 3),), (0, 0, 0, 0, 1), 5),
)


def defining_words(m, generators):
    basis = [{m + t, *generator} for t, generator in enumerate(generators)]
    words = []
    for size in range(1, len(basis) + 1):
        for subset in combinations(basis, size):
            word = set()
            for generator in subset:
                word ^= generator
            words.append(word)
    return words


def test_construction_and_word_length_pattern():
    for m, generators, expected, resolution in CASES:
        X = fractional_factorial(m, generators)
        n, k = X.shape
        assert (n, k) == (2**m, m + len(generators))
        assert torch.equal(X[:, :m], ff2n(m))
        rows = X.to(torch.int64).tolist()
        words = defining_words(m, generators)
        fractions = []
        for size in range(1, k + 1):
            numerator = sum(sum(prod(row[c] for c in term) for row in rows)**2
                            for term in combinations(range(k), size))
            fractions.append(Fraction(numerator, n**2))
        wlp = tuple(word_length_pattern(X).tolist())
        assert [a * n**2 for a in wlp] == [a * n**2 for a in fractions]
        assert wlp == expected == tuple(sum(len(w) == j for w in words) for j in range(1, k + 1))
        assert next(j for j, a in enumerate(wlp, 1) if a) == resolution


@pytest.mark.parametrize("case,count", [(0, 21), (1, 6), (2, 0), (None, None)])
def test_alias_matrix(case, count):
    m, generators, _, _ = CASES[0 if case is None else case]
    X = ff2n(3)[:6] if case is None else fractional_factorial(m, generators)
    n, k = X.shape
    pairs = list(combinations(range(k), 2))
    F = torch.cat((X.new_ones((n, 1)), X), dim=1)
    G = torch.stack([X[:, a] * X[:, b] for a, b in pairs], dim=1)
    A = design_diagnostics(F, torch.arange(n), alias=G)["alias"]
    if case is None:
        torch.testing.assert_close(A, torch.linalg.lstsq(F, G).solution, atol=1e-10, rtol=0)
        return
    words = defining_words(m, generators)
    terms = [set(), *({c} for c in range(k))]
    expected = X.new_tensor([[t ^ set(pair) in words for pair in pairs] for t in terms])
    assert torch.equal(F.mT @ G / n, expected)
    assert A.count_nonzero().item() == count
    # Cholesky solves can round a mathematical 1 by a few ulps.
    torch.testing.assert_close(A, expected, rtol=0, atol=2 * torch.finfo(X.dtype).eps)


def test_fold_over():
    for m, generators, expected, _ in CASES[:2]:
        X = fractional_factorial(m, generators)
        folded = fold_over(X)
        assert torch.equal(folded, torch.cat((X, -X)))
        wlp = tuple(word_length_pattern(folded).tolist())
        assert wlp == tuple(a if j % 2 == 0 else 0 for j, a in enumerate(expected, 1))
        assert wlp[:3] == (0, 0, 0)

    m, generators, _, _ = CASES[0]
    X = fractional_factorial(m, generators)
    words = defining_words(m, generators)
    for c in range(X.shape[1]):
        folded = fold_over(X, c)
        assert torch.equal(folded[:len(X)], X)
        expected = [sum(len(w) == j and c not in w for w in words) for j in range(1, X.shape[1] + 1)]
        assert word_length_pattern(folded).tolist() == expected

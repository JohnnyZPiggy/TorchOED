import itertools

import torch

from torchoed import lhs
from torchoed.sampling import _lhs_swap_deltas


def dense(x, mode):
    a, b = torch.triu_indices(len(x), len(x), 1)
    squared = (x[a] - x[b]).square()
    return squared.sum(-1).pow(-7.5).sum() if mode == "maximin" else squared.prod(-1).reciprocal().sum()


def test_swap_delta_oracle():
    g = torch.Generator().manual_seed(0)
    x = torch.stack([(torch.randperm(9, generator=g).double() + 0.5) / 9 for _ in range(3)], 1)
    for mode in ("maximin", "maxpro"):
        squared = (x[:, None] - x[None, :]).square()
        squared.diagonal(dim1=0, dim2=1).fill_(1)
        cache = squared.sum(-1) if mode == "maximin" else squared.log().sum(-1)
        transform = (lambda d: d.pow(-7.5)) if mode == "maximin" else (lambda l: (-l).exp())
        for column in range(3):
            changes, (aa, bb), _ = _lhs_swap_deltas(x, cache, column, mode, transform)
            for index, (a, b) in enumerate(zip(aa, bb)):
                y = x.clone()
                y[a, column], y[b, column] = x[b, column], x[a, column]
                torch.testing.assert_close(dense(x, mode) + changes[index], dense(y, mode), rtol=1e-12, atol=0)


def exhaustive(n, k, mode):
    levels = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    permutations = torch.tensor(list(itertools.permutations(levels.tolist())), dtype=torch.float64)
    designs = torch.stack([torch.stack((levels, *cols), 1)
                           for cols in itertools.product(permutations, repeat=k - 1)])
    a, b = torch.triu_indices(n, n, 1)
    squared = (designs[:, a] - designs[:, b]).square()
    scores = squared.sum(-1).pow(-7.5).sum(-1) if mode == "maximin" else squared.prod(-1).reciprocal().sum(-1)
    result = lhs(n, k, mode=mode, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(dense(result, mode), scores.min(), rtol=1e-12, atol=0)
    torch.testing.assert_close(result.sort(dim=0).values, levels[:, None].expand(n, k), rtol=0, atol=0)


def test_exhaustive_maximin():
    exhaustive(6, 2, "maximin")


def test_exhaustive_maxpro():
    exhaustive(5, 3, "maxpro")


from fractions import Fraction as Q
import math

from torchoed import space_filling_diagnostics
from torchoed.sampling import _cd2_kernel, _cd2_squared, _cd2_swap_deltas


def fraction_cd2(x):
    n, k = len(x), len(x[0])
    h = lambda v: 1 + abs(v - Q(1, 2)) / 2 - (v - Q(1, 2))**2 / 2
    g = lambda v, w: 1 + (abs(v - Q(1, 2)) + abs(w - Q(1, 2)) - abs(v - w)) / 2
    return (Q(13, 12)**k - Q(2, n) * sum(math.prod(map(h, row)) for row in x)
            + Q(1, n*n) * sum(math.prod(g(v, w) for v, w in zip(a, b)) for a in x for b in x))


def test_diagnostics_fraction_oracle():
    ranks = [[0, 3, 1], [1, 5, 4], [2, 0, 5], [3, 2, 0], [4, 1, 3], [5, 4, 2]]
    x = [[Q(2*r + 1, 12) for r in row] for row in ranks]
    result = space_filling_diagnostics(torch.tensor(x, dtype=torch.float64))
    squared = [[(v-w)**2 for v, w in zip(a, b)] for a, b in itertools.combinations(x, 2)]
    expected = {"min_distance": math.sqrt(min(map(sum, squared))),
                "phi_p": sum(float(sum(s))**(-7.5) for s in squared)**(1/15),
                "maxpro": float(sum(1/math.prod(s) for s in squared)/15)**(1/3)}
    for key, value in expected.items():
        torch.testing.assert_close(result[key], torch.tensor(value, dtype=torch.float64), rtol=1e-12, atol=0)
    torch.testing.assert_close(result["cd2"].square(), torch.tensor(float(fraction_cd2(x)), dtype=torch.float64), rtol=1e-14, atol=0)
    one = space_filling_diagnostics(((torch.arange(4, dtype=torch.float64)+0.5)/4)[:, None])
    torch.testing.assert_close(one["cd2"].square(), torch.tensor(1/192, dtype=torch.float64), rtol=1e-14, atol=0)


def test_cd2_swap_delta_oracle():
    gen = torch.Generator().manual_seed(0)
    x = torch.stack([(torch.randperm(9, generator=gen).double()+0.5)/9 for _ in range(3)], 1)
    cache = _cd2_kernel(x)
    for column in range(3):
        changes, (aa, bb), _ = _cd2_swap_deltas(x, cache, column)
        for i, (a, b) in enumerate(zip(aa, bb)):
            y = x.clone()
            y[a, column], y[b, column] = x[b, column], x[a, column]
            torch.testing.assert_close(_cd2_squared(x, cache) + changes[i], _cd2_squared(y, _cd2_kernel(y)), rtol=1e-12, atol=0)


def test_uniform_default_exact_local_optimum():
    x = lhs(8, 3, mode="uniform", generator=torch.Generator().manual_seed(0))
    exact = [[Q(float(v)) for v in row] for row in x]
    score = fraction_cd2(exact)
    for column in range(3):
        for a, b in itertools.combinations(range(8), 2):
            y = [row.copy() for row in exact]
            y[a][column], y[b][column] = y[b][column], y[a][column]
            assert fraction_cd2(y) >= score


def test_exhaustive_uniform():
    levels = [Q(2*r+1, 12) for r in range(6)]
    optimum = min(fraction_cd2(list(zip(levels, p))) for p in itertools.permutations(levels))
    x = lhs(6, 2, mode="uniform", generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(space_filling_diagnostics(x)["cd2"].square(), torch.tensor(float(optimum), dtype=torch.float64), rtol=1e-12, atol=0)


from torchoed import fast_flexible_filling, feasible_candidates, sobol
from torchoed.sampling import _kmeans, _maxpro_exchange
import pytest


def filling_setup():
    A = torch.tensor([[1., 1.]], dtype=torch.float64)
    b = torch.tensor([1.], dtype=torch.float64)
    S = sobol(256, 2, scramble=True, seed=0)
    F = S[feasible_candidates(S, A=A, b=b)]
    assign, centroids, clustered = _kmeans(F, 6, 100)
    chosen, exchanged = _maxpro_exchange(F, assign, centroids, 100)
    return A, b, S, F, assign, centroids, chosen, clustered, exchanged


def test_filling_lloyd_and_exchange_oracles():
    _, _, S, F, assign, centroids, chosen, clustered, exchanged = filling_setup()
    assert clustered and exchanged
    assert all(len(S[:, c].unique()) == 256 for c in range(2))
    distances = torch.cdist(F, centroids)
    torch.testing.assert_close(distances[torch.arange(len(F)), assign], distances.min(1).values,
                               rtol=0, atol=1e-12)
    X = F[chosen]
    # Public MaxPro is (Q / number_of_pairs) ** (1/k).
    Q = space_filling_diagnostics(X)["maxpro"].pow(2) * 15
    for c in range(6):
        torch.testing.assert_close(centroids[c], F[assign == c].mean(0), rtol=0, atol=1e-12)
        for member in F[assign == c]:
            Y = X.clone()
            Y[c] = member
            replacement = space_filling_diagnostics(Y)["maxpro"].pow(2) * 15
            assert replacement >= Q * (1 - 1e-12)


def test_filling_public_feasibility():
    A, b, S, F, _, _, chosen, _, _ = filling_setup()
    X = fast_flexible_filling(6, 2, A=A, b=b, n_candidates=256, seed=0)
    assert (X @ A.T <= b).all()
    assert (X[:, None] == S).all(-1).any(-1).all()
    assert len(X.unique(dim=0)) == 6
    torch.testing.assert_close(X, F[chosen], rtol=0, atol=0)


def test_filling_errors():
    with pytest.raises(ValueError, match="power of two"):
        fast_flexible_filling(6, 2, n_candidates=100)
    with pytest.raises(ValueError, match="too few feasible"):
        fast_flexible_filling(6, 2, A=torch.tensor([[1., 1.]], dtype=torch.float64),
                              b=torch.tensor([-1.], dtype=torch.float64), n_candidates=256)

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

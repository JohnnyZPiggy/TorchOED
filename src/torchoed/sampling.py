import torch

from ._validation import budget, floating, integer


def lhs(n_samples: int, n_factors: int, *, generator: torch.Generator | None = None, device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64, mode: str = "random", restarts: int = 10, max_iter: int | None = None, max_elements: int = 1_000_000) -> torch.Tensor:
    """Random jittered LHS or midpoint maximin / MaxPro LHS by column swaps."""
    n = integer(n_samples, "n_samples")
    k = integer(n_factors, "n_factors")
    floating(dtype)
    budget((n, k), max_elements)
    if mode not in ("random", "maximin", "maxpro"):
        raise ValueError("mode must be random, maximin or maxpro")
    if n > (2**22 if dtype == torch.float32 else 2**51):
        raise ValueError("strata exceed safe floating-point resolution")
    target = torch.device("cpu" if device is None else device)
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None")
        if generator.device.type != target.type:
            raise ValueError("generator and output must use the same device backend")
    if mode != "random":
        transform = (lambda d: d.pow(-7.5)) if mode == "maximin" else (lambda l: (-l).exp())
        best, best_score = None, float("inf")
        for _ in range(restarts):
            x = torch.stack([(torch.randperm(n, generator=generator, device=target).to(dtype) + 0.5) / n
                             for _ in range(k)], dim=1)
            squared = (x[:, None] - x[None, :]).square()
            squared.diagonal(dim1=0, dim2=1).fill_(1)
            cache = squared.sum(-1) if mode == "maximin" else squared.log().sum(-1)
            delta = lambda x, cache, column: _lhs_swap_deltas(x, cache, column, mode, transform)
            x, score = _lhs_search(x, cache, delta, transform, max_iter)
            if best is None or score < best_score:
                best, best_score = x, score
        return best
    columns = []
    for _ in range(k):
        ids = torch.randperm(n, generator=generator, device=target)
        jitter = torch.rand(n, generator=generator, device=target, dtype=dtype)
        x = (ids.to(dtype) + jitter) / n
        # Rounding can place a jitter at the next boundary; retain exact strata.
        x = torch.where(torch.floor(n * x).to(torch.int64) == ids,
                        x, (ids.to(dtype) + 0.5) / n)
        columns.append(x)
    return torch.stack(columns, dim=1)


def _lhs_swap_deltas(x, cache, column, mode, transform):
    """All column-swap deltas from the two affected rows, O(n**3)."""
    n = x.shape[0]
    a, b = torch.triu_indices(n, n, 1, device=x.device)
    values = x[:, column]
    old_a = (values[a, None] - values).square()
    old_b = (values[b, None] - values).square()
    j = torch.arange(n, device=x.device)
    affected = (j != a[:, None]) & (j != b[:, None])
    # Self and swapped-pair entries never contribute to the delta.
    old_a = torch.where(affected, old_a, 1)
    old_b = torch.where(affected, old_b, 1)
    if mode == "maxpro":
        old_a, old_b = old_a.log(), old_b.log()
    new_a = cache[a] - old_a + old_b
    new_b = cache[b] - old_b + old_a
    change = transform(new_a) - transform(cache[a]) + transform(new_b) - transform(cache[b])
    return torch.where(affected, change, 0).sum(-1), (a, b), (new_a, new_b)


def _lhs_search(x, cache, swap_deltas, transform, max_iter):
    """Steepest column-swap descent; the objective supplies batched deltas."""
    n, k = x.shape
    upper = torch.triu_indices(n, n, 1, device=x.device)
    score = transform(cache[upper[0], upper[1]]).sum()
    iteration = 0
    while n > 1 and (max_iter is None or iteration < max_iter):
        best = None
        for column in range(k):
            changes, (a, b), rows = swap_deltas(x, cache, column)
            value, index = changes.min(0)
            if score + value < score and (best is None or value < best[0]):
                best = value, column, a[index], b[index], rows[0][index], rows[1][index]
        if best is None:
            break
        value, column, a, b, row_a, row_b = best
        x[a, column], x[b, column] = x[b, column].clone(), x[a, column].clone()
        cache[a], cache[b] = row_a, row_b
        cache[:, a], cache[:, b] = row_a, row_b
        score = transform(cache[upper[0], upper[1]]).sum()
        iteration += 1
    return x, score


def sobol(n: int, d: int, *, scramble: bool = False, seed: int | None = None, skip: int = 0, balance: str = "exact",
          dtype: torch.dtype = torch.float64, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor:
    """Native CPU Sobol generation followed by an explicit device transfer."""
    integer(n, "n", 0)
    integer(d, "d")
    integer(skip, "skip", 0)
    floating(dtype)
    budget((n, d), max_elements)
    if type(scramble) is not bool:
        raise TypeError("scramble must be bool")
    if seed is not None:
        integer(seed, "seed", 0)
        if seed > 2**63 - 1:
            raise ValueError("seed must fit signed int64")
    engine_type = torch.quasirandom.SobolEngine
    if d > engine_type.MAXDIM or skip + n > 2**engine_type.MAXBIT:
        raise ValueError("Sobol dimension or sequence horizon exceeded")
    if balance != "exact":
        raise ValueError("only balance='exact' is implemented; never pads n")
    if n == 0:
        return torch.empty((0, d), dtype=dtype, device=device)
    engine = engine_type(d, scramble=scramble, seed=seed)
    if skip:
        engine.fast_forward(skip)
    return engine.draw(n, dtype=dtype).to(device=device)

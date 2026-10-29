import torch

from ._validation import budget, floating, integer


def lhs(n_samples, n_factors, *, generator=None, device=None,
        dtype=torch.float64, mode="random", max_elements=1_000_000):
    """Basic random Latin hypercube; consumes the supplied generator normally."""
    n = integer(n_samples, "n_samples")
    k = integer(n_factors, "n_factors")
    floating(dtype)
    budget((n, k), max_elements)
    if mode != "random":
        raise ValueError("only mode='random' is implemented")
    if n > (2**22 if dtype == torch.float32 else 2**51):
        raise ValueError("strata exceed safe floating-point resolution")
    target = torch.device("cpu" if device is None else device)
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None")
        if generator.device.type != target.type:
            raise ValueError("generator and output must use the same device backend")
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


def sobol(n, d, *, scramble=False, seed=None, skip=0, balance="exact",
          dtype=torch.float64, device=None, max_elements=1_000_000):
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

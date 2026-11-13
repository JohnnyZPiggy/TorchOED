import torch

from ._validation import budget, floating, integer


def full_factorial(levels: list[int] | tuple[int, ...], *, device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor:
    """Integer index grid; last factor varies fastest. See docs/M1_API.md."""
    if not isinstance(levels, (tuple, list)) or not levels:
        raise ValueError("levels must be a nonempty tuple/list of positive Python ints")
    rows = 1
    for q in levels:
        rows *= integer(q, "level")
    budget((rows, len(levels)), max_elements)
    axes = [torch.arange(q, device=device, dtype=torch.int64) for q in levels]
    return torch.cartesian_prod(*axes).reshape(rows, len(levels))


def ff2n(k: int, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float64, max_elements: int = 1_000_000) -> torch.Tensor:
    """Signed corners, in full_factorial order, with explicit floating dtype."""
    integer(k, "k")
    floating(dtype)
    # Bound k before constructing an exponentially large Python integer/list.
    if k >= 63:
        raise ValueError("signed grid exceeds int64 output limit")
    budget((2**k, k), max_elements)
    return (2 * full_factorial((2,) * k, device=device,
                              max_elements=max_elements) - 1).to(dtype=dtype)


def definitive_screening(k: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a three-level definitive screening design from a Paley conference matrix."""
    integer(k, "k")
    q = k if k % 2 else k - 1
    if q not in (3, 5, 7, 11, 13, 17, 19, 23):
        raise ValueError("k must be a supported Paley factor count")
    residues = {i * i % q for i in range(1, q)}
    chi = [0] + [1 if i in residues else -1 for i in range(1, q)]
    rows = [[0] + [1] * (k - 1)]
    rows += [[chi[-1]] + [chi[(j - i) % q] for j in range(k - 1)] for i in range(q)]
    return torch.tensor(rows + [[-x for x in row] for row in rows] + [[0] * k],
                        dtype=dtype, device=device)


def simplex_lattice(q: int, m: int, *, dtype: torch.dtype = torch.float64,
                    device: torch.device | str | None = None, max_elements: int = 1_000_000) -> torch.Tensor:
    """Return the {q, m} simplex lattice in full_factorial order.
    For a Scheffé quadratic, drop the q square columns of polynomial_features(X, 2, include_bias=False).
    """
    integer(q, "q")
    integer(m, "m")
    floating(dtype)
    counts = full_factorial([m + 1] * q, device=device, max_elements=max_elements)
    return counts[counts.sum(dim=1) == m].to(dtype) / m

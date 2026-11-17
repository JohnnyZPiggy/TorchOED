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


def fractional_factorial(
    m: int, generators: list[list[int] | tuple[int, ...]] | tuple[list[int] | tuple[int, ...], ...],
    *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None,
    max_elements: int = 1_000_000,
) -> torch.Tensor:
    """Append unsigned products of base columns to the signed factorial grid."""
    if integer(m, "m") >= 63:
        raise ValueError("signed grid exceeds int64 output limit")
    if not isinstance(generators, (list, tuple)) or not generators:
        raise ValueError("generators must be a nonempty list or tuple")
    terms = []
    for generator in generators:
        if (not isinstance(generator, (list, tuple)) or not generator
                or any(type(c) is not int or not 0 <= c < m for c in generator)):
            raise ValueError("each generator must list distinct Python ints in range(m)")
        term = set(generator)
        if len(term) != len(generator):
            raise ValueError("each generator must list distinct Python ints in range(m)")
        terms.append(term)
    budget((2**m, m + len(generators)), max_elements)
    base = ff2n(m, dtype=dtype, device=device, max_elements=max_elements)
    signs = base.new_tensor([[-1 if c in term else 1 for term in terms] for c in reversed(range(m))])
    generated = base.new_ones((1, len(terms)))
    # Prepend slower factors in ff2n order, with the -1 level first.
    for sign in signs:
        generated = torch.cat((generated * sign, generated))
    return torch.cat((base, generated), dim=1)


def fold_over(X: torch.Tensor, factor: int | None = None) -> torch.Tensor:
    """Append rows with all factors reversed, or just the chosen factor."""
    if not isinstance(X, torch.Tensor) or X.ndim != 2 or not X.is_floating_point():
        raise ValueError("X must be a 2-D floating tensor")
    if factor is None:
        return torch.cat((X, -X))
    if type(factor) is not int or not 0 <= factor < X.shape[1]:
        raise ValueError("factor must be a Python int in range(X.shape[1])")
    reversed_rows = X.clone()
    reversed_rows[:, factor] = -X[:, factor]
    return torch.cat((X, reversed_rows))


def word_length_pattern(X: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor:
    """Return the float64 generalized word-length pattern of a signed two-level design."""
    if not isinstance(X, torch.Tensor) or X.ndim != 2 or not X.is_floating_point():
        raise ValueError("X must be a 2-D floating tensor")
    n, k = X.shape
    if n == 0:
        raise ValueError("the word-length pattern requires at least one row")
    if k >= 63:
        raise ValueError("word-length pattern exceeds int64 output limit")
    budget((n, 2**k), max_elements)
    if not ((X == -1) | (X == 1)).all():
        raise ValueError("X must contain only -1 and +1")
    products = X.new_ones((n, 1), dtype=torch.float64)
    for c in range(k):
        products = torch.cat((products, products * X[:, c:c + 1]), dim=1)
    lengths = torch.tensor([s.bit_count() for s in range(2**k)], device=X.device)
    squared_j = products.sum(dim=0).square()
    return torch.bincount(lengths, weights=squared_j, minlength=k + 1)[1:] / n**2


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

import torch

from ._validation import budget, floating, integer


def full_factorial(levels, *, device=None, max_elements=1_000_000):
    """Integer index grid; last factor varies fastest. See docs/M1_API.md."""
    if not isinstance(levels, (tuple, list)) or not levels:
        raise ValueError("levels must be a nonempty tuple/list of positive Python ints")
    rows = 1
    for q in levels:
        rows *= integer(q, "level")
    budget((rows, len(levels)), max_elements)
    axes = [torch.arange(q, device=device, dtype=torch.int64) for q in levels]
    return torch.cartesian_prod(*axes).reshape(rows, len(levels))


def ff2n(k, *, device=None, dtype=torch.float64, max_elements=1_000_000):
    """Signed corners, in full_factorial order, with explicit floating dtype."""
    integer(k, "k")
    floating(dtype)
    # Bound k before constructing an exponentially large Python integer/list.
    if k >= 63:
        raise ValueError("signed grid exceeds int64 output limit")
    budget((2**k, k), max_elements)
    return (2 * full_factorial((2,) * k, device=device,
                              max_elements=max_elements) - 1).to(dtype=dtype)

import torch

from ._validation import budget, floating


def polynomial_features(C, degree=2, include_bias=True, *, max_elements=1_000_000):
    """[bias, linear, squares, pair products i<j]; preserves autograd."""
    if not isinstance(C, torch.Tensor):
        raise TypeError("C must be a floating torch.Tensor")
    floating(C.dtype)
    if C.ndim not in (2, 3) or C.shape[-1] < 1:
        raise ValueError("C must have shape (m,k) or (B,m,k), k>=1")
    if type(degree) is not int or degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    if type(include_bias) is not bool:
        raise TypeError("include_bias must be bool")
    k = C.shape[-1]
    p = int(include_bias) + k + (k + k * (k - 1) // 2 if degree == 2 else 0)
    budget((*C.shape[:-1], p), max_elements)
    if not bool(torch.isfinite(C).all()):
        raise ValueError("C must contain only finite values")
    parts = [torch.ones_like(C[..., :1])] if include_bias else []
    parts.append(C)
    if degree == 2:
        i, j = torch.triu_indices(k, k, offset=1, device=C.device)
        parts.extend((C.square(), C[..., i] * C[..., j]))
    return torch.cat(parts, dim=-1)

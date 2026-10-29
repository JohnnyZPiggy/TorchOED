"""Maximization-oriented D/A/E scores without implicit regularization."""
import torch

from ._validation import budget, floating
from .information import _reject


def _symmetric(M, max_elements):
    if not isinstance(M, torch.Tensor):
        raise TypeError("M must be a torch.Tensor")
    floating(M.dtype)
    if (M.ndim not in (2, 3) or M.shape[-1] < 1 or M.shape[-2] != M.shape[-1]
            or (M.ndim == 3 and M.shape[0] == 0)):
        raise ValueError("M must be square (p,p) or (B,p,p), p>=1 and B>=1")
    budget(M.shape, max_elements)
    _reject(~torch.isfinite(M).all(dim=(-2, -1)), "nonfinite M")
    scale = M.abs().amax(dim=(-2, -1))
    tolerance = 64 * M.shape[-1] * torch.finfo(M.dtype).eps * scale
    _reject((M - M.mT).abs().amax(dim=(-2, -1)) > tolerance, "M must be symmetric")
    # Average only within the accepted rounding tolerance; avoids triangle bias.
    return M * 0.5 + M.mT * 0.5, tolerance


def _cholesky(M, max_elements):
    symmetric, tolerance = _symmetric(M, max_elements)
    L, info = torch.linalg.cholesky_ex(symmetric)
    _reject(info != 0, "M must be positive definite (Cholesky failed)")
    # Roundoff can give exactly singular Gram matrices a successful factorization.
    pivots = L.diagonal(dim1=-2, dim2=-1).square()
    _reject((pivots <= tolerance[..., None]).any(dim=-1),
            "M is numerically singular/ill-conditioned (Cholesky pivot threshold)")
    return L


def _score(value):
    _reject(~torch.isfinite(value), "criterion overflow/nonfinite output")
    return value


def d_optimality(M, *, max_elements=1_000_000):
    """Maximize logdet(M); singular/indefinite input raises ValueError."""
    L = _cholesky(M, max_elements)
    return _score(2 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1))


def a_optimality(M, *, max_elements=1_000_000):
    """Maximize -trace(M^-1), evaluated through Cholesky solves, not inverse."""
    L = _cholesky(M, max_elements)
    identity = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device).expand(L.shape)
    solution = torch.cholesky_solve(identity, L)
    return _score(-solution.diagonal(dim1=-2, dim2=-1).sum(dim=-1))


def e_optimality(M, *, max_elements=1_000_000):
    """Maximize minimum eigenvalue; numerical PSD zeros allowed, no ridge."""
    symmetric, tolerance = _symmetric(M, max_elements)
    value = torch.linalg.eigvalsh(symmetric)[..., 0]
    _reject(value < -tolerance, "M must be positive semidefinite")
    return _score(value.clamp_min(0))

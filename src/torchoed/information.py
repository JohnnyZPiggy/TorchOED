"""Homoscedastic linear information, with exact observation-event IDs."""
import torch

from ._validation import budget, floating


def _reject(invalid, reason):
    if bool(invalid.any()):
        where = (f" at batch index {invalid.nonzero()[0, 0].item()}"
                 if invalid.ndim else "")
        raise ValueError(reason + where)


def _features(F, max_elements):
    if not isinstance(F, torch.Tensor):
        raise TypeError("F must be a torch.Tensor")
    floating(F.dtype)
    if F.ndim not in (2, 3) or F.shape[-1] < 1 or (F.ndim == 3 and F.shape[0] == 0):
        raise ValueError("F must be (m,p) or (B,m,p), p>=1 and B>=1")
    budget(F.shape, max_elements)
    budget((*F.shape[:-2], F.shape[-1], F.shape[-1]), max_elements)
    _reject(~torch.isfinite(F).all(dim=(-2, -1)), "nonfinite F")


def _finite_output(M):
    _reject(~torch.isfinite(M).all(dim=(-2, -1)), "information overflow/nonfinite output")
    return M


def exact_information(F, ids, *, normalized=False, allow_repeats=False,
                      max_elements=1_000_000):
    """Raw X.T@X, optionally divided by event count; never implicit repeats."""
    _features(F, max_elements)
    if type(normalized) is not bool or type(allow_repeats) is not bool:
        raise TypeError("normalized and allow_repeats must be bool")
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64:
        raise TypeError("ids must be an int64 torch.Tensor")
    if ids.device != F.device:
        raise ValueError("ids and F must have the same device")
    if ids.ndim != F.ndim - 1 or ids.shape[:-1] != F.shape[:-2]:
        raise ValueError("ids must match F exactly: (n,) or (B,n); no broadcasting")
    n = ids.shape[-1]
    budget((*ids.shape, F.shape[-1]), max_elements)
    if normalized and n == 0:
        raise ValueError("normalized information requires n>0")
    _reject(((ids < 0) | (ids >= F.shape[-2])).any(dim=-1), "ID out of range")
    if not allow_repeats:
        ordered = ids.sort(dim=-1).values
        _reject((ordered[..., 1:] == ordered[..., :-1]).any(dim=-1), "duplicate ID")
    if F.ndim == 2:
        X = F[ids]
    else:
        X = torch.gather(F, 1, ids[..., None].expand(-1, -1, F.shape[-1]))
    M = X.mT @ X
    return _finite_output(M / n if normalized else M)


def weighted_information(F, w, *, max_elements=1_000_000):
    """Probability-weighted information, not sample counts; no dense diagonal."""
    _features(F, max_elements)
    if not isinstance(w, torch.Tensor):
        raise TypeError("w must be a torch.Tensor")
    if w.dtype != F.dtype or w.device != F.device:
        raise ValueError("w and F must have identical dtype and device")
    if w.shape != F.shape[:-1]:
        raise ValueError("w must match (m,) or (B,m); no broadcasting")
    _reject((~torch.isfinite(w) | (w < 0)).any(dim=-1), "weights must be finite and nonnegative")
    tolerance = 64 * torch.finfo(F.dtype).eps
    _reject((w.sum(dim=-1) - 1).abs() > tolerance, "weights must sum to 1")
    return _finite_output(F.mT @ (w[..., None] * F))

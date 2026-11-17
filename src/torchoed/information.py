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


def _exact_rows(F, ids, normalized=False, allow_repeats=False, max_elements=1_000_000):
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
        return F[ids]
    return torch.gather(F, 1, ids[..., None].expand(-1, -1, F.shape[-1]))


def exact_information(F: torch.Tensor, ids: torch.Tensor, *, normalized: bool = False, allow_repeats: bool = False,
                      max_elements: int = 1_000_000) -> torch.Tensor:
    """Raw X.T@X, optionally divided by event count; never implicit repeats."""
    X = _exact_rows(F, ids, normalized, allow_repeats, max_elements)
    M = X.mT @ X
    return _finite_output(M / X.shape[-2] if normalized else M)


def weighted_information(F: torch.Tensor, w: torch.Tensor, *, max_elements: int = 1_000_000) -> torch.Tensor:
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


def _coefficient_power(noncentrality: torch.Tensor, df: int, alpha: float) -> torch.Tensor:
    # Central beta quantile via the b-recurrence in P2_GATE §11.
    b = df / 2
    steps = torch.arange(0.5 if df % 2 else 1.0, b, device=noncentrality.device,
                         dtype=noncentrality.dtype)
    log_u = (steps + 0.5).lgamma() - (steps + 1).lgamma() - noncentrality.new_tensor(0.5).lgamma()
    lo, hi = noncentrality.new_tensor(0.0), noncentrality.new_tensor(1.0)
    for _ in range(64):
        x = (lo + hi) / 2
        root = x.sqrt()
        central = (2 / torch.pi) * root.asin() if df % 2 else root
        central = central + (log_u + 0.5 * x.log() + steps * (-x).log1p()).exp().sum()
        below = central < 1 - alpha
        lo = torch.where(below, x, lo)
        hi = torch.where(below, hi, x)

    # Sum beta survival probabilities against the truncated Poisson weights.
    mu = noncentrality[:, None] / 2
    cutoff = (mu + 12 * mu.sqrt() + 30).ceil()
    k = torch.arange(int(cutoff.max()) + 1, device=mu.device, dtype=mu.dtype)
    a = k[:-1] + 0.5
    log_t = ((a + b).lgamma() - (a + 1).lgamma() - mu.new_tensor(b).lgamma()
             + a * x.log() + b * (-x).log1p())
    q = (1 - central) + torch.cat((k[:1], log_t.exp().cumsum(dim=0)))
    weights = (torch.xlogy(k, mu) - mu - (k + 1).lgamma()).exp()
    return (weights * q).masked_fill(k > cutoff, 0).sum(dim=1)


def design_diagnostics(
    F: torch.Tensor, ids: torch.Tensor, *, region: torch.Tensor | None = None,
    effect: float | torch.Tensor | None = None, alpha: float = 0.05,
    blocks: list[int] | tuple[int, ...] | None = None,
    whole_plots: list[int] | tuple[int, ...] | None = None, eta: float = 1.0,
) -> dict[str, torch.Tensor | list[int]]:
    """Return design correlations, VIFs, scaled prediction variances and their sorted values.

    Set either `blocks` or `whole_plots` to sizes of contiguous groups in `ids`.
    Blocks absorb the intercept and center prediction contrasts over `region`.
    Whole plots use GLS with positive variance ratio `eta`.

    With `effect`, return `power` at level `alpha` for two-sided t-tests with
    unit error variance. Structured results also include the per-coefficient
    residual degrees of freedom in `df`. Split-plot power is exact only in
    the P2_GATE section 19.5 case, otherwise approximate (containment rule).
    """
    structured = blocks is not None or whole_plots is not None
    if structured:
        S = _exact_rows(F, ids)
    else:
        M = exact_information(F, ids)
    if effect is not None:
        if type(alpha) is not float or not 0.001 <= alpha <= 0.5:
            raise ValueError("alpha must be a Python float in [0.001, 0.5]")
        if not structured and ids.shape[-1] <= F.shape[-1]:
            raise ValueError("power requires n > p")
    if not structured:
        L = torch.linalg.cholesky(M)
        S = F[ids]
    n, p = S.shape
    R = F if region is None else region
    df = None
    if structured:
        # Search imports exact_information from this module.
        from .d_optimal_design import _block_information, _split_information, _valid_plot_sizes

        if blocks is not None and whole_plots is not None:
            raise ValueError("blocks and whole_plots are mutually exclusive")
        sizes = blocks if blocks is not None else whole_plots
        if not _valid_plot_sizes(sizes, n):
            raise ValueError("group sizes must be positive Python ints summing to len(ids)")
        if blocks is not None:
            M, _ = _block_information(F, ids, blocks)
            C = M
            R = R - R.mean(0)
            if effect is not None:
                df = ids.new_full((p,), n - len(blocks) - p)
        else:
            if type(eta) not in (int, float) or not 0 < eta < float("inf"):
                raise ValueError("eta must be a finite positive Python int or float")
            counts = F.new_tensor(whole_plots)
            inverse_scale = ((counts + 1 / eta).reciprocal() / eta if eta >= 1
                             else (1 + eta * counts).reciprocal())
            M, sums = _split_information(F, ids, whole_plots, inverse_scale / counts)
            weighted_sum = inverse_scale @ sums
            C = M - torch.outer(weighted_sum, weighted_sum / (inverse_scale @ counts))
            if effect is not None:
                whole = torch.stack([(plot == plot[0]).all(0) for plot in S.split(whole_plots)]).all(0)
                whole_count = whole.sum()
                df = torch.where(whole, len(whole_plots) - whole_count,
                                 n - len(whole_plots) - (p - whole_count))
        if df is not None and (df < 1).any():
            raise ValueError("structured power requires positive residual degrees of freedom")
        L = torch.linalg.cholesky(M)

    columns = (S != S[0]).any(dim=0).nonzero().flatten().tolist()
    if structured:
        C = C[columns][:, columns]
        scale = C.diagonal().rsqrt()
        correlation = scale[:, None] * C * scale
    else:
        X = S[:, columns]
        X = torch.nn.functional.normalize(X - X.mean(dim=0), dim=0, eps=0)
        correlation = X.mT @ X if len(columns) != 1 else X.new_ones((1, 1))
    vif = torch.cholesky_inverse(torch.linalg.cholesky(correlation)).diagonal()
    solved = torch.cholesky_solve(R.mT, L)
    spv = n * torch.einsum("ij,ji->i", R, solved)
    result: dict[str, torch.Tensor | list[int]] = {
        "columns": columns, "correlation": correlation, "vif": vif, "spv": spv, "fds": spv.sort().values,
    }
    if effect is not None:
        delta = torch.as_tensor(effect, dtype=torch.float64, device=F.device).broadcast_to((p,))
        noncentrality = delta.square() / torch.cholesky_inverse(L).diagonal()
        if df is None:
            result["power"] = _coefficient_power(noncentrality, n - p, alpha).to(F.dtype)
        else:
            power = F.new_empty(p)
            for dof in df.unique().tolist():
                mask = df == dof
                power[mask] = _coefficient_power(noncentrality[mask], dof, alpha).to(F.dtype)
            result.update(power=power, df=df)
    return result

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


def _coefficient_power(noncentrality: torch.Tensor, df: int, alpha: float, r: int = 1) -> torch.Tensor:
    # Central beta quantile via the b-recurrence in P2_GATE §11.
    b = df / 2
    a0 = 0.5 if r % 2 else 1.0
    steps = torch.arange(0.5 if df % 2 else 1.0, b, device=noncentrality.device,
                         dtype=noncentrality.dtype)
    log_u = (steps + a0).lgamma() - (steps + 1).lgamma() - noncentrality.new_tensor(a0).lgamma()
    shifts = torch.arange(a0, r / 2, device=noncentrality.device, dtype=noncentrality.dtype)
    log_shift = ((shifts + b).lgamma() - (shifts + 1).lgamma()
                 - noncentrality.new_tensor(b).lgamma())
    lo, hi = noncentrality.new_tensor(0.0), noncentrality.new_tensor(1.0)
    for _ in range(64):
        x = (lo + hi) / 2
        root = x.sqrt()
        central = (2 / torch.pi) * root.asin() if df % 2 else root
        if r % 2 == 0:
            central = 1 - (1 - x).sqrt() if df % 2 else x
        central = central + (log_u + a0 * x.log() + steps * (-x).log1p()).exp().sum()
        q0 = 1 - central
        if r > 1:
            q0 = q0 + (log_shift + shifts * x.log() + b * (-x).log1p()).exp().sum()
        below = central < 1 - alpha if r == 1 else q0 > alpha
        lo = torch.where(below, x, lo)
        hi = torch.where(below, hi, x)

    # Sum beta survival probabilities against the truncated Poisson weights.
    mu = noncentrality[:, None] / 2
    cutoff = (mu + 12 * mu.sqrt() + 30).ceil()
    k = torch.arange(int(cutoff.max()) + 1, device=mu.device, dtype=mu.dtype)
    a = k[:-1] + r / 2
    log_t = ((a + b).lgamma() - (a + 1).lgamma() - mu.new_tensor(b).lgamma()
             + a * x.log() + b * (-x).log1p())
    q = q0 + torch.cat((k[:1], log_t.exp().cumsum(dim=0)))
    weights = (torch.xlogy(k, mu) - mu - (k + 1).lgamma()).exp()
    return (weights * q).masked_fill(k > cutoff, 0).sum(dim=1)


def design_diagnostics(
    F: torch.Tensor, ids: torch.Tensor, *, region: torch.Tensor | None = None,
    effect: float | torch.Tensor | None = None, alpha: float = 0.05,
    blocks: list[int] | tuple[int, ...] | None = None,
    whole_plots: list[int] | tuple[int, ...] | None = None, eta: float = 1.0,
    subplots: list[int] | tuple[int, ...] | None = None, eta_sub: float = 1.0,
    covariates: torch.Tensor | None = None, centered: bool = True,
    alias: torch.Tensor | None = None, groups: list[list[int]] | None = None,
) -> dict[str, torch.Tensor | list[int]]:
    """Return design correlations, VIFs, scaled prediction variances and their sorted values.

    Set either `blocks` or `whole_plots` to sizes of contiguous groups in `ids`.
    Blocks absorb the intercept and center prediction contrasts over `region`.
    Whole plots use GLS with positive variance ratio `eta`.
    Nested `subplots` add a stratum with variance ratio `eta_sub` (§114–§115).

    With `effect`, return `power` at level `alpha` for two-sided t-tests with
    unit error variance. Structured results also include the per-coefficient
    residual degrees of freedom in `df`. Split-plot power is exact only in
    the P2_GATE section 19.5 case, otherwise approximate (containment rule).
    With `groups`, return effect-group F-test `group_power`; structured results
    also include `group_df`. Whole-plot groups must lie in one stratum.
    With `alias`, return omitted-term aliasing using unstructured information.
    """
    if subplots is not None and (whole_plots is None or blocks is not None or covariates is not None or alias is not None):
        raise ValueError("subplots requires whole_plots without blocks, covariates or alias")
    if covariates is not None and (blocks is not None or whole_plots is not None or alias is not None):
        raise ValueError("covariates cannot be combined with blocks, whole_plots or alias")
    structured = blocks is not None or whole_plots is not None or covariates is not None
    if not isinstance(centered, bool) or (not centered and structured):
        raise ValueError("centered must be bool; centered=False is unstructured only")
    if alias is not None and structured:
        raise ValueError("alias is unavailable with blocks or whole_plots")
    if structured:
        S = _exact_rows(F, ids)
    else:
        M = exact_information(F, ids)
    if alias is not None and (not isinstance(alias, torch.Tensor) or alias.ndim != 2
                              or alias.shape[0] != F.shape[0] or alias.dtype != F.dtype
                              or alias.device != F.device):
        raise ValueError("alias must be a 2-D tensor with F's row count, dtype and device")
    if effect is not None:
        if type(alpha) is not float or not 0.001 <= alpha <= 0.5:
            raise ValueError("alpha must be a Python float in [0.001, 0.5]")
        if not structured and ids.shape[-1] <= F.shape[-1]:
            raise ValueError("power requires n > p")
    if not structured:
        L = torch.linalg.cholesky(M)
        S = F[ids]
    n, p = S.shape
    if groups is not None:
        if (effect is None or type(groups) is not list or not groups
                or any(type(g) is not list or not g or any(type(j) is not int or not 0 <= j < p for j in g)
                       or len(set(g)) != len(g) for g in groups)):
            raise ValueError("groups requires effect and a non-empty list of non-empty distinct coefficient lists")
    R = F if region is None else region
    df = None
    if structured:
        # Search imports exact_information from this module.
        from .d_optimal_design import _block_information, _split_information, _nested_information, _valid_plot_sizes

        if covariates is not None:
            Z = covariates
            if (not isinstance(Z, torch.Tensor) or Z.ndim != 2 or Z.shape[0] != n
                    or Z.shape[1] < 1 or Z.dtype != F.dtype or Z.device != F.device
                    or not torch.isfinite(Z).all()):
                raise ValueError("covariates must be finite (len(ids), c), c >= 1, with F's dtype and device")
            X = torch.cat((F.new_ones(n, 1), Z), 1)
            gram = X.mT @ X
            L, info = torch.linalg.cholesky_ex(gram)
            if info != 0 or (L.diagonal().square() <= 64*len(gram)*torch.finfo(F.dtype).eps*gram.abs().amax()).any():
                raise ValueError("[1 covariates] must have full column rank")
            U = torch.linalg.qr(X, mode="reduced").Q
            residual = S - U @ (U.mT @ S)
            M = C = residual.mT @ residual
            R = R - R.mean(0)
            if effect is not None:
                df = ids.new_full((p,), n - 1 - Z.shape[1] - p)
        else:
            sizes = whole_plots if whole_plots is not None else blocks
            if not _valid_plot_sizes(sizes, n):
                raise ValueError("group sizes must be positive Python ints summing to len(ids)")
            if whole_plots is None:
                M, _ = _block_information(F, ids, blocks)
                C = M
                R = R - R.mean(0)
                if effect is not None:
                    df = ids.new_full((p,), n - len(blocks) - p)
            else:
                if type(eta) not in (int, float) or not 0 < eta < float("inf"):
                    raise ValueError("eta must be a finite positive Python int or float")
                if subplots is not None:
                    ends = torch.tensor(subplots).cumsum(0).tolist() if _valid_plot_sizes(subplots, n) else []
                    if (not ends or any(b not in ends for b in torch.tensor(whole_plots).cumsum(0).tolist())
                            or type(eta_sub) not in (int, float) or not 0 < eta_sub < float("inf")):
                        raise ValueError("subplots must align with whole plots; eta_sub must be finite positive")
                    plot_ids = torch.arange(n, device=F.device).split(subplots)
                    labels = torch.repeat_interleave(ids.new_tensor(range(len(whole_plots))), ids.new_tensor(whole_plots))
                    owner = torch.stack([labels[g[0]] for g in plot_ids])
                    a = 1 / (1 + eta_sub * F.new_tensor(subplots))
                    T = F.new_zeros(len(whole_plots)).index_add(0, owner, F.new_tensor(subplots)*a)
                    scale = 1 / (1 + eta*T)
                    M, (_, totals) = _nested_information(F, ids, plot_ids, owner, a, eta_sub*a, eta*scale)
                    u = scale @ totals
                    C = M - torch.outer(u, u / (scale @ T))
                    if effect is not None:
                        whole = torch.stack([(v == v[0]).all(0) for v in S.split(whole_plots)]).all(0)
                        sub = torch.stack([(v == v[0]).all(0) for v in S.split(subplots)]).all(0) & ~whole
                        stratum = torch.where(whole, 0, torch.where(sub, 1, 2))
                        df = torch.where(whole, len(whole_plots)-whole.sum(),
                             torch.where(sub, len(subplots)-len(whole_plots)-sub.sum(), n-len(subplots)-(~(whole|sub)).sum()))
                else:
                    counts = F.new_tensor(whole_plots)
                    inverse_scale = ((counts + 1 / eta).reciprocal() / eta if eta >= 1
                                     else (1 + eta * counts).reciprocal())
                    if blocks is not None:
                        boundaries = torch.tensor(whole_plots).cumsum(0).tolist()
                        if (not _valid_plot_sizes(blocks, n)
                                or any(b not in boundaries for b in torch.tensor(blocks).cumsum(0).tolist())):
                            raise ValueError("blocks must sum to len(ids) with boundaries at whole-plot boundaries")
                        M, _ = _split_information(F, ids, whole_plots, inverse_scale / counts, blocks)
                        C = M
                        R = R - R.mean(0)
                    else:
                        M, sums = _split_information(F, ids, whole_plots, inverse_scale / counts)
                        weighted_sum = inverse_scale @ sums
                        C = M - torch.outer(weighted_sum, weighted_sum / (inverse_scale @ counts))
                    if effect is not None:
                        whole = torch.stack([(plot == plot[0]).all(0) for plot in S.split(whole_plots)]).all(0)
                        whole_count = whole.sum()
                        df = torch.where(whole, len(whole_plots) - whole_count - (len(blocks) if blocks is not None else 0),
                                         n - len(whole_plots) - (p - whole_count))
        if df is not None and (df < 1).any():
            raise ValueError("structured power requires positive residual degrees of freedom")
        L = torch.linalg.cholesky(M)

    columns = (S != S[0]).any(dim=0).nonzero().flatten().tolist()
    if structured:
        C = C[columns][:, columns]
        scale = C.diagonal().rsqrt()
        correlation = scale[:, None] * C * scale
    elif not centered:
        columns = list(range(p))
        scale = M.diagonal().rsqrt()
        correlation = scale[:, None] * M * scale
    else:
        X = S[:, columns]
        X = torch.nn.functional.normalize(X - X.mean(dim=0), dim=0, eps=0)
        correlation = X.mT @ X if len(columns) != 1 else X.new_ones((1, 1))
    vif = (torch.cholesky_inverse(torch.linalg.cholesky(correlation)).diagonal() if centered
           else M.diagonal() * torch.cholesky_inverse(L).diagonal())
    solved = torch.cholesky_solve(R.mT, L)
    spv = n * torch.einsum("ij,ji->i", R, solved)
    result: dict[str, torch.Tensor | list[int]] = {
        "columns": columns, "correlation": correlation, "vif": vif, "spv": spv, "fds": spv.sort().values,
    }
    if alias is not None:
        result["alias"] = torch.cholesky_solve(S.mT @ alias[ids], L)
    if effect is not None:
        delta = torch.as_tensor(effect, dtype=torch.float64, device=F.device).broadcast_to((p,))
        inverse = torch.cholesky_inverse(L)
        noncentrality = delta.square() / inverse.diagonal()
        if df is None:
            result["power"] = _coefficient_power(noncentrality, n - p, alpha).to(F.dtype)
        else:
            power = F.new_empty(p)
            for dof in df.unique().tolist():
                mask = df == dof
                power[mask] = _coefficient_power(noncentrality[mask], dof, alpha).to(F.dtype)
            result.update(power=power, df=df)
        if groups is not None:
            group_power, group_df = [], []
            for g in groups:
                strata = stratum if subplots is not None else whole if whole_plots is not None else None
                if strata is not None and not (strata[g] == strata[g[0]]).all():
                    raise ValueError("group contains mixed strata")
                dof = n - p if df is None else int(df[g[0]])
                covariance = inverse[g][:, g].to(delta.dtype)
                lam = (delta[g] @ torch.cholesky_solve(delta[g, None], torch.linalg.cholesky(covariance))).reshape(1)
                # Use the existing scalar noncentrality for byte-identical one-term power.
                if len(g) == 1:
                    lam = noncentrality[g]
                group_power.append(_coefficient_power(lam, dof, alpha, len(g))[0])
                group_df.append(dof)
            result["group_power"] = torch.stack(group_power).to(F.dtype)
            if df is not None:
                result["group_df"] = ids.new_tensor(group_df)
    return result


def _glm_fit(X: torch.Tensor, Y: torch.Tensor, family: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched R-style IRLS, freezing each row at its first stopping event."""
    eps = torch.finfo(X.dtype).eps
    mu = (Y + 0.5) / 2 if family == "binomial" else Y + 0.1
    eta = torch.logit(mu) if family == "binomial" else mu.log()
    def deviance(m: torch.Tensor) -> torch.Tensor:
        return (-2 * (torch.xlogy(Y, m) + torch.xlogy(1-Y, 1-m)).sum(1) if family == "binomial"
                else 2 * (torch.xlogy(Y, Y/m) - Y + m).sum(1))
    previous = deviance(mu)
    active = torch.ones(len(Y), dtype=torch.bool, device=X.device)
    converged = torch.zeros_like(active)
    beta = X.new_zeros((len(Y), X.shape[1]))
    covariance = torch.eye(X.shape[1], dtype=X.dtype, device=X.device).expand(len(Y), -1, -1).clone()
    for _ in range(25):
        if not active.any():
            break
        index = active.nonzero().flatten()
        e, m, y = eta[index], mu[index], Y[index]
        t = e.clamp(-30, 30).exp()
        derivative = (torch.where(e.abs() > 30, eps, t/(1+t).square()) if family == "binomial"
                      else e.exp().clamp_min(eps))
        variance = m*(1-m) if family == "binomial" else m
        weights = derivative.square()/variance
        z = e + (y-m)/derivative
        A = X.T @ (weights[:, :, None]*X)
        L, info = torch.linalg.cholesky_ex(A)
        good = info == 0
        active[index[~good]] = False
        index, L = index[good], L[good]
        fitted = torch.cholesky_solve(((weights[good]*z[good]) @ X).unsqueeze(-1), L).squeeze(-1)
        new_eta = fitted @ X.T
        t = torch.where(new_eta < -30, eps, torch.where(new_eta > 30, 1/eps, new_eta.exp()))
        new_mu = t/(1+t) if family == "binomial" else new_eta.exp().clamp_min(eps)
        mu[index] = new_mu
        D = deviance(mu)[index]
        finite = torch.isfinite(fitted).all(1) & torch.isfinite(D)
        done = finite & ((D-previous[index]).abs()/(D.abs()+0.1) < 1e-8)
        beta[index], covariance[index], eta[index] = fitted, torch.cholesky_inverse(L), new_eta
        converged[index] = done
        active[index[done | ~finite]] = False
        previous[index] = D
    return beta, covariance, eta, converged


def _glm_rejections(X: torch.Tensor, Y: torch.Tensor, family: str, alpha: float,
                    groups: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    beta, covariance, eta, converged = _glm_fit(X, Y, family)
    z = beta / covariance.diagonal(dim1=-2, dim2=-1).sqrt()
    reject = (torch.erfc(z.abs()/2**0.5) < alpha) & converged[:, None]
    group_reject = []
    for g in groups:
        if len(g) == 1:
            group_reject.append(reject[:, g[0]])
        else:
            C = covariance[:, g][:, :, g]
            # Failed rows reject nothing and need no covariance factorization.
            C = torch.where(converged[:, None, None], C, torch.eye(len(g), device=X.device, dtype=X.dtype))
            b = torch.where(converged[:, None], beta[:, g], 0)
            W = (b * torch.cholesky_solve(b.unsqueeze(-1), torch.linalg.cholesky(C)).squeeze(-1)).sum(1)
            group_reject.append((torch.special.gammaincc(X.new_tensor(len(g)/2), W/2) < alpha) & converged)
    grouped = torch.stack(group_reject, 1) if groups else reject[:, :0]
    separated = converged & (eta.abs().amax(1) > 10 if family == "binomial" else eta.amin(1) < -10)
    return reject, grouped, converged, separated


def glm_power(
    X: torch.Tensor, beta: torch.Tensor, family: str, *, alpha: float = 0.05,
    nsim: int = 1000, groups: list[list[int]] | None = None,
    generator: torch.Generator | None = None, max_elements: int = 10_000_000,
) -> dict[str, torch.Tensor | int | float | str]:
    """Monte Carlo Wald power from batched float64 logistic or Poisson IRLS.

    Counts all simulations in the denominator; non-converged fits reject nothing.
    Separation is a reporting heuristic. See P2_GATE §111 for the fit convention.
    """
    if (not isinstance(X, torch.Tensor) or X.ndim != 2 or X.dtype != torch.float64
            or X.shape[0] < X.shape[1] or X.shape[1] < 1 or not torch.isfinite(X).all()
            or not isinstance(beta, torch.Tensor) or beta.shape != (X.shape[1],)
            or beta.dtype != X.dtype or beta.device != X.device or not torch.isfinite(beta).all()
            or family not in ("binomial", "poisson") or type(alpha) not in (int, float)
            or not 0 < alpha < 1 or type(nsim) is not int or nsim < 1
            or type(max_elements) is not int or nsim*(X.shape[0]+X.shape[1]**2) > max_elements
            or (generator is not None and (not isinstance(generator, torch.Generator) or generator.device != X.device))):
        raise ValueError("invalid GLM matrix, coefficients, family, simulation count, level, generator or workspace")
    if groups is not None and (type(groups) is not list or any(type(g) is not list or not g
            or any(type(j) is not int or not 0 <= j < X.shape[1] for j in g)
            or len(set(g)) != len(g) for g in groups)):
        raise ValueError("groups must contain non-empty lists of distinct column indices")
    mean = (X @ beta).sigmoid() if family == "binomial" else (X @ beta).exp()
    Y = (torch.bernoulli(mean.expand(nsim, -1), generator=generator) if family == "binomial"
         else torch.poisson(mean.expand(nsim, -1), generator=generator))
    reject, grouped, converged, separated = _glm_rejections(X, Y, family, alpha, groups or [])
    power = reject.double().mean(0)
    result: dict[str, torch.Tensor | int | float | str] = dict(power=power, mc_se=(power*(1-power)/nsim).sqrt(),
        nonconverged=int((~converged).sum()), separated=int(separated.sum()), nsim=nsim, alpha=alpha, family=family)
    if groups is not None:
        gp = grouped.double().mean(0)
        result.update(group_power=gp, group_mc_se=(gp*(1-gp)/nsim).sqrt())
    return result


def compare_designs(
    F: torch.Tensor, designs: list[torch.Tensor | int], *, region: torch.Tensor | None = None,
    effect: float | torch.Tensor = 1.0, alpha: float = 0.05,
    quantiles: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0), criterion: str = "D",
    generator: torch.Generator | None = None,
) -> dict[str, object]:
    """Compare total-information D/A/I/G efficiencies, power and unscaled FDS.

    Integer entries request optimal designs in order, sharing the generator.
    The first design is the reference; efficiencies above one are better.
    """
    from .d_optimal_design import optimal_design

    if (type(designs) is not list or len(designs) < 2
            or any(type(d) is not int and not (isinstance(d, torch.Tensor)
                   and d.ndim == 1 and d.dtype == torch.int64) for d in designs)):
        raise ValueError("designs must be a list of at least two int64 ID tensors or Python ints")
    _features(F, 1_000_000)
    if F.ndim != 2:
        raise ValueError("compare_designs requires unstructured 2-D features")
    F = F.to(torch.float64)
    R = F if region is None else region.to(dtype=F.dtype, device=F.device)
    if R.ndim != 2 or R.shape[1] != F.shape[1] or not torch.isfinite(R).all() or not (R != 0).any():
        raise ValueError("region must be finite, match feature columns and contain a nonzero row")
    q = torch.as_tensor(quantiles, dtype=F.dtype, device=F.device)
    ids, sizes, power, fds, percentiles, metrics = [], [], [], [], [], []
    for entry in designs:
        selected = (optimal_design(F, entry, criterion=criterion, generator=generator)["ids"]
                    if type(entry) is int else entry)
        X = _exact_rows(F, selected, allow_repeats=True)
        n = len(selected)
        # Local event IDs preserve repeated candidate rows without changing diagnostics.
        diag = design_diagnostics(X, torch.arange(n, device=F.device), region=R,
                                  effect=effect, alpha=alpha)
        L = torch.linalg.cholesky(X.mT @ X)
        v = diag["spv"] / n
        metrics.append(torch.stack((2 * L.diagonal().log().sum(),
                                    torch.cholesky_inverse(L).diagonal().sum(), v.mean(), v.max())))
        ids.append(selected)
        sizes.append(n)
        power.append(diag["power"])
        fds.append(v.sort().values)
        percentiles.append(torch.quantile(v, q))
    values = torch.stack(metrics)
    efficiency = {"D": ((values[:, 0] - values[0, 0]) / F.shape[1]).exp()}
    efficiency.update({name: values[0, j] / values[:, j] for j, name in enumerate(("A", "I", "G"), 1)})
    return dict(ids=ids, n=sizes, efficiency=efficiency, power=torch.stack(power),
                fds_quantiles=torch.stack(percentiles), fds=fds)

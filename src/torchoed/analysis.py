"""Float64 least squares (§123–§124) and random-intercept REML (§136–§138)."""
from collections.abc import Callable
import torch


def _f_survival(f, d1, d2):
    # The same log-space U/T recurrences as information._coefficient_power.
    a0, b0, b = (0.5 if d1 % 2 else 1.), (0.5 if d2 % 2 else 1.), d2 / 2
    positive = f.clamp_min(0)
    x = torch.where(positive.isinf(), 1., d1 * positive / (d1 * positive + d2))
    central = ((2 / torch.pi) * x.sqrt().asin() if b0 == .5 else x.sqrt()) if a0 == .5 else (1 - (1 - x).sqrt() if b0 == .5 else x)
    u = torch.arange(b0, b, dtype=f.dtype, device=f.device)
    t = torch.arange(a0, d1 / 2, dtype=f.dtype, device=f.device)
    log_u = (u + a0).lgamma() - (u + 1).lgamma() - f.new_tensor(a0).lgamma()
    log_t = (t + b).lgamma() - (t + 1).lgamma() - f.new_tensor(b).lgamma()
    p = 1 - central - (log_u + a0 * x[..., None].log() + u * (-x[..., None]).log1p()).exp().sum(-1)
    p = p + (log_t + t * x[..., None].log() + b * (-x[..., None]).log1p()).exp().sum(-1)
    return torch.where(f <= 0, 1., torch.where(f.isinf(), 0., p))


def fit_least_squares(X: torch.Tensor, Y: torch.Tensor, *, terms: tuple[int, ...] | None = None,
                      intercept: bool = True) -> dict[str, torch.Tensor | int | dict[str, torch.Tensor | int] | None]:
    """Fit full-rank OLS; return coefficients, type I ANOVA, LOF and diagnostics.

    X must be float64, n > p. Y is (n,) or (n,J) on X's device.
    With intercept=True the first term must be the constant column alone.
    P-values use finite beta recurrences: absolute accuracy degrades at large df.
    """
    if X.ndim != 2 or X.dtype != torch.float64:
        raise ValueError('X must be 2-D float64')
    n, p = X.shape
    if Y.ndim not in (1, 2) or Y.shape[0] != n or Y.dtype != X.dtype or Y.device != X.device or n <= p:
        raise ValueError('Y must match X in rows, dtype and device; require n > p')
    terms = (1,) * p if terms is None else terms
    if not isinstance(terms, tuple) or any(type(c) is not int or c <= 0 for c in terms) or sum(terms) != p:
        raise ValueError('terms must be positive ints summing to p')
    if torch.linalg.matrix_rank(X) != p:
        raise ValueError('X must have full column rank')
    single = Y.ndim == 1
    y = Y[:, None] if single else Y
    Q, R = torch.linalg.qr(X, mode='reduced')
    z = Q.mT @ y
    coef = torch.linalg.solve_triangular(R, z, upper=True)
    e = y - Q @ z
    rss, nu = e.square().sum(0), n - p
    sigma2 = rss / nu
    ri = torch.linalg.solve_triangular(R, torch.eye(p, dtype=X.dtype, device=X.device), upper=True)
    se = ri.square().sum(1).sqrt()[:, None] * sigma2.sqrt()
    stat = coef / se
    h = Q.square().sum(1)
    h = torch.where(h >= 1 - 10 * torch.finfo(X.dtype).eps, 1., h)
    student = e / (sigma2[None, :] * (1 - h[:, None])).sqrt()
    student = student.masked_fill(h[:, None] == 1, torch.nan)
    cooks = student.square() * h[:, None] / (p * (1 - h[:, None]))
    df = torch.tensor(terms, device=X.device)
    ss = torch.stack([v.square().sum(0) for v in z.split(terms)])
    if intercept:
        df, ss = df[1:], ss[1:]
    f = ss / df[:, None] / sigma2
    prob = torch.stack([_f_survival(row, int(d), nu) for row, d in zip(f, df)]) if len(df) else f.clone()
    def out(v):
        return v.squeeze(-1) if single else v
    anova = dict(df=df, ss=out(ss), f=out(f), p=out(prob), ss_resid=out(rss))
    _, groups, counts = torch.unique(X, dim=0, return_inverse=True, return_counts=True)
    G = len(counts)
    lof = None
    if G > p and n > G:
        means = y.new_zeros((G, y.shape[1])).index_add_(0, groups, y) / counts[:, None]
        pe = (y - means[groups]).square().sum(0)
        sl = rss - pe
        fl = sl / (G - p) / (pe / (n - G))
        lof = dict(df_lof=G-p, ss_lof=out(sl), df_pe=n-G, ss_pe=out(pe), f=out(fl), p=out(_f_survival(fl, G-p, n-G)))
    return dict(coef=out(coef), se=out(se), t=out(stat), p_t=out(_f_survival(stat.square(), 1, nu)),
                sigma2=out(sigma2), df_resid=nu, qr_r=R, anova=anova, lack_of_fit=lof,
                residuals=out(e), leverage=h, studentized=out(student), cooks=out(cooks))


def fit_reml(X: torch.Tensor, y: torch.Tensor, groups: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor | int]:
    """Dense CPU float64 REML for 1–3 identifiable random intercepts (§136–§138).

    Return coef, se, Satterthwaite df, t, variance (groups then residual),
    loglik and iterations. Zero/boundary group variances are outside this API.
    """
    if X.ndim != 2 or X.dtype != torch.float64 or X.device.type != 'cpu':
        raise ValueError('X must be 2-D CPU float64')
    n, p = X.shape
    if y.shape != (n,) or y.dtype != X.dtype or y.device != X.device or n <= p:
        raise ValueError('y must match X in rows, dtype and device; require n > p')
    if torch.linalg.matrix_rank(X) != p:
        raise ValueError('X must have full column rank')
    if not isinstance(groups, tuple) or not 1 <= len(groups) <= 3:
        raise ValueError('groups must be a tuple of 1–3 integer label tensors')
    kernels = [torch.eye(n, dtype=X.dtype)]
    for g, labels in enumerate(groups):
        if labels.shape != (n,) or labels.device != X.device or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f'grouping {g} must contain n integer labels on X device')
        _, ids = torch.unique(labels, return_inverse=True)
        Z = torch.nn.functional.one_hot(ids).to(X.dtype)
        K = Z @ Z.T
        if Z.shape[1] == n or torch.linalg.matrix_rank(torch.cat((X, Z), 1)) == p:
            raise ValueError(f'grouping {g} is not identifiable with X')
        if any(torch.equal(K, other) for other in kernels[1:]):
            raise ValueError(f'grouping {g} duplicates a partition')
        kernels.append(K)
    K = torch.stack(kernels)

    def fit(variance):
        chol = torch.linalg.cholesky((variance[:, None, None] * K).sum(0))
        WX = torch.cholesky_solve(X, chol)
        cchol = torch.linalg.cholesky(X.T @ WX)
        C = torch.cholesky_inverse(cchol)
        beta = C @ WX.T @ y
        r = y - X @ beta
        ll = -.5 * ((n-p) * X.new_tensor(2*torch.pi).log() + 2*chol.diag().log().sum()
                    + 2*cchol.diag().log().sum() + r @ torch.cholesky_solve(r[:, None], chol)[:, 0])
        return ll, beta, C, WX, chol

    def loss(phi):
        return -fit(phi.exp())[0]

    resid = y - X @ torch.linalg.lstsq(X, y).solution
    phi = (resid.square().sum() / (n-p)).log().repeat(len(K)).detach()
    last_step = float('inf')
    for iteration in range(1, 201):
        phi.requires_grad_()
        value = loss(phi)
        grad = torch.autograd.grad(value, phi)[0].detach()
        norm = float(grad.abs().max())
        if norm <= 1e-11 * max(1, n-p) and last_step <= 1e-10:
            break
        hess = torch.autograd.functional.hessian(loss, phi).detach()
        hchol, info = torch.linalg.cholesky_ex(hess)
        if info != 0:
            _, _, C, WX, chol = fit(phi.detach().exp())
            P = torch.cholesky_inverse(chol) - WX @ C @ WX.T
            PK = P @ K
            fisher = .5 * torch.einsum('kij,lji->kl', PK, PK)
            hchol = torch.linalg.cholesky(phi.detach().exp()[:, None] * fisher * phi.detach().exp()[None, :])
        step = torch.cholesky_solve(grad[:, None], hchol)[:, 0]
        step = step * (5. / max(5., float(step.abs().max())))
        phi = phi.detach()
        for _ in range(50):
            trial = phi - step
            try:
                new_value = loss(trial)
                accepted = bool(torch.isfinite(new_value) & (new_value <= value.detach() + 1e-13*value.detach().abs()))
            except torch.linalg.LinAlgError:
                accepted = False
            if accepted:
                phi, last_step = trial, float(step.abs().max())
                boundary = (phi[1:] - phi[0]).exp() < 1e-10
                if boundary.any():
                    raise ValueError(f'boundary estimate for grouping {int(boundary.nonzero()[0])}')
                break
            step = step / 2
        else:
            if norm <= 1e-8 * max(1, n-p):
                break
            raise RuntimeError(f'REML line search failed; last gradient norm {norm}')
    else:
        raise RuntimeError(f'REML exceeded 200 iterations; last gradient norm {norm}')
    variance = phi.detach().exp()
    ll, beta, C, WX, chol = fit(variance)
    P = torch.cholesky_inverse(chol) - WX @ C @ WX.T
    PK, u = P @ K, P @ y
    Ku = K @ u
    H = -.5 * torch.einsum('kij,lji->kl', PK, PK) + Ku @ P @ Ku.T
    T = WX @ C
    g = torch.einsum('ij,kil,lj->jk', T, K, T)
    se = C.diag().sqrt()
    df = 2*C.diag().square() / (g * torch.linalg.solve(H, g.T).T).sum(1)
    return dict(coef=beta, se=se, df=df, t=beta/se, variance=torch.cat((variance[1:], variance[:1])),
                loglik=ll, iterations=iteration)


def predict_least_squares(fit: dict[str, torch.Tensor | int], X0: torch.Tensor, *, level: float = 0.95) -> dict[str, torch.Tensor]:
    """Single-response unweighted OLS predictions and intervals (section 139)."""
    if not 0.5 <= level < 1:
        raise ValueError('require 0.5 <= level < 1')
    if fit['coef'].ndim != 1:
        raise ValueError('require a single-response fit')
    w = torch.linalg.solve_triangular(fit['qr_r'].mT, X0.mT, upper=False)
    mean, se = X0 @ fit['coef'], (fit['sigma2'] * w.square().sum(0)).sqrt()
    lo, hi = X0.new_tensor(0.), X0.new_tensor(1.)
    while _f_survival(hi.square(), 1, fit['df_resid']) > 1-level:
        hi = 2*hi
    for _ in range(200):
        mid = (lo+hi)/2
        if _f_survival(mid.square(), 1, fit['df_resid']) > 1-level:
            lo = mid
        else:
            hi = mid
    t = (lo+hi)/2
    ci, pi = t*se, t*(se.square()+fit['sigma2']).sqrt()
    return dict(fit=mean, se=se, ci=torch.stack((mean-ci, mean+ci), -1),
                pi=torch.stack((mean-pi, mean+pi), -1), t=t)


def prediction_trace(fit: dict[str, torch.Tensor | int], rows: Callable[[torch.Tensor], torch.Tensor], f0: torch.Tensor, j: int, grid: torch.Tensor, *, level: float = 0.95) -> dict[str, torch.Tensor]:
    """Vary factor j on grid, holding the other factors at f0."""
    F = f0.repeat(len(grid), 1)
    F[:, j] = grid
    return predict_least_squares(fit, rows(F), level=level)

"""Float64 least squares and sequential tests (gate §123–§124)."""
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


def fit_least_squares(X, Y, *, terms=None, intercept=True):
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
                sigma2=out(sigma2), df_resid=nu, anova=anova, lack_of_fit=lof,
                residuals=out(e), leverage=h, studentized=out(student), cooks=out(cooks))

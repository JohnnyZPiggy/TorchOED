from fractions import Fraction as F
from itertools import product
import math
import pytest
import torch
from torchoed import fit_least_squares
from torchoed.analysis import _f_survival


def exact_fit(x, y):
    n, p = len(x), len(x[0])
    a = [[sum(xi[i]*xi[j] for xi in x) for j in range(p)] + [F(i == j) for j in range(p)] for i in range(p)]
    for i in range(p):
        a[i] = [v/a[i][i] for v in a[i]]
        for j in range(p):
            if i != j:
                c = a[j][i]
                a[j] = [v-c*w for v, w in zip(a[j], a[i])]
    inv = [r[p:] for r in a]
    beta = [sum(inv[i][j]*sum(xi[j]*yi for xi, yi in zip(x, y)) for j in range(p)) for i in range(p)]
    rss = sum((yi-sum(v*b for v, b in zip(xi, beta)))**2 for xi, yi in zip(x, y))
    h = [sum(xi[i]*inv[i][j]*xi[j] for i in range(p) for j in range(p)) for xi in x]
    return beta, rss, h


def close(a, b):
    torch.testing.assert_close(torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64), rtol=1e-12, atol=1e-12)


def test_exact_oracle():
    rows = list(product(range(3), range(2), range(3)))
    x = [[F(v) for v in (1, a == 1, a == 2, b)] for a, b, r in rows]
    ys = [[F(2+a*a+3*b+a*b, 3)+F(r-1, 4+j) for a, b, r in rows] for j in range(2)]
    X, Y = torch.tensor(x, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64).T
    result = fit_least_squares(X, Y, terms=(1, 2, 1))
    for j, y in enumerate(ys):
        beta, rss, h = exact_fit(x, y)
        ss = [exact_fit([r[:c] for r in x], y)[1] for c in (1, 3, 4)]
        pe = sum(sum((y[i]-sum(y[k] for k in range(g, g+3))/3)**2 for i in range(g, g+3)) for g in range(0, 18, 3))
        close(result['coef'][:, j], list(map(float, beta)))
        close(result['anova']['ss'][:, j], [float(ss[0]-ss[1]), float(ss[1]-ss[2])])
        close(result['anova']['ss_resid'][j], float(rss))
        close(result['lack_of_fit']['ss_pe'][j], float(pe))
        close(result['lack_of_fit']['ss_lof'][j], float(rss-pe))
        close(result['leverage'], list(map(float, h)))
        separate = fit_least_squares(X, Y[:, j], terms=(1, 2, 1))
        for key, value in separate.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    close(result[key][k] if k.startswith('df') else result[key][k][..., j], v)
            else:
                close(result[key] if key in ('leverage', 'df_resid') else result[key][..., j], value)
    assert result['df_resid'] == 14 and result['anova']['df'].tolist() == [2, 1]
    assert (result['lack_of_fit']['df_lof'], result['lack_of_fit']['df_pe']) == (2, 12)


def test_p_values():
    t = torch.tensor([0., 1e-6, .5, 1., 3., 20.], dtype=torch.float64)
    close(_f_survival(t*t, 1, 1), 1-2/math.pi*t.atan())
    close(_f_survival(t*t, 1, 2), 1-t/(2+t*t).sqrt())
    for d in (3, 7, 48):
        close(_f_survival(t, 2, d), (1+2*t/d)**(-d/2))
    for d in (3, 5, 8):
        close(_f_survival(t, d, 2), 1-(d*t/(d*t+2))**(d/2))
    close(_f_survival(-t, 3, 7), torch.ones_like(t))


def test_diagnostics_boundary():
    X = torch.tensor([[1., x] for x in range(6)], dtype=torch.float64)
    y = torch.tensor([1., 3., 2., 5., 4., 8.], dtype=torch.float64)
    r = fit_least_squares(X, y)
    H = X @ torch.linalg.solve(X.T@X, X.T)
    h, e, s2 = H.diag(), y-H@y, r['sigma2']
    close(r['studentized'], e/(s2*(1-h)).sqrt())
    close(r['cooks'], e*e*h/(2*s2*(1-h)**2))
    X[:, 1] = torch.tensor([1., 0., 0., 0., 0., 0.])
    r = fit_least_squares(X, y)
    assert r['leverage'][0] == 1 and r['studentized'][0].isnan() and r['cooks'][0].isnan()
    with pytest.raises(ValueError, match='rank'):
        fit_least_squares(torch.ones_like(X), y)

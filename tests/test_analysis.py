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
                close(result[key] if key in ('leverage', 'df_resid', 'qr_r') else result[key][..., j], value)
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


def test_reml_fraction_ems():
    import csv
    from pathlib import Path
    from torchoed import fit_reml
    root = Path(__file__).resolve().parents[1] / 'reports/w17-reml-check'
    for name, crossed in (('Oats', False), ('strip', True)):
        rows = list(csv.DictReader((root / f'{name}.csv').open()))
        bkey, akey, ckey, ykey = ('block', 'A', 'B', 'y') if crossed else ('Block', 'Variety', 'nitro', 'yield')
        b, a, c = [[row[k] for row in rows] for k in (bkey, akey, ckey)]
        y = [F(row[ykey]) for row in rows]
        B, A, D = (len(set(v)) for v in (b, a, c))
        av, cv = sorted(set(a)), sorted(set(c), key=F)
        x = [[F(1)] + [F(i == v) for v in av[1:]] + [F(j == v) for v in cv[1:]] +
             [F(i == v and j == w) for w in cv[1:] for v in av[1:]] for i, j in zip(a, c)]
        n, p = len(y), len(x[0])
        def project(w, keys):
            sums, counts = {}, {}
            for key, val in zip(keys, w):
                sums[key] = sums.get(key, F(0)) + val
                counts[key] = counts.get(key, 0) + 1
            return [sums[key] / counts[key] for key in keys]
        def strata(w):
            pb = project(w, b)
            parts = [pb, [v-z for v, z in zip(project(w, list(zip(b, a))), pb)]]
            if crossed:
                parts += [[v-z for v, z in zip(project(w, list(zip(b, c))), pb)]]
            return parts + [[v-sum(part[i] for part in parts) for i, v in enumerate(w)]]
        # Balanced saturated treatment means remove X exactly before stratum SS.
        e = [v-z for v, z in zip(y, project(y, list(zip(a, c))))]
        dfs = [B-1, (B-1)*(A-1)] + ([(B-1)*(D-1), (B-1)*(A-1)*(D-1)] if crossed else [A*(B-1)*(D-1)])
        ms = [sum(v*v for v in part)/d for part, d in zip(strata(e), dfs)]
        variance = [(ms[0]-ms[1]-(ms[2]-ms[-1] if crossed else 0))/(A*D), (ms[1]-ms[-1])/D]
        variance += ([(ms[2]-ms[-1])/A] if crossed else []) + [ms[-1]]
        # Fraction Gauss–Jordan; OLS weights suffice since X is stratum invariant.
        aug = [[sum(row[i]*row[j] for row in x) for j in range(p)] + [F(i == j) for j in range(p)] for i in range(p)]
        det = F(1)
        for i in range(p):
            pivot = aug[i][i]
            det *= pivot
            aug[i] = [v/pivot for v in aug[i]]
            for j in range(p):
                if j != i:
                    scale = aug[j][i]
                    aug[j] = [v-scale*w for v, w in zip(aug[j], aug[i])]
        weights = [[sum(aug[i][p+j]*row[j] for j in range(p)) for row in x] for i in range(p)]
        beta = [sum(w*v for w, v in zip(ws, y)) for ws in weights]
        se, df = [], []
        for ws in weights:
            terms = [sum(v*v for v in part)*lam for part, lam in zip(strata(ws), ms)]
            se.append(math.sqrt(sum(terms)))
            df.append(float(sum(terms)**2 / sum(t*t/d for t, d in zip(terms, dfs))))
        ll = -.5*((n-p)*(math.log(2*math.pi)+1) + math.log(det) + sum(d*math.log(lam) for d, lam in zip(dfs, ms)))
        labels = [b, list(zip(b, a))] + ([list(zip(b, c))] if crossed else [])
        groups = tuple(torch.tensor([sorted(set(v)).index(key)*7+3 for key in v]) for v in labels)
        out = fit_reml(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), groups)
        expected = dict(coef=list(map(float, beta)), se=se, df=df, variance=list(map(float, variance)), loglik=ll)
        errors = {}
        for key, value in expected.items():
            ref = torch.as_tensor(value, dtype=torch.float64)
            errors[key] = float(((out[key]-ref).abs()/ref.abs()).max())
            torch.testing.assert_close(out[key], ref, rtol=1e-10, atol=0)
        close(out['t'], out['coef']/out['se'])
        print(name, 'exact relative errors', errors, 'iterations', out['iterations'])
        # Frozen R is external, under W18's measured reference rule, never refit.
        r = {}
        for row in csv.DictReader((root / 'r.csv').open()):
            if row['case'] == name:
                r.setdefault(row['group'], []).append(float(row['value']))
        order = [2, 1, 0, 3] if crossed else [1, 0, 2]
        r['variance'] = [r['variance'][i] for i in order]
        for key, rkey in (('coef', 'fixed'), ('se', 'SE'), ('df', 'df'), ('variance', 'variance'), ('loglik', 'logLik')):
            ref = torch.tensor(r[rkey], dtype=torch.float64).reshape(out[key].shape)
            print(name, 'R relative', key, float(((out[key]-ref).abs()/ref.abs()).max()))
            torch.testing.assert_close(out[key], ref, rtol=1e-8 if key in ('coef', 'loglik') else 1e-4, atol=0)


def test_reml_unbalanced_score():
    from torchoed import fit_reml
    # Seed 17; all fitted group/residual ratios > .1 (interior).
    rng = torch.Generator().manual_seed(17)
    labels = torch.tensor(list(product(range(5), range(3), range(3))))
    b, a, c = labels.T
    X = torch.stack([torch.ones(len(b))] + [(a == i).double() for i in (1, 2)] +
                    [(c == j).double() for j in (1, 2)] + [(a == i).double()*(c == j) for j in (1, 2) for i in (1, 2)], 1).double()
    y = X @ torch.arange(1., 10., dtype=torch.float64) + 3*torch.randn(5, generator=rng, dtype=torch.float64)[b]
    y += 2*torch.randn(15, generator=rng, dtype=torch.float64)[3*b+a] + .5*torch.randn(45, generator=rng, dtype=torch.float64)
    X, y, groups = X[1:-1], y[1:-1], (b[1:-1], (3*b+a)[1:-1])
    out = fit_reml(X, y, groups)
    assert (out['variance'][:-1]/out['variance'][-1] > .1).all()
    K = torch.stack([torch.eye(len(y), dtype=torch.float64)] + [(g[:, None] == g).double() for g in groups])
    variance = torch.cat((out['variance'][-1:], out['variance'][:-1]))
    def dense(v):
        V = (v[:, None, None]*K).sum(0)
        W = torch.linalg.inv(V)
        C = torch.linalg.inv(X.T @ W @ X)
        P = W - W @ X @ C @ X.T @ W
        ll = -.5*((len(y)-X.shape[1])*math.log(2*math.pi) + torch.linalg.slogdet(V)[1] +
                  torch.linalg.slogdet(X.T @ W @ X)[1] + y @ P @ y)
        return ll, P
    ll, P = dense(variance)
    trace = torch.einsum('ij,kji->k', P, K)
    quadratic = torch.einsum('i,kij,j->k', P@y, K, P@y)
    assert float((.5*(trace-quadratic)).abs().max()/quadratic.norm()) <= 1e-8
    for k in range(len(K)):
        for factor in (.99, 1.01):
            trial = variance.clone()
            trial[k] *= factor
            assert dense(trial)[0] < ll
    print('unbalanced variance ratios', out['variance'][:-1]/out['variance'][-1], 'score', .5*(quadratic-trace))


def test_reml_unidentifiable():
    from torchoed import fit_reml
    X = torch.ones((8, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match='grouping 0.*identifiable'):
        fit_reml(X, torch.arange(8, dtype=torch.float64), (torch.zeros(8, dtype=torch.int64),))


def test_prediction_dense():
    from torchoed import predict_least_squares
    g = torch.Generator().manual_seed(3)
    X, y = torch.randn(12, 3, generator=g, dtype=torch.float64), torch.randn(12, generator=g, dtype=torch.float64)
    fit = fit_least_squares(X, y, intercept=False)
    out = predict_least_squares(fit, X)
    torch.testing.assert_close(out['se'].square()/fit['sigma2'], (X @ torch.linalg.inv(X.T@X)*X).sum(1), rtol=1e-12, atol=0)
    torch.testing.assert_close(out['se'].square()/fit['sigma2'], fit['leverage'], rtol=1e-12, atol=0)
    torch.testing.assert_close(((out['pi'][:, 1]-out['pi'][:, 0])/2).square()-((out['ci'][:, 1]-out['ci'][:, 0])/2).square(), (out['t'].square()*fit['sigma2']).expand(len(X)))


def test_prediction_quantile():
    from torchoed import predict_least_squares
    for nu, level in product((1, 2), (.5, .9, .95, .99)):
        fit = fit_least_squares(torch.ones(nu+1, 1, dtype=torch.float64), torch.arange(nu+1, dtype=torch.float64))
        t = predict_least_squares(fit, torch.ones(1, 1, dtype=torch.float64), level=level)['t']
        a = 1-level
        expected = math.tan(math.pi*level/2) if nu == 1 else math.sqrt(2/(a*(2-a))-2)
        assert math.isclose(float(t), expected, rel_tol=1e-12)
    with pytest.raises(ValueError):
        predict_least_squares(fit, torch.ones(1, 1, dtype=torch.float64), level=.4)


def test_prediction_trace():
    from torchoed import predict_least_squares, prediction_trace
    X = torch.tensor([[1., -1.], [1., 0.], [1., 1.]], dtype=torch.float64)
    fit = fit_least_squares(X, torch.tensor([0., 2., 3.], dtype=torch.float64))
    actual = prediction_trace(fit, lambda f: f, X[1], 1, X[:, 1])
    for key, value in predict_least_squares(fit, X).items():
        torch.testing.assert_close(actual[key], value)


def test_desirability_fraction():
    from torchoed import d_max, d_min, d_target, d_overall
    low, target, high = F(-1), F(1, 3), F(2)
    points = sorted({v+d for v in (low, target, high) for d in (F(-1, 10), F(0), F(1, 10))} | {F(0), F(1)})
    def exact(y):
        a = F(0) if y < low else F(1) if y > high else ((y-low)/(high-low))**2
        b = F(1) if y < low else F(0) if y > high else ((y-high)/(low-high))**3
        c = F(0) if y < low or y > high else ((y-low)/(target-low))**2 if y <= target else ((y-high)/(target-high))**3
        return a, b, c
    y = torch.tensor([float(v) for v in points], dtype=torch.float64)
    refs = torch.tensor([[float(v) for v in exact(p)] for p in points], dtype=torch.float64).T
    actual = (d_max(y, -1, 2, 2), d_min(y, -1, 2, 3), d_target(y, -1, 1/3, 2, 2, 3))
    torch.testing.assert_close(torch.stack(actual), refs, rtol=0, atol=1e-15)
    overall = torch.tensor([float(math.prod(exact(p)))**(1/3) for p in points], dtype=torch.float64)
    torch.testing.assert_close(d_overall(*actual), overall, rtol=0, atol=1e-15)
    with pytest.raises(ValueError):
        d_max(y, 1, 1)


def test_desirability_closed_form():
    from torchoed import d_max, d_min, d_overall, maximize_desirability
    lo, hi = torch.tensor([-1.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)
    for objective, x, value in ((lambda f: d_overall(d_max(f[:, 0], -1, 1), d_min(f[:, 0], -1, 1)), 0., .5),
                                (lambda f: d_max(f[:, 0], -1, 1), 1., 1.)):
        out = maximize_desirability(objective, lo, hi)
        assert abs(float(out['value'])-value) <= 1e-8
        assert abs(float(out['x'][0])-x) <= 1e-5

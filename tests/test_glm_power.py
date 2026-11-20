import csv
import itertools
import json
from pathlib import Path

import torch
from torchoed import glm_power
from torchoed.information import _glm_fit, _glm_rejections

torch.set_num_threads(1)


def cases():
    f = torch.tensor([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=torch.float64).repeat(4, 1)
    B = torch.column_stack((torch.ones(16), f))
    P = torch.tensor([[1, -1], [1, -1], [1, 1], [1, 1]], dtype=torch.float64)
    return [(B, B.new_tensor([0, 1, .5]), 'binomial'), (P, P.new_tensor([.5, .5]), 'poisson')]


def test_exact_oracle():
    for X, beta, family in cases():
        Y = X.new_tensor(list(itertools.product(range(2 if family == 'binomial' else 21), repeat=len(X))))
        mu = (X @ beta).sigmoid() if family == 'binomial' else (X @ beta).exp()
        prob = ((Y*mu.log()+(1-Y)*(-mu).log1p()).sum(1).exp() if family == 'binomial'
                else (torch.xlogy(Y, mu)-mu-(Y+1).lgamma()).sum(1).exp())
        groups = [[1, 2], [1]] if family == 'binomial' else [[1]]
        reject, grouped, _, _ = _glm_rejections(X, Y, family, .05, groups)
        exact = prob @ torch.cat((reject, grouped), 1).double()
        sanity = X.new_tensor([.001945, .108394, .011026, .020456, .108394] if family == 'binomial' else [.211672, .067840, .067840])
        torch.testing.assert_close(exact, sanity, atol=1e-6, rtol=0)
        r = glm_power(X, beta, family, nsim=20000, groups=groups, generator=torch.Generator().manual_seed(0))
        observed = torch.cat((r['power'], r['group_power']))
        assert ((observed-exact).abs() <= 4*(exact*(1-exact)/20000).sqrt()+3.9e-12).all()
        assert r['group_power'][-1] == r['power'][1]
        print(f'{family} exact={exact.tolist()} MC={observed.tolist()}', flush=True)


def test_independent_newton():
    for X, beta, family in cases():
        mu = (X @ beta).sigmoid() if family == 'binomial' else (X @ beta).exp()
        draw = torch.bernoulli if family == 'binomial' else torch.poisson
        Y = draw(mu.expand(200, -1), generator=torch.Generator().manual_seed(0))
        fitted, _, eta, ok = _glm_fit(X, Y, family)
        separated = eta.abs().amax(1) > 10 if family == 'binomial' else eta.amin(1) < -10
        for y, expected in zip(Y[ok & ~separated], fitted[ok & ~separated]):
            b = torch.zeros_like(beta)
            for _ in range(50):
                m = (X @ b).sigmoid() if family == 'binomial' else (X @ b).exp()
                w = m*(1-m) if family == 'binomial' else m
                step = torch.linalg.solve(X.T @ (w[:, None]*X), X.T @ (y-m))
                b += step
                if step.abs().max() < 1e-12:
                    break
            torch.testing.assert_close(expected, b, atol=1e-6, rtol=0)


def test_skpr_references():
    root = Path(__file__).resolve().parents[1] / 'reports/w15-glmpower-ext'
    with (root/'designs.csv').open() as file:
        rows = list(csv.DictReader(file))
    ref = json.loads((root/'results.json').read_text())['cases']
    for name, beta in [('G1', [0, .5, 1, 0]), ('G2', [1, .25, .5, 0]), ('G3', [.5, 1, .5, .25])]:
        c = torch.tensor([[float(r[k]) for k in ('X1', 'X2', 'X3') if r[k]] for r in rows if r['case'] == name], dtype=torch.float64)
        X = torch.column_stack((torch.ones(len(c)), c, c[:, 0]*c[:, 1])) if name == 'G3' else torch.column_stack((torch.ones(len(c)), c))
        r = glm_power(X, X.new_tensor(beta), 'poisson' if name == 'G2' else 'binomial', nsim=10000, generator=torch.Generator().manual_seed(0))
        p = X.new_tensor([v['power'] for v in ref[name]['parameter_power']])
        z = (r['power']-p)/(r['mc_se'].square()+p*(1-p)/10000).sqrt()
        print(name, 'TorchOED', r['power'].tolist(), 'skpr', p.tolist(), 'z', z.tolist(), flush=True)
        assert (z.abs() <= 4).all(), (name, r['power'], p, z)

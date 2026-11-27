import json
from fractions import Fraction as F
from pathlib import Path

import torch
from torchoed import residual_bootstrap, fit_least_squares_batched, fit_least_squares
from test_analysis import exact_fit


def test_r_references():
    refs = json.loads((Path(__file__).parents[1] / 'reports/w19-bootstrap-refs/r.json').read_text())
    for name, case in refs.items():
        X = torch.tensor(case['model_matrix'], dtype=torch.float64)
        y = torch.tensor(case['y'], dtype=torch.float64)
        indices = torch.tensor(case['indices']) - 1
        result = residual_bootstrap(X, y, indices=indices)
        for key in ('coef', 'lower', 'upper'):
            expected = torch.tensor(case['replicates' if key == 'coef' else key], dtype=torch.float64)
            error = ((result[key] - expected).abs() / expected.abs().clamp_min(1)).max().item()
            print(name, key, 'max relative', error)
            assert error <= 1e-10


def test_batched_loop_and_fraction():
    rows = [[F(1), F(a), F(a*a)] for a in (-2, -1, 0, 1, 3)]
    y = list(map(F, (1, 0, 2, 5, 4)))
    X = torch.tensor(rows, dtype=torch.float64)
    Y = torch.tensor(y, dtype=torch.float64)
    designs = torch.stack((X, X * X.new_tensor([1, 2, 3])))
    responses = torch.stack((Y, Y + 2))
    result = fit_least_squares_batched(designs, responses)
    for d in range(2):
        single = fit_least_squares(designs[d], responses[d])
        for key in ('coef', 'se'):
            torch.testing.assert_close(result[key][d], single[key], rtol=1e-12, atol=0)
    beta, rss, _ = exact_fit(rows, y)
    # Exact normal-equation inverse diagonal from the same Fraction solver.
    inv = [exact_fit(rows, [F(i == j) for i in range(5)])[0] for j in range(5)]
    diag = [sum(col[k]**2 for col in inv) for k in range(3)]
    se = torch.tensor([float(rss / 2 * v)**.5 for v in diag], dtype=torch.float64)
    torch.testing.assert_close(result['coef'][0], torch.tensor(beta, dtype=torch.float64), rtol=1e-12, atol=0)
    torch.testing.assert_close(result['se'][0], se, rtol=1e-12, atol=0)


def test_centred_residuals_and_generator():
    X = torch.tensor([[1., a, a*a] for a in (-2, -1, 0, 1, 3)], dtype=torch.float64)
    y = X.new_tensor([1, 0, 2, 5, 4])
    first = residual_bootstrap(X, y, B=37, generator=torch.Generator().manual_seed(619))
    second = residual_bootstrap(X, y, B=37, generator=torch.Generator().manual_seed(619))
    assert first['coef'].shape == (37, 3) and first['indices'].shape == (5, 37)
    assert torch.equal(first['indices'], second['indices'])
    fit = fit_least_squares(X, y)
    r = fit['residuals'] / (1 - fit['leverage']).sqrt()
    r = r - r.mean()
    assert r.sum().abs() <= 1e-12 * r.abs().sum()
    expected = fit['coef'][:, None] + torch.linalg.solve_triangular(
        fit['qr_r'], torch.linalg.qr(X).Q.T @ r[first['indices']], upper=True)
    torch.testing.assert_close(first['coef'], expected.T, rtol=1e-12, atol=1e-12)

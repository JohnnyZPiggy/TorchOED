"""Frozen desirability 2.1 vignette comparison; run from the repository root."""
import json
from pathlib import Path
import torch
from torchoed import d_max, d_target, d_overall, maximize_desirability

torch.set_num_threads(1)
ref = json.loads(Path('reports/w18-r-refs/r.json').read_text())['desirability']


def responses(x):
    # Literal transcription of the two frozen R source strings in r.json.
    a, b, c = x.unbind(1)
    conversion = (81.09+1.0284*a+4.043*b+6.2037*c-1.8366*a*a+2.9382*b*b
                  -5.1915*c*c+2.215*a*b+11.375*a*c-3.875*b*c)
    activity = (59.85+3.583*a+.2546*b+2.2298*c+.83479*a*a+.07484*b*b
                +.05716*c*c-.3875*a*b-.375*a*c+.3125*b*c)
    return conversion, activity


def individual(conversion, activity):
    return d_max(conversion, 80, 97), d_target(activity, 55, 57.5, 60)


def overall(x):
    return d_overall(*individual(*responses(x)))


if __name__ == '__main__':
    y = torch.tensor(ref['responses'], dtype=torch.float64)
    ds = individual(*y.unbind(1))
    actual = torch.stack((*ds, d_overall(*ds)), 1)
    expected = torch.tensor([ref['individual']['conversion'], ref['individual']['activity'], ref['overall']], dtype=torch.float64).T
    error = float((actual-expected).abs().max())
    lo = torch.full((3,), -1.682, dtype=torch.float64)
    hi = -lo
    result = maximize_desirability(overall, lo, hi)
    axis = torch.linspace(-1.682, 1.682, 101, dtype=torch.float64)
    with torch.no_grad():
        grid_max = float(overall(torch.cartesian_prod(axis, axis, axis)).max())
    best = float(result['value'])
    check_a = error <= 1e-10
    check_b = best >= .942509403229837-1e-6 and best >= grid_max-1e-6
    data = dict(reference='reports/w18-r-refs/r.json; R 4.5.2, desirability 2.1; run 45936707367',
                environment=dict(torch=torch.__version__, dtype='float64', device='cpu', threads=1),
                check_a=dict(verdict='PASS' if check_a else 'FAIL', max_absolute=error, actual=actual.tolist(), expected=expected.tolist()),
                check_b=dict(verdict='PASS' if check_b else 'FAIL', point=result['x'].tolist(), value=best,
                             reference_optimum=.942509403229837, grid_101_cubed_max=grid_max,
                             within_1e6=int((result['values'] >= best-1e-6).sum()),
                             start_count=len(result['values']), values=result['values'].tolist(), iterations=result['iterations'].tolist()))
    dest = Path('reports/w18-desirability')
    (dest/'results.json').write_text(json.dumps(data, indent=2)+'\n')
    (dest/'summary.md').write_text(
        f'# Week 18 desirability — G3\n\n'
        f'- Check (a): {data["check_a"]["verdict"]}; 15 frozen-R individual/overall values, max absolute {error:.16g} (limit 1e-10).\n'
        f'- Check (b): {data["check_b"]["verdict"]}; default midpoint starts, best D = {best:.16g} at {result["x"].tolist()}.\n'
        f'- R documented D = 0.942509403229837; TorchOED 101³ grid maximum = {grid_max:.16g}; both comparisons allow 1e-6.\n'
        f'- {data["check_b"]["within_1e6"]}/{len(result["values"])} starts within 1e-6 of best; all final values and accepted-step counts in results.json.\n'
        '- Frozen conversion/activity models transcribed from `reports/w18-r-refs/r.json` (R 4.5.2, desirability 2.1, workflow 45936707367).\n'
        '- CPU float64, one thread; no timing or global-optimality claim. No new R dispatch or held-out case.\n'
        '- Reproduce from repository root: `PYTHONPATH=src python reports/w18-desirability/run.py`.\n')
    print(json.dumps(dict(check_a=check_a, check_b=check_b, best=best, grid_max=grid_max,
                          near_best=data['check_b']['within_1e6'])))
    assert check_a and check_b

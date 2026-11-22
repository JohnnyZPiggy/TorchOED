"""Re-score the downloaded R outputs; CPU float64, no reference fitting here."""
import csv
import json
import platform
from pathlib import Path
import torch
from torchoed import fit_least_squares

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)


def score(actual, reference, absolute=False):
    a = torch.as_tensor(actual, dtype=torch.float64).reshape(-1)
    values = reference if isinstance(reference, list) else [reference]
    b = torch.tensor([float('nan') if x is None else x for x in values], dtype=torch.float64)
    assert a.shape == b.shape
    finite = a.isfinite() & b.isfinite()
    assert torch.equal(a.isnan(), b.isnan()), 'nonfinite diagnostics differ'
    delta = (a[finite] - b[finite]).abs()
    if not absolute:
        delta = delta / b[finite].abs().clamp_min(1e-300)
    i = int(delta.argmax())
    indices = finite.nonzero().flatten()
    return dict(max_difference=float(delta[i]), index=int(indices[i]),
                torch=float(a[indices[i]]), R=float(b[indices[i]]),
                metric='absolute' if absolute else 'relative')


def main():
    reference = json.loads((ROOT / 'r.json').read_text(encoding='utf-8-sig'))
    runtime = json.loads((ROOT / 'runtime.json').read_text(encoding='utf-8-sig'))
    cases = {}
    for name, expected in [('warpbreaks', (50, 2, 48)), ('ChemReact', (7, 3, 4))]:
        ref = reference[name]
        if 'error' in ref:
            cases[name] = dict(verdict='NOT_RUN', error=ref['error'])
            continue
        with (ROOT / (name + '.csv')).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if name == 'warpbreaks':
            x = [[1, r['wool'] == 'B', r['tension'] == 'M', r['tension'] == 'H'] for r in rows]
            y = [float(r['breaks']) for r in rows]
            terms = (1, 1, 2)
        else:
            x = [[1, r['Block'] == 'B2', (float(r['Time'])-85)/5, (float(r['Temp'])-175)/5] for r in rows]
            x = [[a, b, c, d, c*d, c*c, d*d] for a, b, c, d in x]
            y = [float(r['Yield']) for r in rows]
            terms = (1,)*7
        fit = fit_least_squares(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), terms=terms)
        lof = fit['lack_of_fit']
        dfs = (fit['df_resid'], lof['df_lof'], lof['df_pe'])
        assert dfs == expected
        assert (ref['df_resid'], ref['lack_of_fit']['df_lof'], ref['lack_of_fit']['df_pe']) == expected
        groups = {key: score(fit[key], ref[key], key == 'p_t') for key in
                  ('coef', 'se', 't', 'p_t', 'residuals', 'leverage', 'studentized', 'cooks')}
        for section, keys in [('anova', ('df', 'ss', 'f', 'p', 'ss_resid')),
                              ('lack_of_fit', ('df_lof', 'ss_lof', 'df_pe', 'ss_pe', 'f', 'p'))]:
            for key in keys:
                groups[section + '.' + key] = score(fit[section][key], ref[section][key], key == 'p')
        cases[name] = dict(verdict='MATCH' if all(v['max_difference'] <= 1e-8 for v in groups.values()) else 'NO MATCH',
                           df=list(dfs), groups=groups)
    result = dict(run_id=runtime['run_id'], source_sha=runtime['source_sha'],
                  versions=dict(R=runtime['R_version'], rsm=runtime['packages']['rsm'],
                                python=platform.python_version(), torch=torch.__version__),
                  device='cpu', dtype='float64', tolerance=1e-8, cases=cases,
                  matched=sum(c['verdict'] == 'MATCH' for c in cases.values()), total=2)
    (ROOT / 'results.json').write_bytes((json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print('PASS: frozen df checks and shared scoring; MATCH %d / 2' % result['matched'])


if __name__ == '__main__':
    main()

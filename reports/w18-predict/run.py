"""Rebuild R-column-order models and score frozen predict.lm outputs; no R fitting."""
import csv
import json
import platform
from pathlib import Path
import torch
from torchoed import fit_least_squares, predict_least_squares

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)


def main():
    refs = json.loads((ROOT.parent/'w18-r-refs/r.json').read_text(encoding='utf-8-sig'))
    cases = {}
    for name in ('warpbreaks', 'ChemReact'):
        with (ROOT.parent/'w17-lm-published'/f'{name}.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if name == 'warpbreaks':
            X = [[1, r['wool']=='B', r['tension']=='M', r['tension']=='H'] for r in rows]
            y = [float(r['breaks']) for r in rows]
            X0 = [[1, r['wool']=='B', r['tension']=='M', r['tension']=='H'] for r in refs[name]['points']]
        else:
            factors = [(r['Block']=='B2', (float(r['Time'])-85)/5, (float(r['Temp'])-175)/5) for r in rows]
            X = [[1, b, x, z, x*x, z*z, x*z] for b, x, z in factors]
            y = [float(r['Yield']) for r in rows]
            X0 = [[1, r['Block']=='B2', r['x1'], r['x2'], r['x1']**2, r['x2']**2, r['x1']*r['x2']] for r in refs[name]['points']]
        X, y, X0 = [torch.tensor(v, dtype=torch.float64) for v in (X, y, X0)]
        out = predict_least_squares(fit_least_squares(X, y), X0)
        ref = refs[name]
        expected = torch.tensor([[c[0], se, *c[1:], *p[1:]] for c, se, p in zip(ref['confidence']['fit'], ref['confidence']['se.fit'], ref['prediction']['fit'])], dtype=torch.float64)
        actual = torch.cat((out['fit'][:, None], out['se'][:, None], out['ci'], out['pi']), 1)
        delta = (actual-expected).abs()/expected.abs().clamp_min(1e-300)
        worst = float(delta.max())
        cases[name] = dict(points=len(X0), max_relative_difference=worst, verdict='MATCH' if worst <= 1e-8 else 'NO MATCH', columns=['fit', 'se.fit', 'ci.lwr', 'ci.upr', 'pi.lwr', 'pi.upr'], actual=actual.tolist(), reference=expected.tolist())
    result = dict(reference_run=45936707367, reference='reports/w18-r-refs/r.json', R='4.5.2', rsm='2.10.6', python=platform.python_version(), torch=torch.__version__, device='CPU float64', tolerance=1e-8, cases=cases, matched=sum(c['verdict']=='MATCH' for c in cases.values()), total=2)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    text = '# W18 prediction profiler: frozen R comparison\n\n'
    text += 'R 4.5.2, rsm 2.10.6; reference workflow run [45936707367](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/45936707367). No R re-run.\n\n'
    text += 'Reproduce: `python reports/w18-predict/run.py` (CPU float64; no timing claim). Models rebuilt from W17 CSVs in R column order; ChemReact uses centre and stationary point, Block B1.\n\n'
    text += 'Shared metric: maximum relative difference over fit, se.fit and all four interval ends, denominator |R| (floor 1e-300). Target 1e-8.\n\n'
    text += '\n'.join(f"- {name}: {c['verdict']}, {c['points']} points, max relative difference {c['max_relative_difference']:.6g}." for name,c in cases.items())
    text += '\n\nPublished examples: 2/2 MATCH. H50 is reserved for week end and was not run.\n'
    (ROOT/'summary.md').write_text(text,encoding='utf-8')
    print({name:(c['verdict'],c['max_relative_difference']) for name,c in cases.items()})
    assert result['matched']==2


if __name__ == '__main__':
    main()

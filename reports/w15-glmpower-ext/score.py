import csv
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
runtime = json.loads((root / 'runtime.json').read_text())
reference = json.loads((root / 'skpr.json').read_text())
reference['RD1']['expression_not_executed'] = (root / 'rd1_expression.txt').read_text().rstrip('\n')
reference['RD1']['expression_source'] = 'https://github.com/cran/skpr/blob/699cf957261814894e3e7f59918a777661f6bf2f/man/eval_design_mc.Rd'
designs = list(csv.DictReader((root / 'designs.csv').open()))
normal = torch.distributions.Normal(torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
z = normal.icdf(torch.tensor(.975, dtype=torch.float64))
results = dict(run_id=runtime['run_id'], source_sha=runtime['source_sha'], R=runtime['R_version'], skpr=runtime['packages']['skpr'], cases={})
summary = [f"R run {runtime['run_id']}; {runtime['R_version']}; skpr {results['skpr']}. Asymptotic Wald values are approximations, not verdicts."]
for name, ref in reference.items():
    case = dict(ref)
    if ref['status'] == 'RECORDED':
        if name != 'RD1':
            values = [[float(row[c]) for c in ('X1', 'X2') if row[c]] + ([float(row['X3'])] if row['X3'] else []) for row in designs if row['case'] == name]
            x = torch.tensor(values, dtype=torch.float64)
            x = torch.column_stack((torch.ones(len(x), dtype=torch.float64), x, x[:, 0] * x[:, 1])) if name == 'G3' else torch.column_stack((torch.ones(len(x), dtype=torch.float64), x))
            beta = torch.tensor({'G1': [0, .5, 1, 0], 'G2': [1, .25, .5, 0], 'G3': [.5, 1, .5, .25]}[name], dtype=torch.float64)
            mu = (x @ beta).exp() if name == 'G2' else (x @ beta).sigmoid()
            w = mu if name == 'G2' else mu * (1 - mu)
            se = torch.linalg.inv(x.T @ (w[:, None] * x)).diag().sqrt()
            power = normal.cdf(beta.abs() / se - z) + normal.cdf(-beta.abs() / se - z)
            assert torch.allclose(power[beta == 0], torch.full_like(power[beta == 0], .05), atol=1e-14, rtol=0)
        rows = [row for row in ref['power'] if row['type'].startswith('parameter.power')]
        for j, row in enumerate(rows):
            p = row['power']; row['MC_SE'] = (p * (1 - p) / ref['nsim']) ** .5
            if name != 'RD1': row['asymptotic_approximation'] = power[j].item()
        case['parameter_power'] = rows
        summary.append(name + ': ' + '; '.join(f"{r['parameter']} {r['power']:.6g} Â± {r['MC_SE']:.6g}" + (f" (asymptotic {r['asymptotic_approximation']:.6g})" if name != 'RD1' else '') for r in rows))
    else:
        summary.append(f"{name}: NOT_RUN â€” {ref['error']}")
    results['cases'][name] = case
summary.append('RD1 expression verbatim (JSON string, newlines escaped): ' + json.dumps(reference['RD1'].get('call', reference['RD1']['expression_not_executed'])))
(root / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
(root / 'summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8', newline='\n')

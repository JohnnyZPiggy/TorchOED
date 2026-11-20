"""Re-score the frozen skpr power reference with TorchOED CPU float64."""
import csv
import json
import platform
import sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from torchoed import design_diagnostics

torch.set_num_threads(1)
reference = json.loads((HERE / 'reference.json').read_text())
runtime = json.loads((HERE / 'runtime.json').read_text())
terms = ['x1', 'x2', 'x3', 'x1:x2', 'x1:x3', 'x2:x3']
results = []
for name in ['D8', 'D12', 'D16']:
    with (HERE / (name + '.csv')).open(newline='') as file:
        rows = [[float(row[k]) for k in ['x1', 'x2', 'x3']] for row in csv.DictReader(file)]
    x = torch.tensor(rows, dtype=torch.float64)
    F = torch.cat((torch.ones((len(x), 1), dtype=x.dtype), x,
                   torch.stack((x[:,0]*x[:,1], x[:,0]*x[:,2], x[:,1]*x[:,2]), 1)), 1)
    power = design_diagnostics(F, torch.arange(len(x)), effect=1.0)['power']
    reported = {row['parameter']: row['power'] for row in reference[name] if row['type'] == 'parameter.power'}
    comparisons = [dict(term=t, torch_power=power[j+1].item(), R_power=reported[t],
                        delta=power[j+1].item()-reported[t]) for j,t in enumerate(terms)]
    delta = max(abs(v['delta']) for v in comparisons)
    results.append(dict(design=name, n=len(x), df=len(x)-7, comparisons=comparisons,
                        skpr_intercept=reported['(Intercept)'], max_abs_delta=delta,
                        verdict='MATCH' if delta <= 1e-10 else 'NO MATCH'))
output = dict(run_id=runtime['run_id'], oracle_source_sha=runtime['source_sha'],
              R_version=runtime['R_version'], skpr_version=runtime['packages']['skpr'],
              python=platform.python_version(), torch=torch.__version__, designs=results,
              published_example='NR: optional Rd run omitted; only the three frozen required designs evaluated; count remains 14')
(HERE / 'results.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
summary = ['# W14 skpr power across run sizes', '',
           f"R oracle run [{runtime['run_id']}](https://github.com/JohnnyZPiggy/TorchOED/actions/runs/{runtime['run_id']}), source `{runtime['source_sha']}`.",
           f"{runtime['R_version']}; skpr {runtime['packages']['skpr']}. TorchOED CPU float64, Python {platform.python_version()}, torch {torch.__version__}.",
           '', '| Design | df | max absolute power delta | Verdict |', '|---|---|---|---|']
summary += [f"| {r['design']} | {r['df']} | {r['max_abs_delta']:.17g} | {r['verdict']} |" for r in results]
summary += ['', 'Mapping fixed before dispatch: skpr effectsize=2 means coefficient=1 for continuous terms; TorchOED effect=1, alpha=0.05. Six non-intercept parameter powers per design compared. Intercept excluded; its skpr number is retained as-is in results.json and reference.json.', '', output['published_example']+'. No published-example index increment.', '', 'Reproduce: `python reports/w14-skpr-power/run.py`. No searches, held-out cases, GPU, or production changes.']
(HERE / 'summary.md').write_text('\n'.join(summary)+'\n', encoding='utf-8')
print(json.dumps(results, indent=2))
print(f"Power comparisons: {sum(abs(v['delta']) <= 1e-10 for r in results for v in r['comparisons'])} PASS, {sum(abs(v['delta']) > 1e-10 for r in results for v in r['comparisons'])} FAIL")

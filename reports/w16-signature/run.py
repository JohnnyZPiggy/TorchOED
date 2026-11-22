"""Frozen #541 calls; --check rechecks saved evidence without rerunning search."""
import csv
import json
import sys
from pathlib import Path
import torch
from torchoed import optimal_design, polynomial_features

p = Path(__file__).resolve().parent
source = p.parent / 'w12-published'
torch.set_num_threads(1)
with (source / 'inputs/alg-d.csv').open() as stream:
    C = torch.tensor([[float(v) for v in r.values()] for r in csv.DictReader(stream)], dtype=torch.float64)
F = polynomial_features(C, degree=2)
with (source / 'reference.csv').open() as stream:
    reference = list(csv.DictReader(stream))

def reference_ids(c):
    rows = [r for r in reference if r['id'] == 'alg-' + c.lower()]
    lookup = {(int(r['row']), r['col']): float(r['value']) for r in rows}
    return [C.tolist().index([lookup[i, col] for col in ('A', 'B', 'C')])
            for i in sorted({int(r['row']) for r in rows})]

def metric(ids, c):
    X = F[ids]
    M = X.mT @ X
    if c == 'D':
        return torch.linalg.slogdet(M).logabsdet.item()
    inv = torch.linalg.inv(M)
    return (inv.trace() if c == 'A' else (inv @ (F.mT @ F / len(F))).trace()).item()

if '--check' not in sys.argv:
    results = dict(torch=torch.__version__, device='CPU', dtype='float64', threads=1, criteria={})
    for c in ('D', 'A', 'I'):
        rows = {}
        for name, kwargs in [('published', dict(fixed_ids=reference_ids(c), restarts=1)), ('torchoed', {})]:
            try:
                r = optimal_design(F, 14, criterion=c, generator=torch.Generator().manual_seed(0), **kwargs)
            except Exception as error:
                (p / 'results.json').write_text(json.dumps(dict(error=str(error)), indent=2) + '\n', encoding='utf-8')
                raise
            ids = r['ids'].tolist()
            rows[name] = dict(ids=ids, value=metric(ids, c), objective=r['objective'],
                              floor=r['d_efficiency_lower_bound'] if c == 'D' else r['efficiency_lower_bound'],
                              certificate=r['certificate'], stop_reason=r['stop_reason'])
        results['criteria'][c] = rows
    (p / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n', encoding='utf-8')
results = json.loads((p / 'results.json').read_text(encoding='utf-8'))
expected = dict(D=0.830449826989619, A=0.892857142857143, I=0.613547439052458)
for c, rows in results['criteria'].items():
    ids = reference_ids(c)
    X = F[ids]
    solved = torch.linalg.solve(X.mT @ X, F.mT)
    bound = F.shape[1] / (14 * (F * solved.mT).sum(1)).max().item()
    assert abs(bound - expected[c]) <= 1e-12
    assert set(rows['published']['ids']) == set(ids)
    if c == 'D':
        assert bound - 1e-12 <= rows['published']['floor'] <= 1
    for name, row in rows.items():
        assert abs(metric(row['ids'], c) - row['value']) <= 1e-12
    print(c, rows)
print('13 PASS, 0 FAIL: 3 dense bounds, 3 ID sets, 1 D floor, 6 saved metrics')

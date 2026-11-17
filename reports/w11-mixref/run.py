"""Score external mixture references independently in CPU float64."""
import csv, json, sys
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import torch
from torchoed import optimal_design
out = Path(__file__).resolve().parent
runtime = json.loads((out/'runtime.json').read_text(encoding='utf-8'))
if runtime['reference_errors']:
    print(json.dumps({case: dict(verdict='NOT_RUN', errors=runtime['reference_errors']) for case in ('R1','R3','D')}, indent=2))
    sys.exit(0)
torch.set_num_threads(1)
read = lambda name: list(csv.DictReader((out/name).open()))
tensor = lambda rows: torch.tensor(rows, dtype=torch.float64)
candidates = tensor([[float(r[k]) for k in ('x1','x2','x3')] for r in read('candidates_r1.csv')])
features = lambda x: torch.cat((x, x[:,[0,0,1]]*x[:,[1,2,2]]),1)
d_value = lambda f: float(torch.linalg.det(f.T@f/len(f)).pow(1/f.shape[1]))
results = {}
for region, lower, upper in [('R1',[Q(1,5),Q(1,10),Q(1,10)],[Q(3,5),Q(1,2),Q(3,5)]),('R3',[Q(1,10)]*4,[Q(3,5),Q(1,2),Q(2,5),Q(3,10)])]:
    q, vertices = len(lower), set()
    for free in range(q):
        fixed = [j for j in range(q) if j != free]
        for values in product(*[(lower[j],upper[j]) for j in fixed]):
            v = list(lower)
            for j,x in zip(fixed,values): v[j] = x
            v[free] = 1-sum(values)
            if lower[free] <= v[free] <= upper[free]: vertices.add(tuple(v))
    rows = [r for r in read('xvert.csv') if r['region'] == region]
    dropped = sum(float(r['dimen']) != 0 for r in rows)
    rows = [r for r in rows if float(r['dimen']) == 0]
    x = tensor([[float(r[f'x{j+1}']) for j in range(q)] for r in rows])
    distances = torch.cdist(x,tensor([[float(a) for a in v] for v in sorted(vertices)]))
    gap = float(torch.maximum(distances.min(0).values.max(),distances.min(1).values.max()))
    results[region] = dict(centroids_dropped=dropped, count=len(x), expected=len(vertices), hausdorff=gap, verdict='MATCH' if len(x)==len(vertices) and gap<=1e-12 else 'NO MATCH')
r = features(tensor([[float(row[k]) for k in ('x1','x2','x3')] for row in read('optfederov.csv')]))
f = features(candidates); design = optimal_design(f,10,criterion='D',restarts=10,generator=torch.Generator().manual_seed(11))
dr, dt = d_value(r), d_value(f[design['ids']]); delta = dt-dr
results['D'] = dict(R=dr,TorchOED=dt,delta=delta,efficiency_lower_bound=float(design['efficiency_lower_bound']), verdict='MATCH' if abs(delta)<=1e-10*dr or delta>0 else 'NO MATCH')
print(json.dumps(results,indent=2))

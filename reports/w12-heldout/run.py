"""Frozen #368: one measurement; --score rechecks saved designs without searches."""
import csv, itertools, json, math, platform, subprocess, sys
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
import torch
from torchoed import optimal_design, plackett_burman, scheffe_features, simplex_centroid

torch.set_num_threads(1)
root = Path(__file__).parent
saved = '--score' in sys.argv or '--references' in sys.argv
result = json.loads((root/'results.json').read_text()) if saved else dict(
    source_sha=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    python=platform.python_version(), torch=torch.__version__, device='CPU', dtype='float64',
    threads=1, seed=0, restarts=10, H30_mapping='8*i+k', H32_mapping='3*i+k')
checks = {}
original = {k:result[k] for k in ('H30','H31','H32')} if saved else None
error = 'Error: as.character(packageVersion(pins$package[i])) == pins$version[i] is not TRUE'

def trace(rows):
    p = len(rows[0])
    a = [[sum(row[i]*row[j] for row in rows) for j in range(p)] +
         [Q(i == j) for j in range(p)] for i in range(p)]
    for j in range(p):
        pivot = next((i for i in range(j,p) if a[i][j]), None)
        if pivot is None: return None
        a[j], a[pivot] = a[pivot], a[j]
        d = a[j][j]; a[j] = [v/d for v in a[j]]
        for i in range(p):
            if i != j:
                d = a[i][j]; a[i] = [v-d*w for v,w in zip(a[i],a[j])]
    return sum(a[i][p+i] for i in range(p))

def copies(indices, repeats):
    count = Counter(); ids = []
    for i in indices:
        if count[i] == repeats: return None
        ids.append(repeats*i+count[i]); count[i] += 1
    return ids

def floor(F, ids, **options):
    if ids is None: return 'NOT_RUN: candidate used more than available copies'
    return optimal_design(F,len(ids),fixed_ids=ids,**options)['efficiency_lower_bound']

X = simplex_centroid(3); F0 = scheffe_features(X,'special_cubic'); F = F0.repeat_interleave(8,0)
# Construct exact features from the rational simplex-centroid coordinates.
exact = []
for row in X.tolist():
    a,b,c = [Q(v).limit_denominator(3) for v in row]
    exact.append([a,b,c,a*b,a*c,b*c,a*b*c])
if not saved:
    r = optimal_design(F,8,criterion='A',generator=torch.Generator().manual_seed(0))
    best = None; winners = []; singular = visited = 0
    for selection in itertools.combinations_with_replacement(range(7),8):
        value = trace([exact[i] for i in selection]); visited += 1
        if value is None: singular += 1; continue
        if best is None or value < best: best = value; winners = []
        if value == best: winners.append(selection)
    result['H30'] = dict(ids=r['ids'].tolist(), reference_multisets=winners,
                         visited=visited, singular=singular)
row = result['H30']; ids = row['ids']; refids = copies(row['reference_multisets'][0],8)
tv = trace([exact[i//8] for i in ids]); rv = trace([exact[i//8] for i in refids])
checks['H30'] = bool(torch.linalg.matrix_rank(F[ids]) == 7 and math.isclose(
    torch.linalg.inv(F[ids].mT@F[ids]).trace().item(),float(tv),rel_tol=1e-12))
row.update(torch_exact=str(tv), reference_exact=str(rv), torch_value=float(tv),
           reference_value=float(rv), delta=float(tv-rv), verdict='MATCH' if tv == rv else 'NO MATCH',
           torch_floor=floor(F,ids,criterion='A'), reference_floor=floor(F,refids,criterion='A'))

def pb_metric(X, name):
    X = X.to(torch.int64); triples = list(itertools.combinations(range(27),3))
    dense = torch.einsum('ri,rj,rk->ijk',X,X,X)
    python = [abs(sum(int(r[i])*int(r[j])*int(r[k]) for r in X.tolist())) for i,j,k in triples]
    checks[name] = python == [abs(int(dense[i,j,k])) for i,j,k in triples]
    return dict(orthogonal=bool(torch.equal(X.mT@X,28*torch.eye(27,dtype=torch.int64))),
                J3=dict(sorted(Counter(python).items())), triples=len(triples))

tpb = pb_metric(plackett_burman(28),'H31_torch')
row = result.setdefault('H31',{}); row['torch'] = tpb
if (root/'H31.csv').exists():
    with (root/'H31.csv').open() as f: rows = list(csv.reader(f))[1:]
    assert len(rows) == 28 and all(len(r) == 27 for r in rows)
    rpb = pb_metric(torch.tensor([[int(v) for v in r] for r in rows]),'H31_reference')
    row.update(reference=rpb,verdict='MATCH' if tpb == rpb and tpb['orthogonal'] else 'NO MATCH')
else: row.update(reference='NOT_RUN: '+error,verdict='NOT_RUN')

with (root/'H32-candidates.csv').open() as f: grid = list(csv.reader(f))[1:]
X = torch.tensor([[float(v) for v in r] for r in grid],dtype=torch.float64)
def features(X):
    a,b = X.unbind(1); return torch.stack((a,b,a*b,a*a,b*b),1)
F = features(X).repeat_interleave(3,0)
def blocked_metric(S, name):
    Z = torch.eye(3,dtype=torch.float64).repeat_interleave(4,0)
    M = S.mT@S-S.mT@Z@torch.linalg.solve(Z.mT@Z,Z.mT@S)
    sign, ld = torch.linalg.slogdet(M); joint = torch.cat((Z,S),1)
    alternative = torch.linalg.slogdet(joint.mT@joint).logabsdet-torch.linalg.slogdet(Z.mT@Z).logabsdet
    checks[name] = bool(sign > 0 and math.isclose(ld.item(),alternative.item(),abs_tol=1e-10))
    return dict(logdet=ld.item(), value=math.exp(ld.item()/6))
if not saved:
    r = optimal_design(F,12,criterion='D',blocks=[4,4,4],generator=torch.Generator().manual_seed(0))
    result['H32'] = dict(ids=r['ids'].tolist())
row = result['H32']; row['torch'] = blocked_metric(F[row['ids']],'H32_torch')
row['torch_floor'] = floor(F,row['ids'],criterion='D',blocks=[4,4,4])
if (root/'H32.csv').exists():
    with (root/'H32.csv').open() as f: rows = list(csv.DictReader(f))
    rows.sort(key=lambda r:int(r['block']))
    assert Counter(int(r['block']) for r in rows) == {1:4,2:4,3:4}
    RX = torch.tensor([[float(r['x1']),float(r['x2'])] for r in rows],dtype=torch.float64)
    refids = copies([X.tolist().index(r) for r in RX.tolist()],3)
    row['reference'] = blocked_metric(features(RX),'H32_reference')
    row['reference_floor'] = floor(F,refids,criterion='D',blocks=[4,4,4])
    row['reference_ids'] = refids
    row['delta'] = row['torch']['value']-row['reference']['value']
    row['verdict'] = 'MATCH' if math.isclose(row['delta'],0,abs_tol=1e-10) else 'NO MATCH'
else: row.update(reference='NOT_RUN: '+error,verdict='NOT_RUN')
result['R_workflow'] = dict(run_id=44518822770, conclusion='failure',
    url='https://github.com/JohnnyZPiggy/TorchOED/actions/runs/44518822770',
    source_sha='949bae4ae9070e6c663c178151d076b17dd5c4b0', error=error,
    artifact='NOT_RUN: reference/runtime step skipped; no artifact produced',
    versions='NOT_RUN: sessionInfo.txt and runtime.json were not produced')
result['checks'] = {k:'PASS' if v else 'FAIL' for k,v in checks.items()}
assert result['H30']['visited'] == 3003
if '--score' in sys.argv:
    assert original == json.loads(json.dumps({k:result[k] for k in ('H30','H31','H32')}))
if '--score' not in sys.argv: (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert all(checks.values())

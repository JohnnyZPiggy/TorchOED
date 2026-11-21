import csv, itertools, json, math
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
import torch
from torchoed import full_factorial, optimal_design, minimum_aberration, word_length_pattern
p = Path(__file__).parent
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
def eliminate(a):
    n = len(a); b = [[Q(v) for v in row]+[Q(i==j) for j in range(n)] for i,row in enumerate(a)]; det = Q(1)
    for k in range(n):
        pivot = b[k][k]
        if not pivot: return Q(0), None  # PSD Gram matrices: zero leading pivot implies singularity.
        det *= pivot; b[k] = [v/pivot for v in b[k]]
        for i in range(n):
            if i != k:
                scale = b[i][k]; b[i] = [v-scale*u for v,u in zip(b[i],b[k])]
    return det, [row[n:] for row in b]
def bareiss(a):
    a = [[int(v) for v in row] for row in a]; previous = 1
    for k in range(len(a)-1):
        pivot = a[k][k]
        if not pivot: return 0
        for i in range(k+1,len(a)):
            for j in range(k+1,len(a)): a[i][j] = (pivot*a[i][j]-a[i][k]*a[k][j])//previous
        previous = pivot
    return a[-1][-1]
def features(x): return np.column_stack((np.ones(8,dtype=np.int64), x, x[:,0]*x[:,1], x[:,0]*x[:,2], x[:,1]*x[:,2]))
x = (2*full_factorial([2]*3).double()-1).repeat(2,1); F = torch.tensor(features(x[:8].numpy().astype(np.int64)),dtype=torch.float64).repeat(2,1)
results = {}
for name,first,second,kwargs in [('H40',np.repeat([0,1],4),np.repeat(range(4),2),dict(whole_plots=[4,4],hard=x[:,0].long(),subplots=[2]*4,subplot_hard=x[:,1].long(),eta=1,eta_sub=1)),('H41',np.repeat(range(4),2),np.tile([0,1],4),dict(strip=(4,2),row_hard=x[:,0].long(),col_hard=x[:,1].long(),eta_row=1,eta_col=1))]:
    V = np.eye(8,dtype=np.int64)+(first[:,None]==first)+(second[:,None]==second)
    _, inverse = eliminate(V.tolist()); scale = math.lcm(*(v.denominator for row in inverse for v in row))
    W = np.array([[int(v*scale) for v in row] for row in inverse],dtype=np.int64)
    ids = optimal_design(F,8,criterion='D',generator=torch.Generator().manual_seed(0),**kwargs)['ids'].tolist()
    X = F[ids].numpy().astype(np.int64); ours = Q(bareiss(X.T@W@X),scale**7)
    dense = [[sum(Q(int(X[r,i]))*inverse[r][s]*int(X[s,j]) for r in range(8) for s in range(8)) for j in range(7)] for i in range(7)]
    assert eliminate(dense)[0] == ours
    assert len(set(ids))==8 and all(len(set(x[ids][first==g,0].tolist()))==1 for g in set(first)) and all(len(set(x[ids][second==g,1].tolist()))==1 for g in set(second))
    best = Q(-1); feasible = count = 0
    for bits in itertools.product((-1,1),repeat=14):
        a = np.array(bits[:len(set(first))])[first]; b = np.array(bits[len(set(first)):6])[second]
        rows = list(zip(a,b,bits[6:]))
        if max(Counter(rows).values())>2: continue
        feasible += 1; Y = features(np.array(rows)); value = Q(bareiss(Y.T@W@Y),scale**7)
        if value>best: best,count = value,1
        elif value==best: count += 1
    results[name] = dict(torchoed=str(ours),reference=str(best),delta=str(ours-best),verdict='MATCH' if ours==best else 'NO MATCH',feasible_assignments=feasible,maximum_count=count,total_assignments=16384,ids=ids,independent_fraction_check='PASS',labels='A/B levels as length-16 integer tensors; run-order strata',inverse_scale=scale)
runtime = json.loads((p/'runtime.json').read_text()); assert runtime['R_version'] and runtime['packages']['FrF2']
reference = torch.tensor([[float(v) for v in row] for row in csv.reader((p/'H42-R.csv').open())]); assert reference.shape==(32,12) and ((reference==-1)|(reference==1)).all()
ours = word_length_pattern(minimum_aberration(12,32)['design']).tolist(); ref = word_length_pattern(reference).tolist()
results['H42'] = dict(torchoed=ours,reference=ref,verdict='MATCH' if ours==ref else 'TorchOED better' if ours<ref else 'NO MATCH',first_differing_word_length=next((i+1 for i,(a,b) in enumerate(zip(ours,ref)) if a!=b),None),run_id=runtime['run_id'],oracle_source_sha=runtime['source_sha'],R_version=runtime['R_version'],FrF2=runtime['packages']['FrF2'])
results['environment'] = dict(torch=torch.__version__,numpy=np.__version__,dtype='float64',threads=1)
(p/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v['verdict'] for k,v in results.items() if k.startswith('H')})); print('Independent exact determinants: 2 PASS, 0 FAIL; feasibility: 2 PASS, 0 FAIL')

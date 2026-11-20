import csv, itertools, json, math
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
import torch
from torchoed import full_factorial, optimal_design, sobol, feasible_candidates, fast_flexible_filling, space_filling_diagnostics
p = Path(__file__).parent
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
def bareiss(a):
    a = [[int(v) for v in row] for row in a]; previous = 1
    for k in range(len(a)-1):
        pivot = a[k][k]
        if not pivot: return 0  # PSD Gram: a zero leading pivot implies singularity.
        for i in range(k+1,len(a)):
            for j in range(k+1,len(a)): a[i][j] = (pivot*a[i][j]-a[i][k]*a[k][j])//previous
        previous = pivot
    return a[-1][-1]
def fraction(a, w=None):
    n=len(a); b=[[Q(int(v)) for v in row]+[Q(i==j) for j in range(n)] for i,row in enumerate(a)]; det=Q(1)
    for k in range(n):
        pivot=b[k][k]; det*=pivot
        if not pivot: return Q(0)
        b[k]=[v/pivot for v in b[k]]
        for i in range(n):
            if i!=k:
                scale=b[i][k]; b[i]=[v-scale*u for v,u in zip(b[i],b[k])]
    return det if w is None else sum(b[i][n+j]*Q(int(w[j,i]),27) for i in range(n) for j in range(n))
def search(f, n, **kw): return optimal_design(f,n,generator=torch.Generator().manual_seed(0),**kw)['ids'].tolist()
x=full_factorial([3,3,3]).double()-1; primary=torch.cat((torch.ones(27,1),x),1); K=np.diag([0]*4+[1]*3)
f36=torch.cat((primary,torch.stack((x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]),1)),1)
f37=torch.cat((primary,2*(x.square()-2/3)),1)
ids36=search(f36,6,if_possible=[4,5,6]); ids37=search(f37,6,criterion='I',if_possible=[4,5,6])
sub=np.array(list(itertools.combinations(range(27),6))); a=f36.numpy().astype(np.int64); y=np.rint(3*f37.numpy()).astype(np.int64)
g36=np.einsum('bni,bnj->bij',a[sub],a[sub])+K; values36=[bareiss(m) for m in g36]; best36=max(values36); our36=bareiss(a[ids36].T@a[ids36]+K)
assert fraction(a[ids36].T@a[ids36]+K)==our36
# Y=3F, so M=G/9 and W=Y'Y/243; trace(M^-1 W)=trace(G^-1 Y'Y)/27.
g37=np.einsum('bni,bnj->bij',y[sub],y[sub])+9*K; W=y.T@y; valid=np.array([bareiss(g[:4,:4])>0 for g in g37])
float_values=np.full(len(sub),np.inf); float_values[valid]=np.einsum('bij,ji->b',np.linalg.inv(g37[valid].astype(float)),W)/27
near=np.flatnonzero(np.abs(float_values-float_values.min())<=abs(float_values.min())*1e-8)
exact37=[fraction(g37[i],W) for i in near]; best37=min(exact37); our37=fraction(y[ids37].T@y[ids37]+9*K,W)
x=2*full_factorial([2]*4).double()-1; f38=torch.cat((torch.ones(16,1),x,(x[:,0]*x[:,1])[:,None],(x[:,2]*x[:,3])[:,None]),1)
ids38=search(f38,8,models=[list(range(5)),list(range(7))]); a=f38.numpy().astype(np.int64)
def pair(ids):
    g=a[ids].T@a[ids]; return bareiss(g[:5,:5]),bareiss(g)
pairs=[pair(list(ids)) for ids in itertools.combinations(range(16),8)]; score=lambda d:d[0]**7*d[1]**5
best38=max(pairs,key=score); our38=pair(ids38); g=a[ids38].T@a[ids38]
assert (fraction(g[:5,:5]),fraction(g))==our38
A=torch.tensor([[1.,1.,1.]]); b=torch.tensor([1.5]); S=sobol(2**14,3,scramble=True,seed=0,dtype=torch.float64); S=S[feasible_candidates(S,A=A,b=b)]
saved=torch.tensor([[float(v) for v in row] for row in csv.reader((p/'S.csv').open())]); assert torch.equal(S,saved)
X=fast_flexible_filling(15,3,A=A,b=b); diagnostics={k:v.item() for k,v in space_filling_diagnostics(X).items() if k in ('maxpro','min_distance')}
runtime=json.loads((p/'runtime.json').read_text()); assert '4.5.2' in runtime['R_version'] and runtime['packages']['support']=='0.1.7'
results=dict(H36=dict(torchoed=our36,reference=best36,maximum_count=values36.count(best36),subsets=len(sub),ids=ids36,delta=our36-best36,verdict='MATCH' if our36==best36 else 'NO MATCH',independent_fraction_check='PASS'),H37=dict(torchoed=str(our37),reference=str(best37),minimum_count=exact37.count(best37),subsets=len(sub),nonsingular=int(valid.sum()),float_min=float(float_values.min()),fraction_rescored=len(near),ids=ids37,delta=str(our37-best37),verdict='MATCH' if our37==best37 else 'NO MATCH'),H38=dict(torchoed=our38,reference=best38,maximum_count=sum(score(d)==score(best38) for d in pairs),subsets=len(pairs),nonsingular=sum(all(d) for d in pairs),ids=ids38,delta_score=score(our38)-score(best38),verdict='MATCH' if score(our38)==score(best38) else 'NO MATCH',independent_fraction_check='PASS'),H39=dict(torchoed=diagnostics,design=X.tolist(),feasible_sample=len(S),reference='NOT_RUN',verdict='NOT_RUN',error='Error in file(file, "rt") : cannot open the connection\nIn addition: Warning message:\nIn file(file, "rt") :\n  cannot open file \'reports/w14-heldout/S.csv\': No such file or directory\nExecution halted',run_id=runtime['run_id'],R_version=runtime['R_version'],support=runtime['packages']['support']),environment=dict(torch=torch.__version__,numpy=np.__version__,default_dtype='float64',threads=1))
(p/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v['verdict'] for k,v in results.items() if k.startswith('H')})); print('Independent exact determinants: 2 PASS, 0 FAIL; feasible sample bit-equal: PASS')

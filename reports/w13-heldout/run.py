import csv, itertools, json, math
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
from torchoed import lhs, space_filling_diagnostics
p = Path(__file__).parent
torch.set_num_threads(1)
def ranks(x): return x.argsort(0).argsort(0)
def design(n, k, mode): return ranks(lhs(n, k, mode=mode, generator=torch.Generator().manual_seed(0)))
def metrics(r): return {k: v.item() for k, v in space_filling_diagnostics((r.to(torch.float64)+.5)/len(r)).items() if k in ('maxpro', 'min_distance')}
def exact(r):
    n = len(r); x = [[F(2*int(v)+1,2*n) for v in row] for row in r]
    h = lambda a: 1+abs(a-F(1,2))/2-abs(a-F(1,2))**2/2
    g = lambda a,b: 1+abs(a-F(1,2))/2+abs(b-F(1,2))/2-abs(a-b)/2
    return F(13,12)**2-F(2,n)*sum(math.prod(h(a) for a in row) for row in x)+F(1,n*n)*sum(math.prod(g(a,b) for a,b in zip(u,v)) for u in x for v in x)
def cd2(perms):
    n=perms.shape[1]; a=np.abs(2*np.arange(n)+1-n); h=8*n*n+2*n*a-a*a
    g=4*n+a[:,None]+a[None,:]-2*np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    return [F(169,144)-F(int(v),32*n**5)+F(int(w),16*n**4) for v,w in zip((h[None,:]*h[perms]).sum(1),(g[None,:,:]*g[perms[:,:,None],perms[:,None,:]]).sum((1,2)))]
r33=design(10,2,'maximin'); d=((r33[:,None]-r33[None,:])**2).sum(2); upper=d[torch.triu_indices(10,10,1).unbind()]; ours=int(upper.min()); best=0; count=0
it=itertools.permutations(range(10))
while batch:=list(itertools.islice(it,10000)):
    a=np.array(batch); m=np.full(len(a),1000)
    for i,j in itertools.combinations(range(10),2): m=np.minimum(m,(i-j)**2+(a[:,i]-a[:,j])**2)
    best=max(best,int(m.max())); count+=len(a)
r35=design(7,2,'uniform'); values=cd2(np.array(list(itertools.permutations(range(7))))); ref35=min(values); ours35=exact(r35.tolist())
assert cd2(r35[r35[:,0].argsort(),1].numpy()[None,:])[0]==ours35
r34=design(20,3,'maxpro'); ref34=ranks(torch.tensor([[float(v) for v in row] for row in csv.reader((p/'H34-R.csv').open())],dtype=torch.float64)); a,b=metrics(r34),metrics(ref34)
runtime=json.loads((p/'runtime.json').read_text(encoding='utf-8')); assert runtime['packages']['MaxPro']=='4.1-2' and '4.5.2' in runtime['R_version']
results=dict(H33=dict(torchoed=ours,reference=best,pairs_at_minimum=int((upper==ours).sum()),permutations=count,delta=ours-best,verdict='MATCH' if ours==best else 'NO MATCH',ranks=r33.tolist()),H34=dict(torchoed=a,reference=b,delta=a['maxpro']-b['maxpro'],verdict='MATCH or TorchOED better' if a['maxpro']<=b['maxpro']*(1+1e-9) else 'NO MATCH',ranks=r34.tolist(),reference_ranks=ref34.tolist(),run_id=runtime['run_id'],R_version=runtime['R_version'],MaxPro=runtime['packages']['MaxPro']),H35=dict(torchoed=str(ours35),reference=str(ref35),delta=str(ours35-ref35),permutations=len(values),independent_fraction_check='PASS',verdict='MATCH' if ours35==ref35 else 'NO MATCH',ranks=r35.tolist()),environment=dict(torch=torch.__version__,numpy=np.__version__))
(p/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v['verdict'] for k,v in results.items() if k.startswith('H')})); print('H35 independent Fraction check: 1 PASS, 0 FAIL')

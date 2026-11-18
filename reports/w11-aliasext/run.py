import csv, itertools, json, math, platform, subprocess, sys
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
import torch
from torchoed import optimal_design
torch.set_num_threads(1); path=Path(__file__).with_name('results.json'); points=list(itertools.product((-1,1),repeat=3)); F=torch.tensor([(1,*r) for r in points]*6,dtype=torch.float64); G=torch.tensor([(a*b,a*c,b*c) for a,b,c in points]*6,dtype=F.dtype)
def score(rows):
    x=torch.tensor(rows,dtype=F.dtype); a,b,c=x.T; X=torch.cat((torch.ones((6,1),dtype=F.dtype),x),1); Y=torch.stack((a*b,a*c,b*c),1); M=X.T@X; A=torch.linalg.solve(M,X.T@Y); d=torch.linalg.det(M).item(); t=A.square().sum().item()
    return dict(rows=rows,det_M=d,alias_trace=t,D_efficiency=(d/1024)**.25,Psi=torch.linalg.slogdet(M/6).logabsdet.item()/4-.5*t,on_front=any(math.isclose(d,v,rel_tol=1e-10) and math.isclose(t,w,rel_tol=1e-10) for v,w in [(1024,1.5),(768,1/3)]),verdict='MATCH' if math.isclose(d,768,rel_tol=1e-10) and math.isclose(t,1/3,rel_tol=1e-10) else 'NO MATCH')
def exact(ids):
    x=[list(map(Q,[1,*points[i]])) for i in ids]; y=[[r[1]*r[2],r[1]*r[3],r[2]*r[3]] for r in x]; m=[[sum(r[i]*r[j] for r in x) for j in range(4)]+[sum(r[i]*s[j] for r,s in zip(x,y)) for j in range(3)] for i in range(4)]; det=Q(1)
    for k in range(4):
        pivot=next((j for j in range(k,4) if m[j][k]),None)
        if pivot is None: return None
        if pivot!=k: m[k],m[pivot]=m[pivot],m[k]; det=-det
        v=m[k][k]; det*=v; m[k]=[a/v for a in m[k]]; m=[r if j==k else [a-r[k]*b for a,b in zip(r,m[k])] for j,r in enumerate(m)]
    return det,sum(v*v for r in m for v in r[4:])
if '--score' not in sys.argv:
    r=optimal_design(F,6,alias=G,alias_weight=.5,generator=torch.Generator().manual_seed(0)); pairs=Counter(); visited=0
    for ids in itertools.combinations_with_replacement(range(8),6):
        visited+=1; pair=exact(ids)
        if pair: pairs[pair]+=1
    front={f'{d},{t}':n for (d,t),n in pairs.items() if not any(e>=d and u<=t and (e,u)!=(d,t) for e,u in pairs)}; result=dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=platform.python_version(),torch=torch.__version__,device='CPU',dtype='float64',threads=1,seed=0,restarts=10,TorchOED=dict(score(F[r['ids'],1:].tolist()),ids=r['ids'].tolist(),objective=r['objective']),oracle=dict(visited=visited,distinct_pairs=len(pairs),front=front)); path.write_text(json.dumps(result,indent=2)+'\n')
result=json.loads(path.read_text()); assert result['oracle']==dict(visited=1716,distinct_pairs=19,front={'1024,3/2':12,'768,1/3':4}); assert math.isclose(score(result['TorchOED']['rows'])['Psi'],result['TorchOED']['objective'],rel_tol=1e-10); result['tests']=dict(exhaustive_oracle='PASS',dense_rescore='PASS')
if (p:=path.with_name('skpr.csv')).exists():
    raw=list(csv.DictReader(p.open())); rows=[[float(r[k]) for k in ('x1','x2','x3')] for r in raw]; assert len(rows)==6 and all(v in (-1,1) for row in rows for v in row); result['skpr']=score(rows)
else: result['skpr']=dict(verdict='NOT_RUN',error=path.with_name('sessionInfo.txt').read_text() if path.with_name('sessionInfo.txt').exists() else 'Colab pending')
path.write_text(json.dumps(result,indent=2)+'\n'); print('2 PASS, 0 FAIL',result)

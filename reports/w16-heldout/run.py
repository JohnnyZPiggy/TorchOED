import csv, itertools, json, math, time
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
import sympy as sp
import torch
from torchoed import full_factorial, minimum_aberration, word_length_pattern, glm_power, optimal_design
p = Path(__file__).parent; torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
runtime = json.loads((p/'runtime.json').read_text()); start = time.perf_counter()
def confounded(x, blocks):
    return [list(c) for n in (1,2) for c in itertools.combinations(range(x.shape[1]),n) if all(len(set(x[blocks==b][:,c].prod(1).tolist()))==1 for b in blocks.unique())]
tiny = 2*full_factorial([2,2]).double()-1
assert confounded(tiny,tiny.prod(1)) == [[0,1]]; print('H43 non-held-out AB check: PASS')
def features(x): return np.column_stack((np.ones(8,dtype=np.int64),x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]))
x = torch.tensor([[float(v) for v in row] for row in list(csv.reader((p/'H44-design.csv').open()))[1:]])
F = torch.tensor(features(x[:8].numpy().astype(np.int64)),dtype=torch.float64).repeat(2,1)
if 'torchoed' not in runtime:
    t = time.perf_counter(); a = minimum_aberration(10,64,blocks=4)
    h43 = dict(wlp=word_length_pattern(a['design']).tolist(),confounded=confounded(a['design'],a['block'])); times = dict(H43=time.perf_counter()-t)
    t = time.perf_counter(); b = glm_power(torch.cat((torch.ones(16,1),x),1),beta=torch.tensor([0.5,0.5,-0.25,0.0]),family='poisson',alpha=0.05,nsim=10000,generator=torch.Generator().manual_seed(0)); times['H44']=time.perf_counter()-t
    t = time.perf_counter(); ids = optimal_design(F,8,criterion='A',strip=(2,4),row_hard=x[:,0].long(),col_hard=x[:,1].long(),eta_row=2.0,eta_col=0.5,generator=torch.Generator().manual_seed(0))['ids'].tolist(); times['H45']=time.perf_counter()-t
    runtime.update(torchoed=dict(H43=h43,H44=b['power'].tolist(),H45=ids),torchoed_wall_seconds=times)
    (p/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n',encoding='utf-8')
ours = runtime['torchoed']; results = {}
for name in ('H43','H44'):
    rows = list(csv.DictReader((p/(name+'-R.csv')).open())); error = rows[0].get('error')
    meta = dict(run_id=runtime['run_id'],source_sha=runtime['source_sha'],R_version=runtime['R_version'],FrF2=runtime['packages']['FrF2'],skpr=runtime['packages']['skpr'])
    if error: results[name]=dict(torchoed=ours[name],reference='NOT_RUN',verdict='NOT_RUN',error=error,**meta); continue
    if name=='H43':
        data = torch.tensor([[float(v) for v in r.values()] for r in rows]); ref = dict(wlp=word_length_pattern(data[:,:10]).tolist(),confounded=confounded(data[:,:10],data[:,10]))
        a,b = ours[name],ref; verdict = 'MATCH' if a==b and not a['confounded'] else 'TorchOED better' if not a['confounded'] and not b['confounded'] and a['wlp']<b['wlp'] else 'NO MATCH'
        detail = dict(first_differing_word_length=next((i+1 for i,(u,v) in enumerate(zip(a['wlp'],b['wlp'])) if u!=v),None))
    else:
        values = {r['parameter']:float(r['power']) for r in rows if r['type']=='parameter.power.mc'}; ref = [values[k] for k in ('(Intercept)','A','B','C')]
        z = [abs(a-b)/math.sqrt((a*(1-a)+b*(1-b))/10000) if a!=b else 0.0 for a,b in zip(ours[name],ref)]; verdict='MATCH' if max(z)<=4 else 'NO MATCH'; detail=dict(worst_z=max(z))
    results[name]=dict(torchoed=ours[name],reference=ref,verdict=verdict,**detail,**meta)
r = np.repeat(range(2),4); c = np.tile(range(4),2)
V = sp.eye(8)+2*sp.Matrix((r[:,None]==r).astype(int))+sp.Rational(1,2)*sp.Matrix((c[:,None]==c).astype(int)); inv=V.inv(); scale=sp.ilcm(*[v.q for v in inv]); W=np.array(inv*scale,dtype=np.int64)
def metric(y):
    gram=sp.Matrix((y.T@W@y).tolist()); det=gram.det(method='domain-ge')
    return Q(str(sp.trace(gram.inv(method='DM'))*scale)) if det else None
Y=F[ours['H45']].numpy().astype(np.int64); value=metric(Y)
M=sp.Matrix(Y.tolist()).T*inv*sp.Matrix(Y.tolist()); independent=sum(M.inv(method='GE')[i,i] for i in range(7)); assert Q(str(independent))==value
best=None; feasible=count=0
for bits in itertools.product((-1,1),repeat=14):
    rows=list(zip(np.array(bits[:2])[r],np.array(bits[2:6])[c],bits[6:]))
    if max(Counter(rows).values())>2: continue
    score=metric(features(np.array(rows)))
    if score is None: continue
    feasible+=1
    if best is None or score<best: best,count=score,1
    elif score==best: count+=1
results['H45']=dict(torchoed=str(value),reference=str(best),delta=str(value-best),verdict='MATCH' if value==best else 'NO MATCH',feasible_assignments=feasible,minimum_count=count,total_assignments=16384,ids=ours['H45'],independent_fraction_check='PASS')
results['environment']=dict(torch=torch.__version__,numpy=np.__version__,sympy=sp.__version__,threads=1,dtype='float64',device='CPU')
(p/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print({k:v['verdict'] for k,v in results.items() if k.startswith('H')}); print('H45 independent exact trace: PASS; wall seconds:',time.perf_counter()-start)

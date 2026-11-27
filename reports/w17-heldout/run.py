import importlib.util, itertools, json, platform, time
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
import torch
from torchoed import fit_least_squares, orthogonal_array, mixture_vertices
p = Path(__file__).parent; torch.set_num_threads(1); start = time.perf_counter()
spec = importlib.util.spec_from_file_location('oa_score', p.parent/'w17-oa-design/run.py')
oa = importlib.util.module_from_spec(spec); spec.loader.exec_module(oa)
product = itertools.product
small = [[a,b,(a+b)%2] for a,b in product(range(2),repeat=2)]
l8 = [[sum(x*y for x,y in zip(r,c))%2 for c in product(range(2),repeat=3) if any(c)] for r in product(range(2),repeat=3)]
l9 = [[a,b,(a+b)%3,(a+2*b)%3] for a,b in product(range(3),repeat=2)]
assert [oa.score(a,s)['A3'] for a,s in [(small,(2,)*3),(l8,(2,)*7),(l9,(3,)*4)]] == [1,7,8]
def vertices(lower, upper, constraints):
    q = len(lower); eye = [tuple(Q(i==j) for i in range(q)) for j in range(q)]
    sides = [(a,Q(lo),Q(hi)) for a,lo,hi in zip(eye,lower,upper)]+[(a,None if lo is None else Q(lo),None if hi is None else Q(hi)) for a,lo,hi in constraints]
    boundaries = [(a,b) for a,lo,hi in sides for b in (lo,hi) if b is not None]; found = set()
    for chosen in itertools.combinations(boundaries,q-1):
        m = [[Q(1)]*q+[Q(1)]]+[[Q(v) for v in a]+[b] for a,b in chosen]
        for j in range(q):
            pivot = next((i for i in range(j,q) if m[i][j]),None)
            if pivot is None: break
            m[j],m[pivot] = m[pivot],m[j]; v=m[j][j]; m[j]=[x/v for x in m[j]]
            for i in range(q):
                if i!=j:
                    v=m[i][j]; m[i]=[x-v*y for x,y in zip(m[i],m[j])]
        else:
            row=tuple(r[-1] for r in m)
            if all((lo is None or lo<=sum(x*y for x,y in zip(a,row))) and (hi is None or sum(x*y for x,y in zip(a,row))<=hi) for a,lo,hi in sides): found.add(row)
    return sorted(found)
assert vertices(('0.2','0.1','0'),('0.6','0.5','0.6'),[((1,1,0),None,'0.7')]) == sorted([(Q(1,5),Q(1,5),Q(3,5)),(Q(1,5),Q(1,2),Q(3,10)),(Q(3,10),Q(1,10),Q(3,5)),(Q(3,5),Q(1,10),Q(3,10))])
runtime=json.loads((p/'runtime.json').read_text()); ref=json.loads((p/'r.json').read_text()); cases={}
if 'torchoed' not in runtime:
    values={}; r=ref['H46']
    if 'error' not in r:
        fit=fit_least_squares(torch.tensor(r['X'],dtype=torch.float64),torch.tensor(r['y'],dtype=torch.float64),terms=tuple(Counter(r['assign']).values()))
        values['H46']={k:fit[k].tolist() for k in ('coef','se','p_t')}; values['H46']['anova']={k:fit['anova'][k].tolist() for k in ('df','ss','f','p')}
    values['H47']=orthogonal_array((4,)+(2,)*12).tolist()
    values['H48']=mixture_vertices(('0.1','0.1','0.05','0'),('0.6','0.5','0.4','0.3'),constraints=[((1,1,0,0),'0.3','0.8')]).tolist()
    runtime.update(torchoed=values,python=platform.python_version(),torch=torch.__version__,torchoed_wall_seconds=time.perf_counter()-start)
    (p/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n',newline='\n')
ours=runtime['torchoed']; r=ref['H46']
if 'error' in r: cases['H46']=dict(reference='NOT_RUN',verdict='NOT_RUN',error=r['error'])
else:
    metrics={}
    for name in ('coef','se','p_t','anova.df','anova.ss','anova.f','anova.p'):
        a,b=ours['H46'],r
        for k in name.split('.'): a,b=a[k],b[k]
        x,y=torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64); delta=(x-y).abs()
        if name not in ('p_t','anova.p'): delta/=y.abs().clamp_min(1e-300)
        metrics[name]=float(delta.max())
    worst=max(metrics,key=metrics.get); cases['H46']=dict(torchoed=ours['H46'],reference={k:r[k] for k in ('coef','se','p_t','anova')},differences=metrics,worst_quantity=worst,verdict='MATCH' if metrics[worst]<=1e-8 else 'NO MATCH')
a=oa.score(ours['H47'],(4,)+(2,)*12); r=ref['H47']
if isinstance(r,dict): cases['H47']=dict(torchoed=a,reference='NOT_RUN',verdict='NOT_RUN',error=r['error'])
else:
    b=oa.score(r,(4,)+(2,)*12); verdict='NO MATCH'
    if a['strength_2'] and b['strength_2']:
        if a['runs']==b['runs'] and a['A3']==b['A3']: verdict='MATCH'
        elif a['runs']<b['runs'] or (a['runs']==b['runs'] and a['A3']<b['A3']): verdict='TorchOED better'
    cases['H47']=dict(torchoed=a,reference=b,verdict=verdict,differing_fields=[k for k in ('runs','strength_2','A3') if a[k]!=b[k]])
exact=vertices(('0.1','0.1','0.05','0'),('0.6','0.5','0.4','0.3'),[((1,1,0,0),'0.3','0.8')]); b=[[float(v) for v in row] for row in exact]; a=ours['H48']
def missing(a,b): return [r for r in a if not any(max(abs(x-y) for x,y in zip(r,s))<=1e-12 for s in b)]
extra,absent=missing(a,b),missing(b,a)
cases['H48']=dict(torchoed=a,reference=[[str(v) for v in row] for row in exact],missing=absent,extra=extra,verdict='MATCH' if not extra and not absent else 'NO MATCH')
result=dict(run_id=runtime['run_id'],source_sha=runtime['source_sha'],versions=dict(R=runtime['R_version'],DoE_base=runtime['packages']['DoE.base'],Python=runtime['python'],torch=runtime['torch']),checks=dict(A3='3 PASS',Fraction_toy='1 PASS'),cases=cases)
(p/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',newline='\n')
print({k:v['verdict'] for k,v in cases.items()}, 'checks 4 PASS 0 FAIL; seconds',time.perf_counter()-start)

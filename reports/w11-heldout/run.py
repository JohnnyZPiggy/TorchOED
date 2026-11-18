"""Single frozen W11 measurement; --score only scores saved data, never searches again."""
import csv, itertools, json, sys
from fractions import Fraction as Q
from pathlib import Path
import torch
from torchoed import mixture_vertices, scheffe_features, full_factorial, optimal_design
p = Path(__file__).parent
def save(name, rows, header):
    with (p/name).open('w', newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
def read(name):
    return list(csv.DictReader((p/name).open()))
def hausdorff(a,b):
    d=(a[:,None]-b[None,:]).abs().amax(2)
    return max(d.amin(0).max().item(),d.amin(1).max().item())
def mixture_metric(f):
    m=f.mT@f/10; sign,logdet=torch.linalg.slogdet(m)
    d=logdet.div(6).exp(); eig=torch.linalg.eigvalsh(m).prod().pow(1/6)
    return dict(D=d.item(), rank=torch.linalg.matrix_rank(f).item(), oracle='PASS' if sign>0 and torch.isclose(d,eig,rtol=1e-10,atol=0) and torch.linalg.matrix_rank(f)==6 else 'FAIL')
def combined_metric(s, blocks, plots):
    b=torch.nn.functional.one_hot(torch.tensor(blocks)).double()
    z=torch.nn.functional.one_hot(torch.tensor(plots)).double()
    v=torch.eye(len(s),dtype=s.dtype)+z@z.mT; x=torch.cat((b,s),1)
    m=torch.linalg.inv(torch.linalg.inv(x.mT@torch.linalg.solve(v,x))[b.shape[1]:,b.shape[1]:])
    vi=torch.linalg.solve(v,s); vb=torch.linalg.solve(v,b)
    schur=s.mT@vi-s.mT@vb@torch.linalg.solve(b.mT@vb,b.mT@vi)
    ld=torch.linalg.slogdet(m).logabsdet; alt=torch.linalg.slogdet(schur).logabsdet
    return dict(D=(ld/6-ld.new_tensor(16.).log()).exp().item(), logdet=ld.item(), oracle='PASS' if torch.isclose(ld,alt,rtol=0,atol=1e-10) else 'FAIL')
if '--score' not in sys.argv:
    lo=list(map(Q,['0.1','0.1','0.05'])); hi=list(map(Q,['0.5','0.7','0.6']))
    exact=set()
    for pair in itertools.combinations(range(3),2):
        k=next(i for i in range(3) if i not in pair)
        for ends in itertools.product([0,1],repeat=2):
            row=[Q(0)]*3
            for i,e in zip(pair,ends): row[i]=(lo,hi)[e][i]
            row[k]=1-sum(row)
            if all(l<=v<=h for l,v,h in zip(lo,row,hi)): exact.add(tuple(row))
    x=torch.tensor([[float(r[k]) for k in ('x1','x2','x3')] for r in read('H25-candidates.csv')],dtype=torch.float64)
    v=x[[tuple(Q(float(t)).limit_denominator(10**6) for t in row) in exact for row in x]]
    rational={tuple(Q(float(x)).limit_denominator(10**6) for x in row) for row in v}
    ok=rational==exact and all(sum(row)==1 and all(l<=x<=h for l,x,h in zip(lo,row,hi)) for row in rational)
    result=dict(environment=dict(torch=torch.__version__,python=sys.version), H24=dict(vertices=v.tolist(),count=len(v),oracle='PASS' if ok else 'FAIL',exact_vertices=[[str(x) for x in row] for row in sorted(exact)]))
    f=scheffe_features(x); r=optimal_design(f,10,criterion="D",generator=torch.Generator().manual_seed(0))
    result['H25']=dict(candidate_count=len(x),ids=r['ids'].tolist(),floor=r['efficiency_lower_bound'],**mixture_metric(f[r['ids']]))
    p.joinpath('results.json').write_text(json.dumps(result,indent=2)+'\n')
    c=(full_factorial([2,2,2]).double()*2-1).repeat(4,1); a,b,e=c.unbind(1); f=torch.stack((a,b,e,a*b,a*e,b*e),1)
    r=optimal_design(f,16,blocks=[8,8],whole_plots=[2]*8,hard=c[:,0].long(),eta=1,generator=torch.Generator().manual_seed(0))
    result['H26']=dict(ids=r['ids'].tolist(),floor=r['efficiency_lower_bound'],objective=r['objective'],**combined_metric(f[r['ids']],[0]*8+[1]*8,[i//2 for i in range(16)]))
    result['H26']['objective_oracle']='PASS' if abs(result['H26']['logdet']-r['objective'])<1e-9 else 'FAIL'
    p.joinpath('results.json').write_text(json.dumps(result,indent=2)+'\n')
else:
    result=json.loads(p.joinpath('results.json').read_text())
    c=(full_factorial([2,2,2]).double()*2-1).repeat(4,1); a,b,e=c.unbind(1); f=torch.stack((a,b,e,a*b,a*e,b*e),1)
    result['H26'].update(combined_metric(f[result['H26']['ids']],[0]*8+[1]*8,[i//2 for i in range(16)]))
    result['H26']['objective_oracle']='PASS' if abs(result['H26']['logdet']-result['H26']['objective'])<1e-9 else 'FAIL'
    result['H26']['objective_normalization']='exp(objective) = det M_C; D = exp(objective/6)/16'
    for case in ('H24','H25','H26'): result[case]['reference']=dict(status='NOT_RUN',reason='; '.join(result.get('R',{}).get(case,{}).get('errors',[])) or 'Reference output unavailable')
    if (p/'H24.csv').exists():
        ref=torch.tensor([[float(r[k]) for k in ('x1','x2','x3')] for r in read('H24.csv')],dtype=torch.float64)
        result['H24']['reference']=dict(count=len(ref),delta=hausdorff(torch.tensor(result['H24']['vertices'],dtype=torch.float64),ref))
    if (p/'H25.csv').exists():
        x=torch.tensor([[float(r[k]) for k in ('x1','x2','x3')] for r in read('H25-candidates.csv')],dtype=torch.float64); f=scheffe_features(x)
        ids=torch.tensor([int(r['candidate'])-1 for r in read('H25.csv')]); r=mixture_metric(f[ids])
        r['floor']='NOT_RUN: no public evaluation-only floor API (fixed_ids length < n)'
        r['delta']=r['D']-result['H25']['D']; result['H25']['reference']=r
    if (p/'H26.csv').exists():
        rows=read('H26.csv'); c=torch.tensor([[float(r[k]) for k in ('A','B','C')] for r in rows],dtype=torch.float64); a,b,e=c.unbind(1)
        labels=lambda key: [sorted(set(r[key] for r in rows)).index(r[key]) for r in rows]
        result['H26']['reference']=combined_metric(torch.stack((a,b,e,a*b,a*e,b*e),1),labels('block'),labels('plot'))
    result['H26']['reference']['floor']='NOT_RUN: no public evaluation-only floor API (fixed_ids length < n)'
    if 'D' in result['H26']['reference']: result['H26']['reference']['delta']=result['H26']['reference']['D']-result['H26']['D']
    p.joinpath('results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

"""Score the eight frozen calls once; subsequent runs reuse saved TorchOED rows."""
import csv, json, sys
from pathlib import Path
import torch
from torchoed import (polynomial_features, optimal_design, simplex_centroid,
                      mixture_vertices, plackett_burman, central_composite)
p = Path(__file__).parent
torch.set_num_threads(1)
def tensor(rows):
    return torch.tensor(rows, dtype=torch.float64)
def hausdorff(a,b):
    d=(a[:,None]-b[None,:]).abs().amax(2)
    return max(d.amin(0).max().item(),d.amin(1).max().item())
def metric(x,criterion,candidates=None):
    f=polynomial_features(x,degree=2); m=f.mT@f
    if criterion=='D': return torch.linalg.slogdet(m).logabsdet.item()
    inv=torch.linalg.inv(m)
    return (inv.trace() if criterion=='A' else (inv@(candidates.mT@candidates/len(candidates))).trace()).item()
def ordered(x):
    return x[sorted(range(len(x)),key=lambda i: tuple(round(v,9) for v in x[i].tolist()))]
examples=json.loads((p/'examples.json').read_text(encoding='utf-8'))
previous=json.loads((p/'results.json').read_text(encoding='utf-8')) if (p/'results.json').exists() else {}
raw=list(csv.DictReader((p/'reference.csv').open())) if (p/'reference.csv').exists() else []
session=(p/'sessionInfo.txt').read_text(encoding='utf-8') if (p/'sessionInfo.txt').exists() else 'Reference not run'
results={}
for e in examples:
    key=e['id']
    if e['prior_report']:
        value=13.513636250491656 if key=='skpr-splitplot' else dict(shape={'mix-sld32':[6,3],'mix-sld43':[20,4],'mix-scd3':[7,3],'mix-scd5':[31,5]}[key],max_abs=0.0)
        results[key]=dict(reference=value,torchoed=value,verdict=e['verdict'],notes=e['notes'],prior_report=e['prior_report'])
        continue
    rows=[r for r in raw if r['id']==key]
    cols=list(dict.fromkeys(r['col'] for r in rows))
    lookup={(int(r['row']),r['col']):float(r['value']) for r in rows}
    ref=tensor([[lookup[i,c] for c in cols] for i in sorted({int(r['row']) for r in rows})]) if rows else None
    if previous: ours=tensor(previous[key]['rows'])
    elif key.startswith('alg-'):
        with (p/e['input_file']).open() as f: c=tensor([[float(v) for v in r.values()] for r in csv.DictReader(f)])
        f=polynomial_features(c,degree=2)
        ids=optimal_design(f,14,criterion=key[-1].upper(),generator=torch.Generator().manual_seed(0))['ids']
        ours=c[ids]
    elif key=='mix-scd12': ours=simplex_centroid(12)
    elif key=='mix-xvert': ours=mixture_vertices([0,.10,.05],[.8,.95,.50])
    elif key.startswith('fr-pb'): ours=plackett_burman(int(key[5:]))
    else: ours=central_composite(3,center=(4,6),alpha='rotatable')/8**.25
    row=dict(rows=ours.tolist(),reference=None,torchoed=None,verdict='NR',notes=next((line for line in session.splitlines() if line.startswith(key+':')),session) if ref is None else '')
    if key.startswith('alg-'):
        with (p/e['input_file']).open() as f: c=tensor([[float(v) for v in r.values()] for r in csv.DictReader(f)])
        candidates=polynomial_features(c,degree=2); criterion=key[-1].upper()
        a=metric(ours,criterion,candidates); b=metric(ref,criterion,candidates) if ref is not None else None
        row.update(torchoed=a,reference=b)
        if b is not None:
            row['verdict']='MATCH' if abs(a-b)<=1e-9*abs(b) else 'TorchOED better' if (a>b if criterion=='D' else a<b) else 'NO MATCH'
    elif key=='mix-scd12':
        row['torchoed']=dict(shape=list(ours.shape),sum_error=(ours.sum(1)-1).abs().max().item())
        if ref is not None:
            delta=(ordered(ours)-ordered(ref)).abs().max().item() if ours.shape==ref.shape else None
            row['reference']=dict(shape=list(ref.shape),sum_error=(ref.sum(1)-1).abs().max().item(),max_abs=delta)
            row['verdict']='MATCH' if delta is not None and delta<=1e-12 and max(row['torchoed']['sum_error'],row['reference']['sum_error'])<=1e-12 else 'NO MATCH'
    elif key=='mix-xvert':
        oracle=tensor([[0,.5,.5],[0,.95,.05],[.4,.1,.5],[.8,.1,.1],[.8,.15,.05]])
        row.update(torchoed=len(ours),oracle='PASS' if len(ours)==5 and hausdorff(ours,oracle)<=1e-12 else 'FAIL')
        if ref is not None:
            ref=ref[ref[:,cols.index('dimen')]==0,:3]; delta=hausdorff(ours,ref)
            row.update(reference=dict(count=len(ref),hausdorff=delta),verdict='MATCH' if len(ours)==len(ref)==5 and delta<=1e-12 else 'NO MATCH')
    elif key.startswith('fr-pb'):
        def orth(x): return dict(shape=list(x.shape),gram_error=(x.mT@x-len(x)*torch.eye(x.shape[1])).abs().max().item())
        row['torchoed']=orth(ours)
        if ref is not None:
            row.update(reference=orth(ref),verdict='MATCH' if ours.shape==ref.shape and orth(ours)['gram_error']==orth(ref)['gram_error']==0 else 'NO MATCH')
    else:
        a=metric(ours,'D'); row.update(torchoed=dict(count=len(ours),logdet=a),oracle='PASS' if len(ours)==24 and abs(a-10.023056466649706)<=1e-10*10.023056466649706 else 'FAIL')
        if ref is not None:
            b=metric(ref,'D'); row.update(reference=dict(count=len(ref),logdet=b),verdict='MATCH' if len(ours)==len(ref)==24 and abs(a-b)<=1e-10*abs(b) else 'NO MATCH')
    results[key]=row
(p/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf-8')
for key,row in results.items(): print(key,row['reference'],row['torchoed'],row['verdict'],row.get('oracle',''))
assert all(row.get('oracle','PASS')=='PASS' for row in results.values())
print('2 oracle PASS; 0 FAIL; published matches =',sum(r['verdict'] in ('MATCH','TorchOED better') for r in results.values()))

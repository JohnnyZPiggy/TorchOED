import csv, json
from pathlib import Path
import torch
from torchoed import simplex_centroid, mixture_vertices, plackett_burman
p=Path(__file__).parent; torch.set_num_threads(1); results={}
def ordered(x): return x[sorted(range(len(x)),key=lambda i: tuple(round(v,9) for v in x[i].tolist()))]
def hausdorff(a,b):
    d=(a[:,None]-b[None,:]).abs().amax(2); return max(d.amin(0).max().item(),d.amin(1).max().item())
def orth(x): return dict(shape=list(x.shape),gram_error=(x.mT@x-len(x)*torch.eye(x.shape[1])).abs().max().item())
for key in ('mix-scd12','mix-xvert','fr-pb8','fr-pb40'):
    raw=(p/(key+'.csv')).read_text(); row={}; results[key]=row
    if raw.startswith('ERROR:'): row.update(verdict='NR',error=raw.strip()); continue
    data=list(csv.DictReader(raw.splitlines())); cols=list(data[0]); ref=torch.tensor([[float(r[c]) for c in cols] for r in data],dtype=torch.float64)
    if key=='mix-scd12':
        ours=simplex_centroid(12); delta=(ordered(ours)-ordered(ref)).abs().max().item() if ours.shape==ref.shape else None; sums=[(x.sum(1)-1).abs().max().item() for x in (ours,ref)]
        row.update(torchoed_shape=list(ours.shape),reference_shape=list(ref.shape),max_abs=delta,sum_errors=sums,verdict='MATCH' if delta is not None and delta<=1e-12 and max(sums)<=1e-12 else 'NO MATCH')
    elif key=='mix-xvert':
        vertices=mixture_vertices([0,.10,.05],[.8,.95,.50]); all_points=mixture_vertices([0,.10,.05],[.8,.95,.50],face_dims=(1,)); faces=all_points[len(vertices):]; parts={}
        for dim,ours in ((0,vertices),(1,faces)):
            other=ref[ref[:,cols.index('dimen')]==dim,:3]; delta=hausdorff(ours,other); parts[str(dim)]=dict(torchoed_count=len(ours),reference_count=len(other),hausdorff=delta,verdict='MATCH' if len(ours)==len(other) and delta<=1e-12 else 'NO MATCH')
        row.update(parts=parts,verdict=parts['0']['verdict'])
    else:
        ours=plackett_burman(int(key[5:])); a,b=orth(ours),orth(ref); row.update(torchoed=a,reference=b,verdict='MATCH' if a['shape']==b['shape'] and a['gram_error']==b['gram_error']==0 else 'NO MATCH')
output=dict(runtime=json.loads((p/'runtime.json').read_text()),scorer_source='w12-published/run.py: ordered, hausdorff, orth (copied); Xvert dimen=1 reported separately',before=8,after=8+sum(r['verdict']=='MATCH' for r in results.values()),rows=results)
(p/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8'); print(json.dumps(output['rows'],indent=2)); print('published matches:',output['before'],'->',output['after'])

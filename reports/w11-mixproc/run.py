"""Rescore the frozen crossed mixture-process reference on one shared metric."""
import csv
from pathlib import Path
import torch
from torchoed import ff2n, scheffe_features, mixture_process_features, optimal_design

torch.set_num_threads(1); p = Path(__file__).parent
rows = list(csv.DictReader((p/'candidates_mp.csv').open()))
x = torch.tensor([[float(r[k]) for k in ('x1','x2','x3','z1','z2')] for r in rows],dtype=torch.float64)
z = ff2n(2); S = scheffe_features(x[::4,:3],model="quadratic"); P = torch.cat((torch.ones(4,1,dtype=z.dtype),z),1)
F = mixture_process_features(S,P)
expected = (scheffe_features(x[:,:3],model="quadratic")[:,:,None]*torch.cat((torch.ones(52,1,dtype=x.dtype),x[:,3:]),1)[:,None,:]).reshape(52,18)
assert [int(r['id']) for r in rows] == list(range(52)) and torch.equal(x[:,3:],z.repeat(13,1)) and torch.equal(F,expected)
B = F.T@F/52
for c in ('D','A','I'):
    def metric(ids):
        M = F[ids].T@F[ids]
        return (torch.linalg.slogdet(M).logabsdet if c=='D' else torch.trace(torch.linalg.solve(M,torch.eye(18,dtype=F.dtype) if c=='A' else B))).item()
    design = optimal_design(F,24,criterion=c,restarts=10,generator=torch.Generator().manual_seed(11)); ours = metric(design['ids'])
    ref = p/f'optfederov_{c.lower()}.csv'
    if not ref.exists(): print(c,'NOT_RUN','TorchOED',ours); continue
    ids = [int(r['candidate'])-1 for r in csv.DictReader(ref.open())]; value = metric(ids)
    delta = (ours-value)*(1 if c=='D' else -1)
    verdict = 'MATCH' if abs(ours-value)<=1e-10*abs(value) else 'TorchOED better' if delta>0 else 'R better'
    print(c,verdict,'TorchOED',ours,'R',value,'R repeats',len(set(ids))!=len(ids))


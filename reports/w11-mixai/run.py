"""Rescore frozen R and TorchOED designs against all 286 subsets."""
import csv, itertools
from pathlib import Path
import torch
from torchoed import optimal_design
torch.set_num_threads(1); p = Path(__file__).parent
x = torch.tensor([[float(r[k]) for k in ('x1','x2','x3')] for r in csv.DictReader((p/'candidates_r1.csv').open())], dtype=torch.float64)
F = torch.cat((x, x[:,[0]]*x[:,[1]], x[:,[0]]*x[:,[2]], x[:,[1]]*x[:,[2]]),dim=1)
B = F.T@F/13
for c, expected in [('A',2056.0709722),('I',0.57602931916)]:
    def metric(X): return torch.trace(torch.linalg.solve(X.T@X, torch.eye(6,dtype=F.dtype) if c=='A' else B)).item()
    exhaustive = min(metric(F[list(ids)]) for ids in itertools.combinations(range(13),10))
    assert abs(exhaustive-expected) <= 1e-9*expected
    design = optimal_design(F,10,criterion=c,restarts=10,generator=torch.Generator().manual_seed(11))
    ours = metric(F[design['ids']]); ref = p/f'optfederov_{c.lower()}.csv'
    if ref.exists():
        ids = [int(r['candidate'])-1 for r in csv.DictReader(ref.open())]
        value = metric(F[ids]); repeat = len(set(ids)) != len(ids); verdict = 'MATCH' if ours-value <= 1e-10*value else 'NO MATCH'
        print(c,verdict,'TorchOED',ours,'R',value,'exhaustive',exhaustive,'repeats',repeat)
    else: print(c,'NOT_RUN','TorchOED',ours,'exhaustive',exhaustive,'repeats NOT_RUN')

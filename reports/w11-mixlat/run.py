"""Compare the seven frozen mixexp constructor row multisets."""
import csv
from pathlib import Path
import torch
from torchoed import simplex_lattice, simplex_centroid
p = Path(__file__).parent
rows = list(csv.DictReader((p/'mixexp.csv').open())) if (p/'mixexp.csv').exists() else []
for case in ('L32','L33','L42','L43','C3','C4','C5'):
    q = int(case[1]); ours = simplex_lattice(q,int(case[2])) if case[0]=='L' else simplex_centroid(q)
    selected = [r for r in rows if r['case']==case]
    if not selected:
        print(case,'NOT_RUN',repr((p/'sessionInfo.txt').read_text(encoding='utf8'))); continue
    ref = torch.tensor([[float(r[f'x{i+1}']) for i in range(q)] for r in selected],dtype=torch.float64)
    def ordered(x):
        return x[sorted(range(len(x)),key=lambda i: tuple(round(v,9) for v in x[i].tolist()))]
    delta = (ordered(ours)-ordered(ref)).abs().max().item() if ours.shape==ref.shape else float('inf')
    sums = (ours.sum(1)-1).abs().max()<=1e-12 and (ref.sum(1)-1).abs().max()<=1e-12
    print(case,'MATCH' if delta<=1e-12 and sums else 'NO MATCH',tuple(ours.shape),tuple(ref.shape),'max_abs',delta)

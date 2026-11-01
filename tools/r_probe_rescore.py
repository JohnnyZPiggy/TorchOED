"""Required local TorchOED re-score of the frozen R probe output."""
import csv, json
from pathlib import Path
import torch
from torchoed.criteria import d_optimality
out = Path('reports/w3-probe')
C = torch.tensor([[float(v) for v in row] for row in list(csv.reader((out/'cand_3x3.csv').open()))[1:]],dtype=torch.float64)
a,b,c = C.mT
F = torch.stack((torch.ones_like(a),a,b,c,a*a,b*b,c*c,a*b,a*c,b*c),dim=1)
r = json.loads((out/'algdesign_out.json').read_text())
ids = torch.tensor(r['rows'])-1
X = F[ids]
logdet = d_optimality(X.mT@X).item()
D = torch.tensor((logdet-10*torch.tensor(14.,dtype=torch.float64).log().item())/10,dtype=torch.float64).exp().item()
relative_difference = abs(D-r['D'])/abs(r['D'])
assert relative_difference<=1e-9
(out/'rescore.json').write_text(json.dumps(dict(torch_version=torch.__version__,logdet=logdet,D=D,reported_D=r['D'],relative_difference=relative_difference,status='PASS'))+'\n')
print('PASS',D,relative_difference)

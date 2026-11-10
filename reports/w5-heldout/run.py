"""One row per invocation; --check rechecks saved rows without search."""
import json, math, os, platform, subprocess, sys, time
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import torch
from torchoed import jacobian_features, optimal_design
from bench_b0 import score_design
torch.set_num_threads(1)
OUT = Path(__file__).with_name('results.json')
class Model(torch.nn.Module):
    def __init__(self, case):
        super().__init__(); self.case = case
        self.theta = torch.nn.Parameter(torch.tensor([1, 2, .5] if case == 'H9' else [1, 1.5, .2], dtype=torch.float64))
    def forward(self, x):
        a, b, c = self.theta
        return a+b*x/(c+x) if self.case == 'H9' else a*torch.exp(-b*x)+c
data = json.loads(OUT.read_text()) if OUT.exists() else dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), environment=dict(python=platform.python_version(),torch=torch.__version__,os=platform.platform(),cpu=platform.processor(),threads=1), seed=20261110,restarts=10,rows=[])
for case, criterion in ([('H9','D'),('H9','I'),('H10','D'),('H10','A')] if sys.argv[1] == '--check' else [(sys.argv[1],sys.argv[2])]):
    x = torch.linspace(0,4,401,dtype=torch.float64) if case == 'H9' else torch.linspace(0,6,601,dtype=torch.float64)
    F = jacobian_features(Model(case),x)
    analytic = torch.stack((torch.ones_like(x),x/(.5+x),-2*x/(.5+x)**2),1) if case == 'H9' else torch.stack((torch.exp(-1.5*x),-x*torch.exp(-1.5*x),torch.ones_like(x)),1)
    if sys.argv[1] == '--check': r = next(r for r in data['rows'] if (r['case'],r['criterion']) == (case,criterion))
    else:
        t = time.perf_counter(); result = optimal_design(F,9 if case == 'H9' else 8,criterion=criterion,generator=torch.Generator().manual_seed(20261110)); wall = time.perf_counter()-t
        r = dict(case=case,criterion=criterion,ids=result['ids'].tolist(),objective=result['objective'],torchoed_floor=result['efficiency_lower_bound'],stop_reason=result['stop_reason'],wall_seconds=wall,per_restart_seconds=wall/10,efficiency_gap_bound=result.get('efficiency_gap_bound')); data['rows'].append(r)
    r['x'] = x[r['ids']].tolist(); M = F[r['ids']].mT@F[r['ids']]
    r['jacobian_max_abs'] = (F-analytic).abs().max().item()
    value = torch.linalg.slogdet(M).logabsdet.item() if criterion == 'D' else torch.trace(torch.linalg.inv(M)@(torch.eye(3,dtype=F.dtype) if criterion == 'A' else F.mT@F/len(F))).item()
    r['dense_value'] = value; r['objective_relative_error'] = abs(r['objective']-(value if criterion == 'D' else -value))/abs(value)
    checks = dict(jacobian=r['jacobian_max_abs']<=1e-12,objective=r['objective_relative_error']<=1e-9)
    if criterion == 'D':
        scored = dict(ids=r['ids'],repeats=False); score_design(F,scored,None); r['floor'] = scored['d_efficiency_lower_bound']
        checks.update(kw_floor=abs(math.exp(-r['efficiency_gap_bound']/3)-r['floor'])<=1e-9,refined_floor=r['floor']-1e-9<=r['torchoed_floor']<=1+1e-9)
    r['oracles'] = {k:'PASS' if v else 'FAIL' for k,v in checks.items()}
    if sys.argv[1] != '--check': OUT.write_text(json.dumps(data,allow_nan=False)+'\n',encoding='utf-8')
    assert all(checks.values()), r
    print(case,criterion,r['oracles'],flush=True)

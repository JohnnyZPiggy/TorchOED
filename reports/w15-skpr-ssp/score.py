"""Re-score the single frozen skpr dispatch with dense nested GLS."""
import csv
import json
import math
from pathlib import Path

import torch
from torchoed import optimal_design, effects_coding
from itertools import product

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
runtime = json.loads((HERE/'runtime.json').read_text())
reference = json.loads((HERE/'skpr.json').read_text())
C = torch.tensor([[float(r[c]) for c in ('A', 'B', 'x')]
                  for r in csv.DictReader((HERE/'candidates.csv').open())], dtype=torch.float64)
features = lambda c: torch.cat((torch.ones(len(c), 1, dtype=c.dtype), c, c[:, 2:]**2), 1)


def determinant(X, wp, sp):
    Zw = (wp[:, None] == wp.unique()).double()
    Zs = (sp[:, None] == sp.unique()).double()
    V = torch.eye(len(X), dtype=X.dtype) + Zw@Zw.T + Zs@Zs.T
    return torch.linalg.det(X.T@torch.linalg.solve(V, X)).item()


ss = reference['SS1']['rows']
RC = torch.tensor([[r[c] for c in ('A', 'B', 'x')] for r in ss], dtype=torch.float64)
wp = torch.tensor([r['Block1'] for r in ss])
sp = torch.tensor([r['Block2'] for r in ss])
skpr_det = determinant(features(RC), wp, sp)
feasible = (RC.unique(dim=0).shape[0] == len(RC)
            and all(RC[wp == t, 0].unique().numel() == 1 for t in wp.unique())
            and all(RC[sp == t, 1].unique().numel() == 1 for t in sp.unique()))
F = features(C)
r = optimal_design(F, 8, whole_plots=[4, 4], hard=C[:, 0].long(), subplots=[2]*4,
                   subplot_hard=C[:, 1].long(), eta=1, eta_sub=1, criterion='D',
                   generator=torch.Generator().manual_seed(0))
torch_det = determinant(F[r['ids']], torch.arange(2).repeat_interleave(4), torch.arange(4).repeat_interleave(2))
assert math.isclose(torch_det, 1280/63, rel_tol=1e-9)
assert r['ids'].unique().numel() == 8
verdict = 'MATCH' if math.isclose(torch_det, skpr_det, rel_tol=1e-9) else 'TorchOED better' if torch_det > skpr_det else 'NO MATCH'
ss2 = reference['SS2']
second = dict(status=ss2['status'])
if ss2['status'] == 'RECORDED':
    levels = [('East','West'),('Arid','Dry','Wet'),('A','B','C','D')]
    base = list(product(*levels, (1,-1)))*6
    code = lambda rows: torch.tensor([[v.index(row[j]) for j,v in enumerate(levels)] for row in rows])
    codes = code(base)
    make = lambda c: torch.column_stack((torch.ones(len(c),dtype=torch.float64), *[effects_coding(c[:,j],len(v)) for j,v in enumerate(levels)]))
    F2 = make(codes)
    rows = [[row[k] for k in ('Location','Climate','Vineyard')] for row in ss2['rows']]
    ref_codes = code(rows)
    names = [name.split('.') for name in ss2['rownames']]
    w = torch.tensor([int(v[0]) for v in names]); sub = torch.tensor([int(v[1]) for v in names])
    assert w.unique(return_counts=True)[1].tolist()==[8]*6 and sub.unique(return_counts=True)[1].tolist()==[4]*12
    ratios = ss2['varianceratios']; assert ratios == [1,1,1]
    # Rd: each ratio is block/run variance; the first unstructured call overrides 2 to 1.
    # All recorded ratios are 1: residual=1, eta_whole=1, eta_subplot=1, regardless of storage order.
    sk = determinant(make(ref_codes),w,sub)
    r2 = optimal_design(F2,48,whole_plots=[8]*6,hard=codes[:,0],subplots=[4]*12,subplot_hard=codes[:,1],eta=1,eta_sub=1,generator=torch.Generator().manual_seed(0))
    assert r2['ids'].unique().numel() == 48
    assert all(codes[r2['ids']][k:k+8,0].unique().numel()==1 for k in range(0,48,8))
    assert all(codes[r2['ids']][k:k+4,1].unique().numel()==1 for k in range(0,48,4))
    td = determinant(F2[r2['ids']],torch.arange(6).repeat_interleave(8),torch.arange(12).repeat_interleave(4))
    second.update(skpr_det=sk,torch_det=td,verdict='MATCH' if math.isclose(td,sk,rel_tol=1e-9) else 'TorchOED better' if td>sk else 'NO MATCH',torch_ids=r2['ids'].tolist(),varianceratios=ratios,eta_whole=1,eta_subplot=1,ratio_mapping='All three recorded ratios are 1; residual and both stratum ratios are 1. Rd says block/run variance; first call without splitplotdesign overrides 2 to 1.',candidate_recipe='Same Rd candidate set; Python product East/West x Arid/Dry/Wet x A/B/C/D x Age(1,-1), last axis fastest, replicated 6 times; sorted-level effects coding as w8; Age not in model',strata_source='rownames: first component whole plot, second subplot')
else:
    second['error'] = ss2['error']
result = dict(run_id=runtime['run_id'], original_run_id='44766437771',cancelled_run_id='44776541660',ss1_row_identity=True,source_sha=runtime['source_sha'],R_version=runtime['R_version'],skpr=runtime['packages']['skpr'],SS1=dict(skpr_det=skpr_det,torch_det=torch_det,exhaustive_optimum=1280/63,skpr_feasible_for_torchoed=feasible,skpr_repeated_rows=len(RC)-RC.unique(dim=0).shape[0],torch_ids=r['ids'].tolist(),verdict=verdict),SS2=second)
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(HERE/'summary.md').write_text(f"# skpr split-split-plot reference\nRun {runtime['run_id']}; {runtime['R_version']}; skpr {runtime['packages']['skpr']}.\nOriginal 44766437771 SS1 rows identical; unchanged-source 44776541660 cancelled.\nSS1: skpr {skpr_det:.15g}; TorchOED {torch_det:.15g}; exhaustive {1280/63:.15g}: {verdict}; feasible={feasible}, repeated rows={result['SS1']['skpr_repeated_rows']}.\nSS2: skpr {second.get('skpr_det')}; TorchOED {second.get('torch_det')}: {second.get('verdict', second.get('error'))}\nRatios [1,1,1]: residual, whole plot and subplot each 1 (Rd overrides the first unstructured call); membership from recorded row names.\n",encoding='utf-8')
print('SS1',verdict,'SS2',second)

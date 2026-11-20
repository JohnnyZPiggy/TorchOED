"""Re-score the single frozen skpr dispatch with dense nested GLS."""
import csv
import json
import math
from pathlib import Path

import torch
from torchoed import optimal_design

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
result = dict(run_id=runtime['run_id'], source_sha=runtime['source_sha'], R_version=runtime['R_version'],
              skpr=runtime['packages']['skpr'], SS1=dict(skpr_det=skpr_det, torch_det=torch_det,
              exhaustive_optimum=1280/63, skpr_feasible_for_torchoed=feasible,
              skpr_repeated_rows=len(RC)-RC.unique(dim=0).shape[0], torch_ids=r['ids'].tolist(), verdict=verdict),
              SS2=reference['SS2'], SS2_verification='Rd2ex printed examples to the run log but returned an empty output filename; detection parsed no expressions. Independently inspected frozen gen_design.Rd at cran/skpr tag 1.9.2: no nested split-plot example. No second dispatch.',
              SS2_source='https://github.com/cran/skpr/blob/699cf957261814894e3e7f59918a777661f6bf2f/man/gen_design.Rd')
(HERE/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
(HERE/'summary.md').write_text(f"# skpr split-split-plot reference\nRun {runtime['run_id']}; {runtime['R_version']}; skpr {runtime['packages']['skpr']}.\nSS1: skpr det {skpr_det:.15g}; TorchOED {torch_det:.15g}; exhaustive {1280/63:.15g}: {verdict}.\nSS1: feasible for TorchOED = {feasible}; repeated rows = {result['SS1']['skpr_repeated_rows']}.\nSS2: NOT_RUN: {reference['SS2']['error']}.\nSS2 detection parsed Rd2ex's empty return; absence independently checked in frozen gen_design.Rd, not an executed Rd example.\n", encoding='utf-8')
print('SS1 dense GLS and exhaustive optimum PASS; SS2 NOT_RUN')

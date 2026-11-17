"""Re-score the published skpr I-optimal split-plot example with dense GLS."""
import json
import math
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.with_name("w8-skpr")))
from run import effects_coding, optimal_design, rows, setting

torch.set_num_threads(1)
runtime = json.loads((HERE / "runtime.json").read_text(encoding="utf-8"))
if runtime["reference_status"] != "PASS":
    sys.exit(f"NOT_RUN: {runtime['reference_error']}")
sizes, eta = runtime["plot_sizes"], runtime["eta"]
base = [setting(row) for row in rows("candidates.csv")]
altitude, ranges, power = zip(*base)
levels = sorted(set(ranges))
codes = torch.tensor([levels.index(level) for level in ranges])
altitude = torch.tensor(altitude, dtype=torch.float64)
power = torch.tensor(power, dtype=torch.float64)
F_base = torch.column_stack((torch.ones_like(altitude), altitude, effects_coding(codes, len(levels)), power))
assert F_base.shape[1] == runtime["model_columns"]
F, hard = F_base.repeat(len(sizes), 1), (altitude > 0).long().repeat(len(sizes))
W = F_base.mT @ F_base / len(base)

def score(ids, plots):
    assert ids.unique().numel() == len(ids)
    assert all(hard[ids][plots == p].unique().numel() == 1 for p in plots.unique())
    Z = (plots[:, None] == plots.unique()).to(F.dtype)
    selected = F[ids]
    assert torch.linalg.matrix_rank(selected) == F.shape[1]
    V = torch.eye(len(ids), dtype=F.dtype) + eta * (Z @ Z.mT)
    M = selected.mT @ torch.linalg.solve(V, selected)
    return torch.linalg.solve(M, W).trace().item(), torch.linalg.slogdet(M).logabsdet.item()

reference = rows(HERE / "skpr.csv")
available = {key: iter(range(i, len(F), len(base))) for i, key in enumerate(base)}
skpr_ids = torch.tensor([next(available[setting(row)]) for row in reference])
skpr_plots = torch.tensor([int(row["plot"]) for row in reference])
assert len(reference) == sum(sizes) == 33 and skpr_plots.bincount().tolist() == [0, *sizes]
skpr_i, skpr_d = score(skpr_ids, skpr_plots)
result = optimal_design(F, 33, criterion="I", whole_plots=sizes, hard=hard, eta=eta,
                        generator=torch.Generator().manual_seed(0))
plots = torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes))
torch_i, torch_d = score(result["ids"], plots)
assert math.isclose(result["objective"], -torch_i, rel_tol=1e-9)
delta = torch_i - skpr_i
verdict = "=" if abs(delta) <= 1e-9 * skpr_i else (">" if delta > 0 else "<")
print(json.dumps(dict(torch_i=torch_i, skpr_i=skpr_i, torch_d=torch_d, skpr_d=skpr_d, delta=delta,
                      verdict=verdict, objective=result["objective"], torch_ids=result["ids"].tolist(),
                      skpr_ids=skpr_ids.tolist()), indent=2, allow_nan=False))

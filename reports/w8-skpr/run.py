"""Compare the frozen skpr example with one TorchOED search."""
import csv
import json
import math
import platform
import sys
from collections import defaultdict, deque
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from torchoed import effects_coding, optimal_design


def rows(name):
    with (HERE / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def setting(row):
    return float(row["Altitude"]), row["Range"], float(row["Power"])


def dense_score(F, hard, ids, plots, eta):
    assert len(set(ids.tolist())) == len(ids)
    assert all(hard[ids][plots == p].unique().numel() == 1 for p in plots.unique())
    Z = (plots[:, None] == plots.unique()).to(F.dtype)
    V = torch.eye(len(ids), dtype=F.dtype) + eta * (Z @ Z.mT)
    selected = F[ids]
    assert torch.linalg.matrix_rank(selected) == F.shape[1]
    sign, logdet = torch.linalg.slogdet(selected.mT @ torch.linalg.solve(V, selected))
    assert sign == 1
    return logdet.item()


def main():
    torch.set_num_threads(1)
    runtime = json.loads((HERE / "runtime.json").read_text(encoding="utf-8"))
    sizes, eta = runtime["plot_sizes"], runtime["eta"]
    base = [setting(row) for row in rows("candidates.csv")]
    candidates = base * len(sizes)
    altitude, ranges, power = zip(*candidates)
    numeric = torch.tensor(list(zip(altitude, power)), dtype=torch.float64)
    lo, hi = numeric.amin(0), numeric.amax(0)
    scaled = 2 * (numeric - lo) / (hi - lo) - 1
    levels = sorted(set(ranges))
    codes = torch.tensor([levels.index(level) for level in ranges])
    F = torch.column_stack((torch.ones_like(scaled[:, 0]), scaled[:, 0],
                            effects_coding(codes, len(levels)), scaled[:, 1]))
    assert F.shape[1] == runtime["model_columns"]
    hard = (scaled[:, 0] > 0).long()

    available = defaultdict(deque)
    for i, key in enumerate(candidates):
        available[key].append(i)
    reference = rows("skpr.csv")
    assert all(available[setting(row)] for row in reference)
    skpr_ids = torch.tensor([available[setting(row)].popleft() for row in reference])
    skpr_plots = torch.tensor([int(row["plot"]) for row in reference])
    n = len(reference)
    assert skpr_plots.bincount()[1:].tolist() == sizes and sum(sizes) == n
    skpr = dense_score(F, hard, skpr_ids, skpr_plots, eta)

    result = optimal_design(F, n, whole_plots=sizes, hard=hard, eta=eta,
                            generator=torch.Generator().manual_seed(0))
    plots = torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes))
    torch_score = dense_score(F, hard, result["ids"], plots, eta)
    assert math.isclose(result["objective"], torch_score, rel_tol=1e-9)
    delta = torch_score - skpr
    verdict = "=" if abs(delta) <= 1e-9 * abs(skpr) else (">" if delta > 0 else "<")
    print(json.dumps(dict(
        n=n, torch_score=torch_score, skpr_score=skpr, delta=delta, verdict=verdict,
        objective=result["objective"], torch_ids=result["ids"].tolist(),
        skpr_ids=skpr_ids.tolist(), python=platform.python_version(), torch=torch.__version__,
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

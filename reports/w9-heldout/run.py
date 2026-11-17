"""Run frozen H19/H20 once, then optionally score the Colab artifact."""
import csv
import inspect
import io
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from torchoed import full_factorial, optimal_design, polynomial_features

SEED = 20261117


def metric(case, x, rows, groups):
    blocked = case == "H19"
    f = polynomial_features(x, 2, include_bias=not blocked)
    s = polynomial_features(rows, 2, include_bias=not blocked)
    if blocked:
        f = f - f.mean(0)
        s = torch.cat([g - g.mean(0) for g in s.split(groups)])
        m = s.T @ s
    else:
        z = torch.eye(len(groups), dtype=s.dtype).repeat_interleave(torch.tensor(groups), dim=0)
        v = torch.eye(len(s), dtype=s.dtype) + z @ z.T
        m = s.T @ torch.linalg.inv(v) @ s
    return torch.trace((f.T @ f / len(f)) @ torch.linalg.inv(m)).item()


def search(case, x):
    blocked = case == "H19"
    groups = [8, 8] if blocked else [3, 3, 3, 3]
    kwargs = {"blocks": groups} if blocked else {"whole_plots": groups, "hard": x[:, 0].long(), "eta": 1.0}
    f = polynomial_features(x, 2, include_bias=not blocked)
    start = perf_counter()
    result = optimal_design(f, sum(groups), criterion="I", generator=torch.Generator().manual_seed(SEED), **kwargs)
    seconds = perf_counter() - start
    ids = torch.as_tensor(result["ids"])
    rows = x[ids]
    phi = metric(case, x, rows, groups)
    layout = (ids.unique().numel() == 16 if blocked else
              all(bool((g[:, 0] == g[0, 0]).all()) for g in rows.split(groups)))
    floor = 0 < (result["efficiency_lower_bound"] or 0) <= 1
    oracle = math.isclose(result["objective"], -phi, rel_tol=1e-9)
    row = {k: result[k] for k in ("objective", "stop_reason", "certificate", "efficiency_lower_bound")}
    row.update(ids=[g.tolist() for g in ids.split(groups)], phi=phi, wall_seconds=seconds,
               per_restart_seconds=seconds / inspect.signature(optimal_design).parameters["restarts"].default,
               checks={k: "PASS" if v else "FAIL" for k, v in
                       dict(metric=oracle, layout=layout, floor=floor).items()},
               status="PASS" if oracle and layout and floor else "FAIL")
    if not blocked:
        row.update(A=[g[0, 0].item() for g in rows.split(groups)], hard_labels="X[:, 0].long()")
    return row


def reference(case, x, local, remote):
    if "csv" not in remote:
        return remote
    groups = {}
    for row in csv.DictReader(io.StringIO(remote["csv"])):
        groups.setdefault(row["plot"], []).append(tuple(float(row[k]) for k in "ABC"))
    points = [p for group in groups.values() for p in group]
    rows = torch.tensor(points, dtype=x.dtype)
    rank = int(torch.linalg.matrix_rank(polynomial_features(rows, 2)))
    if case == "H20" and rank != 10:
        return dict(remote, status="FAIL", rank=rank, rank_check="FAIL")
    phi = metric(case, x, rows, [len(group) for group in groups.values()])
    return dict(status="PASS", phi=phi, delta=phi-local["phi"] if "phi" in local else None,
                rows_by_plot=groups, rank=rank, rank_check="PASS" if case == "H20" else None,
                single_A=all(len({p[0] for p in g}) == 1 for g in groups.values()),
                max_repeat=max(Counter(points).values()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    x = full_factorial([3, 3, 3]).double() - 1
    if len(sys.argv) == 2:
        print(json.dumps(search(sys.argv[1], x), allow_nan=False))
    elif sys.argv[1] == "score":
        data = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        remote = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
        for case in ("H19", "H20"):
            data[case]["reference"] = reference(case, x, data[case], remote[case])
        print(json.dumps(data, indent=2, allow_nan=False))
    else:
        output = Path(sys.argv[2])
        with output.open("x", encoding="utf-8") as handle:
            data = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                        dtype="float64", threads=torch.get_num_threads(), seed=SEED)
            for case in ("H19", "H20"):
                try:
                    run = subprocess.run([sys.executable, __file__, case], timeout=10800, capture_output=True, text=True)
                    data[case] = (json.loads(run.stdout) if run.returncode == 0 else
                                  dict(status="FAIL", error=run.stderr.strip()))
                except subprocess.TimeoutExpired:
                    data[case] = dict(status="TIMEOUT", limit_seconds=10800)
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")

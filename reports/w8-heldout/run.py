"""Run frozen H17 and H18 once against the checkout source."""
import json
import math
import platform
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from torchoed import ff2n, optimal_design, simplex_lattice

def search(F, n, **kwargs):
    start = perf_counter()
    result = optimal_design(F, n, generator=torch.Generator().manual_seed(20261122), **kwargs)
    seconds = (perf_counter() - start) / len(result["restart_objectives"])
    row = {k: result[k] for k in ("objective", "stop_reason", "certificate")}
    return result["ids"], dict(row, per_restart_seconds=seconds)

if __name__ == "__main__":
    torch.set_num_threads(1)
    with (HERE / "results.json").open("x", encoding="utf-8") as output:
        X = torch.cat([ff2n(4)] * 2)
        a, b, c, d = X.T
        F = torch.stack([torch.ones_like(a), a, b, c, d, *(x * y for x, y in combinations(X.T, 2))], dim=1)
        ids, h17 = search(F, 16, whole_plots=[4] * 4, hard=(a > 0).long(), eta=0.5)
        A, S = a[ids].reshape(4, 4), F[ids]
        Z = torch.eye(4, dtype=F.dtype).repeat_interleave(4, dim=0)
        V = torch.eye(16, dtype=F.dtype) + 0.5 * (Z @ Z.T)
        sign, dense = torch.linalg.slogdet(S.T @ torch.linalg.solve(V, S))
        pass17 = bool(sign == 1 and (A == A[:, :1]).all() and math.isclose(h17["objective"], dense.item(), rel_tol=1e-9))
        h17.update(A_by_plot=A.tolist(), dense_logdet=dense.item(), status="PASS" if pass17 else "FAIL")
        L = simplex_lattice(4, 6)
        F = torch.stack([*L.T, *(x * y for x, y in combinations(L.T, 2)),
                         *(x * y * z for x, y, z in combinations(L.T, 3))], dim=1)
        ids, h18 = search(F, 14)
        S = F[ids]
        sign, dense = torch.linalg.slogdet(S.T @ S)
        pass18 = bool(sign == 1 and math.isclose(h18["objective"], dense.item(), rel_tol=1e-10))
        h18.update(integer_sixths=(6 * L[ids]).round().long().tolist(), dense_logdet=dense.item(), status="PASS" if pass18 else "FAIL")
        data = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                    dtype="float64", threads=torch.get_num_threads(), seed=20261122, H17=h17, H18=h18)
        output.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
    assert pass17 and pass18, (h17["status"], h18["status"])

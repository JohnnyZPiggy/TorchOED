"""Run the frozen W7 cases once, or score the saved skpr reference."""
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results.json"
CASES = ("H14", "H15", "H16")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(ROOT / "src"))

import torch
from torchoed import design_diagnostics, full_factorial, optimal_design, polynomial_features, simplex_lattice
from torchoed.d_optimal_design import _split_information

torch.set_num_threads(1)
SIZES = [4] * 6


def search(F, n, **kwargs):
    start = time.perf_counter()
    result = optimal_design(F, n, generator=torch.Generator().manual_seed(20261114), **kwargs)
    wall = time.perf_counter() - start
    restarts = len(result["restart_objectives"])
    row = {key: result[key] for key in ("objective", "stop_reason", "certificate")}
    return result["ids"], dict(row, restarts=restarts, wall_seconds=wall, per_restart_seconds=wall / restarts)


def split_scores(rows):
    F = polynomial_features(rows, 2)
    Z = torch.eye(6, dtype=F.dtype).repeat_interleave(4, dim=0)
    shared, _ = _split_information(F, torch.arange(24), SIZES, F.new_full((6,), 1 / 20))
    V = torch.eye(24, dtype=F.dtype) + Z @ Z.mT
    dense = F.mT @ torch.linalg.inv(V) @ F
    sign, value = torch.linalg.slogdet(shared)
    dense_sign, oracle = torch.linalg.slogdet(dense)
    err = (value - oracle).abs().item()
    plots = rows[:, 0].reshape(6, 4)
    return dict(logdet=value.item(), dense_logdet=oracle.item(), dense_abs_error=err,
                metric_oracle=bool(sign == 1 and dense_sign == 1 and err <= 1e-10),
                A_by_plot=plots.tolist(), single_A=bool((plots == plots[:, :1]).all()),
                max_repeat=int(rows.unique(dim=0, return_counts=True)[1].max()))


def h14(X):
    X4 = torch.cat([X] * 4)
    ids, row = search(polynomial_features(X4, 2), 24, whole_plots=SIZES, hard=(X4[:, 0] + 1).long(), eta=1.0)
    score = split_scores(X4[ids])
    match = math.isclose(row["objective"], score["logdet"], rel_tol=1e-9)
    passed = score["metric_oracle"] and score["single_A"] and match
    return dict(row, **score, objective_oracle=match, ids_by_plot=ids.reshape(6, 4).tolist(),
                status="PASS" if passed else "FAIL")


def h15(_):
    L = simplex_lattice(5, 4)
    P = polynomial_features(L, 2, include_bias=False)
    F = torch.cat((P[:, :5], P[:, 10:]), 1)
    ids, row = search(F, 15)
    S = F[ids]
    sign, value = torch.linalg.slogdet(S.mT @ S)
    bound = -20 * math.log(4)
    gap = row["objective"] - bound
    selected = set(map(tuple, (L[ids] * 4).round().long().tolist()))
    small = full_factorial([3] * 5)
    target = set(map(tuple, (2 * small[small.sum(dim=1) == 2]).tolist()))
    passed = sign == 1 and abs(row["objective"] - value.item()) <= 1e-10 and gap <= 1e-10
    return dict(row, ids=ids.tolist(), dense_logdet=value.item(), kiefer_bound=bound, gap=gap,
                same_lattice=selected == target, status="PASS" if passed else "FAIL")


def h16(X):
    F = polynomial_features(X, 2)
    ids, _ = search(F, 16)
    power = design_diagnostics(F, ids, effect=1.0, alpha=0.05)["power"]
    S = F[ids]
    M = S.mT @ S
    lam = torch.linalg.inv(M).diagonal().reciprocal()
    used = torch.cholesky_inverse(torch.linalg.cholesky(M)).diagonal().reciprocal()
    equal = torch.isclose(lam[:, None], lam, rtol=1e-12, atol=0)
    equal_power = (power[:, None] - power).abs()[equal].max().item()
    passed = (torch.allclose(lam, used, rtol=1e-12, atol=0)
              and bool(((power >= 0.05 - 1e-12) & (power <= 1 + 1e-10)).all()) and equal_power <= 1e-14)
    scipy_check = {"status": "NOT_RUN"}
    try:
        import scipy
        from scipy.stats import f, ncf
    except ImportError:
        pass
    else:
        ref = ncf.sf(f.ppf(0.95, 1, 6), 1, 6, lam.tolist())
        error = (power - torch.tensor(ref, dtype=power.dtype)).abs().max().item()
        scipy_check = dict(version=scipy.__version__, max_abs_error=error, status="PASS" if error <= 1e-8 else "FAIL")
        passed = passed and error <= 1e-8
    return dict(ids=ids.tolist(), power=power.tolist(), lambda_dense=lam.tolist(), lambda_used=used.tolist(),
                equal_lambda_power_max_abs=equal_power, scipy=scipy_check, status="PASS" if passed else "FAIL")


def main():
    if len(sys.argv) == 2:
        X = full_factorial([3, 3, 3]).double() - 1
        print(json.dumps(dict(zip(CASES, (h14, h15, h16)))[sys.argv[1]](X), allow_nan=False))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--reference":
        data = json.loads(OUT.read_text(encoding="utf-8"))
        artifact = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        assert artifact["source_sha"] == data["measurement_sha"]
        ref = dict(status=artifact["reference_status"])
        if ref["status"] == "PASS":
            blocks = artifact["rows_by_plot"]
            rows = torch.tensor(blocks, dtype=torch.float64).reshape(24, 3)
            ref.update(split_scores(rows), rows_by_plot=blocks)
            if not ref["metric_oracle"]:
                ref["status"] = data["H14"]["status"] = "FAIL"
            (HERE / "sessionInfo.txt").write_text(artifact["sessionInfo"], encoding="utf-8")
        else:
            ref["reason"] = artifact["reference_error"]
        data["H14"]["skpr"] = ref
    else:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        data = dict(measurement_sha=sha, platform=platform.platform(), python=platform.python_version(),
                    torch=torch.__version__, dtype="float64", threads=1, seed=20261114)
        with OUT.open("x", encoding="utf-8") as file:
            for case in CASES:
                try:
                    result = subprocess.run([sys.executable, "-B", __file__, case], capture_output=True,
                                            text=True, timeout=10800)
                    data[case] = json.loads(result.stdout) if result.returncode == 0 else {
                        "status": "FAIL", "error": result.stderr.strip()}
                except subprocess.TimeoutExpired:
                    data[case] = dict(status="TIMEOUT", limit_seconds=10800)
                print(case, data[case]["status"], flush=True)
            file.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
        return
    OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

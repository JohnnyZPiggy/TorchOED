"""Run the three frozen W6 cases once, or attach a saved R reference."""
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torchoed import definitive_screening, design_diagnostics, full_factorial, optimal_design, polynomial_features
from torchoed.d_optimal_design import _block_information

torch.set_num_threads(1)
OUT = HERE / "results.json"
SEED = 20261112
BLOCKS = [6, 6, 6]


def h11_features():
    with (HERE / "H11.csv").open(newline="") as file:
        grid = torch.tensor([list(map(int, row.values())) for row in csv.DictReader(file)]).double()
    assert torch.equal(grid, full_factorial([3] * 3).double() - 1)
    return polynomial_features(grid, 2, include_bias=False)


def block_scores(F, ids):
    M, _ = _block_information(F, ids, BLOCKS)
    Z = torch.eye(3, dtype=F.dtype).repeat_interleave(torch.tensor(BLOCKS), dim=0)
    S = F[ids]
    dense = S.mT @ (torch.eye(len(ids), dtype=F.dtype) - Z @ torch.linalg.inv(Z.mT @ Z) @ Z.mT) @ S
    sign, value = torch.linalg.slogdet(M)
    dense_sign, oracle = torch.linalg.slogdet(dense)
    return dict(logdet=value.item(), dense_logdet=oracle.item(),
                metric_oracle=bool(sign == 1 and dense_sign == 1 and (value - oracle).abs() <= 1e-10))


def search(F, n, **kwargs):
    generator = torch.Generator().manual_seed(SEED)
    start = time.perf_counter()
    result = optimal_design(F, n, generator=generator, **kwargs)
    wall = time.perf_counter() - start
    restarts = len(result["restart_objectives"])
    record = {key: result[key] for key in ("objective", "stop_reason", "certificate")}
    return result, dict(record, wall_seconds=wall, restarts=restarts, per_restart_seconds=wall / restarts)


def h11():
    F = h11_features()
    result, row = search(F, 18, blocks=BLOCKS)
    ids = result["ids"]
    score = block_scores(F, ids)
    objective = math.isclose(row["objective"], score["logdet"], rel_tol=1e-9)
    return dict(row, **score, ids_by_block=[b.tolist() for b in ids.split(BLOCKS)],
                objective_oracle=objective, status="PASS" if score["metric_oracle"] and objective else "FAIL")


def h12():
    D = definitive_screening(13).long()
    C = D[:14]
    eye = torch.eye(13, dtype=C.dtype)
    a, b = torch.triu_indices(13, 13)
    second = D[:, a] * D[:, b]
    conference = (C.mT @ C - 13 * eye).abs().max().item()
    main = ((D.mT @ D) * (1 - eye)).abs().max().item()
    interaction = (D.mT @ second).abs().max().item()
    corr = torch.corrcoef(second.double().mT)
    i, j = torch.triu_indices(len(a), len(a), offset=1)
    return dict(runs=len(D), conference_max_abs=conference, main_main_max_abs=main,
                main_second_order_max_abs=interaction, max_abs_second_order_correlation=corr[i, j].abs().max().item(),
                status="PASS" if len(D) == 29 and conference == main == interaction == 0 else "FAIL")


def h13():
    F = polynomial_features(full_factorial([3] * 4).double() - 1, 2)
    result, row = search(F, 20)
    ids = result["ids"]
    diagnostics = design_diagnostics(F, ids)
    spv = diagnostics["spv"]
    S = F[ids]
    dense = len(ids) * (F @ torch.linalg.inv(S.mT @ S) @ F.mT).diag()
    error = (spv - dense).abs().max().item()
    kw_floor = F.shape[1] / spv.max().item()
    floor = result["efficiency_lower_bound"]
    consistent = floor >= kw_floor * (1 - 1e-12)
    return dict(row, ids=ids.tolist(), max_vif=diagnostics["vif"].max().item(),
                spv_quantiles=spv.quantile(spv.new_tensor([0, .25, .5, .75, 1])).tolist(),
                efficiency_lower_bound=floor, kw_floor=kw_floor, spv_max_abs_error=error,
                spv_oracle=error <= 1e-10, floor_oracle=consistent,
                status="PASS" if error <= 1e-10 and consistent else "FAIL")


def record_reference(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    reference = {"status": artifact["reference_status"]}
    if artifact["reference_status"] == "PASS":
        blocks = artifact["algdesign"]
        ids = torch.tensor(blocks).flatten()
        score = block_scores(h11_features(), ids)
        reference.update(score, ids_by_block=blocks,
                         repeats_within_block=any(len(set(b)) != len(b) for b in blocks),
                         repeats_across_blocks=len(set(ids.tolist())) < sum(len(set(b)) for b in blocks),
                         status="PASS" if score["metric_oracle"] else "FAIL")
    else:
        reference["reason"] = artifact["reference_error"]
    data = json.loads(OUT.read_text(encoding="utf-8"))
    data["rows"]["H11"]["algdesign"] = reference
    return data


if len(sys.argv) == 2:
    print(json.dumps({"H11": h11, "H12": h12, "H13": h13}[sys.argv[1]](), allow_nan=False))
    raise SystemExit
if len(sys.argv) == 3 and sys.argv[1] == "--reference":
    data = record_reference(sys.argv[2])
else:
    if OUT.exists():
        raise SystemExit("Held-out results already exist")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    data = dict(source_sha=sha, seed=SEED, environment=dict(python=platform.python_version(),
                torch=torch.__version__, os=platform.platform(), cpu=platform.processor(), threads=1), rows={})
    for case in ("H11", "H12", "H13"):
        try:
            row = subprocess.check_output([sys.executable, "-B", __file__, case], text=True, timeout=10800)
            data["rows"][case] = json.loads(row)
        except subprocess.TimeoutExpired:
            data["rows"][case] = {"status": "TIMEOUT", "wall_seconds_cap": 10800}
        OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")

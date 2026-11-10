from pathlib import Path
import runpy

import torch


def test_closed_loop_example():
    example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/closed_loop_sim.py"), run_name="example")
    result = example["main"](seed=0)
    assert result == example["main"](seed=0)
    assert len(result["run_ids"]) == len(set(result["run_ids"])) == 8
    x = torch.linspace(0.1, 10, 100, dtype=torch.float64)
    # Independent analytic Jacobian in (V, K); log coordinates preserve D IDs.
    F = torch.stack((x / (2.5 + x), -2.0 * x / (2.5 + x).square()), dim=1)
    pairs = torch.combinations(torch.arange(100), r=2)
    rows = F[pairs]
    determinants = torch.linalg.det(rows.transpose(1, 2) @ rows)
    best_pair = pairs[determinants.argmax()].tolist()
    assert best_pair == [16, 99]
    assert result["run_ids"][:2] == best_pair
    V, K = result["theta"][-1]
    assert max(abs(V - 3.0) / 3.0, abs(K - 1.2) / 1.2) <= 0.25

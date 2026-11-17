import csv
from pathlib import Path

import pytest
import torch

from torchoed import box_behnken, central_composite, polynomial_features


REFERENCES = Path(__file__).resolve().parents[1] / "reports/w10-rref"


def reference(file, key, value):
    with (REFERENCES / file).open(newline="", encoding="utf-8") as csv_file:
        return [row for row in csv.DictReader(csv_file) if row[key] == str(value)]


@pytest.mark.parametrize("case,k,alpha,center", [
    ("A", 2, "rotatable", (3, 3)), ("B", 4, "orthogonal", (4, 2)),
    ("C", 5, "faces", (2, 2)),
])
def test_ccd_rsm(case, k, alpha, center):
    rows = reference("rsm_ccd.csv", "case", case)
    expected = torch.tensor([[float(r[f"x{i}"]) for i in range(1, k + 1)]
                             for r in rows], dtype=torch.float64)
    torch.testing.assert_close(central_composite(k, alpha=alpha, center=center),
                               expected, atol=1e-12, rtol=0)
    assert [r["block"] for r in rows] == (["1"] * (2**k + center[0])
                                         + ["2"] * (2 * k + center[1]))


@pytest.mark.parametrize("k", [3, 5])
def test_bbd_rsm(k):
    rows = reference("rsm_bbd.csv", "k", k)
    expected = [[float(r[f"x{i}"]) for i in range(1, k + 1)] for r in rows]
    actual = box_behnken(k).tolist()
    assert (actual == expected if k == 3 else sorted(actual) == sorted(expected))


@pytest.mark.parametrize("k,center,alpha", [
    (2, (3, 3), "orthogonal"), (4, (4, 2), "orthogonal"), (4, (4, 2), 1.0),
])
def test_ccd_block_information(k, center, alpha):
    features = polynomial_features(central_composite(k, alpha=alpha, center=center),
                                   2, include_bias=False)
    cube, axial = features.split((2**k + center[0], 2 * k + center[1]))
    within = torch.cat((cube - cube.mean(dim=0), axial - axial.mean(dim=0)))
    overall = features - features.mean(dim=0)
    m_c = overall.T @ overall
    difference = (m_c - within.T @ within).abs().max()
    if alpha == "orthogonal":
        assert difference <= 1e-12 * m_c.abs().max()
    else:
        assert difference > 1e-6


@pytest.mark.parametrize("k", [2, 3, 4])
def test_ccd_rotatability(k):
    x, y = central_composite(k, alpha="rotatable")[:, :2].T
    assert x.pow(4).sum().item() == pytest.approx(3 * (x * y).square().sum().item(),
                                                 abs=1e-12, rel=0)


@pytest.mark.parametrize("k", [3, 4, 5])
def test_bbd_rotatability(k):
    x, y = box_behnken(k, center=0)[:, :2].T.to(torch.int64)
    assert bool(x.pow(4).sum() == 3 * (x * y).square().sum()) == (k == 4)


@pytest.mark.parametrize("constructor,k,kwargs", [
    (central_composite, 1, {}), (central_composite, 2, {"alpha": "spherical"}),
    (central_composite, 2, {"alpha": 0}), (central_composite, 2, {"alpha": True}),
    (central_composite, 2, {"center": (-1, 4)}),
    (box_behnken, 2, {}), (box_behnken, 8, {}), (box_behnken, 3, {"center": -1}),
])
def test_invalid_response_surface_input(constructor, k, kwargs):
    with pytest.raises(ValueError):
        constructor(k, **kwargs)


@pytest.mark.parametrize("k", [6, 7])
def test_bbd_three_factor_rsm(k):
    rows = reference("rsm_bbd67.csv", "k", k)
    expected = [[int(r[f"x{i}"]) for i in range(1, k + 1)] for r in rows]
    assert box_behnken(k).tolist() == expected


@pytest.mark.parametrize("k,j,mixed", [(6, 3, 16), (7, 1, 8)])
def test_bbd_three_factor_rotatability(k, j, mixed):
    rows = box_behnken(k, center=0).to(torch.int64).tolist()
    fourth_moment = sum(row[0] ** 4 for row in rows)
    mixed_moment = sum(row[0] ** 2 * row[j] ** 2 for row in rows)
    assert (fourth_moment, mixed_moment) == (24, mixed)
    assert (fourth_moment == 3 * mixed_moment) == (k == 7)

"""Tiny CPU documentation smoke; no benchmark or optimality claim."""

import torch

from torchoed import (
    d_optimality, exact_information, ff2n, fedorov_exchange,
    polynomial_features, score_two_swaps,
)


def main():
    corners = ff2n(2, device="cpu", dtype=torch.float64)
    features = polynomial_features(corners, degree=1)
    ids = torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu")
    information = exact_information(features, ids)
    assert features.shape == (4, 3)
    torch.testing.assert_close(torch.linalg.det(information), torch.tensor(16., dtype=torch.float64))
    print("Initial logdet:", d_optimality(information).item())
    result = fedorov_exchange(features, ids, max_iter=1, tile_size=2)
    print("1-swap stop:", result["stop_reason"], "certificate:", result["certificate"])

    # Six scalar candidates provide a tiny nonempty two-swap neighborhood.
    coordinates = torch.tensor([[-2.], [-1.], [0.], [1.], [2.], [3.]], dtype=torch.float64)
    linear = polynomial_features(coordinates, degree=1)
    scores = score_two_swaps(linear, ids, tile_size=2)
    valid = scores["status"] == 0
    assert scores["evaluated_pairs"] == 9
    assert bool(torch.isfinite(scores["gains"][valid]).all())
    assert bool(torch.isnan(scores["gains"][~valid]).all())
    print("Two-swap statuses:", scores["counts"])
    print("Valid gains (no swap selected or applied):", scores["gains"][valid].tolist())


if __name__ == "__main__":
    main()

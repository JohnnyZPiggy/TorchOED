import torch

from torchoed import d_optimal_design, full_factorial, optimal_design


def test_seeded_restarts_match_before_consolidation():
    # Recorded at ff33d80 on the same six calls.
    X = full_factorial([3, 4]).double() - 1
    F = torch.column_stack((torch.ones_like(X[:, 0]), X))
    for design, features, n, options, expected in (
        (d_optimal_design, F, 5, {}, ([2, 0, 8, 3, 11], 5.393627546352361, "LOCAL_2_NUMERICAL")),
        (d_optimal_design, F, 5, {"fixed_ids": [1, 7]}, ([1, 0, 8, 7, 11], 5.099866427824199, "LOCAL_2_NUMERICAL")),
        (optimal_design, F, 5, {"criterion": "A"}, ([3, 1, 0, 8, 11], -0.5454545454545454, "LOCAL_2_NUMERICAL")),
        (optimal_design, F, 5, {"criterion": "I"}, ([2, 0, 8, 3, 11], -0.48484848484848486, "LOCAL_2_NUMERICAL")),
        (optimal_design, X, 8, {"blocks": [4, 4]}, ([0, 2, 9, 11, 1, 3, 8, 10], 4.382026634673881, "LOCAL_BLOCK_EXCHANGE")),
        (optimal_design, F, 8, {"whole_plots": [4, 4], "hard": X[:, 0].long(), "eta": 0.5},
         ([8, 9, 10, 11, 0, 1, 2, 3], 4.264243599017498, "LOCAL_SPLIT_PLOT_EXCHANGE")),
    ):
        result = design(features, n, restarts=3, generator=torch.Generator().manual_seed(47), **options)
        assert (result["ids"].tolist(), result["objective"], result["stop_reason"]) == expected

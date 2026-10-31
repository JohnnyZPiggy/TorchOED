"""Torch-only M1 constructors and polynomial features; CPU acceptance only."""

__version__ = "0.1.0"

from .factorial import ff2n, full_factorial
from .features import polynomial_features
from .sampling import lhs, sobol
from .information import exact_information, weighted_information
from .criteria import d_optimality, a_optimality, e_optimality, c_optimality
from .exchange import fedorov_exchange
from .two_swap import score_two_swaps
from .k2_exchange import k2_exchange
from .d_optimal_design import d_optimal_design

__all__ = ["full_factorial", "ff2n", "lhs", "sobol", "polynomial_features",
           "exact_information", "weighted_information", "d_optimality",
           "a_optimality", "e_optimality", "c_optimality", "fedorov_exchange", "score_two_swaps", "k2_exchange", "d_optimal_design"]

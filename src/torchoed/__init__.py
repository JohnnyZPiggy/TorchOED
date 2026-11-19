"""Torch-only M1 constructors and polynomial features; CPU acceptance only."""

__version__ = "0.8.0"

from .factorial import (
    box_behnken, central_composite, definitive_screening, ff2n, fold_over, fractional_factorial,
    fractional_factorial_generators, full_factorial, plackett_burman, simplex_centroid, mixture_vertices, simplex_lattice, word_length_pattern,
)
from .features import mixture_process_features, scheffe_features, polynomial_features, effects_coding, feasible_candidates, jacobian_features
from .sampling import lhs, sobol
from .information import design_diagnostics, exact_information, weighted_information
from .criteria import d_optimality, a_optimality, e_optimality, c_optimality, g_optimality, v_optimality, i_optimality
from .exchange import fedorov_exchange
from .two_swap import score_two_swaps
from .k2_exchange import k2_exchange
from .d_optimal_design import d_optimal_design, optimal_design, next_batch

__all__ = ["full_factorial", "ff2n", "definitive_screening", "plackett_burman", "central_composite", "box_behnken", "simplex_lattice", "simplex_centroid", "mixture_vertices", "lhs", "sobol", "polynomial_features", "scheffe_features", "mixture_process_features", "effects_coding", "feasible_candidates", "jacobian_features",
           "fractional_factorial", "fractional_factorial_generators", "fold_over", "word_length_pattern", "design_diagnostics",
           "exact_information", "weighted_information", "d_optimality",
           "a_optimality", "e_optimality", "c_optimality", "g_optimality", "v_optimality", "i_optimality", "fedorov_exchange", "score_two_swaps", "k2_exchange", "d_optimal_design", "optimal_design", "next_batch"]

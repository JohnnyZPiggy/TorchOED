"""Torch-only M1 constructors and polynomial features; CPU acceptance only."""

__version__ = "0.0.1"

from .factorial import ff2n, full_factorial
from .features import polynomial_features
from .sampling import lhs, sobol

__all__ = ["full_factorial", "ff2n", "lhs", "sobol", "polynomial_features"]

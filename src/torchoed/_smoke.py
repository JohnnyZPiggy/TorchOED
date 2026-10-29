"""Private M0 smoke operation; not a public information-matrix API."""

import torch


def _gram(features: torch.Tensor) -> torch.Tensor:
    return features.mT @ features

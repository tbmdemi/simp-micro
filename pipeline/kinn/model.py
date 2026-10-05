"""KAN-informed neural field for differentiable mechanics."""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from efficient_kan import EfficientKANLinear


class KINN(nn.Module):
    """Map reference coordinates and material parameters to displacement.

    Args:
        coordinate_dim: Number of spatial coordinates.
        material_dim: Number of material features appended to coordinates.
        hidden_dim: Width of each KAN layer.
        depth: Number of hidden KAN layers.

    Returns:
        Forward output of shape ``(B, N, 2)`` containing displacement.
    """

    def __init__(
        self,
        coordinate_dim: int = 2,
        material_dim: int = 1,
        hidden_dim: int = 64,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if (
            coordinate_dim < 1
            or material_dim < 0
            or hidden_dim < 1
            or depth < 1
        ):
            raise ValueError("KINN dimensions must be positive")
        layers: list[nn.Module] = []
        input_dim = coordinate_dim + material_dim
        for index in range(depth):
            layers.append(
                EfficientKANLinear(
                    input_dim if index == 0 else hidden_dim, hidden_dim
                )
            )
            layers.append(nn.Tanh())
        layers.append(EfficientKANLinear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(
        self, coordinates: torch.Tensor, material: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate displacement at coordinates for a material state."""
        if coordinates.ndim != 3 or material.ndim != 3:
            raise ValueError(
                "coordinates and material must have shape (B, N, D)"
            )
        if coordinates.shape[:2] != material.shape[:2]:
            raise ValueError(
                "coordinates and material batch/point counts differ"
            )
        batch, points = coordinates.shape[:2]
        inputs = torch.cat([coordinates, material], dim=-1).reshape(
            batch * points, -1
        )
        return self.network(inputs).reshape(batch, points, 2)

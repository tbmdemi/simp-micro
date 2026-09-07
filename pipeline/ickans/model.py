"""Convex constitutive energy model with positive curvature."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ICKAN(nn.Module):
    """Learn a convex elastic energy with strictly positive tangent stiffness.

    Positive softplus weights and Softplus activations provide convex hidden
    paths. A positive quadratic term supplies a strict curvature floor, making
    the Hessian positive definite even when learned hidden weights collapse.

    Args:
        input_dim: Number of strain/material features.
        hidden_dim: Width of the convex hidden layers.
        quadratic_floor: Minimum diagonal curvature contribution.

    Returns:
        Scalar energy per input row, shape ``(N, 1)``.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        quadratic_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or quadratic_floor <= 0:
            raise ValueError("ICKAN dimensions must be positive")
        self.input_dim = input_dim
        self.quadratic_floor = quadratic_floor
        self.weight1 = nn.Parameter(torch.empty(2 * input_dim, hidden_dim))
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(hidden_dim))
        self.output_weight = nn.Parameter(torch.empty(hidden_dim, 1))
        self.output_bias = nn.Parameter(torch.zeros(1))
        for parameter in (self.weight1, self.weight2, self.output_weight):
            nn.init.normal_(parameter, mean=-3.0, std=0.1)

    def positive_weights(self):
        """Return the strictly positive effective weights."""
        return tuple(
            F.softplus(weight)
            for weight in (self.weight1, self.weight2, self.output_weight)
        )

    def forward(self, strain: torch.Tensor) -> torch.Tensor:
        """Evaluate convex strain energy."""
        if strain.ndim != 2 or strain.size(1) != self.input_dim:
            raise ValueError(f"strain must have shape (N, {self.input_dim})")
        weight1, weight2, output_weight = self.positive_weights()
        signed_features = torch.cat([strain, -strain], dim=1)
        hidden1 = F.softplus(signed_features @ weight1 + self.bias1)
        hidden2 = F.softplus(hidden1 @ weight2 + self.bias2)
        learned = hidden2 @ output_weight + self.output_bias
        quadratic = (
            0.5
            * self.quadratic_floor
            * strain.square().sum(dim=1, keepdim=True)
        )
        return learned + quadratic

    def tangent_hessian(self, strain: torch.Tensor) -> torch.Tensor:
        """Return the per-sample Hessian of the energy with autograd."""
        strain = strain.detach().requires_grad_(True)
        energies = self(strain).squeeze(-1)
        rows = []
        for index in range(energies.numel()):
            gradient = torch.autograd.grad(
                energies[index], strain, create_graph=True, retain_graph=True
            )[0][index]
            row = []
            for component in gradient:
                row.append(
                    torch.autograd.grad(component, strain, retain_graph=True)[
                        0
                    ][index]
                )
            rows.append(torch.stack(row))
        return torch.stack(rows)

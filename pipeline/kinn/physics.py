"""Differentiable energy losses and optional JAX-AMG bridge."""

from __future__ import annotations

import torch


def deep_energy_loss(
    model,
    coordinates: torch.Tensor,
    material: torch.Tensor,
    young_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
) -> torch.Tensor:
    """Compute a small-strain elastic energy plus displacement regularization.

    The spatial gradient is obtained with autograd, keeping the loss fully
    differentiable with respect to KINN parameters.
    """
    if not coordinates.requires_grad:
        coordinates = coordinates.detach().requires_grad_(True)
    displacement = model(coordinates, material)
    gradients = []
    for component in range(displacement.size(-1)):
        gradients.append(
            torch.autograd.grad(
                displacement[..., component].sum(),
                coordinates,
                create_graph=True,
                retain_graph=True,
            )[0]
        )
    grad_u = torch.stack(gradients, dim=-2)
    strain = 0.5 * (grad_u + grad_u.transpose(-1, -2))
    trace = strain.diagonal(dim1=-2, dim2=-1).sum(-1)
    lame_mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    energy = (
        lame_mu * (strain.square().sum(dim=(-1, -2)))
        + 0.5 * lame_lambda * trace.square()
    )
    return energy.mean() + 1e-4 * displacement.square().mean()


def solve_linear_amg(matrix, rhs):
    """Solve a linear system through optional JAX-AMG or SciPy fallback.

    This adapter deliberately keeps the mechanics model in PyTorch; it is a
    boundary for experiments with JAX-AMG and does not silently detach training
    gradients.
    """
    try:
        import jax.numpy as jnp
        from jax.scipy.sparse.linalg import cg

        solution, info = cg(jnp.asarray(matrix), jnp.asarray(rhs))
        return solution, info, "jax-cg"
    except ImportError:
        import scipy.sparse.linalg as spla

        solution = spla.spsolve(matrix, rhs)
        return solution, 0, "scipy-spsolve"

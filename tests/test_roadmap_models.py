"""Smoke tests for the newly added roadmap model modules."""

import torch

from pipeline.ickans import ICKAN
from pipeline.kinn import KINN, deep_energy_loss
from pipeline.mno import MambaNeuralOperator


def test_mno_preserves_grid_and_outputs_18_fields():
    model = MambaNeuralOperator(d_model=16, layers=1, use_mamba=False)
    output = model(torch.rand(2, 1, 8, 10))
    assert output.shape == (2, 18, 8, 10)


def test_kinn_energy_is_differentiable():
    model = KINN(hidden_dim=12, depth=2)
    coordinates = torch.rand(2, 10, 2, requires_grad=True)
    material = torch.ones(2, 10, 1)
    loss = deep_energy_loss(model, coordinates, material)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_phase5_kinn_prior_keeps_density_gradient():
    from pipeline.phase5_cvae.losses import kinn_prior_loss

    kinn = KINN(hidden_dim=8, depth=1)
    density = torch.rand(1, 8, 8, requires_grad=True).unsqueeze(0)
    loss = kinn_prior_loss(density, kinn)
    loss.backward()
    assert density.grad_fn is not None
    assert torch.isfinite(loss)


def test_ickan_tangent_is_positive_definite():
    model = ICKAN(input_dim=3, hidden_dim=10, quadratic_floor=1e-2)
    strain = torch.rand(4, 3)
    hessian = model.tangent_hessian(strain)
    eigenvalues = torch.linalg.eigvalsh(hessian)
    assert torch.all(eigenvalues > 0)

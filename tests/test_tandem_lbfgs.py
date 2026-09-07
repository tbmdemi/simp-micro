"""Tests for differentiable tandem latent optimization."""

import torch
from torch import nn

from pipeline.phase5_cvae.tandem_lbfgs import tandem_inverse_design_lbfgs


class _Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 2
        self.anchor = nn.Parameter(torch.zeros(1))

    def decoder(self, z, condition):
        return z[:, :1].view(-1, 1, 1, 1).expand(-1, 1, 4, 4)


class _Surrogate(nn.Module):
    n_seeds = 1

    def forward(self, image, seed_vec):
        value = image.mean(dim=(1, 2, 3))
        return torch.stack([value, value * 2.0, value], dim=1)


def test_tandem_lbfgs_reduces_poisson_error():
    result = tandem_inverse_design_lbfgs(
        torch.tensor([0.75, 1.50]),
        _Generator(),
        _Surrogate(),
        steps=20,
    )

    assert result["image"].shape == (1, 1, 4, 4)
    assert result["loss"] < 1e-6
    assert len(result["history"]) >= 1
    assert result["history"][-1] <= result["history"][0]
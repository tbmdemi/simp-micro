"""Tests for differentiable tandem latent optimization."""

import pytest
import torch
from torch import nn

from pipeline.phase5_cvae.tandem_lbfgs import tandem_inverse_design_lbfgs

FE_PARAMS_SMALL = dict(
    nelx=6, nely=6, penal=3.0, E0=199.0, Emin=1e-9, nu=0.3, rho0=1.0
)


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


class _TinyFEDecoder(nn.Module):
    """Decoder tối giản: z -> ảnh mật độ 6x6 qua 1 lớp Linear+Sigmoid, đủ
    nhỏ để `real_physics_loss` giải FE thật nhanh trong test."""

    def __init__(self, latent_dim, resolution):
        super().__init__()
        self.resolution = resolution
        self.fc = nn.Linear(latent_dim, resolution * resolution)

    def forward(self, z, condition):
        rho = torch.sigmoid(self.fc(z))
        return rho.view(-1, 1, self.resolution, self.resolution)


class _TinyFEGenerator(nn.Module):
    def __init__(self, latent_dim=4, resolution=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = _TinyFEDecoder(latent_dim, resolution)


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


class TestGuidanceSourceRealPhysics:
    """guidance_source="real_physics": gradient từ RealPhysicsNu (FE thật)
    thay vì surrogate CNN - xem docstring module tandem_lbfgs.py."""

    def test_runs_without_surrogate_model(self):
        torch.manual_seed(0)
        generator = _TinyFEGenerator(latent_dim=4, resolution=6)
        result = tandem_inverse_design_lbfgs(
            torch.tensor([-0.3, -0.3]),
            generator,
            surrogate_model=None,
            steps=5,
            guidance_source="real_physics",
            fe_params=FE_PARAMS_SMALL,
        )
        assert result["image"].shape == (1, 1, 6, 6)
        assert result["prediction"].shape == (1, 2)
        assert len(result["history"]) >= 1
        assert all(torch.isfinite(torch.tensor(v)) for v in result["history"])

    def test_history_does_not_increase(self):
        torch.manual_seed(0)
        generator = _TinyFEGenerator(latent_dim=4, resolution=6)
        result = tandem_inverse_design_lbfgs(
            torch.tensor([-0.3, -0.3]),
            generator,
            steps=5,
            guidance_source="real_physics",
            fe_params=FE_PARAMS_SMALL,
        )
        assert result["history"][-1] <= result["history"][0] + 1e-6

    def test_defaults_fe_params_to_verify_fe_constants_when_omitted(self):
        """fe_params=None (mặc định) -> dùng verify_fe.FE_PARAMS (nelx=
        nely=50) - test này chỉ chạy 1 step (chậm hơn lưới nhỏ) để xác
        nhận không lỗi, không kiểm tra hội tụ."""
        torch.manual_seed(0)
        generator = _TinyFEGenerator(latent_dim=4, resolution=8)
        result = tandem_inverse_design_lbfgs(
            torch.tensor([-0.3, -0.3]),
            generator,
            steps=1,
            guidance_source="real_physics",
        )
        assert result["prediction"].shape == (1, 2)

    def test_invalid_guidance_source_raises(self):
        generator = _TinyFEGenerator(latent_dim=4, resolution=6)
        with pytest.raises(ValueError, match="guidance_source"):
            tandem_inverse_design_lbfgs(
                torch.tensor([-0.3, -0.3]),
                generator,
                guidance_source="not_a_real_source",
            )

    def test_surrogate_guidance_without_surrogate_model_raises(self):
        generator = _TinyFEGenerator(latent_dim=4, resolution=6)
        with pytest.raises(ValueError, match="surrogate_model"):
            tandem_inverse_design_lbfgs(
                torch.tensor([-0.3, -0.3]),
                generator,
                surrogate_model=None,
                guidance_source="surrogate",
            )

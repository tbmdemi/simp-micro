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


class TestProjectionStages:
    """Refine nhận thức nhị phân hóa (plan.md v3 P1.1e)."""

    def test_runs_with_projection_and_periodic(self):
        torch.manual_seed(0)
        gen = _TinyFEGenerator(latent_dim=4, resolution=6)
        result = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]),
            gen,
            guidance_source="real_physics",
            fe_params=FE_PARAMS_SMALL,
            steps=4,
            learning_rate=0.1,
            projection_betas=(1.0, 8.0),
            periodic=True,
        )
        assert result["image"].shape == (1, 1, 6, 6)
        assert len(result["history"]) >= 4
        assert torch.isfinite(result["prediction"]).all()

    def test_no_projection_keeps_legacy_single_stage(self):
        """projection_betas=None phải cho đúng kết quả như trước (cùng z0)."""
        gen = _TinyFEGenerator(latent_dim=4, resolution=6)
        z0 = torch.randn(1, 4, generator=torch.Generator().manual_seed(1))
        kw = dict(
            guidance_source="real_physics",
            fe_params=FE_PARAMS_SMALL,
            steps=3,
            learning_rate=0.1,
            initial_z=z0,
        )
        a = tandem_inverse_design_lbfgs(torch.tensor([-0.2, -0.2]), gen, **kw)
        b = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]), gen, projection_betas=None, **kw
        )
        assert torch.allclose(a["z"], b["z"])


class TestManufPenalty:
    """Phạt chế tạo trong refine (plan.md mục K1)."""

    def _kw(self):
        z0 = torch.randn(1, 4, generator=torch.Generator().manual_seed(2))
        return dict(
            guidance_source="real_physics",
            fe_params=FE_PARAMS_SMALL,
            steps=4,
            learning_rate=0.1,
            initial_z=z0,
            projection_betas=(1.0, 8.0),
        )

    def test_zero_weights_keep_previous_result(self):
        """Mặc định 0 phải cho đúng kết quả cũ (số đã báo không đổi)."""
        torch.manual_seed(0)
        gen = _TinyFEGenerator(latent_dim=4, resolution=6)
        a = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]), gen, **self._kw()
        )
        b = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]),
            gen,
            corner_weight=0.0,
            thin_weight=0.0,
            **self._kw(),
        )
        assert torch.equal(a["z"], b["z"])

    def test_penalty_changes_objective(self):
        torch.manual_seed(0)
        gen = _TinyFEGenerator(latent_dim=4, resolution=6)
        a = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]), gen, **self._kw()
        )
        b = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]),
            gen,
            corner_weight=1.0,
            thin_weight=1.0,
            **self._kw(),
        )
        assert a["history"][0] != b["history"][0]

    def test_penalty_without_projection_raises(self):
        gen = _TinyFEGenerator(latent_dim=4, resolution=6)
        kw = self._kw()
        kw["projection_betas"] = None
        with pytest.raises(ValueError):
            tandem_inverse_design_lbfgs(
                torch.tensor([-0.2, -0.2]), gen, corner_weight=1.0, **kw
            )


class TestRealizationObjective:
    """Objective N1 (hiện thực hóa dịch lệch lưới) + đối chứng lưới mịn."""

    def _kw(self):
        z0 = torch.randn(1, 4, generator=torch.Generator().manual_seed(3))
        return dict(
            guidance_source="real_physics",
            fe_params=FE_PARAMS_SMALL,
            steps=4,
            learning_rate=0.1,
            initial_z=z0,
            projection_betas=(1.0, 8.0),
        )

    def test_defaults_keep_previous_result(self):
        """Tắt N1 (mặc định) phải cho đúng kết quả cũ."""
        gen = _TinyFEGenerator(latent_dim=4, resolution=6)
        a = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]), gen, **self._kw()
        )
        b = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]),
            gen,
            realization_shifts=None,
            fe_upsample=1,
            **self._kw(),
        )
        assert torch.equal(a["z"], b["z"])

    def test_realization_and_upsample_change_objective(self):
        gen = _TinyFEGenerator(latent_dim=4, resolution=6)
        a = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]), gen, **self._kw()
        )
        b = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]),
            gen,
            realization_shifts=[(0.0, 0.5), (0.5, 0.0)],
            **self._kw(),
        )
        c = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]), gen, fe_upsample=2, **self._kw()
        )
        assert a["history"][0] != b["history"][0]
        assert a["history"][0] != c["history"][0]
        assert torch.isfinite(b["z"]).all() and torch.isfinite(c["z"]).all()

    def test_robust_and_last_only_upsample(self):
        """N2: robust formulation đổi objective; lưới mịn chỉ mức cuối khác
        lưới mịn mọi mức ở mức β đầu nhưng vẫn chạy hết."""
        gen = _TinyFEGenerator(latent_dim=4, resolution=6)
        a = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]), gen, **self._kw()
        )
        r = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]),
            gen,
            robust_etas=(0.25, 0.75),
            **self._kw(),
        )
        last = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]),
            gen,
            fe_upsample=2,
            fe_upsample_last_only=True,
            **self._kw(),
        )
        assert a["history"][0] != r["history"][0]
        # mức β đầu vẫn ở lưới verify → giá trị đầu trùng bản không mịn
        assert last["history"][0] == pytest.approx(a["history"][0])
        assert torch.isfinite(r["z"]).all() and torch.isfinite(last["z"]).all()

    def test_design_filter_changes_objective_and_validates(self):
        """N2-E1′: thiết kế đã lọc đổi objective; không trộn với periodic."""
        gen = _TinyFEGenerator(latent_dim=4, resolution=6)
        a = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]), gen, **self._kw()
        )
        f = tandem_inverse_design_lbfgs(
            torch.tensor([-0.2, -0.2]),
            gen,
            design_filter_sigma=1.0,
            robust_etas=(0.25, 0.75),
            **self._kw(),
        )
        assert a["history"][0] != f["history"][0]
        assert torch.isfinite(f["z"]).all()
        with pytest.raises(ValueError):
            tandem_inverse_design_lbfgs(
                torch.tensor([-0.2, -0.2]),
                gen,
                design_filter_sigma=1.0,
                periodic=True,
                **self._kw(),
            )

    def test_requires_projection(self):
        gen = _TinyFEGenerator(latent_dim=4, resolution=6)
        kw = self._kw()
        kw["projection_betas"] = None
        with pytest.raises(ValueError):
            tandem_inverse_design_lbfgs(
                torch.tensor([-0.2, -0.2]), gen, robust_etas=(0.25,), **kw
            )
        with pytest.raises(ValueError):
            tandem_inverse_design_lbfgs(
                torch.tensor([-0.2, -0.2]),
                gen,
                realization_shifts=[(0.0, 0.5)],
                **kw,
            )
        with pytest.raises(ValueError):
            tandem_inverse_design_lbfgs(
                torch.tensor([-0.2, -0.2]), gen, fe_upsample=2, **kw
            )

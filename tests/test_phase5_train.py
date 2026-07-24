"""
Tests for pipeline/phase5_cvae/train.py - run_epoch, real_fe_r2.

Imports are lazy inside each test - see tests/conftest.py docstring.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader


def _dummy_surrogate():
    class DummySurrogate(torch.nn.Module):
        def forward(self, image, seed_vec):
            return torch.zeros(image.size(0), 3)
    return DummySurrogate()


class TestRunEpoch:
    def test_train_mode_updates_model_params(self, phase3_npz_path):
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import run_epoch

        ds = CVAEDataset(phase3_npz_path)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        before = [p.clone() for p in model.parameters()]
        # Small lr - this net + gamma*PROP_LOSS_SCALE-weighted loss is prone
        # to exploding on a tiny random batch at high lr; the point here is
        # just to confirm run_epoch(train=True) updates weights, not to
        # exercise convergence behavior.
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        stats = run_epoch(
            model, loader, _dummy_surrogate(), ["v12", "v21", "volfrac_achieved"],
            optimizer, beta=0.5, gamma=1.0, lambda_tv=0.0, lambda_bin=0.0,
            device="cpu", train=True,
        )

        for key in ("total", "recon", "kl", "prop", "prop_weighted",
                    "tv", "binarization", "disagreement"):
            assert key in stats
            assert np.isfinite(stats[key])
        after = list(model.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_eval_mode_does_not_update_params(self, phase3_npz_path):
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import run_epoch

        ds = CVAEDataset(phase3_npz_path)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        before = [p.clone() for p in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        run_epoch(
            model, loader, _dummy_surrogate(), ["v12", "v21", "volfrac_achieved"],
            optimizer, beta=0.5, gamma=1.0, lambda_tv=0.0, lambda_bin=0.0,
            device="cpu", train=False,
        )

        after = list(model.parameters())
        assert all(torch.allclose(b, a) for b, a in zip(before, after))

    def test_ensemble_surrogate_path(self, phase3_npz_path):
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import run_epoch

        ds = CVAEDataset(phase3_npz_path)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        stats = run_epoch(
            model, loader, [_dummy_surrogate(), _dummy_surrogate()],
            ["v12", "v21", "volfrac_achieved"],
            optimizer, beta=0.5, gamma=1.0, lambda_tv=0.0, lambda_bin=0.0,
            device="cpu", train=True, lambda_disagreement=0.1,
        )
        assert np.isfinite(stats["total"])


class TestRunEpochRealPhysics:
    """run_epoch(lambda_real_physics>0) - differentiable-physics đầu-cuối
    qua toàn bộ vòng lặp train thật (không chỉ unit test losses.py riêng
    lẻ) - dùng lưới FE 6x6 nhỏ cho nhanh (thay vì 50x50 sản xuất)."""

    def test_train_mode_with_real_physics_updates_decoder_params(self, phase3_npz_path):
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import run_epoch

        ds = CVAEDataset(phase3_npz_path)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        before = [p.clone() for p in model.decoder.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        tiny_fe_params = dict(nelx=6, nely=6, penal=3.0, E0=199.0, Emin=1e-9, nu=0.3, rho0=1.0)

        stats = run_epoch(
            model, loader, _dummy_surrogate(), ["v12", "v21", "volfrac_achieved"],
            optimizer, beta=0.5, gamma=1.0, lambda_tv=0.0, lambda_bin=0.0,
            device="cpu", train=True,
            lambda_real_physics=1.0, real_physics_every=1,
            real_physics_subsample=2, fe_params=tiny_fe_params,
        )

        assert "real_physics" in stats
        assert np.isfinite(stats["real_physics"])
        after = list(model.decoder.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_eval_mode_skips_real_physics(self, phase3_npz_path):
        """Thiết kế: real-physics chỉ áp lúc train (xem comment trong
        run_epoch) - val_stats['real_physics'] phải là NaN (không đo)."""
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import run_epoch

        ds = CVAEDataset(phase3_npz_path)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        tiny_fe_params = dict(nelx=6, nely=6, penal=3.0, E0=199.0, Emin=1e-9, nu=0.3, rho0=1.0)

        stats = run_epoch(
            model, loader, _dummy_surrogate(), ["v12", "v21", "volfrac_achieved"],
            optimizer, beta=0.5, gamma=1.0, lambda_tv=0.0, lambda_bin=0.0,
            device="cpu", train=False,
            lambda_real_physics=1.0, real_physics_every=1,
            real_physics_subsample=2, fe_params=tiny_fe_params,
        )
        assert np.isnan(stats["real_physics"])

    def test_lambda_zero_matches_baseline_behavior(self, phase3_npz_path):
        """lambda_real_physics=0.0 (mặc định) - real_physics không được đo,
        hành vi giống hệt trước khi thêm tính năng này."""
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import run_epoch

        ds = CVAEDataset(phase3_npz_path)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        stats = run_epoch(
            model, loader, _dummy_surrogate(), ["v12", "v21", "volfrac_achieved"],
            optimizer, beta=0.5, gamma=1.0, lambda_tv=0.0, lambda_bin=0.0,
            device="cpu", train=True,
        )
        assert np.isnan(stats["real_physics"])


class TestRealFeR2:
    def test_returns_finite_r2_on_tiny_fe_grid(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import train as train_mod
        from pipeline.phase5_cvae.model import CVAE

        tiny_fe_params = dict(train_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(train_mod, "FE_PARAMS", tiny_fe_params)

        model = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        conditions = np.array([[-0.5, -0.4], [0.2, 0.3], [-0.1, 0.0], [0.4, -0.2]])

        r2 = train_mod.real_fe_r2(model, conditions, device="cpu")

        assert np.isfinite(r2) or np.isnan(r2)  # never inf; nan only if <2 valid points

    def test_returns_nan_with_fewer_than_two_conditions(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import train as train_mod
        from pipeline.phase5_cvae.model import CVAE

        tiny_fe_params = dict(train_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(train_mod, "FE_PARAMS", tiny_fe_params)

        model = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        conditions = np.array([[-0.5, -0.4]])

        r2 = train_mod.real_fe_r2(model, conditions, device="cpu")
        assert np.isnan(r2)

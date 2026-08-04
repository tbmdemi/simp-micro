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


class TestApplyConditionDropout:
    """Classifier-free-guidance-style dropout on the 2 optional columns
    (volfrac at (2,3), void_size_frac at (4,5)) - see train.py docstring.
    Columns 0,1 (v12,v21) must never be touched."""

    def test_never_touches_v12_v21_columns(self):
        from pipeline.phase5_cvae.train import apply_condition_dropout
        torch.manual_seed(0)
        condition = torch.tensor([[-0.5, 0.3, 0.4, 1.0, 0.2, 1.0]] * 20)
        dropped = apply_condition_dropout(condition, dropout_p=1.0)
        assert torch.allclose(dropped[:, 0], condition[:, 0])
        assert torch.allclose(dropped[:, 1], condition[:, 1])

    def test_dropout_p_one_always_zeros_optional_columns(self):
        from pipeline.phase5_cvae.train import apply_condition_dropout
        torch.manual_seed(0)
        condition = torch.tensor([[-0.5, 0.3, 0.4, 1.0, 0.2, 1.0]] * 20)
        dropped = apply_condition_dropout(condition, dropout_p=1.0)
        assert torch.all(dropped[:, 2:] == 0.0)

    def test_dropout_p_zero_never_touches_optional_columns(self):
        from pipeline.phase5_cvae.train import apply_condition_dropout
        torch.manual_seed(0)
        condition = torch.tensor([[-0.5, 0.3, 0.4, 1.0, 0.2, 1.0]] * 20)
        dropped = apply_condition_dropout(condition, dropout_p=0.0)
        assert torch.allclose(dropped, condition)

    def test_value_and_mask_always_dropped_together(self):
        """Nếu mask=0, value đi kèm PHẢI cũng =0 (và ngược lại mask=1 thì
        value giữ nguyên) - không được lệch pha value/mask cho cùng 1 mẫu."""
        from pipeline.phase5_cvae.train import apply_condition_dropout
        torch.manual_seed(1)
        condition = torch.tensor([[-0.5, 0.3, 0.4, 1.0, 0.2, 1.0]] * 50)
        dropped = apply_condition_dropout(condition, dropout_p=0.5)
        volfrac_dropped = dropped[:, 3] == 0.0
        assert torch.all((dropped[:, 2] == 0.0) == volfrac_dropped)
        void_dropped = dropped[:, 5] == 0.0
        assert torch.all((dropped[:, 4] == 0.0) == void_dropped)

    def test_does_not_mutate_input_in_place(self):
        from pipeline.phase5_cvae.train import apply_condition_dropout
        torch.manual_seed(0)
        condition = torch.tensor([[-0.5, 0.3, 0.4, 1.0, 0.2, 1.0]] * 10)
        original = condition.clone()
        apply_condition_dropout(condition, dropout_p=1.0)
        assert torch.allclose(condition, original)


class TestRunEpochExtendedCondition:
    """run_epoch(extended_condition=True, lambda_volfrac>0) - end-to-end
    through the real training loop (not just losses.py's unit test), same
    spirit as TestRunEpochRealPhysics above."""

    def test_extended_condition_updates_decoder_params(self, make_phase3_npz):
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import run_epoch

        path = make_phase3_npz("val.npz", n_samples=12)
        ds = CVAEDataset(path, extended_condition=True)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = CVAE(condition_dim=6, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        before = [p.clone() for p in model.decoder.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        stats = run_epoch(
            model, loader, _dummy_surrogate(), ["v12", "v21", "volfrac_achieved"],
            optimizer, beta=0.5, gamma=1.0, lambda_tv=0.0, lambda_bin=0.0,
            device="cpu", train=True,
            extended_condition=True, lambda_volfrac=1.0, optional_dropout_p=0.5,
        )

        assert "volfrac_loss" in stats
        assert np.isfinite(stats["volfrac_loss"])
        after = list(model.decoder.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_lambda_volfrac_zero_gives_zero_volfrac_loss(self, make_phase3_npz):
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import run_epoch

        path = make_phase3_npz("val.npz", n_samples=8)
        ds = CVAEDataset(path, extended_condition=True)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = CVAE(condition_dim=6, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        stats = run_epoch(
            model, loader, _dummy_surrogate(), ["v12", "v21", "volfrac_achieved"],
            optimizer, beta=0.5, gamma=1.0, lambda_tv=0.0, lambda_bin=0.0,
            device="cpu", train=True,
            extended_condition=True, lambda_volfrac=0.0,
        )
        assert stats["volfrac_loss"] == 0.0

    def test_non_extended_condition_ignores_new_args(self, phase3_npz_path):
        """Backward-compat: extended_condition=False (mặc định) - lambda_volfrac
        bị bỏ qua hoàn toàn dù truyền >0, không lỗi shape trên condition 2 chiều."""
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import run_epoch

        ds = CVAEDataset(phase3_npz_path)  # extended_condition=False
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        stats = run_epoch(
            model, loader, _dummy_surrogate(), ["v12", "v21", "volfrac_achieved"],
            optimizer, beta=0.5, gamma=1.0, lambda_tv=0.0, lambda_bin=0.0,
            device="cpu", train=True,
            extended_condition=False, lambda_volfrac=1.0,
        )
        assert stats["volfrac_loss"] == 0.0


class TestResizeConditionDimWeights:
    """resize_condition_dim_weights() - --resume-from một checkpoint có
    condition_dim khác model hiện tại (vd fine-tune cvae_realphysics.pt
    condition_dim=2 thành --extended-condition condition_dim=6, đúng lệnh
    khuyến nghị ở README mục 5.1). Trước fix này, model.load_state_dict()
    strict=True crash ngay với RuntimeError size mismatch trên 3 layer fc
    (encoder.fc_mu/fc_logvar, decoder.fc) - tái hiện bằng cách chạy chính
    lệnh README ở advisor-test session 2026-08-03, xem EXPERIMENT_LOG.md."""

    def test_same_dim_is_noop(self):
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import resize_condition_dim_weights

        model = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        sd = model.state_dict()
        out = resize_condition_dim_weights(sd, 2, 2)
        assert out is sd

    def test_widen_preserves_base_and_v12_v21_columns(self):
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import resize_condition_dim_weights

        small = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        old_sd = small.state_dict()
        new_sd = resize_condition_dim_weights(old_sd, 2, 6)

        for key in ["encoder.fc_mu.weight", "encoder.fc_logvar.weight",
                    "decoder.fc.weight"]:
            old_w = old_sd[key]
            new_w = new_sd[key]
            base_dim = old_w.shape[1] - 2
            assert new_w.shape == (old_w.shape[0], base_dim + 6)
            assert torch.equal(new_w[:, :base_dim], old_w[:, :base_dim])
            assert torch.equal(new_w[:, base_dim:base_dim + 2],
                                old_w[:, base_dim:base_dim + 2])
            # nouveau condition cols (volfrac/mask/void/mask) must be
            # freshly initialized, not left as zeros/garbage.
            assert new_w[:, base_dim + 2:].std().item() > 0

        # unrelated params (biases, conv backbone) untouched.
        for key in ["encoder.fc_mu.bias", "encoder.fc_logvar.bias",
                    "decoder.fc.bias"]:
            assert torch.equal(old_sd[key], new_sd[key])

    def test_widened_state_dict_loads_into_new_model_without_crash(self):
        """End-to-end: đúng lệnh README --extended-condition --resume-from -
        model.load_state_dict(strict=True) trên state_dict đã resize phải
        THÀNH CÔNG (trước fix: RuntimeError size mismatch ngay tại đây)."""
        from pipeline.phase5_cvae.model import CVAE
        from pipeline.phase5_cvae.train import resize_condition_dim_weights

        small = CVAE(condition_dim=2, latent_dim=4, resolution=64,
                      channels=(4, 8, 16, 32))
        big = CVAE(condition_dim=6, latent_dim=4, resolution=64,
                    channels=(4, 8, 16, 32))
        resized = resize_condition_dim_weights(small.state_dict(), 2, 6)
        big.load_state_dict(resized)  # must not raise


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

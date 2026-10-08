"""
Tests for pipeline/phase5_cvae/losses.py.

losses.py imports torch/torch.nn/torch.nn.functional plus real_physics
(RealPhysicsNu - safe leaf module, no sys.path tricks of its own) at module
level; the phase4-surrogate import is done lazily via importlib inside
load_frozen_surrogate(), specifically to dodge the bare-import collision
documented in tests/conftest.py), so a top-level import here is safe.
"""
import pytest
import torch

from pipeline.phase5_cvae.losses import (
    binarization_loss,
    cvae_loss,
    kl_beta_schedule,
    kl_divergence,
    periodicity_loss,
    prior_sample_regularization,
    property_consistency_loss,
    property_consistency_loss_ensemble,
    real_physics_loss,
    real_physics_prior_loss,
    reconstruction_loss,
    tv_loss,
    volfrac_consistency_loss,
)
from pipeline.phase4_surrogate.model import SurrogateCNN


class _ConstantSurrogate(torch.nn.Module):
    """Stub surrogate: predicts `value` across all `n_outputs` columns,
    ignoring the actual input - used where tests only need a deterministic,
    input-independent prediction (was reimplemented ad-hoc under 5 different
    names across this file)."""
    def __init__(self, value=0.0, n_outputs=3):
        super().__init__()
        self.value, self.n_outputs = value, n_outputs

    def forward(self, image, seed_vec):
        return torch.full((image.size(0), self.n_outputs), self.value)


class _PerfectSurrogate(torch.nn.Module):
    """Stub surrogate: predicts exactly `._target` (caller wires this to the
    condition tensor) - used to test the zero-loss-when-correct case."""
    def forward(self, image, seed_vec):
        return self._target


class _Nu0EchoSurrogate(torch.nn.Module):
    """Stub surrogate with include_nu0=True: echoes the received nu0 value
    into v12/v21 (ignoring image/seed_vec) so a test can assert the exact
    per-sample nu0 that property_consistency_loss() passed through."""
    include_nu0 = True

    def forward(self, image, seed_vec, nu0=None):
        if nu0 is None:
            raise ValueError("include_nu0=True nhung forward() khong nhan nu0")
        return torch.stack([nu0, nu0, torch.zeros_like(nu0)], dim=1)


def _write_surrogate_export(path, n_seeds=4, channels=(8, 16), fc_hidden=16,
                             n_outputs=3, target_names=None):
    model = SurrogateCNN(n_seeds=n_seeds, channels=channels, fc_hidden=fc_hidden,
                          n_outputs=n_outputs)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_seeds": n_seeds,
        "channels": channels,
        "fc_hidden": fc_hidden,
        "n_outputs": n_outputs,
        "target_names": target_names or ["v12", "v21", "volfrac_achieved"],
    }, path)


class TestKlBetaSchedule:
    @pytest.mark.parametrize("epoch,warmup_epochs,beta_max,expected", [
        (5, 0, 1.0, 1.0),      # zero warmup -> beta_max immediately
        (5, 10, 1.0, 0.5),     # linear ramp, midpoint
        (0, 10, 1.0, 0.0),     # linear ramp, start
        (100, 10, 1.0, 1.0),   # clamped after warmup ends
        (5, 10, 2.0, 1.0),     # scales with beta_max
    ], ids=["zero-warmup", "ramp-mid", "ramp-start", "clamped", "scaled-beta-max"])
    def test_schedule(self, epoch, warmup_epochs, beta_max, expected):
        assert kl_beta_schedule(epoch=epoch, warmup_epochs=warmup_epochs, beta_max=beta_max) == expected


class TestReconstructionLoss:
    def test_perfect_reconstruction_near_zero(self):
        target = torch.rand(4, 1, 8, 8).clamp(0.01, 0.99)
        loss = reconstruction_loss(target, target)
        assert loss.item() >= 0.0

    def test_worse_reconstruction_higher_loss(self):
        target = torch.full((4, 1, 8, 8), 0.9)
        good = torch.full((4, 1, 8, 8), 0.9)
        bad = torch.full((4, 1, 8, 8), 0.1)
        assert reconstruction_loss(bad, target) > reconstruction_loss(good, target)

    def test_scales_by_batch_not_by_all_elements(self):
        # sum-per-pixel / batch_size (not mean over everything) - batch of 1
        # vs batch of 2 with identical per-sample content should give the
        # same per-sample loss value.
        img = torch.full((1, 1, 4, 4), 0.7)
        target = torch.full((1, 1, 4, 4), 0.3)
        loss_b1 = reconstruction_loss(img, target)
        loss_b2 = reconstruction_loss(img.repeat(2, 1, 1, 1), target.repeat(2, 1, 1, 1))
        assert torch.isclose(loss_b1, loss_b2, atol=1e-4)


class TestKlDivergence:
    def test_zero_for_standard_normal_posterior(self):
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)
        kl = kl_divergence(mu, logvar)
        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_positive_when_posterior_deviates(self):
        mu = torch.full((4, 8), 2.0)
        logvar = torch.zeros(4, 8)
        kl = kl_divergence(mu, logvar)
        assert kl.item() > 0.0


class TestPropertyConsistencyLoss:
    def test_zero_when_surrogate_predicts_target_exactly(self):
        surrogate = _PerfectSurrogate()
        recon = torch.rand(3, 1, 8, 8)
        seed_vec = torch.zeros(3, 2)
        condition = torch.tensor([[-0.5, 0.2], [0.1, -0.3], [0.0, 0.0]])
        surrogate._target = torch.stack([
            condition[:, 0], condition[:, 1], torch.zeros(3),
        ], dim=1)
        loss = property_consistency_loss(
            recon, condition, seed_vec, surrogate, ["v12", "v21", "volfrac_achieved"]
        )
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_nonzero_when_prediction_is_off(self):
        surrogate = _ConstantSurrogate(0.0)
        recon = torch.rand(2, 1, 8, 8)
        seed_vec = torch.zeros(2, 2)
        condition = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = property_consistency_loss(
            recon, condition, seed_vec, surrogate, ["v12", "v21", "volfrac_achieved"]
        )
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_include_nu0_surrogate_receives_per_sample_nu0(self):
        """Bug đã sửa 2026-08-19 (Giai đoạn A): surrogate.include_nu0=True
        (vd surrogate_a4_nu0.pt) nhưng property_consistency_loss() gọi
        surrogate(recon, seed_vec) không có nu0 -> SurrogateCNN.forward()
        raise ValueError. nu0_col trỏ đúng cột [nu0, nu0_mask] trong
        condition (giống nu0_col của real_physics_loss(), xem A6)."""
        surrogate = _Nu0EchoSurrogate()
        recon = torch.rand(2, 1, 8, 8)
        seed_vec = torch.zeros(2, 2)
        # condition = [v12, v21, nu0, nu0_mask]; mẫu 0 có mask=1 (nu0=0.25
        # thật), mẫu 1 có mask=0 (không chỉ định -> fallback 0.3).
        condition = torch.tensor([[0.0, 0.0, 0.25, 1.0], [0.0, 0.0, 0.99, 0.0]])
        loss = property_consistency_loss(
            recon, condition, seed_vec, surrogate,
            ["v12", "v21", "volfrac_achieved"], nu0_col=2,
        )
        # _Nu0EchoSurrogate dự đoán v12=v21=nu0 nhận được -> so khớp condition[:, :2]=0
        # => loss = mean(nu0_used^2) với nu0_used=[0.25, 0.3] (mẫu 1 fallback default).
        expected = torch.tensor([0.25, 0.3]).pow(2).mean()
        assert loss.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_include_nu0_surrogate_without_nu0_col_uses_default(self):
        """nu0_col=None (caller không truyền, vd cVAE condition_dim=2 không
        có cột nu0) nhưng surrogate.include_nu0=True vẫn phải chạy được -
        fallback toàn batch về 0.3, không crash."""
        surrogate = _Nu0EchoSurrogate()
        recon = torch.rand(2, 1, 8, 8)
        seed_vec = torch.zeros(2, 2)
        condition = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        loss = property_consistency_loss(
            recon, condition, seed_vec, surrogate,
            ["v12", "v21", "volfrac_achieved"], nu0_col=None,
        )
        assert loss.item() == pytest.approx(0.3 ** 2, abs=1e-5)


class TestPropertyConsistencyLossEnsemble:
    def test_matches_single_model_when_ensemble_of_one_copy(self):
        s1, s2 = _ConstantSurrogate(0.3), _ConstantSurrogate(0.3)
        recon = torch.rand(2, 1, 8, 8)
        seed_vec = torch.zeros(2, 2)
        condition = torch.zeros(2, 2)
        target_names = ["v12", "v21", "volfrac_achieved"]

        single = property_consistency_loss(recon, condition, seed_vec, s1, target_names)
        mse_ens, disagreement = property_consistency_loss_ensemble(
            recon, condition, seed_vec, [s1, s2], target_names, lambda_disagreement=0.0
        )
        assert torch.isclose(single, mse_ens, atol=1e-6)
        # Two identical constant models never disagree.
        assert torch.isclose(disagreement, torch.tensor(0.0), atol=1e-6)

    def test_disagreement_penalizes_divergent_surrogates(self):
        low, high = _ConstantSurrogate(0.0), _ConstantSurrogate(2.0)
        recon = torch.rand(2, 1, 8, 8)
        seed_vec = torch.zeros(2, 2)
        condition = torch.ones(2, 2)  # mean pred = 1.0 -> MSE with target 1.0 is 0
        target_names = ["v12", "v21", "volfrac_achieved"]

        mse_only, disagreement = property_consistency_loss_ensemble(
            recon, condition, seed_vec, [low, high],
            target_names, lambda_disagreement=0.0,
        )
        assert torch.isclose(mse_only, torch.tensor(0.0), atol=1e-6)
        assert disagreement.item() > 0.0

        penalized, _ = property_consistency_loss_ensemble(
            recon, condition, seed_vec, [low, high],
            target_names, lambda_disagreement=1.0,
        )
        assert penalized.item() > mse_only.item()


class TestTvAndBinarizationLoss:
    def test_tv_loss_zero_for_uniform_image(self):
        img = torch.full((2, 1, 8, 8), 0.5)
        assert torch.isclose(tv_loss(img), torch.tensor(0.0), atol=1e-6)

    def test_tv_loss_positive_for_checkerboard(self):
        img = torch.zeros(1, 1, 4, 4)
        img[:, :, ::2, ::2] = 1.0
        img[:, :, 1::2, 1::2] = 1.0
        assert tv_loss(img).item() > 0.0

    def test_binarization_loss_zero_for_binary_image(self):
        img = (torch.rand(2, 1, 8, 8) > 0.5).float()
        assert torch.isclose(binarization_loss(img), torch.tensor(0.0), atol=1e-6)

    def test_binarization_loss_maximal_at_half(self):
        img = torch.full((1, 1, 4, 4), 0.5)
        # (x * (1-x)) maximized at x=0.5 -> value 0.25
        assert torch.isclose(binarization_loss(img), torch.tensor(0.25), atol=1e-6)


class TestPeriodicityLoss:
    def test_zero_when_opposite_edges_match(self):
        img = torch.zeros(2, 1, 8, 8)
        img[:, :, :, 0] = 1.0
        img[:, :, :, -1] = 1.0   # left col == right col
        img[:, :, 0, :] = 0.3
        img[:, :, -1, :] = 0.3   # top row == bottom row
        assert torch.isclose(periodicity_loss(img), torch.tensor(0.0), atol=1e-6)

    def test_positive_when_opposite_edges_mismatch(self):
        img = torch.zeros(2, 1, 8, 8)
        img[:, :, :, 0] = 1.0    # left col solid, right col left at 0 (void)
        assert periodicity_loss(img).item() > 0.0

    def test_uses_only_boundary_pixels_not_interior(self):
        img_a = torch.zeros(1, 1, 8, 8)
        img_b = img_a.clone()
        img_b[:, :, 3:5, 3:5] = 1.0  # interior-only change
        assert torch.isclose(periodicity_loss(img_a), periodicity_loss(img_b), atol=1e-6)


class TestPriorSampleRegularization:
    """prior_sample_regularization() must decode from a PRIOR z ~ N(0,1),
    the same regime CVAE.generate() uses at inference - NOT the posterior
    z the rest of cvae_loss operates on. See the function's docstring for
    why this distinction was added: empirically, regularizing posterior
    reconstructions didn't move generation-time manufacturability at all
    (~0-3.5% pass rate before AND after a 25-epoch fine-tune)."""

    def _make_decoder(self, latent_dim=4, resolution=8):
        from pipeline.phase5_cvae.model import Decoder
        return Decoder(condition_dim=2, latent_dim=latent_dim,
                        channels=(8, 4), resolution=resolution)

    def test_returns_all_expected_keys_and_is_differentiable(self):
        decoder = self._make_decoder()
        condition = torch.zeros(3, 2, requires_grad=False)
        total, stats = prior_sample_regularization(
            decoder, latent_dim=4, condition=condition,
            lambda_tv=0.1, lambda_bin=0.1, lambda_periodic=0.1,
        )
        for key in ("prior_tv", "prior_binarization", "prior_periodic"):
            assert key in stats
        # total must carry gradient back to decoder params (weights, not z)
        total.backward()
        assert any(p.grad is not None for p in decoder.parameters())

    def test_zero_lambdas_gives_zero_total(self):
        decoder = self._make_decoder()
        condition = torch.zeros(3, 2)
        total, _ = prior_sample_regularization(
            decoder, latent_dim=4, condition=condition,
            lambda_tv=0.0, lambda_bin=0.0, lambda_periodic=0.0,
        )
        assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)

    def test_uses_a_fresh_random_z_each_call(self):
        """Regression guard: if this ever gets refactored to reuse/cache a z
        or accidentally decode from a fixed input, two calls would produce
        identical stats - real randn() sampling almost never does."""
        decoder = self._make_decoder()
        condition = torch.zeros(3, 2)
        _, stats_a = prior_sample_regularization(
            decoder, latent_dim=4, condition=condition, lambda_periodic=1.0,
        )
        _, stats_b = prior_sample_regularization(
            decoder, latent_dim=4, condition=condition, lambda_periodic=1.0,
        )
        assert stats_a["prior_periodic"].item() != stats_b["prior_periodic"].item()


FE_PARAMS_SMALL = dict(nelx=6, nely=6, penal=3.0, E0=199.0, Emin=1e-9, nu=0.3, rho0=1.0)


class TestRealPhysicsLoss:
    """real_physics_loss() dùng FE-solve THẬT (không surrogate) - kiểm tra
    'ống dẫn' (shape/dtype/differentiability/resize/subsample), KHÔNG lặp
    lại kiểm chứng công thức đạo hàm (đã có ở
    tests/test_phase5_real_physics.py::TestGradientCorrectness)."""

    def test_is_differentiable_wrt_density(self):
        density = torch.rand(2, 1, 8, 8, requires_grad=True)
        condition = torch.zeros(2, 2)
        loss = real_physics_loss(density, condition, FE_PARAMS_SMALL)
        assert loss.dim() == 0
        loss.backward()
        assert density.grad is not None
        assert torch.all(torch.isfinite(density.grad))

    def test_accepts_3d_density_without_channel_dim(self):
        density = torch.rand(2, 8, 8, requires_grad=True)
        condition = torch.zeros(2, 2)
        loss = real_physics_loss(density, condition, FE_PARAMS_SMALL)
        loss.backward()
        assert density.grad is not None

    def test_subsample_restricts_batch(self):
        """subsample=1 chỉ nên lan gradient tới 1 trong 3 mẫu trong batch -
        2 mẫu còn lại phải có gradient đúng bằng 0."""
        torch.manual_seed(0)
        density = torch.rand(3, 1, 8, 8, requires_grad=True)
        condition = torch.zeros(3, 2)
        loss = real_physics_loss(density, condition, FE_PARAMS_SMALL, subsample=1)
        loss.backward()
        n_nonzero_samples = (density.grad.abs().sum(dim=(1, 2, 3)) > 0).sum().item()
        assert n_nonzero_samples == 1

    def test_resize_handles_mismatched_resolution(self):
        """decoder output (vd 16x16) khác lưới FE (6x6 trong FE_PARAMS_SMALL) -
        phải resize khả vi (F.interpolate) chứ không lỗi shape."""
        density = torch.rand(1, 1, 16, 16, requires_grad=True)
        condition = torch.zeros(1, 2)
        loss = real_physics_loss(density, condition, FE_PARAMS_SMALL)
        assert torch.isfinite(loss)
        loss.backward()
        assert density.grad.shape == density.shape


class TestRealPhysicsLossNu0Col:
    """A6 (docs/archive/PROJECT_PLAN.md Nhóm 1): nu0_col phải khiến
    real_physics_loss
    dùng ĐÚNG ν0 per-sample (khi mask=1) thay vì fe_params['nu'] cố định cho
    cả batch - kiểm tra bằng cách đặt target = giá trị THẬT tính sẵn qua
    solve_nu_with_grad với đúng ν0 của từng mẫu, rồi xác nhận loss ~0 (nếu
    code dùng sai ν0, Q khác -> v12/v21 khác -> loss KHÔNG thể ~0)."""

    def _fe_kw(self, nu):
        return dict(penal=3.0, E0=199.0, Emin=1e-9, nu=nu, rho0=1.0)

    def test_uses_per_sample_nu_when_mask_set(self):
        from pipeline.phase5_cvae.real_physics import solve_nu_with_grad
        torch.manual_seed(0)
        density = torch.rand(2, 1, 6, 6)
        nu0, nu1 = 0.15, 0.4
        v12_0, v21_0, _, _ = solve_nu_with_grad(
            density[0, 0].numpy().astype("float64"), **self._fe_kw(nu0))
        v12_1, v21_1, _, _ = solve_nu_with_grad(
            density[1, 0].numpy().astype("float64"), **self._fe_kw(nu1))

        # condition = [v12_target, v21_target, nu0, nu0_mask] - target ĐÚNG
        # bằng giá trị giải với ν0 riêng của từng mẫu.
        condition = torch.tensor([
            [v12_0, v21_0, nu0, 1.0],
            [v12_1, v21_1, nu1, 1.0],
        ], dtype=torch.float32)

        loss = real_physics_loss(density, condition, FE_PARAMS_SMALL, nu0_col=2)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_ignoring_nu0_col_gives_wrong_loss_for_non_default_nu(self):
        """Regression guard: nếu KHÔNG truyền nu0_col (hành vi cũ), loss
        phải KHÁC 0 rõ rệt cho cùng input ở test trên (dùng fe_params['nu']
        =0.3 sai cho cả 2 mẫu có ν0 thật =0.15/0.4) - xác nhận test trên
        thực sự đang kiểm tra đúng thứ (không phải loss ~0 do trùng hợp)."""
        from pipeline.phase5_cvae.real_physics import solve_nu_with_grad
        torch.manual_seed(0)
        density = torch.rand(2, 1, 6, 6)
        nu0, nu1 = 0.15, 0.4
        v12_0, v21_0, _, _ = solve_nu_with_grad(
            density[0, 0].numpy().astype("float64"), **self._fe_kw(nu0))
        v12_1, v21_1, _, _ = solve_nu_with_grad(
            density[1, 0].numpy().astype("float64"), **self._fe_kw(nu1))
        condition = torch.tensor([
            [v12_0, v21_0, nu0, 1.0],
            [v12_1, v21_1, nu1, 1.0],
        ], dtype=torch.float32)

        loss = real_physics_loss(density, condition, FE_PARAMS_SMALL)  # nu0_col=None
        assert loss.item() > 1e-4

    def test_masked_off_sample_falls_back_to_fe_params_default(self):
        """mask=0 (nu0 bị condition-dropout/không chỉ định) phải dùng
        fe_params['nu'] LÀM FALLBACK, không dùng giá trị cột (đã bị zero
        theo đúng quy ước dropout) - value=0.0 không có nghĩa vật lý
        "ν0=0", chỉ có nghĩa "không biết"."""
        from pipeline.phase5_cvae.real_physics import solve_nu_with_grad
        torch.manual_seed(1)
        density = torch.rand(1, 1, 6, 6)
        default_nu = FE_PARAMS_SMALL["nu"]  # 0.3
        v12_default, v21_default, _, _ = solve_nu_with_grad(
            density[0, 0].numpy().astype("float64"), **self._fe_kw(default_nu))

        # value=0.0 (đã dropout), mask=0.0 -> PHẢI fallback fe_params['nu'],
        # không dùng nu=0.0 (khác 0.3, sẽ cho Q/v12/v21 khác).
        condition = torch.tensor([[v12_default, v21_default, 0.0, 0.0]], dtype=torch.float32)
        loss = real_physics_loss(density, condition, FE_PARAMS_SMALL, nu0_col=2)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_subsample_slices_nu0_column_consistently(self):
        """subsample phải cắt condition_sub TRƯỚC khi trích cột nu0 - nếu
        không, index sẽ lệch giữa density_sub và nu per-sample."""
        from pipeline.phase5_cvae.real_physics import solve_nu_with_grad
        torch.manual_seed(2)
        density = torch.rand(3, 1, 6, 6, requires_grad=True)
        nus = [0.15, 0.25, 0.4]
        targets = [
            solve_nu_with_grad(density[i, 0].detach().numpy().astype("float64"),
                                **self._fe_kw(nus[i]))[:2]
            for i in range(3)
        ]
        condition = torch.tensor([
            [targets[i][0], targets[i][1], nus[i], 1.0] for i in range(3)
        ], dtype=torch.float32)

        loss = real_physics_loss(density, condition, FE_PARAMS_SMALL,
                                  subsample=3, nu0_col=2)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestRealPhysicsPriorLossNu0Col:
    def test_forwards_nu0_col_to_real_physics_loss(self, monkeypatch):
        """real_physics_prior_loss phải truyền nu0_col xuống real_physics_loss
        nguyên vẹn - kiểm tra 'ống dẫn', không lặp lại kiểm chứng vật lý."""
        import pipeline.phase5_cvae.losses as losses_mod
        captured = {}

        def _fake_real_physics_loss(density, condition, fe_params, subsample=None,
                                     n_workers=0, nu0_col=None):
            captured["nu0_col"] = nu0_col
            return torch.tensor(0.0, requires_grad=True)

        monkeypatch.setattr(losses_mod, "real_physics_loss", _fake_real_physics_loss)
        from pipeline.phase5_cvae.model import Decoder
        decoder = Decoder(condition_dim=4, latent_dim=4, channels=(8, 4), resolution=8)
        condition = torch.zeros(2, 4)
        real_physics_prior_loss(decoder, latent_dim=4, condition=condition,
                                 fe_params=FE_PARAMS_SMALL, nu0_col=2)
        assert captured["nu0_col"] == 2


class TestRealPhysicsPriorLoss:
    """real_physics_prior_loss() - áp lên ảnh decode từ z ~ PRIOR (cùng chế
    độ model.generate()), cùng lý do với prior_sample_regularization()."""

    def _make_decoder(self, latent_dim=4, resolution=8):
        from pipeline.phase5_cvae.model import Decoder
        return Decoder(condition_dim=2, latent_dim=latent_dim,
                        channels=(8, 4), resolution=resolution)

    def test_gradient_flows_to_decoder_params(self):
        decoder = self._make_decoder()
        condition = torch.zeros(2, 2)
        loss = real_physics_prior_loss(decoder, latent_dim=4, condition=condition,
                                        fe_params=FE_PARAMS_SMALL)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in decoder.parameters())

    def test_uses_fresh_random_z_each_call(self):
        decoder = self._make_decoder()
        condition = torch.zeros(2, 2)
        loss_a = real_physics_prior_loss(decoder, latent_dim=4, condition=condition,
                                          fe_params=FE_PARAMS_SMALL)
        loss_b = real_physics_prior_loss(decoder, latent_dim=4, condition=condition,
                                          fe_params=FE_PARAMS_SMALL)
        assert loss_a.item() != loss_b.item()


class TestCvaeLoss:
    def test_returns_all_expected_keys(self):
        recon = torch.rand(2, 1, 8, 8, requires_grad=True)
        image = torch.rand(2, 1, 8, 8)
        mu = torch.zeros(2, 4, requires_grad=True)
        logvar = torch.zeros(2, 4, requires_grad=True)
        condition = torch.zeros(2, 2)
        seed_vec = torch.zeros(2, 2)

        out = cvae_loss(
            recon, image, mu, logvar, condition, seed_vec,
            _ConstantSurrogate(0.0), ["v12", "v21", "volfrac_achieved"],
            beta=0.5, gamma=1.0,
        )
        for key in ("total", "recon", "kl", "prop", "prop_weighted",
                    "tv", "binarization", "periodic", "disagreement", "beta"):
            assert key in out
        assert out["total"].requires_grad  # differentiable wrt recon/mu/logvar

    def test_ensemble_path_when_surrogate_is_a_list(self):
        recon = torch.rand(2, 1, 8, 8)
        image = torch.rand(2, 1, 8, 8)
        mu = torch.zeros(2, 4)
        logvar = torch.zeros(2, 4)
        condition = torch.zeros(2, 2)
        seed_vec = torch.zeros(2, 2)

        out = cvae_loss(
            recon, image, mu, logvar, condition, seed_vec,
            [_ConstantSurrogate(0.0), _ConstantSurrogate(0.0)], ["v12", "v21", "volfrac_achieved"],
            beta=0.5, gamma=1.0, lambda_disagreement=0.1,
        )
        assert torch.isfinite(out["total"])


class TestExtendedConditionSlicing:
    """property_consistency_loss(_ensemble)/real_physics_loss must compare
    against condition[:, :2] (v12,v21) only - surrogate/FE only ever predict
    those 2 - even when condition is wider (extended_condition=True in
    dataset.py, condition_dim=6: [v12,v21,volfrac,mask,void,mask]). Guards
    against a shape-mismatch/silently-wrong-column regression."""

    def test_property_consistency_loss_ignores_extra_condition_columns(self):
        surrogate = _PerfectSurrogate()
        recon = torch.rand(2, 1, 8, 8)
        seed_vec = torch.zeros(2, 2)
        # condition_dim=6: extra columns (volfrac/mask/void/mask) must be
        # ignored, only columns 0,1 (v12,v21) compared.
        condition = torch.tensor([[-0.5, 0.2, 0.4, 1.0, 0.3, 1.0],
                                   [0.1, -0.3, 0.0, 0.0, 0.0, 0.0]])
        surrogate._target = torch.stack([
            condition[:, 0], condition[:, 1], torch.zeros(2),
        ], dim=1)
        loss = property_consistency_loss(
            recon, condition, seed_vec, surrogate, ["v12", "v21", "volfrac_achieved"]
        )
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_real_physics_loss_ignores_extra_condition_columns(self):
        density = torch.rand(2, 1, 8, 8, requires_grad=True)
        condition = torch.zeros(2, 6)  # only cols 0,1 should matter
        fe_params = dict(nelx=6, nely=6, penal=3.0, E0=199.0, Emin=1e-9, nu=0.3, rho0=1.0)
        loss = real_physics_loss(density, condition, fe_params)
        assert torch.isfinite(loss)
        loss.backward()
        assert density.grad is not None

    def test_property_consistency_loss_ensemble_ignores_extra_columns(self):
        recon = torch.rand(2, 1, 8, 8)
        seed_vec = torch.zeros(2, 2)
        condition = torch.full((2, 6), 0.3)  # cols 0,1 = 0.3 (matches), rest irrelevant
        mse, _ = property_consistency_loss_ensemble(
            recon, condition, seed_vec, [_ConstantSurrogate(0.3), _ConstantSurrogate(0.3)],
            ["v12", "v21", "volfrac_achieved"], lambda_disagreement=0.0,
        )
        assert torch.isclose(mse, torch.tensor(0.0), atol=1e-6)


class TestVolfracConsistencyLoss:
    """volfrac là chiều OPTIONAL rẻ nhất trong extended_condition - suy trực
    tiếp từ recon.mean() (không cần surrogate/FE), chỉ tính trên mẫu
    mask=1 (xem docstring hàm) để không mâu thuẫn với condition-dropout ở
    train.py."""

    def test_zero_when_recon_mean_matches_target_exactly(self):
        recon = torch.full((2, 1, 4, 4), 0.3)  # mean pixel = 0.3
        target = torch.tensor([0.3, 0.3])
        mask = torch.tensor([1.0, 1.0])
        loss = volfrac_consistency_loss(recon, target, mask)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_when_recon_mean_differs(self):
        recon = torch.full((2, 1, 4, 4), 0.3)
        target = torch.tensor([0.6, 0.6])
        mask = torch.tensor([1.0, 1.0])
        loss = volfrac_consistency_loss(recon, target, mask)
        assert loss.item() > 0.0

    def test_masked_out_samples_are_excluded(self):
        """mask=0 mẫu 1 (recon.mean=0.9, target=0.0 - sai lệch RẤT lớn nếu
        tính) - loss chỉ nên phản ánh mẫu 0 (mask=1, khớp hoàn hảo) -> 0."""
        recon = torch.stack([
            torch.full((1, 4, 4), 0.3),
            torch.full((1, 4, 4), 0.9),
        ])
        target = torch.tensor([0.3, 0.0])
        mask = torch.tensor([1.0, 0.0])
        loss = volfrac_consistency_loss(recon, target, mask)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_all_masked_out_does_not_divide_by_zero(self):
        recon = torch.rand(3, 1, 4, 4)
        target = torch.zeros(3)
        mask = torch.zeros(3)
        loss = volfrac_consistency_loss(recon, target, mask)
        assert torch.isfinite(loss)

    def test_is_differentiable_wrt_recon(self):
        recon = torch.rand(2, 1, 4, 4, requires_grad=True)
        target = torch.tensor([0.5, 0.5])
        mask = torch.tensor([1.0, 1.0])
        loss = volfrac_consistency_loss(recon, target, mask)
        loss.backward()
        assert recon.grad is not None
        assert torch.all(torch.isfinite(recon.grad))


class TestLoadFrozenSurrogate:
    def test_load_returns_frozen_eval_model(self, tmp_path):
        from pipeline.phase5_cvae.losses import load_frozen_surrogate
        path = tmp_path / "surrogate_for_phase5.pt"
        _write_surrogate_export(path)

        model, target_names = load_frozen_surrogate(device="cpu", path=str(path))

        assert target_names == ["v12", "v21", "volfrac_achieved"]
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())

    def test_ensemble_loads_multiple_independent_models(self, tmp_path):
        from pipeline.phase5_cvae.losses import load_frozen_surrogate_ensemble
        p1, p2 = tmp_path / "s1.pt", tmp_path / "s2.pt"
        _write_surrogate_export(p1)
        _write_surrogate_export(p2)

        models, target_names = load_frozen_surrogate_ensemble(
            [str(p1), str(p2)], device="cpu"
        )

        assert len(models) == 2
        assert target_names == ["v12", "v21", "volfrac_achieved"]
        assert models[0] is not models[1]

    def test_load_5_output_f1f2_checkpoint(self, tmp_path):
        """Bug đã sửa 2026-08-15: load_frozen_surrogate() bỏ qua n_outputs
        trong checkpoint, luôn dựng SurrogateCNN(n_outputs=3 mặc định) ->
        load_state_dict() crash size-mismatch trên checkpoint 5-output
        (--include-f1f2, xem model.py SurrogateCNN docstring)."""
        from pipeline.phase5_cvae.losses import load_frozen_surrogate
        path = tmp_path / "surrogate_f1f2_for_phase5.pt"
        _write_surrogate_export(
            path, n_outputs=5,
            target_names=["v12", "v21", "volfrac_achieved", "f1", "f2"],
        )

        model, target_names = load_frozen_surrogate(device="cpu", path=str(path))

        assert target_names == ["v12", "v21", "volfrac_achieved", "f1", "f2"]
        assert model.n_outputs == 5
        out = model(torch.rand(2, 1, 64, 64), torch.zeros(2, 4))
        assert out.shape == (2, 5)

    def test_load_old_checkpoint_without_n_outputs_defaults_to_3(self, tmp_path):
        """Checkpoint export TỪ TRƯỚC khi field n_outputs tồn tại (không có
        key này trong dict) vẫn phải load đúng - tương thích ngược."""
        from pipeline.phase5_cvae.losses import load_frozen_surrogate
        path = tmp_path / "old_surrogate.pt"
        model_old = SurrogateCNN(n_seeds=4, channels=(8, 16), fc_hidden=16)
        torch.save({
            "model_state_dict": model_old.state_dict(),
            "n_seeds": 4, "channels": (8, 16), "fc_hidden": 16,
            "target_names": ["v12", "v21", "volfrac_achieved"],
            # cố ý KHÔNG có "n_outputs" - mô phỏng checkpoint cũ.
        }, path)

        model, _ = load_frozen_surrogate(device="cpu", path=str(path))
        assert model.n_outputs == 3

    def test_load_nu0_checkpoint(self, tmp_path):
        """Bug đã sửa 2026-08-19 (Giai đoạn A, chạy thí nghiệm A6 thật lần
        đầu): load_frozen_surrogate() bỏ qua field include_nu0 trong
        checkpoint, luôn dựng SurrogateCNN(include_nu0=False mặc định) ->
        load_state_dict() crash size-mismatch ở fc.0.weight trên checkpoint
        include_nu0=True (surrogate_a4_nu0.pt, thiếu 1 chiều input nu0)."""
        from pipeline.phase5_cvae.losses import load_frozen_surrogate
        path = tmp_path / "surrogate_nu0_for_phase5.pt"
        model_nu0 = SurrogateCNN(n_seeds=4, channels=(8, 16), fc_hidden=16,
                                  include_nu0=True)
        torch.save({
            "model_state_dict": model_nu0.state_dict(),
            "n_seeds": 4, "channels": (8, 16), "fc_hidden": 16,
            "n_outputs": 3, "include_nu0": True,
            "target_names": ["v12", "v21", "volfrac_achieved"],
        }, path)

        model, target_names = load_frozen_surrogate(device="cpu", path=str(path))

        assert model.include_nu0 is True
        out = model(torch.rand(2, 1, 64, 64), torch.zeros(2, 4), nu0=torch.tensor([0.3, 0.25]))
        assert out.shape == (2, 3)

    def test_load_old_checkpoint_without_include_nu0_defaults_to_false(self, tmp_path):
        """Checkpoint export TỪ TRƯỚC khi field include_nu0 tồn tại vẫn phải
        load đúng - tương thích ngược, cùng pattern với n_outputs ở trên."""
        from pipeline.phase5_cvae.losses import load_frozen_surrogate
        path = tmp_path / "old_surrogate.pt"
        model_old = SurrogateCNN(n_seeds=4, channels=(8, 16), fc_hidden=16)
        torch.save({
            "model_state_dict": model_old.state_dict(),
            "n_seeds": 4, "channels": (8, 16), "fc_hidden": 16,
            "target_names": ["v12", "v21", "volfrac_achieved"],
            # cố ý KHÔNG có "include_nu0" - mô phỏng checkpoint cũ.
        }, path)

        model, _ = load_frozen_surrogate(device="cpu", path=str(path))
        assert model.include_nu0 is False

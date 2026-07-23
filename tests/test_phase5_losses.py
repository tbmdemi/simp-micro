"""
Tests for pipeline/phase5_cvae/losses.py.

losses.py itself only imports torch/torch.nn/torch.nn.functional at module
level (the phase4-surrogate import is done lazily via importlib inside
load_frozen_surrogate(), specifically to dodge the bare-import collision
documented in tests/conftest.py), so a top-level import here is safe.
"""
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
    reconstruction_loss,
    tv_loss,
)
from pipeline.phase4_surrogate.model import SurrogateCNN


def _write_surrogate_export(path, n_seeds=4, channels=(8, 16), fc_hidden=16):
    model = SurrogateCNN(n_seeds=n_seeds, channels=channels, fc_hidden=fc_hidden)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_seeds": n_seeds,
        "channels": channels,
        "fc_hidden": fc_hidden,
        "target_names": ["v12", "v21", "volfrac_achieved"],
    }, path)


class TestKlBetaSchedule:
    def test_zero_warmup_returns_beta_max(self):
        assert kl_beta_schedule(epoch=5, warmup_epochs=0, beta_max=1.0) == 1.0

    def test_linear_ramp(self):
        assert kl_beta_schedule(epoch=5, warmup_epochs=10, beta_max=1.0) == 0.5
        assert kl_beta_schedule(epoch=0, warmup_epochs=10, beta_max=1.0) == 0.0

    def test_clamped_after_warmup(self):
        assert kl_beta_schedule(epoch=100, warmup_epochs=10, beta_max=1.0) == 1.0

    def test_scales_with_beta_max(self):
        assert kl_beta_schedule(epoch=5, warmup_epochs=10, beta_max=2.0) == 1.0


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
        # sum-per-pixel / batch_size (not mean over everything) — batch of 1
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
        class PerfectSurrogate(torch.nn.Module):
            def forward(self, image, seed_vec):
                # always predicts condition passed in via a closure hack below
                return self._target

        surrogate = PerfectSurrogate()
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
        class ConstantSurrogate(torch.nn.Module):
            def forward(self, image, seed_vec):
                return torch.zeros(image.size(0), 3)

        surrogate = ConstantSurrogate()
        recon = torch.rand(2, 1, 8, 8)
        seed_vec = torch.zeros(2, 2)
        condition = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = property_consistency_loss(
            recon, condition, seed_vec, surrogate, ["v12", "v21", "volfrac_achieved"]
        )
        assert loss.item() == pytest_approx(1.0)


def pytest_approx(value):
    import pytest
    return pytest.approx(value, abs=1e-5)


class TestPropertyConsistencyLossEnsemble:
    def test_matches_single_model_when_ensemble_of_one_copy(self):
        class ConstantSurrogate(torch.nn.Module):
            def forward(self, image, seed_vec):
                return torch.full((image.size(0), 3), 0.3)

        s1, s2 = ConstantSurrogate(), ConstantSurrogate()
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
        class LowSurrogate(torch.nn.Module):
            def forward(self, image, seed_vec):
                return torch.zeros(image.size(0), 3)

        class HighSurrogate(torch.nn.Module):
            def forward(self, image, seed_vec):
                return torch.full((image.size(0), 3), 2.0)

        recon = torch.rand(2, 1, 8, 8)
        seed_vec = torch.zeros(2, 2)
        condition = torch.ones(2, 2)  # mean pred = 1.0 -> MSE with target 1.0 is 0
        target_names = ["v12", "v21", "volfrac_achieved"]

        mse_only, disagreement = property_consistency_loss_ensemble(
            recon, condition, seed_vec, [LowSurrogate(), HighSurrogate()],
            target_names, lambda_disagreement=0.0,
        )
        assert torch.isclose(mse_only, torch.tensor(0.0), atol=1e-6)
        assert disagreement.item() > 0.0

        penalized, _ = property_consistency_loss_ensemble(
            recon, condition, seed_vec, [LowSurrogate(), HighSurrogate()],
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


class TestCvaeLoss:
    def test_returns_all_expected_keys(self):
        class DummySurrogate(torch.nn.Module):
            def forward(self, image, seed_vec):
                return torch.zeros(image.size(0), 3)

        recon = torch.rand(2, 1, 8, 8, requires_grad=True)
        image = torch.rand(2, 1, 8, 8)
        mu = torch.zeros(2, 4, requires_grad=True)
        logvar = torch.zeros(2, 4, requires_grad=True)
        condition = torch.zeros(2, 2)
        seed_vec = torch.zeros(2, 2)

        out = cvae_loss(
            recon, image, mu, logvar, condition, seed_vec,
            DummySurrogate(), ["v12", "v21", "volfrac_achieved"],
            beta=0.5, gamma=1.0,
        )
        for key in ("total", "recon", "kl", "prop", "prop_weighted",
                    "tv", "binarization", "periodic", "disagreement", "beta"):
            assert key in out
        assert out["total"].requires_grad  # differentiable wrt recon/mu/logvar

    def test_ensemble_path_when_surrogate_is_a_list(self):
        class DummySurrogate(torch.nn.Module):
            def forward(self, image, seed_vec):
                return torch.zeros(image.size(0), 3)

        recon = torch.rand(2, 1, 8, 8)
        image = torch.rand(2, 1, 8, 8)
        mu = torch.zeros(2, 4)
        logvar = torch.zeros(2, 4)
        condition = torch.zeros(2, 2)
        seed_vec = torch.zeros(2, 2)

        out = cvae_loss(
            recon, image, mu, logvar, condition, seed_vec,
            [DummySurrogate(), DummySurrogate()], ["v12", "v21", "volfrac_achieved"],
            beta=0.5, gamma=1.0, lambda_disagreement=0.1,
        )
        assert torch.isfinite(out["total"])


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

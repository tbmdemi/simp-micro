"""
Tests for pipeline/phase5_cvae/model.py — CVAE / Encoder / Decoder.
"""
import torch

from pipeline.phase5_cvae.model import CVAE


class TestCVAEForward:
    def test_forward_shapes(self):
        model = CVAE(condition_dim=2, latent_dim=16, resolution=64,
                      channels=(8, 16, 32, 64))
        img = torch.rand(3, 1, 64, 64)
        cond = torch.tensor([[-0.5, -0.5]] * 3, dtype=torch.float32)
        recon, mu, logvar = model(img, cond)
        assert recon.shape == (3, 1, 64, 64)
        assert mu.shape == (3, 16)
        assert logvar.shape == (3, 16)

    def test_recon_in_unit_range(self):
        """Decoder ends in Sigmoid — output must stay in [0, 1]."""
        model = CVAE(condition_dim=2, latent_dim=8, resolution=64,
                      channels=(4, 8, 16, 32))
        img = torch.rand(2, 1, 64, 64)
        cond = torch.zeros(2, 2)
        recon, _, _ = model(img, cond)
        assert recon.min().item() >= 0.0
        assert recon.max().item() <= 1.0

    def test_deterministic_uses_mu_not_sample(self):
        model = CVAE(condition_dim=2, latent_dim=8, resolution=64,
                      channels=(4, 8, 16, 32))
        model.eval()
        img = torch.rand(2, 1, 64, 64)
        cond = torch.zeros(2, 2)
        with torch.no_grad():
            recon_a, mu_a, _ = model(img, cond, deterministic=True)
            recon_b, mu_b, _ = model(img, cond, deterministic=True)
        # deterministic=True -> z=mu every call -> identical reconstruction
        assert torch.allclose(recon_a, recon_b)
        assert torch.allclose(mu_a, mu_b)

    def test_stochastic_forward_varies_across_calls(self):
        model = CVAE(condition_dim=2, latent_dim=8, resolution=64,
                      channels=(4, 8, 16, 32))
        img = torch.rand(2, 1, 64, 64)
        cond = torch.zeros(2, 2)
        torch.manual_seed(0)
        recon_a, _, _ = model(img, cond, deterministic=False)
        recon_b, _, _ = model(img, cond, deterministic=False)
        assert not torch.allclose(recon_a, recon_b)


class TestReparameterize:
    def test_zero_logvar_std_one(self):
        mu = torch.zeros(5, 4)
        logvar = torch.zeros(5, 4)
        torch.manual_seed(42)
        z = CVAE.reparameterize(mu, logvar)
        # std=1 (exp(0.5*0)=1), so z should differ from mu (non-degenerate)
        assert z.shape == (5, 4)
        assert not torch.allclose(z, mu)

    def test_large_negative_logvar_collapses_to_mu(self):
        mu = torch.full((3, 4), 2.0)
        logvar = torch.full((3, 4), -30.0)  # std ~ 0
        z = CVAE.reparameterize(mu, logvar)
        assert torch.allclose(z, mu, atol=1e-3)


class TestGenerate:
    def test_generate_shape_single_condition(self):
        model = CVAE(condition_dim=2, latent_dim=8, resolution=64,
                      channels=(4, 8, 16, 32))
        cond = torch.tensor([-0.6, -0.6], dtype=torch.float32)
        out = model.generate(cond, n_samples=5, device="cpu")
        assert out.shape == (5, 1, 64, 64)

    def test_generate_shape_batched_condition(self):
        model = CVAE(condition_dim=2, latent_dim=8, resolution=64,
                      channels=(4, 8, 16, 32))
        cond = torch.tensor([[-0.6, -0.6]], dtype=torch.float32)
        out = model.generate(cond, n_samples=1, device="cpu")
        assert out.shape == (1, 1, 64, 64)

    def test_generate_output_in_unit_range(self):
        model = CVAE(condition_dim=2, latent_dim=8, resolution=64,
                      channels=(4, 8, 16, 32))
        cond = torch.tensor([0.1, 0.2], dtype=torch.float32)
        out = model.generate(cond, n_samples=3, device="cpu")
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

"""
Tests for pipeline/phase5_cvae/model.py - CVAE / Encoder / Decoder.
"""

import torch

from pipeline.phase5_cvae.model import CVAE


class TestCVAEForward:
    def test_forward_shapes(self):
        model = CVAE(
            condition_dim=2,
            latent_dim=16,
            resolution=64,
            channels=(8, 16, 32, 64),
        )
        img = torch.rand(3, 1, 64, 64)
        cond = torch.tensor([[-0.5, -0.5]] * 3, dtype=torch.float32)
        recon, mu, logvar = model(img, cond)
        assert recon.shape == (3, 1, 64, 64)
        assert mu.shape == (3, 16)
        assert logvar.shape == (3, 16)

    def test_recon_in_unit_range(self):
        """Decoder ends in Sigmoid - output must stay in [0, 1]."""
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
        )
        img = torch.rand(2, 1, 64, 64)
        cond = torch.zeros(2, 2)
        recon, _, _ = model(img, cond)
        assert recon.min().item() >= 0.0
        assert recon.max().item() <= 1.0

    def test_deterministic_uses_mu_not_sample(self):
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
        )
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
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
        )
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
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
        )
        cond = torch.tensor([-0.6, -0.6], dtype=torch.float32)
        out = model.generate(cond, n_samples=5, device="cpu")
        assert out.shape == (5, 1, 64, 64)

    def test_generate_shape_batched_condition(self):
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
        )
        cond = torch.tensor([[-0.6, -0.6]], dtype=torch.float32)
        out = model.generate(cond, n_samples=1, device="cpu")
        assert out.shape == (1, 1, 64, 64)

    def test_generate_output_in_unit_range(self):
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
        )
        cond = torch.tensor([0.1, 0.2], dtype=torch.float32)
        out = model.generate(cond, n_samples=3, device="cpu")
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0


class TestKANRegressionHead:
    """Task 1 - KAN-hóa bộ hồi quy Co-VAE: Encoder.fc_mu/fc_logvar và
    Decoder.fc dùng EfficientKANLinear thay vì nn.Linear."""

    def test_fc_layers_are_kan(self):
        from efficient_kan import EfficientKANLinear

        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
        )
        assert isinstance(model.encoder.fc_mu, EfficientKANLinear)
        assert isinstance(model.encoder.fc_logvar, EfficientKANLinear)
        assert isinstance(model.decoder.fc, EfficientKANLinear)

    def test_spline_weight_is_3d(self):
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
        )
        # spline_weight [out, in, grid_size + spline_order] = [*, *, 8]
        assert model.decoder.fc.spline_weight.dim() == 3
        assert model.decoder.fc.spline_weight.shape[2] == 8

    def test_decoder_fc_gradients_flow(self):
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
        )
        z = torch.randn(2, 8)
        cond = torch.zeros(2, 2)
        out = model.decoder(z, cond)
        assert out.shape == (2, 1, 64, 64)
        out.sum().backward()
        assert model.decoder.fc.base_weight.grad is not None
        assert model.decoder.fc.spline_weight.grad is not None
        assert torch.isfinite(model.decoder.fc.spline_weight.grad).all()


class TestWireDecoder:
    """Task 2 - WIRE INR Decoder: decoder_type='wire' dùng
    WireContinuousDecoder, resolution-agnostic, API (B,1,H,W) giữ nguyên."""

    def test_forward_shapes_unit_range(self):
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
            decoder_type="wire",
        )
        img = torch.rand(2, 1, 64, 64)
        cond = torch.zeros(2, 2)
        recon, mu, logvar = model(img, cond)
        assert recon.shape == (2, 1, 64, 64)
        assert recon.min().item() >= 0.0
        assert recon.max().item() <= 1.0

    def test_generate_resolution_agnostic(self):
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
            decoder_type="wire",
        )
        cond = torch.tensor([-0.6, -0.6], dtype=torch.float32)
        out = model.generate(cond, n_samples=1, device="cpu", resolution=128)
        assert out.shape == (1, 1, 128, 128)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_forward_coords_continuous_api(self):
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
            decoder_type="wire",
        )
        coords = torch.rand(3, 100, 2) * 2 - 1  # tọa độ liên tục tùy ý
        z = torch.randn(3, 8)
        cond = torch.zeros(3, 2)
        rho = model.decoder.forward_coords(coords, z, cond)
        assert rho.shape == (3, 100, 1)
        assert rho.min().item() >= 0.0
        assert rho.max().item() <= 1.0

    def test_conv_is_default_backward_compat(self):
        from pipeline.phase5_cvae.model import Decoder

        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
        )
        assert model.decoder_type == "conv"
        assert isinstance(model.decoder, Decoder)

    def test_gradients_flow(self):
        model = CVAE(
            condition_dim=2,
            latent_dim=8,
            resolution=64,
            channels=(4, 8, 16, 32),
            decoder_type="wire",
        )
        img = torch.rand(2, 1, 64, 64)
        cond = torch.zeros(2, 2)
        recon, _, _ = model(img, cond)
        recon.sum().backward()
        assert model.decoder.layer1.linear.weight.grad is not None
        assert model.decoder.output_layer.weight.grad is not None
        assert torch.isfinite(model.decoder.layer1.linear.weight.grad).all()

"""
Tests for pipeline/phase4_surrogate/model.py — SurrogateCNN.
"""
import torch

from pipeline.phase4_surrogate.model import SurrogateCNN


class TestSurrogateCNN:
    def test_forward_output_shape(self):
        model = SurrogateCNN(n_seeds=11)
        img = torch.randn(4, 1, 64, 64)
        seed_vec = torch.zeros(4, 11)
        seed_vec[:, 0] = 1.0
        out = model(img, seed_vec)
        assert out.shape == (4, 3)

    def test_forward_batch_size_one(self):
        model = SurrogateCNN(n_seeds=5)
        img = torch.randn(1, 1, 64, 64)
        seed_vec = torch.zeros(1, 5)
        out = model(img, seed_vec)
        assert out.shape == (1, 3)

    def test_custom_channels_and_fc_hidden(self):
        model = SurrogateCNN(n_seeds=3, channels=(8, 16), fc_hidden=32)
        img = torch.randn(2, 1, 64, 64)
        seed_vec = torch.zeros(2, 3)
        out = model(img, seed_vec)
        assert out.shape == (2, 3)

    def test_n_seeds_affects_param_count(self):
        model_a = SurrogateCNN(n_seeds=2)
        model_b = SurrogateCNN(n_seeds=20)
        n_params_a = sum(p.numel() for p in model_a.parameters())
        n_params_b = sum(p.numel() for p in model_b.parameters())
        # Only the first FC layer's input width changes with n_seeds.
        assert n_params_b > n_params_a

    def test_output_not_nan(self):
        model = SurrogateCNN(n_seeds=4)
        model.eval()
        img = torch.rand(6, 1, 64, 64)
        seed_vec = torch.zeros(6, 4)
        seed_vec[:, 1] = 1.0
        with torch.no_grad():
            out = model(img, seed_vec)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradients_flow(self):
        """Backward pass should populate gradients on all parameters —
        guards against an accidental detach/no_grad creeping into forward()."""
        model = SurrogateCNN(n_seeds=3)
        img = torch.randn(2, 1, 64, 64)
        seed_vec = torch.zeros(2, 3)
        seed_vec[:, 0] = 1.0
        out = model(img, seed_vec)
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"

"""
Tests for pipeline/phase4_surrogate/model.py - SurrogateCNN.
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
        """Backward pass should populate gradients on all parameters -
        guards against an accidental detach/no_grad creeping into forward()."""
        model = SurrogateCNN(n_seeds=3)
        img = torch.randn(2, 1, 64, 64)
        seed_vec = torch.zeros(2, 3)
        seed_vec[:, 0] = 1.0
        out = model(img, seed_vec)
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"


class TestSurrogateCNNNu0:
    """include_nu0=True (Giai đoạn A/A4, docs/archive/PROJECT_PLAN.md Nhóm 1) -
    nu0
    concat như 1 input phụ sau GAP, cùng seed one-hot."""

    def test_default_forward_unaffected(self):
        """include_nu0=False (mặc định) - forward(image, seed_vec) không đổi,
        tương thích ngược với mọi checkpoint hiện có."""
        model = SurrogateCNN(n_seeds=4)
        img = torch.randn(3, 1, 64, 64)
        seed_vec = torch.zeros(3, 4)
        seed_vec[:, 0] = 1.0
        out = model(img, seed_vec)
        assert out.shape == (3, 3)

    def test_include_nu0_forward_shape(self):
        model = SurrogateCNN(n_seeds=4, include_nu0=True)
        img = torch.randn(3, 1, 64, 64)
        seed_vec = torch.zeros(3, 4)
        seed_vec[:, 0] = 1.0
        nu0 = torch.tensor([0.2, 0.3, 0.4])
        out = model(img, seed_vec, nu0=nu0)
        assert out.shape == (3, 3)
        assert not torch.isnan(out).any()

    def test_include_nu0_without_nu0_arg_raises(self):
        model = SurrogateCNN(n_seeds=4, include_nu0=True)
        img = torch.randn(2, 1, 64, 64)
        seed_vec = torch.zeros(2, 4)
        import pytest
        with pytest.raises(ValueError):
            model(img, seed_vec)

    def test_include_nu0_adds_one_input_param(self):
        """fc_in phải tăng đúng 1 chiều khi include_nu0=True - so tham số FC
        đầu tiên giữa 2 model cùng n_seeds/channels/fc_hidden."""
        model_no_nu0 = SurrogateCNN(n_seeds=4, channels=(8, 16), fc_hidden=32)
        model_with_nu0 = SurrogateCNN(n_seeds=4, channels=(8, 16), fc_hidden=32, include_nu0=True)
        fc_in_no_nu0 = model_no_nu0.fc[0].in_features
        fc_in_with_nu0 = model_with_nu0.fc[0].in_features
        assert fc_in_with_nu0 == fc_in_no_nu0 + 1

    def test_include_nu0_gradients_flow(self):
        model = SurrogateCNN(n_seeds=3, include_nu0=True)
        img = torch.randn(2, 1, 64, 64)
        seed_vec = torch.zeros(2, 3)
        seed_vec[:, 0] = 1.0
        nu0 = torch.tensor([0.25, 0.35], requires_grad=False)
        out = model(img, seed_vec, nu0=nu0)
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"


class TestSurrogateCNNKAN:
    """ConvKAN head preserves the surrogate contract and old default path."""

    def test_kan_forward_and_gradients(self):
        model = SurrogateCNN(n_seeds=3, channels=(8, 16), fc_hidden=12, use_kan=True)
        image = torch.randn(2, 1, 64, 64)
        seed_vec = torch.zeros(2, 3)
        seed_vec[:, 0] = 1.0

        output = model(image, seed_vec)
        assert output.shape == (2, 3)
        output.sum().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())

    def test_kan_head_has_expected_spline_parameters(self):
        linear = SurrogateCNN(n_seeds=11, channels=(8, 16), fc_hidden=32)
        kan = SurrogateCNN(n_seeds=11, channels=(8, 16), fc_hidden=32, use_kan=True)
        linear_fc_params = sum(parameter.numel() for parameter in linear.fc.parameters())
        kan_fc_params = sum(parameter.numel() for parameter in kan.fc.parameters())
        # EfficientKANLinear carries base and spline weights, so equal-width
        # KAN heads intentionally have more parameters than Linear heads.
        assert kan_fc_params > linear_fc_params

    def test_default_checkpoint_architecture_remains_linear(self):
        model = SurrogateCNN(n_seeds=3, channels=(8, 16), fc_hidden=12)
        assert model.use_kan is False
        assert isinstance(model.fc[0], torch.nn.Linear)

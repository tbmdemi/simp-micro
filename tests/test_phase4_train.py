"""
Tests for pipeline/phase4_surrogate/train.py — weighted_mse, run_epoch.

Imports done lazily inside each test — see tests/conftest.py docstring.
"""
import torch
from torch.utils.data import DataLoader

from pipeline.phase4_surrogate.dataset import AuxeticDataset
from pipeline.phase4_surrogate.model import SurrogateCNN


class TestWeightedMse:
    def test_basic_weighting(self):
        from pipeline.phase4_surrogate.train import weighted_mse
        pred = torch.zeros(4, 3)
        target = torch.ones(4, 3)
        weights = torch.tensor([1.0, 1.0, 0.0])
        loss, per_target = weighted_mse(pred, target, weights)
        # per-target MSE is 1.0 for each column (pred=0, target=1)
        assert torch.allclose(per_target, torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)
        # weighted sum: 1*1 + 1*1 + 0*1 = 2.0
        assert torch.isclose(loss, torch.tensor(2.0), atol=1e-6)

    def test_clamps_extreme_values_to_avoid_nan(self):
        from pipeline.phase4_surrogate.train import weighted_mse
        pred = torch.tensor([[1e30, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 0.0]])
        weights = torch.tensor([1.0, 1.0, 1.0])
        loss, per_target = weighted_mse(pred, target, weights)
        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_nan_input_is_sanitized(self):
        from pipeline.phase4_surrogate.train import weighted_mse
        pred = torch.tensor([[float("nan"), 1.0, 1.0]])
        target = torch.tensor([[0.0, 1.0, 1.0]])
        weights = torch.tensor([1.0, 1.0, 1.0])
        loss, _ = weighted_mse(pred, target, weights)
        assert not torch.isnan(loss)


class TestRunEpoch:
    def _tiny_loader(self, make_phase3_npz, n_samples=8, batch_size=4):
        path = make_phase3_npz("train.npz", n_samples=n_samples)
        ds = AuxeticDataset(path)
        return ds, DataLoader(ds, batch_size=batch_size, shuffle=False)

    def test_train_mode_updates_weights(self, make_phase3_npz):
        from pipeline.phase4_surrogate.train import run_epoch
        ds, loader = self._tiny_loader(make_phase3_npz)
        model = SurrogateCNN(n_seeds=ds.n_seeds, channels=(4, 8), fc_hidden=8)
        before = [p.clone() for p in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        loss, per_target = run_epoch(model, loader, optimizer, "cpu", train=True)

        assert loss == loss  # not NaN
        assert per_target.shape == (3,)
        after = list(model.parameters())
        changed = any(
            not torch.allclose(b, a) for b, a in zip(before, after)
        )
        assert changed, "run_epoch(train=True) should update model parameters"

    def test_eval_mode_does_not_update_weights(self, make_phase3_npz):
        from pipeline.phase4_surrogate.train import run_epoch
        ds, loader = self._tiny_loader(make_phase3_npz)
        model = SurrogateCNN(n_seeds=ds.n_seeds, channels=(4, 8), fc_hidden=8)
        before = [p.clone() for p in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        run_epoch(model, loader, optimizer, "cpu", train=False)

        after = list(model.parameters())
        assert all(torch.allclose(b, a) for b, a in zip(before, after))

    def test_returns_finite_loss(self, make_phase3_npz):
        from pipeline.phase4_surrogate.train import run_epoch
        ds, loader = self._tiny_loader(make_phase3_npz, n_samples=6, batch_size=3)
        model = SurrogateCNN(n_seeds=ds.n_seeds, channels=(4, 8), fc_hidden=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss, _ = run_epoch(model, loader, optimizer, "cpu", train=True)
        assert torch.isfinite(torch.tensor(loss))

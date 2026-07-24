"""
Tests for pipeline/phase4_surrogate/bootstrap_ci.py — adds a confidence
interval on top of evaluate.py's point-estimate R2 (test.npz has 1,184
samples, so we expect a much tighter CI than Phase 5's best_of_n_eval.py,
which only has 19-24 conditions; see pipeline/phase5_cvae/bootstrap_ci.py).

Import is done lazily inside each test (not at module top) because
bootstrap_ci.py does `sys.path.insert(...)` + bare `from dataset import X`
/ `from model import X` at import time — same phase4/phase5 bare-import
landmine as evaluate.py (see tests/test_phase4_evaluate.py docstring and
tests/conftest.py's `_isolate_pipeline_bare_imports`).
"""
import numpy as np
import pytest
import torch


def _write_fake_checkpoint(path, n_seeds=4, channels=(4, 8), fc_hidden=8):
    from pipeline.phase4_surrogate.model import SurrogateCNN
    model = SurrogateCNN(n_seeds=n_seeds, channels=channels, fc_hidden=fc_hidden)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_seeds": n_seeds,
        "seed_classes": ["a", "b", "c", "d"],
        "channels": channels,
        "fc_hidden": fc_hidden,
        "target_names": ["v12", "v21", "volfrac_achieved"],
    }, path)


def _write_fake_test_npz(path, n_samples=40, resolution=16, n_seeds=4, seed=0):
    rng = np.random.default_rng(seed)
    images = rng.random((n_samples, resolution, resolution)).astype(np.float32)
    seed_idx = rng.integers(0, n_seeds, size=n_samples)
    seed_onehot = np.eye(n_seeds, dtype=np.float32)[seed_idx]
    np.savez(
        path,
        images=images,
        v12=rng.uniform(-0.8, 0.2, size=n_samples).astype(np.float32),
        v21=rng.uniform(-0.8, 0.2, size=n_samples).astype(np.float32),
        volfrac_achieved=rng.uniform(0.3, 0.7, size=n_samples).astype(np.float32),
        seed_onehot=seed_onehot,
        seed_classes=np.array(["a", "b", "c", "d"]),
    )


class TestR2Score:
    def test_perfect_prediction_gives_r2_one(self):
        from pipeline.phase4_surrogate.bootstrap_ci import r2_score
        y = np.array([-0.5, -0.3, -0.1, 0.2])
        assert r2_score(y, y.copy()) == pytest.approx(1.0)

    def test_constant_target_returns_nan_not_a_huge_number(self):
        """Same float-rounding pitfall as pipeline/phase5_cvae/bootstrap_ci.py:
        a naive `ss_tot == 0` check lets near-zero-but-nonzero variance
        through, producing a bogus enormous R2 instead of NaN."""
        from pipeline.phase4_surrogate.bootstrap_ci import r2_score
        y_true = np.array([-0.4, -0.4, -0.4])
        y_pred = np.array([-0.3, -0.5, -0.4])
        assert np.isnan(r2_score(y_true, y_pred))


class TestBootstrapR2:
    def test_point_estimate_matches_r2_score(self):
        from pipeline.phase4_surrogate.bootstrap_ci import r2_score, bootstrap_r2
        rng = np.random.default_rng(1)
        y_true = rng.uniform(-0.8, 0.2, size=50)
        y_pred = y_true + rng.normal(0, 0.05, size=50)
        stats = bootstrap_r2(y_true, y_pred, n_boot=500, seed=0)
        assert stats["r2_point_estimate"] == pytest.approx(r2_score(y_true, y_pred))
        assert stats["n_samples"] == 50

    def test_larger_n_gives_narrower_ci_than_smaller_n(self):
        """Core motivation for this script: Phase 4's test set (1,184
        samples) should give a much tighter CI than Phase 5's 19-24
        conditions — verify the mechanism actually behaves that way."""
        from pipeline.phase4_surrogate.bootstrap_ci import bootstrap_r2
        rng = np.random.default_rng(2)

        y_true_small = rng.uniform(-0.8, 0.2, size=20)
        y_pred_small = y_true_small + rng.normal(0, 0.1, size=20)
        stats_small = bootstrap_r2(y_true_small, y_pred_small, n_boot=3000, seed=1)

        y_true_large = rng.uniform(-0.8, 0.2, size=1000)
        y_pred_large = y_true_large + rng.normal(0, 0.1, size=1000)
        stats_large = bootstrap_r2(y_true_large, y_pred_large, n_boot=3000, seed=1)

        width_small = stats_small["r2_ci95_hi"] - stats_small["r2_ci95_lo"]
        width_large = stats_large["r2_ci95_hi"] - stats_large["r2_ci95_lo"]
        assert width_large < width_small

    def test_same_seed_is_reproducible(self):
        from pipeline.phase4_surrogate.bootstrap_ci import bootstrap_r2
        y_true = np.array([-0.5, -0.3, -0.1, 0.2, -0.6, 0.1])
        y_pred = np.array([-0.4, -0.35, -0.05, 0.25, -0.5, 0.05])
        a = bootstrap_r2(y_true, y_pred, n_boot=300, seed=42)
        b = bootstrap_r2(y_true, y_pred, n_boot=300, seed=42)
        assert a["r2_ci95_lo"] == b["r2_ci95_lo"]
        assert a["r2_ci95_hi"] == b["r2_ci95_hi"]


class TestGetTestPredictions:
    def test_shapes_match_dataset_and_target_count(self, tmp_path):
        from pipeline.phase4_surrogate.bootstrap_ci import get_test_predictions
        ckpt_path = tmp_path / "surrogate_best.pt"
        npz_path = tmp_path / "test.npz"
        _write_fake_checkpoint(ckpt_path)
        _write_fake_test_npz(npz_path, n_samples=40, resolution=16)

        preds, targets = get_test_predictions(str(ckpt_path), str(npz_path), device="cpu")

        assert preds.shape == (40, 3)
        assert targets.shape == (40, 3)

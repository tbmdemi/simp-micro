"""
Tests for pipeline/phase5_cvae/verify_fe.py — resize_to_fe_grid,
evaluate_density_field.

Imports are lazy inside each test — verify_fe.py does `sys.path.insert(...)`
+ bare `from model import CVAE` / `from dataset import CVAEDataset` at
import time (see tests/conftest.py docstring).
"""
import numpy as np


class TestResizeToFeGrid:
    def test_same_size_is_near_identity(self):
        from pipeline.phase5_cvae.verify_fe import resize_to_fe_grid
        img = np.zeros((8, 8), dtype=np.float32)
        img[:4, :] = 1.0
        out = resize_to_fe_grid(img, nely=8, nelx=8)
        assert out.shape == (8, 8)
        assert np.allclose(out[:4, :], 1.0)
        assert np.allclose(out[4:, :], 0.0)

    def test_downsample_shape(self):
        from pipeline.phase5_cvae.verify_fe import resize_to_fe_grid
        img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
        out = resize_to_fe_grid(img, nely=16, nelx=16)
        assert out.shape == (16, 16)

    def test_upsample_shape_and_range(self):
        from pipeline.phase5_cvae.verify_fe import resize_to_fe_grid
        img = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        out = resize_to_fe_grid(img, nely=32, nelx=32)
        assert out.shape == (32, 32)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_nearest_neighbor_preserves_binary_values(self):
        """Binarized images resized via NEAREST must stay exactly {0,1} —
        any blurring here would corrupt the FE input silently."""
        from pipeline.phase5_cvae.verify_fe import resize_to_fe_grid
        rng = np.random.default_rng(1)
        img_bin = (rng.random((16, 16)) > 0.5).astype(np.float32)
        out = resize_to_fe_grid(img_bin, nely=10, nelx=12)
        assert set(np.unique(out).tolist()) <= {0.0, 1.0}


class TestEvaluateDensityField:
    def test_uniform_density_gives_symmetric_finite_tensor(self):
        """Smoke test on a tiny FE grid (cheap): uniform density field
        should solve without error and produce a symmetric, finite Q."""
        from pipeline.phase5_cvae.verify_fe import evaluate_density_field
        fe_params = {
            "nelx": 6, "nely": 6, "E0": 199.0, "Emin": 1e-9,
            "nu": 0.3, "penal": 3.0, "rho0": 1.0,
        }
        xPhys = np.full((6, 6), 0.5, dtype=np.float64)
        v12, v21, Q = evaluate_density_field(xPhys, fe_params)

        assert np.isfinite(v12)
        assert np.isfinite(v21)
        assert Q.shape == (3, 3)
        assert np.allclose(Q, Q.T, atol=1e-8)

    def test_penal_changes_result(self):
        """`penal` (SIMP penalization exponent) must actually affect the
        homogenized tensor — a stale/ignored fe_params['penal'] would make
        every self-play/best-of-N call silently equivalent regardless of
        the penal value passed in."""
        from pipeline.phase5_cvae.verify_fe import evaluate_density_field
        base_params = {
            "nelx": 6, "nely": 6, "E0": 199.0, "Emin": 1e-9,
            "nu": 0.3, "rho0": 1.0,
        }
        rng = np.random.default_rng(2)
        xPhys = rng.uniform(0.2, 0.8, size=(6, 6))

        _, _, Q_low = evaluate_density_field(xPhys, dict(base_params, penal=1.0))
        _, _, Q_high = evaluate_density_field(xPhys, dict(base_params, penal=5.0))

        assert not np.allclose(Q_low, Q_high)

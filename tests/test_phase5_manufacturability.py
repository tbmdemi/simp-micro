"""Tests for pipeline/phase5_cvae/manufacturability.py (roadmap 6.2 connectivity/
min-feature-size + 6.3 periodicity checks)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                 "pipeline", "phase5_cvae"))


def _import_manufacturability():
    import manufacturability
    return manufacturability


class TestConnectivity:
    def test_single_solid_block_is_connected(self):
        m = _import_manufacturability()
        img = np.zeros((20, 20))
        img[5:15, 5:15] = 1.0
        result = m.check_connectivity(img)
        assert result["n_components"] == 1
        assert result["is_connected"] is True

    def test_two_disjoint_islands_not_connected(self):
        m = _import_manufacturability()
        img = np.zeros((20, 20))
        img[1:4, 1:4] = 1.0
        img[15:18, 15:18] = 1.0
        result = m.check_connectivity(img)
        assert result["n_components"] == 2
        assert result["is_connected"] is False
        assert result["manufacturable"] is False

    def test_diagonal_touch_counts_as_connected_8conn(self):
        m = _import_manufacturability()
        img = np.zeros((10, 10))
        img[3, 3] = 1.0
        img[4, 4] = 1.0  # touches only at corner
        result = m.check_connectivity(img)
        assert result["n_components"] == 1
        assert result["is_connected"] is True

    def test_empty_image_zero_components(self):
        m = _import_manufacturability()
        img = np.zeros((10, 10))
        result = m.check_connectivity(img)
        assert result["n_components"] == 0
        assert result["is_connected"] is True

    def test_thin_single_pixel_strut_fails_min_feature(self):
        m = _import_manufacturability()
        img = np.zeros((20, 20))
        img[10, :] = 1.0  # 1px-wide strut
        result = m.check_connectivity(img, min_feature_px=4)
        assert result["min_feature_ok"] is False
        assert result["manufacturable"] is False

    def test_thick_block_passes_min_feature(self):
        m = _import_manufacturability()
        img = np.zeros((20, 20))
        img[5:15, 5:15] = 1.0  # 10px-thick block
        result = m.check_connectivity(img, min_feature_px=4)
        assert result["min_feature_ok"] is True
        assert result["manufacturable"] is True


class TestPeriodicity:
    def test_matching_edges_is_periodic_ok(self):
        m = _import_manufacturability()
        img = np.zeros((20, 20))
        img[:, 0] = 1.0
        img[:, -1] = 1.0  # left/right columns identical (both solid)
        result = m.check_periodicity(img)
        assert result["edge_mismatch_lr"] == 0.0
        assert result["periodic_ok"] is True

    def test_mismatched_edges_fails_periodicity(self):
        m = _import_manufacturability()
        img = np.zeros((20, 20))
        img[:, 0] = 1.0  # left column solid, right column void
        result = m.check_periodicity(img, tol=0.05)
        assert result["edge_mismatch_lr"] == 1.0
        assert result["periodic_ok"] is False

    def test_tolerance_allows_small_mismatch(self):
        m = _import_manufacturability()
        img = np.zeros((20, 20))
        img[:, 0] = 1.0
        img[:, -1] = 1.0
        img[0, -1] = 0.0  # 1/20 = 5% mismatch on right edge vs left
        result = m.check_periodicity(img, tol=0.1)
        assert result["periodic_ok"] is True
        result_strict = m.check_periodicity(img, tol=0.0)
        assert result_strict["periodic_ok"] is False


class TestForcePeriodic:
    """force_periodic() (thêm từ nhánh research/auxetic-breakthrough, xem
    EXPERIMENT_LOG.md mục "Phase 6") - ép cứng left==right, top==bottom
    bằng 1 phép gán, đảm bảo periodic_ok=True TUYỆT ĐỐI, không phải thứ có
    thể "gần đúng"."""

    def test_edges_match_exactly_after_forcing(self):
        m = _import_manufacturability()
        rng = np.random.default_rng(0)
        img = rng.random((64, 64)).astype(np.float32)
        out = m.force_periodic(img)
        assert np.array_equal(out[:, 0], out[:, -1])
        assert np.array_equal(out[0, :], out[-1, :])

    def test_shape_and_dtype_preserved(self):
        m = _import_manufacturability()
        img = np.random.default_rng(1).random((64, 64)).astype(np.float32)
        out = m.force_periodic(img)
        assert out.shape == img.shape
        assert out.dtype == img.dtype

    def test_interior_untouched(self):
        m = _import_manufacturability()
        img = np.random.default_rng(2).random((64, 64)).astype(np.float32)
        out = m.force_periodic(img)
        assert np.array_equal(out[1:-1, 1:-1], img[1:-1, 1:-1])

    def test_passes_periodicity_check(self):
        m = _import_manufacturability()
        rng = np.random.default_rng(3)
        img = rng.random((64, 64)).astype(np.float32)
        out = m.force_periodic(img)
        result = m.check_periodicity((out > 0.5).astype(np.float32))
        assert result["periodic_ok"] is True
        assert result["edge_mismatch_lr"] == 0.0
        assert result["edge_mismatch_tb"] == 0.0

    def test_does_not_mutate_input(self):
        m = _import_manufacturability()
        img = np.random.default_rng(4).random((64, 64)).astype(np.float32)
        img_copy = img.copy()
        m.force_periodic(img)
        assert np.array_equal(img, img_copy)


class TestCombined:
    def test_check_manufacturability_combines_both(self):
        m = _import_manufacturability()
        img = np.zeros((20, 20))
        img[5:15, 5:15] = 1.0
        result = m.check_manufacturability(img)
        assert "n_components" in result
        assert "periodic_ok" in result
        assert "passes_all" in result

    def test_passes_all_false_when_either_check_fails(self):
        m = _import_manufacturability()
        img = np.zeros((20, 20))
        img[1:4, 1:4] = 1.0
        img[15:18, 15:18] = 1.0  # two islands -> connectivity fails
        result = m.check_manufacturability(img)
        assert result["passes_all"] is False

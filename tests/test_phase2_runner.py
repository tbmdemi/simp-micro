"""
Tests for pipeline/phase2_multi_batch/runner.py — manufacturability
instrumentation (roadmap 6.2/6.3) added at generation time, xem
EXPERIMENT_LOG.md mục "Phase 2 — Manufacturability".

evaluate_single()'s actual physics call (run_simp) is monkeypatched — this
suite covers the new manufacturability wiring, not the SIMP solve itself
(still untested I/O-heavy code, see README Known Limitations #6).
"""
import os
import numpy as np
import pytest
from PIL import Image

from pipeline.phase2_multi_batch.runner import (
    _compute_manufacturability_from_saved_png,
    evaluate_single,
)


def _write_iteration_png(sample_dir, n_iters, img_64x64):
    os.makedirs(sample_dir, exist_ok=True)
    arr = (img_64x64 * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(
        os.path.join(sample_dir, f"iteration_{n_iters:05d}.png")
    )


class TestComputeManufacturabilityFromSavedPng:
    def test_solid_connected_periodic_block_passes_all(self, tmp_path):
        img = np.ones((64, 64), dtype=np.float32)  # toàn vật liệu, rõ ràng liên thông + tuần hoàn
        _write_iteration_png(tmp_path, 150, img)

        result = _compute_manufacturability_from_saved_png(str(tmp_path), 150)

        assert result is not None
        assert result["is_connected"] is True
        assert result["periodic_ok"] is True
        assert result["passes_all"] is True

    def test_disconnected_islands_fails_connectivity(self, tmp_path):
        img = np.zeros((64, 64), dtype=np.float32)
        img[5:15, 5:15] = 1.0
        img[45:55, 45:55] = 1.0  # 2 mảnh rời nhau
        _write_iteration_png(tmp_path, 150, img)

        result = _compute_manufacturability_from_saved_png(str(tmp_path), 150)

        assert result is not None
        assert result["is_connected"] is False
        assert result["passes_all"] is False

    def test_missing_png_returns_none(self, tmp_path):
        result = _compute_manufacturability_from_saved_png(str(tmp_path), 999)
        assert result is None


class TestEvaluateSinglePopulatesManufacturability:
    def _fake_run_simp_factory(self, img_64x64, n_iters=150):
        """Trả về 1 hàm giả lập run_simp(): ghi PNG (mô phỏng đúng hành vi
        save_density_image() thật) rồi trả về output dict tối thiểu mà
        evaluate_single() cần."""
        def _fake_run_simp(params):
            _write_iteration_png(params["output_dir"], n_iters, img_64x64)
            return {
                "v12": -0.5, "v21": -0.5, "objective": 0.1,
                "n_iters": n_iters, "converged": True,
            }
        return _fake_run_simp

    def test_manufacturable_image_sets_passes_all_true(self, tmp_path, monkeypatch):
        from pipeline.phase2_multi_batch import runner as runner_mod
        img = np.ones((64, 64), dtype=np.float32)
        monkeypatch.setattr(runner_mod, "run_simp", self._fake_run_simp_factory(img))

        params = {"output_dir": str(tmp_path / "sample_0000"), "seed": "circle", "objective": "auxetic"}
        result = evaluate_single((params, 0))

        assert result["success"] is True
        assert result["passes_all"] is True
        assert result["is_connected"] is True
        assert result["periodic_ok"] is True

    def test_non_manufacturable_image_sets_passes_all_false(self, tmp_path, monkeypatch):
        from pipeline.phase2_multi_batch import runner as runner_mod
        img = np.zeros((64, 64), dtype=np.float32)
        img[5:15, 5:15] = 1.0
        img[45:55, 45:55] = 1.0
        monkeypatch.setattr(runner_mod, "run_simp", self._fake_run_simp_factory(img))

        params = {"output_dir": str(tmp_path / "sample_0001"), "seed": "circle", "objective": "auxetic"}
        result = evaluate_single((params, 1))

        assert result["success"] is True
        assert result["passes_all"] is False
        assert result["is_connected"] is False

    def test_run_simp_exception_leaves_manufacturability_none(self, tmp_path, monkeypatch):
        from pipeline.phase2_multi_batch import runner as runner_mod

        def _raising_run_simp(params):
            raise RuntimeError("forced FE failure")

        monkeypatch.setattr(runner_mod, "run_simp", _raising_run_simp)

        params = {"output_dir": str(tmp_path / "sample_0002"), "seed": "circle", "objective": "auxetic"}
        result = evaluate_single((params, 2))

        assert result["success"] is False
        assert result["passes_all"] is None
        assert result["is_connected"] is None

    def test_missing_saved_png_leaves_manufacturability_none_but_still_success(
        self, tmp_path, monkeypatch,
    ):
        """run_simp succeeding but NOT writing the expected PNG (shouldn't
        happen in practice, but evaluate_single must not crash) — success
        stays True (FE itself succeeded), manufacturability fields stay None."""
        from pipeline.phase2_multi_batch import runner as runner_mod

        def _fake_run_simp_no_png(params):
            return {"v12": -0.5, "v21": -0.5, "objective": 0.1,
                    "n_iters": 150, "converged": True}

        monkeypatch.setattr(runner_mod, "run_simp", _fake_run_simp_no_png)

        params = {"output_dir": str(tmp_path / "sample_0003"), "seed": "circle", "objective": "auxetic"}
        result = evaluate_single((params, 3))

        assert result["success"] is True
        assert result["passes_all"] is None

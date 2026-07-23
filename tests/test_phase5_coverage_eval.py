"""
Tests for pipeline/phase5_cvae/coverage_eval.py (roadmap 7.4: property-space
coverage map, dead-zone detection across a v12 target grid instead of
best_of_n_eval.py's 24 random test.npz conditions).

Imports are lazy inside each test, same reasoning as
tests/test_phase5_best_of_n_eval.py (bare-import collision, see
tests/conftest.py).
"""
import sys

import numpy as np
import pytest
import torch


def _write_cvae_checkpoint(path, latent_dim=4, resolution=64,
                            channels=(4, 8, 16, 32)):
    from pipeline.phase5_cvae.model import CVAE
    model = CVAE(condition_dim=2, latent_dim=latent_dim,
                 resolution=resolution, channels=channels)
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": latent_dim,
        "condition_dim": 2,
        "resolution": resolution,
        "channels": channels,
    }, path)


class TestCoverageEvalGridAndHitLogic:
    def test_grid_spans_requested_range(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import coverage_eval as cov_mod

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe_params = dict(cov_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(cov_mod, "FE_PARAMS", tiny_fe_params)

        result = cov_mod.coverage_eval(
            str(ckpt_path), n_samples=2, grid_size=5, device="cpu", seed=1,
            v12_range=(-0.8, 0.4),
        )
        targets = [t["target_v12"] for t in result["per_target"]]
        assert len(targets) == 5
        assert targets[0] == pytest.approx(-0.8)
        assert targets[-1] == pytest.approx(0.4)

    def test_dead_zone_detected_when_best_of_n_always_wrong_sign(
        self, tmp_path, monkeypatch,
    ):
        """Force every real-FE evaluation to return a POSITIVE value - every
        auxetic (v12<0) grid target should then be flagged as a dead zone
        (hit=False), and none should be marked hit."""
        from pipeline.phase5_cvae import coverage_eval as cov_mod

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe_params = dict(cov_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(cov_mod, "FE_PARAMS", tiny_fe_params)

        def always_positive(img_fe, fe_params):
            return 0.5, 0.5, np.eye(3)

        monkeypatch.setattr(cov_mod, "evaluate_density_field", always_positive)

        result = cov_mod.coverage_eval(
            str(ckpt_path), n_samples=2, grid_size=4, device="cpu", seed=1,
            v12_range=(-0.6, -0.1),  # all-auxetic grid
        )

        assert result["hit_rate"] == 0.0
        assert len(result["dead_zone_targets"]) == result["n_auxetic_targets"]

    def test_no_dead_zone_when_fe_matches_target_sign(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import coverage_eval as cov_mod

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe_params = dict(cov_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(cov_mod, "FE_PARAMS", tiny_fe_params)

        # returns a value with the SAME sign as whatever the target is by
        # closing over the target via a mutable box that best_of_n-style
        # tests can't easily do here — simplest deterministic stand-in:
        # always return a small negative number, which is a hit for every
        # auxetic target in a grid restricted to v12 < 0.
        def always_negative(img_fe, fe_params):
            return -0.05, -0.05, np.eye(3)

        monkeypatch.setattr(cov_mod, "evaluate_density_field", always_negative)

        result = cov_mod.coverage_eval(
            str(ckpt_path), n_samples=2, grid_size=4, device="cpu", seed=1,
            v12_range=(-0.6, -0.1),
        )

        assert result["hit_rate"] == 1.0
        assert result["dead_zone_targets"] == []

    def test_non_auxetic_targets_excluded_from_hit_rate(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import coverage_eval as cov_mod

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe_params = dict(cov_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(cov_mod, "FE_PARAMS", tiny_fe_params)

        def always_negative(img_fe, fe_params):
            return -0.05, -0.05, np.eye(3)

        monkeypatch.setattr(cov_mod, "evaluate_density_field", always_negative)

        # grid spans both auxetic (<0) and non-auxetic (>=0) targets
        result = cov_mod.coverage_eval(
            str(ckpt_path), n_samples=2, grid_size=5, device="cpu", seed=1,
            v12_range=(-0.4, 0.4),
        )
        non_auxetic = [t for t in result["per_target"] if not t["is_auxetic_target"]]
        assert len(non_auxetic) > 0
        for t in non_auxetic:
            assert t["hit"] is None
        assert result["n_auxetic_targets"] < result["grid_size"]

    def test_all_fe_solves_failing_yields_nan_hit_rate(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import coverage_eval as cov_mod

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        def always_fail(img_fe, fe_params):
            raise RuntimeError("forced FE failure")

        monkeypatch.setattr(cov_mod, "evaluate_density_field", always_fail)

        result = cov_mod.coverage_eval(
            str(ckpt_path), n_samples=2, grid_size=3, device="cpu", seed=1,
            v12_range=(-0.5, -0.1),
        )
        assert np.isnan(result["hit_rate"])
        assert np.isnan(result["mean_abs_error"])
        assert all(t["best_v12"] is None for t in result["per_target"])


class TestCoverageEvalManufacturabilityFlag:
    def test_check_manufacturability_populates_frac(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import coverage_eval as cov_mod

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe_params = dict(cov_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(cov_mod, "FE_PARAMS", tiny_fe_params)

        def always_negative(img_fe, fe_params):
            return -0.05, -0.05, np.eye(3)

        monkeypatch.setattr(cov_mod, "evaluate_density_field", always_negative)
        monkeypatch.setattr(cov_mod, "check_manufacturability",
                             lambda img_bin: {"passes_all": True})

        result = cov_mod.coverage_eval(
            str(ckpt_path), n_samples=3, grid_size=2, device="cpu", seed=1,
            v12_range=(-0.5, -0.2), check_manuf=True,
        )
        for t in result["per_target"]:
            assert t["frac_manufacturable"] == pytest.approx(1.0)

    def test_disabled_leaves_frac_manufacturable_none(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import coverage_eval as cov_mod

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe_params = dict(cov_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(cov_mod, "FE_PARAMS", tiny_fe_params)

        result = cov_mod.coverage_eval(
            str(ckpt_path), n_samples=2, grid_size=2, device="cpu", seed=1,
            v12_range=(-0.5, -0.2), check_manuf=False,
        )
        for t in result["per_target"]:
            assert t["frac_manufacturable"] is None


class TestCoverageEvalCli:
    def test_main_writes_result_json(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import coverage_eval as cov_mod

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe_params = dict(cov_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(cov_mod, "FE_PARAMS", tiny_fe_params)
        out_path = tmp_path / "result.json"

        monkeypatch.setattr(sys, "argv", [
            "coverage_eval.py",
            "--cvae-ckpt", str(ckpt_path),
            "--grid-size", "3",
            "--n-samples", "2",
            "--seed", "1",
            "--out", str(out_path),
        ])

        cov_mod.main()

        assert out_path.exists()
        import json
        with open(out_path) as f:
            data = json.load(f)
        assert data["grid_size"] == 3

"""
Tests for pipeline/phase5_cvae/active_learning.py (roadmap 7.1-7.3 + 7.5:
the active-learning loop that self_play.py is NOT a substitute for - see
module docstring). Covers the pure/unit-testable pieces
(find_weak_targets, generate_and_verify_candidates,
filter_good_candidates, save_good_candidates_npz, should_stop_loop); the
subprocess orchestration in run()/main() spawns real training jobs and is
intentionally out of scope for a unit test (same call this project already
made for self_play.py's run(), see tests/test_phase5_self_play.py).

Imports are lazy inside each test - see tests/conftest.py docstring on the
bare-import collision between phase4_surrogate/ and phase5_cvae/.
"""
import numpy as np
import pytest
import torch

SEED_CLASSES = np.array(["circle", "square", "hexagonal"], dtype=object)


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


class TestFindWeakTargets:
    def test_none_abs_error_ranked_above_finite_error(self, tmp_path, monkeypatch):
        """A target where every FE eval fails (abs_error=None, coverage_eval's
        own dead_zone_targets list actually EXCLUDES these - see
        coverage_eval.py's auxetic_rows filter on hit is not None) must
        still be treated as the WORST case by find_weak_targets's own
        ranking (None -> +inf), since it's exactly the case
        active-learning should prioritize sampling more data for.

        NOTE: find_weak_targets delegates to coverage_eval() from
        coverage_eval.py, so FE_PARAMS/evaluate_density_field must be
        patched on THAT module's namespace, not on active_learning's -
        active_learning only re-exports the `coverage_eval` function
        object, it doesn't call evaluate_density_field for this path
        itself. Patching via sys.modules["coverage_eval"] is unreliable
        across tests: active_learning.py imports it via a BARE
        `from coverage_eval import ...` (sys.path.insert trick), which
        only registers sys.modules["coverage_eval"] the FIRST time
        `pipeline.phase5_cvae.active_learning` itself gets imported in
        the whole pytest session - later tests get the cached dotted
        module without re-running that bare import, so the bare-name
        key can vanish once conftest's autouse fixture pops it. Patching
        `al_mod.coverage_eval.__globals__` directly instead is robust to
        import order/caching, since it's the exact dict the function
        looks up at call time, however it got bound."""
        from pipeline.phase5_cvae import active_learning as al_mod
        cov_globals = al_mod.coverage_eval.__globals__

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe = dict(cov_globals["FE_PARAMS"], nelx=6, nely=6)
        monkeypatch.setitem(cov_globals, "FE_PARAMS", tiny_fe)

        calls = {"n": 0}

        def flaky_eval(img_fe, fe_params):
            calls["n"] += 1
            # Fail deterministically on every sample of the FIRST grid
            # target processed (the most-negative v12), succeed elsewhere
            # with a large, but finite, error.
            if calls["n"] <= 2:
                raise RuntimeError("forced FE failure")
            return 0.5, 0.5, np.eye(3)

        monkeypatch.setitem(cov_globals, "evaluate_density_field", flaky_eval)

        weak_targets, cov = al_mod.find_weak_targets(
            str(ckpt_path), n_samples=2, grid_size=4, device="cpu",
            n_weak=4, v12_range=(-0.6, -0.1),
        )
        none_error_targets = [
            t["target_v12"] for t in cov["per_target"]
            if t["is_auxetic_target"] and t["abs_error"] is None
        ]
        assert len(none_error_targets) >= 1
        assert weak_targets[0] in none_error_targets

    def test_respects_n_weak_limit(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import active_learning as al_mod
        cov_globals = al_mod.coverage_eval.__globals__

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe = dict(cov_globals["FE_PARAMS"], nelx=6, nely=6)
        monkeypatch.setitem(cov_globals, "FE_PARAMS", tiny_fe)

        weak_targets, _cov = al_mod.find_weak_targets(
            str(ckpt_path), n_samples=2, grid_size=8, device="cpu",
            n_weak=3, v12_range=(-0.8, -0.1),
        )
        assert len(weak_targets) <= 3


class TestGenerateAndVerifyCandidates:
    def test_skips_failed_fe_evals_and_labels_succeed(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import active_learning as al_mod

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe = dict(al_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(al_mod, "FE_PARAMS", tiny_fe)

        calls = {"n": 0}

        def half_fail(img_fe, fe_params):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("forced FE failure")
            return -0.4, -0.3, np.eye(3)

        monkeypatch.setattr(al_mod, "evaluate_density_field", half_fail)

        candidates = al_mod.generate_and_verify_candidates(
            str(ckpt_path), weak_targets=[-0.5, -0.3], n_samples=4,
            device="cpu", seed_classes=SEED_CLASSES, seed=1,
        )
        # Half of the 2*4=8 attempts fail -> exactly 4 survive.
        assert len(candidates) == 4
        for c in candidates:
            assert c["v12"] == pytest.approx(-0.4)
            assert c["seed_onehot"].sum() == 1.0
            assert c["image"].shape == (64, 64)

    def test_no_candidates_when_all_fe_evals_fail(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import active_learning as al_mod

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        tiny_fe = dict(al_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(al_mod, "FE_PARAMS", tiny_fe)

        def always_fail(img_fe, fe_params):
            raise RuntimeError("forced FE failure")

        monkeypatch.setattr(al_mod, "evaluate_density_field", always_fail)

        candidates = al_mod.generate_and_verify_candidates(
            str(ckpt_path), weak_targets=[-0.5], n_samples=3,
            device="cpu", seed_classes=SEED_CLASSES, seed=1,
        )
        assert candidates == []


class TestFilterGoodCandidates:
    def _candidate(self, img_bin):
        return {
            "image": img_bin.astype(np.float32), "img_bin": img_bin,
            "v12": -0.4, "v21": -0.3, "volfrac": 0.4,
            "seed_onehot": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }

    def test_keeps_only_manufacturable(self):
        from pipeline.phase5_cvae import active_learning as al_mod

        # A solid block: connected, thick, and trivially periodic (uniform).
        good_img = np.ones((16, 16), dtype=np.float32)
        # A checkerboard: disconnected single-pixel islands -> not manufacturable.
        bad_img = np.zeros((16, 16), dtype=np.float32)
        bad_img[::2, ::2] = 1.0

        candidates = [self._candidate(good_img), self._candidate(bad_img)]
        good = al_mod.filter_good_candidates(candidates)

        assert len(good) == 1
        assert good[0]["image"] is candidates[0]["image"]

    def test_empty_input_returns_empty(self):
        from pipeline.phase5_cvae import active_learning as al_mod

        assert al_mod.filter_good_candidates([]) == []


class TestSaveGoodCandidatesNpz:
    def test_schema_matches_adversarial_dataset_convention(self, tmp_path):
        from pipeline.phase5_cvae import active_learning as al_mod

        rng = np.random.default_rng(0)
        candidates = [{
            "image": rng.random((64, 64)).astype(np.float32),
            "v12": -0.4, "v21": -0.3, "volfrac": 0.45,
            "seed_onehot": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        } for _ in range(3)]

        out_path = tmp_path / "good.npz"
        al_mod.save_good_candidates_npz(candidates, SEED_CLASSES, str(out_path))

        loaded = np.load(out_path, allow_pickle=True)
        for key in ("images", "v12", "v21", "volfrac_achieved",
                    "seed_onehot", "seed_classes"):
            assert key in loaded
        assert loaded["images"].shape == (3, 64, 64)
        assert loaded["v12"].shape == (3,)
        assert loaded["seed_onehot"].shape == (3, 3)
        assert list(loaded["seed_classes"]) == list(SEED_CLASSES)


class TestShouldStopLoop:
    def _row(self, round_idx, hit_rate, n_dead):
        return {
            "round": round_idx, "hit_rate": hit_rate,
            "dead_zone_targets": [0.0] * n_dead, "mean_abs_error": 0.1,
            "n_good_candidates": 5,
        }

    def test_stops_at_max_rounds(self):
        from pipeline.phase5_cvae import active_learning as al_mod

        summary = [self._row(0, 0.5, 3), self._row(1, 0.7, 1)]
        stop, reason = al_mod.should_stop_loop(summary, round_idx=3, max_rounds=3)
        assert stop is True
        assert "max_rounds" in reason

    def test_continues_when_clear_improvement(self):
        from pipeline.phase5_cvae import active_learning as al_mod

        summary = [self._row(0, 0.5, 5), self._row(1, 0.7, 2)]
        stop, reason = al_mod.should_stop_loop(summary, round_idx=1, max_rounds=5,
                                                min_hit_rate_gain=0.02)
        assert stop is False
        assert reason is None

    def test_stops_when_dead_zone_does_not_shrink(self):
        from pipeline.phase5_cvae import active_learning as al_mod

        summary = [self._row(0, 0.5, 3), self._row(1, 0.9, 3)]
        stop, reason = al_mod.should_stop_loop(summary, round_idx=1, max_rounds=5)
        assert stop is True
        assert "dead_zone" in reason

    def test_stops_when_hit_rate_gain_below_threshold(self):
        from pipeline.phase5_cvae import active_learning as al_mod

        summary = [self._row(0, 0.5, 5), self._row(1, 0.505, 2)]
        stop, reason = al_mod.should_stop_loop(summary, round_idx=1, max_rounds=5,
                                                min_hit_rate_gain=0.02)
        assert stop is True
        assert "hit_rate" in reason

    def test_single_round_never_stops_early(self):
        """With only 1 entry in summary (round 0 baseline), there's nothing
        to compare against yet - must not stop before round 1 even runs."""
        from pipeline.phase5_cvae import active_learning as al_mod

        summary = [self._row(0, 0.5, 5)]
        stop, reason = al_mod.should_stop_loop(summary, round_idx=0, max_rounds=5)
        assert stop is False
        assert reason is None

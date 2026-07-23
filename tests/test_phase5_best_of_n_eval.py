"""
Tests for pipeline/phase5_cvae/best_of_n_eval.py — the "fix that works" for
Phase 5 surrogate exploitation (best-of-N + real-FE selection at inference;
see README §5 and outputs/phase5/self_play/best_of_n_result.json). This is
the highest-priority file to protect from regression: it is now the
officially recommended way to get a trustworthy geometry out of the cVAE
(see pipeline/phase5_cvae/sample.py's warning banner), so a silent breakage
here (wrong best-candidate selection, wrong hit-rate/R2 bookkeeping, wrong
FE-call budget in --k-fe-verify mode) would be worse than a crash.

Imports are lazy inside each test — best_of_n_eval.py does
`sys.path.insert(...)` + bare `from dataset import X` / `from self_play
import load_cvae` / etc. at import time (see tests/conftest.py docstring).
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


def _write_test_npz(path, v12_values, n_seeds=1):
    """Write a test.npz with EXACTLY len(v12_values) samples, so
    `rng.choice(len(test_ds), size=n_conditions, replace=False)` is forced
    to select all of them (in some order) — makes which target conditions
    get used in the test fully deterministic regardless of the RNG seed."""
    n = len(v12_values)
    seed_classes = np.array([f"seed{i}" for i in range(n_seeds)], dtype=object)
    seed_onehot = np.zeros((n, n_seeds), dtype=np.float32)
    seed_onehot[:, 0] = 1.0
    np.savez(
        path,
        images=np.random.default_rng(0).random((n, 64, 64)).astype(np.float32),
        v12=np.array(v12_values, dtype=np.float32),
        v21=np.array(v12_values, dtype=np.float32),
        volfrac_achieved=np.full(n, 0.4, dtype=np.float32),
        seed_onehot=seed_onehot,
        seed_classes=seed_classes,
    )


def _write_surrogate_export(path, n_seeds=1, channels=(8, 16), fc_hidden=16):
    from pipeline.phase4_surrogate.model import SurrogateCNN
    model = SurrogateCNN(n_seeds=n_seeds, channels=channels, fc_hidden=fc_hidden)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_seeds": n_seeds,
        "channels": channels,
        "fc_hidden": fc_hidden,
        "target_names": ["v12", "v21", "volfrac_achieved"],
    }, path)


class TestBestOfNOracleSelectionLogic:
    """Fully deterministic test of the oracle (k_fe_verify=None) path: real
    FE calls are stubbed out with a preset, ordered sequence of return
    values so the best-candidate/hit-rate/R2 math can be checked exactly,
    independent of what the (untrained) cVAE actually generates."""

    def test_selects_closest_candidate_and_computes_correct_metrics(
        self, tmp_path, monkeypatch,
    ):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        # -0.5 and -0.3 are both auxetic targets (v12 < 0).
        _write_test_npz(test_npz, v12_values=[-0.5, -0.3])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        # Preset FE results, consumed in call order. best_of_n() iterates
        # conditions in the order returned by np.random.default_rng(seed)
        # .choice(...) — with exactly 2 samples in test.npz requesting 2,
        # both are always selected, but order can vary; key by target value
        # instead of call position so the test doesn't depend on that order.
        fe_results_by_target = {
            -0.5: [0.1, -0.55, -0.9],     # first(single-shot)=+0.1 -> miss; best=-0.55 (closest) -> hit
            -0.3: [-0.35, -0.2, 5.0],     # first=-0.35 -> hit; best=-0.35 (closest) -> hit
        }
        current_target = {}
        # best_of_n() re-evaluates FE on imgs[0] a SECOND time (as
        # `v12_first`, after already scoring it once inside the `fe_order`
        # loop) — real FE is deterministic given the same image, so index
        # by per-target call count and replay values[0] once the preset
        # sequence for that target is exhausted, instead of a pop() queue
        # (which would desync on that extra call).
        call_idx_by_target = {k: 0 for k in fe_results_by_target}

        def fake_evaluate_density_field(img_fe, fe_params):
            target = current_target["v12"]
            values = fe_results_by_target[target]
            idx = call_idx_by_target[target]
            v = values[idx] if idx < len(values) else values[0]
            call_idx_by_target[target] = idx + 1
            return v, v, np.eye(3)

        monkeypatch.setattr(boe_mod, "evaluate_density_field", fake_evaluate_density_field)

        # `evaluate_density_field` doesn't see which condition it's scoring
        # (best_of_n() only passes it the resized image), so patch
        # CVAE.generate to record the condition currently being processed,
        # so the fake evaluate_density_field knows which target's preset FE
        # sequence to consume next.
        #
        # IMPORTANT: `pipeline.phase5_cvae.model` (dotted import) and the
        # `model` module that best_of_n_eval.py's own call chain actually
        # uses (self_play -> adversarial_dataset -> bare `from model import
        # CVAE`, per tests/conftest.py's docstring) are TWO DISTINCT module
        # objects/class objects for the same file — patching the dotted
        # CVAE would silently patch a class nothing at runtime uses. Import
        # boe_mod first (already done above) to trigger the bare-import
        # chain, then grab the CVAE class from sys.modules["model"].
        CVAE_bare = sys.modules["model"].CVAE
        real_generate = CVAE_bare.generate

        def tracking_generate(self, condition, n_samples=1, device="cpu"):
            cond_np = condition.detach().cpu().numpy()
            cond_np = cond_np[0] if cond_np.ndim == 2 else cond_np
            # round: cond arrives as float32 (e.g. -0.3 -> -0.30000001...),
            # which would otherwise miss the float64 dict keys below.
            current_target["v12"] = round(float(cond_np[0]), 3)
            return real_generate(self, condition, n_samples=n_samples, device=device)

        monkeypatch.setattr(CVAE_bare, "generate", tracking_generate)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=2, n_samples=3, device="cpu", seed=123,
        )

        assert result["n_auxetic_targets"] == 2
        assert result["hit_rate_single_shot"] == pytest.approx(0.5)   # only -0.3 case
        assert result["hit_rate_best_of_n"] == pytest.approx(1.0)     # both cases
        assert result["r2_fe_v12_best_of_n"] == pytest.approx(0.75, abs=1e-6)
        assert result["n_fe_calls_total"] == 6  # 2 conditions x 3 samples, oracle mode
        assert result["k_fe_verify"] == 3  # defaults to n_samples when not set

        # target_v12 in the result retains float32 storage precision (e.g.
        # -0.3 -> -0.30000001192092896) — round for lookup, same as
        # tracking_generate() above.
        by_target = {round(c["target_v12"], 3): c for c in result["per_condition"]}
        assert by_target[-0.5]["v12_best"] == pytest.approx(-0.55)
        assert by_target[-0.5]["v12_first"] == pytest.approx(0.1)
        assert by_target[-0.3]["v12_best"] == pytest.approx(-0.35)


class TestBestOfNPracticalMode:
    """k_fe_verify=K: only K (of N) candidates should ever reach the real
    FE solver — the whole point of the 'practical' mode is bounding FE cost."""

    def test_k_fe_verify_bounds_real_fe_call_count(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.4, -0.2, 0.1])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        surrogate_path = tmp_path / "surrogate_for_phase5.pt"
        _write_surrogate_export(surrogate_path, n_seeds=1)

        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=3, n_samples=10, device="cpu", seed=5,
            k_fe_verify=4, surrogate_path=str(surrogate_path),
        )

        assert result["k_fe_verify"] == 4
        # <= because a real FE solve can (rarely) fail on a tiny/degenerate
        # grid and get skipped — but it must never exceed the K budget.
        assert result["n_fe_calls_total"] <= 3 * 4
        assert result["n_fe_calls_total"] > 0

    def test_k_fe_verify_uses_surrogate_ranking_not_all_n_samples(
        self, tmp_path, monkeypatch,
    ):
        """With k_fe_verify < n_samples, evaluate_density_field must be
        called close to K times (K from the ranked loop, +1 for the
        separate `v12_first` re-scoring of candidate 0 that best_of_n()
        always does) rather than N times — regression guard against
        silently falling back to scoring every candidate (which would
        defeat the whole cost-saving point of --k-fe-verify)."""
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.4])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        surrogate_path = tmp_path / "surrogate_for_phase5.pt"
        _write_surrogate_export(surrogate_path, n_seeds=1)

        call_count = {"n": 0}
        real_evaluate = boe_mod.evaluate_density_field

        def counting_evaluate(img_fe, fe_params):
            call_count["n"] += 1
            return real_evaluate(img_fe, fe_params)

        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)
        monkeypatch.setattr(boe_mod, "evaluate_density_field", counting_evaluate)

        boe_mod.best_of_n(
            str(ckpt_path), n_conditions=1, n_samples=8, device="cpu", seed=1,
            k_fe_verify=3, surrogate_path=str(surrogate_path),
        )

        # k_fe_verify (3) candidates scored inside the ranked loop, plus 1
        # more for the always-recomputed `v12_first` on candidate 0.
        assert call_count["n"] == 4


class TestBestOfNEdgeCases:
    def test_all_fe_solves_failing_does_not_crash(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.4, 0.2])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        def always_fail(img_fe, fe_params):
            raise RuntimeError("forced FE failure")

        monkeypatch.setattr(boe_mod, "evaluate_density_field", always_fail)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=2, n_samples=3, device="cpu", seed=1,
        )

        assert result["n_fe_calls_total"] == 0
        assert result["per_condition"] == []
        assert np.isnan(result["r2_fe_v12_best_of_n"])

    def test_reproducible_with_fixed_seed(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.5, -0.2, 0.3, -0.1])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))
        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        result_a = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=2, n_samples=2, device="cpu", seed=99,
        )
        result_b = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=2, n_samples=2, device="cpu", seed=99,
        )

        assert result_a["per_condition"] == result_b["per_condition"]
        assert result_a["hit_rate_best_of_n"] == result_b["hit_rate_best_of_n"]


class TestBestOfNCli:
    def test_main_writes_result_json(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.5, -0.2])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))
        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        out_path = tmp_path / "result.json"

        monkeypatch.setattr(sys, "argv", [
            "best_of_n_eval.py",
            "--cvae-ckpt", str(ckpt_path),
            "--n-conditions", "2",
            "--n-samples", "2",
            "--seed", "123",
            "--out", str(out_path),
        ])

        boe_mod.main()

        assert out_path.exists()
        import json
        with open(out_path) as f:
            data = json.load(f)
        assert data["n_conditions"] == 2

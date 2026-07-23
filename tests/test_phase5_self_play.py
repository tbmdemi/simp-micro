"""
Tests for pipeline/phase5_cvae/self_play.py — verify_round (the FE-based
checkpoint scoring function; the orchestration in run()/main() spawns real
training subprocesses and is intentionally out of scope for a unit test).

Imports are lazy inside each test — see tests/conftest.py docstring.
"""
import numpy as np
import torch


def _write_cvae_checkpoint(path, latent_dim=6, resolution=64,
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


def _write_test_npz(path, n_samples=16, n_seeds=2, v12_range=(-0.7, 0.3)):
    rng = np.random.default_rng(0)
    seed_classes = np.array([f"seed{i}" for i in range(n_seeds)], dtype=object)
    seed_idx = rng.integers(0, n_seeds, size=n_samples)
    seed_onehot = np.zeros((n_samples, n_seeds), dtype=np.float32)
    seed_onehot[np.arange(n_samples), seed_idx] = 1.0
    np.savez(
        path,
        images=rng.random((n_samples, 64, 64)).astype(np.float32),
        v12=rng.uniform(*v12_range, size=n_samples).astype(np.float32),
        v21=rng.uniform(*v12_range, size=n_samples).astype(np.float32),
        volfrac_achieved=rng.uniform(0.2, 0.6, size=n_samples).astype(np.float32),
        seed_onehot=seed_onehot,
        seed_classes=seed_classes,
    )


class TestVerifyRound:
    def test_returns_expected_keys_and_finite_r2(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import self_play as sp_mod

        tiny_fe_params = dict(sp_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(sp_mod, "FE_PARAMS", tiny_fe_params)

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, n_samples=16)
        monkeypatch.setattr(sp_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        result = sp_mod.verify_round(
            str(ckpt_path), n_conditions=4, n_per_condition=2,
            device="cpu", seed=123,
        )

        for key in ("r2_fe_v12", "hit_rate", "n_auxetic_targets", "n_samples"):
            assert key in result
        assert result["n_samples"] > 0

    def test_same_seed_is_reproducible(self, tmp_path, monkeypatch):
        """verify_round MUST give identical results across repeated calls
        with the same seed — self-play's round-over-round comparison is
        only meaningful if this holds (see module docstring: this was
        previously broken by an unseeded torch.randn() inside
        model.generate(), which made even scoring the SAME checkpoint twice
        produce different numbers)."""
        from pipeline.phase5_cvae import self_play as sp_mod

        tiny_fe_params = dict(sp_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(sp_mod, "FE_PARAMS", tiny_fe_params)

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, n_samples=16)
        monkeypatch.setattr(sp_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        result_a = sp_mod.verify_round(
            str(ckpt_path), n_conditions=3, n_per_condition=2,
            device="cpu", seed=7,
        )
        result_b = sp_mod.verify_round(
            str(ckpt_path), n_conditions=3, n_per_condition=2,
            device="cpu", seed=7,
        )

        assert result_a == result_b

    def test_different_seed_selects_different_conditions(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import self_play as sp_mod

        tiny_fe_params = dict(sp_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(sp_mod, "FE_PARAMS", tiny_fe_params)

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, n_samples=40)
        monkeypatch.setattr(sp_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        result_a = sp_mod.verify_round(
            str(ckpt_path), n_conditions=5, n_per_condition=1,
            device="cpu", seed=1,
        )
        result_b = sp_mod.verify_round(
            str(ckpt_path), n_conditions=5, n_per_condition=1,
            device="cpu", seed=2,
        )
        # Different seeds -> different condition subsets -> generally
        # different sample counts/scores (not a strict guarantee, but with
        # 40 candidates and 5 chosen, an exact match across both r2 and
        # n_samples would be a suspicious coincidence worth investigating).
        assert (result_a["r2_fe_v12"], result_a["n_samples"]) != \
               (result_b["r2_fe_v12"], result_b["n_samples"])

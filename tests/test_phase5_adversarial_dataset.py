"""
Tests for pipeline/phase5_cvae/adversarial_dataset.py — load_cvae,
generate_adversarial_npz.

Imports are lazy inside each test — see tests/conftest.py docstring.
Uses tiny FE grids (via monkeypatched FE_PARAMS) so the real FE solve stays
fast; this module's whole point is to score generated images with the real
solver, so we exercise that path rather than mocking it away.
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


def _write_train_npz(path, n_samples=10, n_seeds=3):
    rng = np.random.default_rng(0)
    seed_classes = np.array([f"seed{i}" for i in range(n_seeds)], dtype=object)
    seed_idx = rng.integers(0, n_seeds, size=n_samples)
    seed_onehot = np.zeros((n_samples, n_seeds), dtype=np.float32)
    seed_onehot[np.arange(n_samples), seed_idx] = 1.0
    np.savez(
        path,
        images=rng.random((n_samples, 64, 64)).astype(np.float32),
        v12=rng.uniform(-0.7, 0.3, size=n_samples).astype(np.float32),
        v21=rng.uniform(-0.7, 0.3, size=n_samples).astype(np.float32),
        volfrac_achieved=rng.uniform(0.2, 0.6, size=n_samples).astype(np.float32),
        seed_onehot=seed_onehot,
        seed_classes=seed_classes,
    )


class TestLoadCvae:
    def test_load_cvae_returns_eval_model(self, tmp_path):
        from pipeline.phase5_cvae.adversarial_dataset import load_cvae
        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        model = load_cvae(str(ckpt_path), "cpu")

        assert not model.training
        assert model.latent_dim == 6


class TestGenerateAdversarialNpz:
    def test_generates_npz_matching_phase3_schema(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import adversarial_dataset as adv_mod

        # Small FE grid so the real solve stays fast in a unit test.
        tiny_fe_params = dict(adv_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(adv_mod, "FE_PARAMS", tiny_fe_params)

        train_npz = tmp_path / "train.npz"
        _write_train_npz(train_npz, n_samples=10, n_seeds=3)
        monkeypatch.setattr(adv_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)
        out_path = tmp_path / "adversarial.npz"

        adv_mod.generate_adversarial_npz(
            str(ckpt_path), str(out_path),
            n_conditions=2, seeds_per_condition=2, device="cpu", seed=0,
        )

        assert out_path.exists()
        data = np.load(out_path, allow_pickle=True)
        for key in ("images", "v12", "v21", "volfrac_achieved",
                    "seed_onehot", "seed_classes"):
            assert key in data
        n = len(data["v12"])
        assert n > 0
        assert data["images"].shape == (n, 64, 64)
        assert data["seed_onehot"].shape == (n, 3)
        # images stored continuous [0,1] (not binarized) per module docstring
        assert data["images"].min() >= 0.0
        assert data["images"].max() <= 1.0

    def test_raises_if_all_fe_solves_fail(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import adversarial_dataset as adv_mod

        def _always_fail(*args, **kwargs):
            raise RuntimeError("forced FE failure")

        monkeypatch.setattr(adv_mod, "evaluate_density_field", _always_fail)

        train_npz = tmp_path / "train.npz"
        _write_train_npz(train_npz, n_samples=5, n_seeds=2)
        monkeypatch.setattr(adv_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        import pytest
        with pytest.raises(RuntimeError, match="Không sinh được"):
            adv_mod.generate_adversarial_npz(
                str(ckpt_path), str(tmp_path / "out.npz"),
                n_conditions=1, seeds_per_condition=1, device="cpu",
            )

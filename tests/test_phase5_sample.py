"""
Tests for pipeline/phase5_cvae/sample.py — load_model, save_png, main().

Imports are lazy inside each test — sample.py does `sys.path.insert(...)` +
bare `from model import CVAE` at import time (see tests/conftest.py
docstring).
"""
import numpy as np
import torch
from PIL import Image


def _write_cvae_checkpoint(path, latent_dim=8, condition_dim=2,
                            resolution=64, channels=(4, 8, 16, 32)):
    from pipeline.phase5_cvae.model import CVAE
    model = CVAE(condition_dim=condition_dim, latent_dim=latent_dim,
                 resolution=resolution, channels=channels)
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": latent_dim,
        "condition_dim": condition_dim,
        "resolution": resolution,
        "channels": channels,
    }, path)


class TestLoadModel:
    def test_load_model_returns_eval_mode_cvae(self, tmp_path):
        from pipeline.phase5_cvae.sample import load_model
        ckpt_path = tmp_path / "cvae_best.pt"
        _write_cvae_checkpoint(ckpt_path)

        model = load_model(device="cpu", ckpt_path=str(ckpt_path))

        assert not model.training
        assert model.latent_dim == 8

    def test_loaded_model_generates_correct_shape(self, tmp_path):
        from pipeline.phase5_cvae.sample import load_model
        ckpt_path = tmp_path / "cvae_best.pt"
        _write_cvae_checkpoint(ckpt_path)
        model = load_model(device="cpu", ckpt_path=str(ckpt_path))

        cond = torch.tensor([-0.5, -0.5], dtype=torch.float32)
        out = model.generate(cond, n_samples=4, device="cpu")
        assert out.shape == (4, 1, 64, 64)


class TestSavePng:
    def test_save_png_roundtrip(self, tmp_path):
        from pipeline.phase5_cvae.sample import save_png
        img = torch.zeros(1, 64, 64)
        img[:, :32, :] = 1.0
        out_path = tmp_path / "sample.png"

        save_png(img, str(out_path))

        assert out_path.exists()
        loaded = np.array(Image.open(str(out_path)))
        assert loaded.shape == (64, 64)
        assert loaded[:32, :].min() == 255
        assert loaded[32:, :].max() == 0


class TestMainCli:
    def test_main_writes_expected_number_of_samples(self, tmp_path, monkeypatch, capsys):
        import sys
        from pipeline.phase5_cvae import sample as sample_module
        ckpt_path = tmp_path / "cvae_best.pt"
        _write_cvae_checkpoint(ckpt_path)
        out_dir = tmp_path / "out"

        monkeypatch.setattr(sample_module, "CKPT_PATH", str(ckpt_path))
        monkeypatch.setattr(sys, "argv", [
            "sample.py", "--v12", "-0.5", "--v21", "-0.5",
            "--n", "3", "--out", str(out_dir),
        ])

        sample_module.main()

        pngs = sorted(out_dir.glob("sample_*.png"))
        assert len(pngs) == 3

    def test_main_warns_single_shot_is_unreliable(self, tmp_path, monkeypatch, capsys):
        """sample.py generates ONE candidate with no FE filtering — the CLI
        must steer users toward best_of_n_eval.py (the actual, FE-verified
        pipeline; see README Phase 5 / outputs/phase5/fe_verification_report.json),
        not let them silently trust a single-shot sample."""
        import sys
        from pipeline.phase5_cvae import sample as sample_module
        ckpt_path = tmp_path / "cvae_best.pt"
        _write_cvae_checkpoint(ckpt_path)

        monkeypatch.setattr(sample_module, "CKPT_PATH", str(ckpt_path))
        monkeypatch.setattr(sys, "argv", [
            "sample.py", "--v12", "-0.5", "--v21", "-0.5",
            "--n", "1", "--out", str(tmp_path / "out"),
        ])

        sample_module.main()

        out = capsys.readouterr().out
        assert "best_of_n_eval.py" in out

    def test_main_raises_if_checkpoint_missing(self, tmp_path, monkeypatch):
        import sys
        from pipeline.phase5_cvae import sample as sample_module

        monkeypatch.setattr(sample_module, "CKPT_PATH", str(tmp_path / "nope.pt"))
        monkeypatch.setattr(sys, "argv", [
            "sample.py", "--v12", "-0.5", "--v21", "-0.5",
        ])

        import pytest
        with pytest.raises(FileNotFoundError):
            sample_module.main()

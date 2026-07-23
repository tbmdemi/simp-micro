"""
Tests for pipeline/phase4_surrogate/export_for_phase5.py — export_surrogate().
"""
import json

import torch

from pipeline.phase4_surrogate.export_for_phase5 import export_surrogate
from pipeline.phase4_surrogate.model import SurrogateCNN


def _write_fake_train_checkpoint(path, n_seeds=4, channels=(8, 16), fc_hidden=32):
    model = SurrogateCNN(n_seeds=n_seeds, channels=channels, fc_hidden=fc_hidden)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_seeds": n_seeds,
        "seed_classes": ["a", "b", "c", "d"],
        "channels": channels,
        "fc_hidden": fc_hidden,
        "target_names": ["v12", "v21", "volfrac_achieved"],
        "val_loss": 0.1234,
        "epoch": 7,
    }, path)


class TestExportSurrogate:
    def test_export_contains_expected_fields(self, tmp_path):
        src = tmp_path / "surrogate_best.pt"
        dst = tmp_path / "surrogate_for_phase5.pt"
        _write_fake_train_checkpoint(src)

        export_surrogate(str(src), str(dst))

        assert dst.exists()
        export = torch.load(str(dst), map_location="cpu", weights_only=False)
        assert export["n_seeds"] == 4
        assert export["seed_classes"] == ["a", "b", "c", "d"]
        assert export["channels"] == (8, 16)
        assert export["fc_hidden"] == 32
        assert export["target_names"] == ["v12", "v21", "volfrac_achieved"]
        assert export["training_val_loss"] == 0.1234
        assert export["training_epoch"] == 7
        assert "usage_note" in export and export["usage_note"]
        assert export["input_spec"]["image_shape"] == [1, 64, 64]

    def test_exported_state_dict_loads_into_fresh_model(self, tmp_path):
        src = tmp_path / "surrogate_best.pt"
        dst = tmp_path / "surrogate_for_phase5.pt"
        _write_fake_train_checkpoint(src, n_seeds=5, channels=(8, 16), fc_hidden=32)

        export_surrogate(str(src), str(dst))
        export = torch.load(str(dst), map_location="cpu", weights_only=False)

        model = SurrogateCNN(
            n_seeds=export["n_seeds"], channels=export["channels"],
            fc_hidden=export["fc_hidden"],
        )
        model.load_state_dict(export["model_state_dict"])  # should not raise

    def test_missing_eval_report_still_exports(self, tmp_path):
        src = tmp_path / "surrogate_best.pt"
        dst = tmp_path / "surrogate_for_phase5.pt"
        _write_fake_train_checkpoint(src)

        export_surrogate(
            str(src), str(dst),
            eval_report_path=str(tmp_path / "does_not_exist.json"),
        )

        export = torch.load(str(dst), map_location="cpu", weights_only=False)
        assert export["evaluation_report"] is None

    def test_eval_report_is_embedded_when_present(self, tmp_path):
        src = tmp_path / "surrogate_best.pt"
        dst = tmp_path / "surrogate_for_phase5.pt"
        eval_report_path = tmp_path / "evaluation_report.json"
        _write_fake_train_checkpoint(src)
        report = {"overall": {"v12": {"r2": 0.91, "mae": 0.03},
                               "v21": {"r2": 0.90, "mae": 0.04}}}
        eval_report_path.write_text(json.dumps(report))

        export_surrogate(str(src), str(dst), eval_report_path=str(eval_report_path))

        export = torch.load(str(dst), map_location="cpu", weights_only=False)
        assert export["evaluation_report"] == report

"""
Tests for pipeline/phase5_cvae/evaluate.py.

Bug đã sửa 2026-08-15: evaluate.py trước đây LUÔN dùng CVAEDataset mặc định
(condition_dim=2), bất kể checkpoint train với --extended-condition
(condition_dim=6) hay không - crash shape mismatch trong
model.decoder()/property_accuracy() khi đánh giá checkpoint extended.
Import lazy trong từng test - xem tests/conftest.py docstring (bare-import
collision giữa phase4_surrogate/ và phase5_cvae/).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader


def _write_surrogate_export(path, n_seeds=4, channels=(8, 16), fc_hidden=16):
    from pipeline.phase4_surrogate.model import SurrogateCNN
    model = SurrogateCNN(n_seeds=n_seeds, channels=channels, fc_hidden=fc_hidden)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_seeds": n_seeds,
        "channels": channels,
        "fc_hidden": fc_hidden,
        "n_outputs": 3,
        "target_names": ["v12", "v21", "volfrac_achieved"],
    }, path)


class TestPropertyAccuracyExtendedCondition:
    """property_accuracy() phải hoạt động cả khi condition có 6 chiều
    (extended_condition=True) - surrogate chỉ dự đoán (v12,v21), nên phải so
    khớp đúng 2 cột đầu của condition/target, không phải toàn bộ vector."""

    def test_2dim_condition_still_works(self, tmp_path, make_phase3_npz, make_cvae_checkpoint):
        from pipeline.phase5_cvae.evaluate import property_accuracy
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.losses import load_frozen_surrogate

        surrogate_path = tmp_path / "surrogate.pt"
        _write_surrogate_export(surrogate_path)
        surrogate, target_names = load_frozen_surrogate(device="cpu", path=str(surrogate_path))

        ckpt_path = make_cvae_checkpoint(condition_dim=2)
        from pipeline.phase5_cvae.sample import load_model
        model = load_model(device="cpu", ckpt_path=ckpt_path)

        npz_path = make_phase3_npz(n_samples=8)
        ds = CVAEDataset(npz_path, extended_condition=False)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        report = property_accuracy(model, surrogate, target_names, loader, "cpu")
        assert set(report.keys()) == {"v12", "v21", "n_samples"}
        assert report["n_samples"] == 8

    def test_6dim_condition_does_not_crash_and_slices_correctly(self, tmp_path, make_phase3_npz, make_cvae_checkpoint):
        """Regression test cho bug: trước đây `targets.append(condition...)`
        không cắt về 2 chiều, làm (targets - preds) broadcast sai/raise khi
        condition có 6 cột."""
        from pipeline.phase5_cvae.evaluate import property_accuracy
        from pipeline.phase5_cvae.dataset import CVAEDataset
        from pipeline.phase5_cvae.losses import load_frozen_surrogate
        from pipeline.phase5_cvae.sample import load_model

        surrogate_path = tmp_path / "surrogate.pt"
        _write_surrogate_export(surrogate_path)
        surrogate, target_names = load_frozen_surrogate(device="cpu", path=str(surrogate_path))

        ckpt_path = make_cvae_checkpoint("cvae_ext.pt", condition_dim=6)
        model = load_model(device="cpu", ckpt_path=ckpt_path)

        npz_path = make_phase3_npz(n_samples=8)
        ds = CVAEDataset(npz_path, extended_condition=True)
        assert ds.condition_dim == 6
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        report = property_accuracy(model, surrogate, target_names, loader, "cpu")
        assert set(report.keys()) == {"v12", "v21", "n_samples"}
        assert report["n_samples"] == 8
        assert np.isfinite(report["v12"]["mae"])

    def test_8dim_condition_extended_plus_nu0_does_not_crash(
        self, tmp_path, make_phase3_npz, make_cvae_checkpoint,
    ):
        """A6 (docs/PROJECT_PLAN.md Nhóm 1): condition_dim=8 (extended_condition
        + include_nu0) - cùng bug class với test 6-chiều ở trên, nhưng ở đây
        kiểm chứng đúng chỗ evaluate.py::main() dùng condition_flags_from_dim()
        để suy (extended, include_nu0) từ model.condition_dim (xem comment
        'Bug tương tự tái phát khi thêm A6' trong evaluate.py main())."""
        from pipeline.phase5_cvae.evaluate import property_accuracy
        from pipeline.phase5_cvae.dataset import CVAEDataset, condition_flags_from_dim
        from pipeline.phase5_cvae.losses import load_frozen_surrogate
        from pipeline.phase5_cvae.sample import load_model

        surrogate_path = tmp_path / "surrogate.pt"
        _write_surrogate_export(surrogate_path)
        surrogate, target_names = load_frozen_surrogate(device="cpu", path=str(surrogate_path))

        ckpt_path = make_cvae_checkpoint("cvae_ext_nu0.pt", condition_dim=8)
        model = load_model(device="cpu", ckpt_path=ckpt_path)

        extended, include_nu0 = condition_flags_from_dim(model.condition_dim)
        assert extended is True and include_nu0 is True

        npz_path = make_phase3_npz(n_samples=8, nu_range=(0.2, 0.4))
        ds = CVAEDataset(npz_path, extended_condition=extended, include_nu0=include_nu0)
        assert ds.condition_dim == 8
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        report = property_accuracy(model, surrogate, target_names, loader, "cpu")
        assert set(report.keys()) == {"v12", "v21", "n_samples"}
        assert report["n_samples"] == 8
        assert np.isfinite(report["v12"]["mae"])


class TestDiversityCheckExtendedCondition:
    def test_6dim_condition_vector_generates_without_crash(self, tmp_path, monkeypatch, make_cvae_checkpoint):
        import pipeline.phase5_cvae.evaluate as evaluate_mod
        from pipeline.phase5_cvae.dataset import build_condition_vector
        from pipeline.phase5_cvae.sample import load_model

        # diversity_check() ghi PNG ra DIAG_DIR (mặc định outputs/phase5/diagnostics/
        # THẬT trong repo, đã có file tracked cùng tên vd condition -0.6/-0.6 -
        # KHÔNG được để test này ghi đè lên đó, xem feedback_smoke_testing.md).
        monkeypatch.setattr(evaluate_mod, "DIAG_DIR", str(tmp_path / "diagnostics"))

        ckpt_path = make_cvae_checkpoint("cvae_ext.pt", condition_dim=6)
        model = load_model(device="cpu", ckpt_path=ckpt_path)

        condition = build_condition_vector(-0.6, -0.6, model.condition_dim)
        assert condition.shape == (6,)

        report = evaluate_mod.diversity_check(model, condition=condition, n_samples=4, device="cpu")
        assert "pixel_std" in report
        assert (tmp_path / "diagnostics").exists()

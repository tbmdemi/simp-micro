"""Tests cho harness benchmark physics-guided refinement (plan.md v3 P1.0):
chế độ --targets (OOD), best-of-N + warm-start refine, và CI paired
bootstrap trong bootstrap_ci.py.

Import module lazily trong fixture và patch THẲNG thuộc tính trên module
benchmark (không patch dotted path của module gốc) - module dùng bare import
kiểu `from tandem_lbfgs import ...`, xem ghi chú landmine phase4/phase5 cùng
tên module trong sys.modules.
"""

import importlib

import numpy as np
import pytest
import torch
from torch import nn


@pytest.fixture
def bench():
    """Module benchmark, import lazily để tránh xung đột sys.modules."""
    return importlib.import_module(
        "pipeline.phase5_cvae.benchmark_physics_guided_refinement"
    )


@pytest.fixture
def bootstrap_mod():
    """Module bootstrap_ci Phase 5."""
    return importlib.import_module("pipeline.phase5_cvae.bootstrap_ci")


class _MeanDecoderModel(nn.Module):
    """Decoder giả: ảnh hằng số = sigmoid(z[:,0]) - để FE giả suy ra "ν"
    trực tiếp từ z, kiểm soát được mẫu nào tốt nhất."""

    latent_dim = 3

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def decoder(self, z, condition):
        """Ảnh (B,1,4,4) hằng số theo z[:,0]."""
        return torch.sigmoid(z[:, :1]).view(-1, 1, 1, 1).expand(-1, 1, 4, 4)


def _install_stubs(bench, monkeypatch, tmp_path, calls):
    """Thay load_cvae/FE/tandem bằng stub rẻ; trả đường dẫn ckpt giả.

    FE giả: v12 = v21 = -mean(ảnh) (ảnh càng đặc -> càng âm).
    Tandem giả: ghi lại initial_z và trả ảnh mean=0,9.
    """
    ckpt = tmp_path / "fake.pt"
    torch.save({"latent_dim": 3, "condition_dim": 2}, ckpt)
    monkeypatch.setattr(
        bench, "load_cvae", lambda path, device: _MeanDecoderModel()
    )

    def fake_verify(
        image_2d, fe_params, apply_force_periodic=False, design_sigma=None
    ):
        calls.setdefault("verify_fp", []).append(apply_force_periodic)
        calls.setdefault("design_sigma", []).append(design_sigma)
        v = -float(np.mean(image_2d))
        return v, v

    def fake_tandem(**kwargs):
        calls.setdefault("initial_z", []).append(kwargs["initial_z"].clone())
        return {
            "image": torch.full((1, 1, 4, 4), 0.9),
            "history": [1.0, 0.5, 0.1],
        }

    monkeypatch.setattr(bench, "_verify_v12_v21", fake_verify)
    monkeypatch.setattr(bench, "tandem_inverse_design_lbfgs", fake_tandem)
    return str(ckpt)


class TestPairedMaeReduction:
    def test_identical_errors_give_zero_and_ci_contains_zero(
        self, bootstrap_mod
    ):
        err = [0.1, 0.2, 0.3, 0.4]
        r = bootstrap_mod.bootstrap_paired_mae_reduction(err, err, n_boot=500)
        assert r["mae_diff"] == pytest.approx(0.0)
        assert r["mae_diff_ci95_lo"] <= 0.0 <= r["mae_diff_ci95_hi"]

    def test_uniform_improvement_ci_excludes_zero(self, bootstrap_mod):
        base = np.linspace(0.2, 0.6, 20)
        r = bootstrap_mod.bootstrap_paired_mae_reduction(
            base, base * 0.5, n_boot=500
        )
        assert r["rel_reduction"] == pytest.approx(0.5)
        assert r["mae_diff_ci95_lo"] > 0.0
        # Giảm đúng 50% ở MỌI condition -> tỉ lệ trên mọi resample = 0,5.
        assert r["rel_reduction_ci95_lo"] == pytest.approx(0.5)

    def test_empty_returns_nan(self, bootstrap_mod):
        r = bootstrap_mod.bootstrap_paired_mae_reduction([], [])
        assert r["n_conditions"] == 0
        assert np.isnan(r["mae_diff"])

    def test_length_mismatch_raises(self, bootstrap_mod):
        with pytest.raises(ValueError):
            bootstrap_mod.bootstrap_paired_mae_reduction([0.1], [0.1, 0.2])


class TestRunBenchmark:
    def test_targets_mode_builds_symmetric_conditions(
        self, bench, monkeypatch, tmp_path
    ):
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        out = bench.run_benchmark(
            ckpt, targets=[-1.5, -2.0], steps=3, n_boot=100
        )
        targets = [
            (c["target_v12"], c["target_v21"]) for c in out["per_condition"]
        ]
        assert targets == [(-1.5, -1.5), (-2.0, -2.0)]
        assert out["config"]["targets"] == [-1.5, -2.0]

    def test_target_v21_makes_anisotropic_conditions(
        self, bench, monkeypatch, tmp_path
    ):
        """Target OOD dị hướng (ν12·ν21 < 1) - đối xứng ν*=-2 là bất khả
        thi vật lý nên quét OOD phải cố định v21 riêng."""
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        out = bench.run_benchmark(
            ckpt, targets=[-2.0], target_v21=-0.05, steps=3, n_boot=50
        )
        row = out["per_condition"][0]
        assert (row["target_v12"], row["target_v21"]) == pytest.approx(
            (-2.0, -0.05)
        )
        # Refined giả ra -0,9 cho cả 2 thành phần -> rel err v12 = 1,1/2.
        assert out["summary"]["mean_rel_err_v12_refined"] == pytest.approx(
            1.1 / 2.0
        )

    def test_best_of_n_picks_min_error_and_warm_starts_refine(
        self, bench, monkeypatch, tmp_path
    ):
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        # Target ν*=-0,9: FE giả = -sigmoid(z0) -> mẫu tốt nhất là mẫu có
        # z0 lớn nhất (sigmoid gần 0,9 nhất trong phân phối N(0,1)).
        out = bench.run_benchmark(
            ckpt, targets=[-0.9], n_samples=8, steps=3, seed=5, n_boot=100
        )
        torch.manual_seed(5)
        z_all = torch.randn(8, 3)
        err = (torch.sigmoid(z_all[:, 0]) - 0.9).abs()
        expected = int(torch.argmin(err))
        row = out["per_condition"][0]
        assert row["baseline_sample_index"] == expected
        assert torch.allclose(
            calls["initial_z"][0], z_all[expected : expected + 1]
        )
        assert row["n_fe_calls_baseline"] == 8
        assert row["n_fe_calls_refine"] == 3 + 2

    def test_single_sample_keeps_legacy_z_draw(
        self, bench, monkeypatch, tmp_path
    ):
        """n_samples=1 phải rút z0 y hệt randn(1,d) bản 2026-09-23 - điều
        kiện để tái lập số MAE 0,587 -> 0,355 cũ."""
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        bench.run_benchmark(ckpt, targets=[-0.5], steps=3, seed=9, n_boot=50)
        torch.manual_seed(9)
        assert torch.allclose(calls["initial_z"][0], torch.randn(1, 3))

    def test_force_periodic_flag_reaches_verify(
        self, bench, monkeypatch, tmp_path
    ):
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        bench.run_benchmark(
            ckpt,
            targets=[-0.5],
            n_samples=2,
            steps=3,
            apply_force_periodic=True,
            n_boot=50,
        )
        assert calls["verify_fp"] and all(calls["verify_fp"])

    def test_summary_reports_paired_ci_and_relative_error(
        self, bench, monkeypatch, tmp_path
    ):
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        out = bench.run_benchmark(
            ckpt, targets=[-0.9, -0.8], steps=3, n_boot=100
        )
        s = out["summary"]
        # Refined giả luôn ra -0,9 (ảnh mean 0,9) -> sai số tương đối
        # = mean(0/0,9, 0,1/0,8).
        assert s["mean_rel_err_refined"] == pytest.approx(
            np.mean([0.0, 0.1 / 0.8])
        )
        assert s["n_paired"] == 2
        assert "rel_reduction_ci95_lo" in s["paired_v12"]
        assert s["n_fe_calls_refine"] == 2 * 5

    def test_guarded_keeps_baseline_when_refine_is_worse(
        self, bench, monkeypatch, tmp_path
    ):
        """Refined giả luôn ra -0,9; target -0,1 -> baseline (z ngẫu nhiên,
        FE giả ~ -0,5) gần hơn -> guarded phải giữ baseline, mức giảm = 0."""
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        out = bench.run_benchmark(
            ckpt, targets=[-0.1, -0.1, -0.1], steps=3, n_boot=50
        )
        s = out["summary"]
        assert s["frac_guarded_accept"] == 0.0
        assert s["paired_v12_guarded"]["mae_diff"] == pytest.approx(0.0)

    def test_projection_betas_forwarded_to_refiner(
        self, bench, monkeypatch, tmp_path
    ):
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        seen = {}
        orig = bench.tandem_inverse_design_lbfgs

        def spy(**kw):
            seen.update(kw)
            return orig(**kw)

        monkeypatch.setattr(bench, "tandem_inverse_design_lbfgs", spy)
        out = bench.run_benchmark(
            ckpt,
            targets=[-0.5],
            steps=3,
            apply_force_periodic=True,
            projection_betas=[1, 16],
            n_boot=50,
        )
        assert seen["projection_betas"] == [1, 16] and seen["periodic"]
        assert out["config"]["projection_betas"] == [1, 16]

    def test_manufacturability_and_saved_images(
        self, bench, monkeypatch, tmp_path
    ):
        """P1.6c: cờ chế tạo được luôn ghi cho baseline/refined; ảnh nhị
        phân chỉ lưu khi save_images (JSON gọn ở chế độ mặc định)."""
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        out = bench.run_benchmark(
            ckpt, targets=[-0.5], steps=3, n_boot=50, save_images=True
        )
        row = out["per_condition"][0]
        # Refined giả = ảnh đặc 0,9 → 1 mảnh liên thông, tuần hoàn.
        assert row["refined_manufacturable"] is True
        assert np.asarray(row["refined_image"]).shape == (4, 4)
        assert np.asarray(row["refined_image"]).min() == 1
        assert out["summary"]["frac_manufacturable_refined"] == 1.0

        plain = bench.run_benchmark(ckpt, targets=[-0.5], steps=3, n_boot=50)
        assert "refined_image" not in plain["per_condition"][0]
        assert "baseline_manufacturable" in plain["per_condition"][0]

    def test_rejects_zero_samples(self, bench, monkeypatch, tmp_path):
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        with pytest.raises(ValueError):
            bench.run_benchmark(ckpt, targets=[-0.5], n_samples=0)


class TestFilteredDesign:
    """N2-E1′: thiết kế = ảnh đã lọc trên lưới FE, guard theo loss robust."""

    def test_binary_design_on_fe_grid(self, bench):
        img = np.zeros((64, 64), dtype=np.float32)
        img[16:48, 16:48] = 1.0
        params = dict(bench.FE_PARAMS)
        d = bench._binary_design(img, False, 1.0, params)
        assert d.shape == (params["nely"], params["nelx"])
        assert set(np.unique(d)) <= {0, 1}
        # Không lọc → hành vi cũ: ảnh 64² ngưỡng 0,5.
        assert bench._binary_design(img).shape == (64, 64)

    def test_filter_removes_isolated_speck(self, bench):
        """Đốm 1 px (lỗi E1) phải biến mất qua bộ lọc σ = 1 phần tử."""
        img = np.zeros((64, 64), dtype=np.float32)
        img[16:48, 16:48] = 1.0
        img[5, 5] = 1.0
        d = bench._binary_design(img, False, 1.0, dict(bench.FE_PARAMS))
        assert d[:8, :8].sum() == 0

    def test_robust_score_finite_and_zero_for_exact_target(
        self, bench, monkeypatch
    ):
        monkeypatch.setattr(
            bench, "evaluate_density_field", lambda d, p: (-0.3, -0.2, None)
        )
        img = np.full((64, 64), 0.9, dtype=np.float32)
        s = bench._robust_score(img, (-0.3, -0.2), -0.3, -0.2, 1.0, [0.25])
        assert s == pytest.approx(0.0)
        s2 = bench._robust_score(img, (-0.5, -0.2), -0.3, -0.2, 1.0, [0.25])
        assert s2 > 0

    def test_design_sigma_reaches_verify(self, bench, monkeypatch, tmp_path):
        calls = {}
        ckpt = _install_stubs(bench, monkeypatch, tmp_path, calls)
        out = bench.run_benchmark(
            ckpt,
            targets=[-0.5],
            n_samples=2,
            steps=3,
            n_boot=50,
            design_filter_sigma=1.0,
        )
        assert calls["design_sigma"] and all(
            s == 1.0 for s in calls["design_sigma"]
        )
        assert out["config"]["design_filter_sigma"] == 1.0

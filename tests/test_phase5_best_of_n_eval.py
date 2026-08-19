"""
Tests for pipeline/phase5_cvae/best_of_n_eval.py - the "fix that works" for
Phase 5 surrogate exploitation (best-of-N + real-FE selection at inference;
see README §5 and outputs/phase5/self_play/best_of_n_result.json). This is
the highest-priority file to protect from regression: it is now the
officially recommended way to get a trustworthy geometry out of the cVAE
(see pipeline/phase5_cvae/sample.py's warning banner), so a silent breakage
here (wrong best-candidate selection, wrong hit-rate/R2 bookkeeping, wrong
FE-call budget in --k-fe-verify mode) would be worse than a crash.

Imports are lazy inside each test - best_of_n_eval.py does
`sys.path.insert(...)` + bare `from dataset import X` / `from self_play
import load_cvae` / etc. at import time (see tests/conftest.py docstring).
"""
import sys

import numpy as np
import pytest
import torch


def _write_cvae_checkpoint(path, latent_dim=4, resolution=64,
                            channels=(4, 8, 16, 32), condition_dim=2):
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


def _write_test_npz(path, v12_values, n_seeds=1, nu_values=None):
    """Write a test.npz with EXACTLY len(v12_values) samples, so
    `rng.choice(len(test_ds), size=n_conditions, replace=False)` is forced
    to select all of them (in some order) - makes which target conditions
    get used in the test fully deterministic regardless of the RNG seed.

    nu_values: nếu truyền vào (độ dài = len(v12_values)), thêm field 'nu' -
    cần cho test condition_dim ∈ {4,8} (A6, CVAEDataset(include_nu0=True))."""
    n = len(v12_values)
    seed_classes = np.array([f"seed{i}" for i in range(n_seeds)], dtype=object)
    seed_onehot = np.zeros((n, n_seeds), dtype=np.float32)
    seed_onehot[:, 0] = 1.0
    extra = {}
    if nu_values is not None:
        extra["nu"] = np.array(nu_values, dtype=np.float32)
    np.savez(
        path,
        images=np.random.default_rng(0).random((n, 64, 64)).astype(np.float32),
        v12=np.array(v12_values, dtype=np.float32),
        v21=np.array(v12_values, dtype=np.float32),
        volfrac_achieved=np.full(n, 0.4, dtype=np.float32),
        seed_onehot=seed_onehot,
        seed_classes=seed_classes,
        **extra,
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
        # .choice(...) - with exactly 2 samples in test.npz requesting 2,
        # both are always selected, but order can vary; key by target value
        # instead of call position so the test doesn't depend on that order.
        fe_results_by_target = {
            -0.5: [0.1, -0.55, -0.9],     # first(single-shot)=+0.1 -> miss; best=-0.55 (closest) -> hit
            -0.3: [-0.35, -0.2, 5.0],     # first=-0.35 -> hit; best=-0.35 (closest) -> hit
        }
        current_target = {}
        # best_of_n() re-evaluates FE on imgs[0] a SECOND time (as
        # `v12_first`, after already scoring it once inside the `fe_order`
        # loop) - real FE is deterministic given the same image, so index
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
        # objects/class objects for the same file - patching the dotted
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

        # w_accuracy=1/w_manuf=0/w_aesthetic=0: pin composite scoring back to
        # pure argmin(|Δv12|) - this test verifies the FE-value/hit-rate/R2
        # bookkeeping against a controlled oracle sequence, independent of
        # manufacturability/aesthetic scores computed on whatever an
        # untrained CVAE happens to generate (see TestCompositeScoring for
        # dedicated tests of the composite-score math itself).
        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=2, n_samples=3, device="cpu", seed=123,
            w_accuracy=1.0, w_manuf=0.0, w_aesthetic=0.0,
        )

        assert result["n_auxetic_targets"] == 2
        assert result["hit_rate_single_shot"] == pytest.approx(0.5)   # only -0.3 case
        assert result["hit_rate_best_of_n"] == pytest.approx(1.0)     # both cases
        assert result["r2_fe_v12_best_of_n"] == pytest.approx(0.75, abs=1e-6)
        assert result["n_fe_calls_total"] == 6  # 2 conditions x 3 samples, oracle mode
        assert result["k_fe_verify"] == 3  # defaults to n_samples when not set

        # target_v12 in the result retains float32 storage precision (e.g.
        # -0.3 -> -0.30000001192092896) - round for lookup, same as
        # tracking_generate() above.
        by_target = {round(c["target_v12"], 3): c for c in result["per_condition"]}
        assert by_target[-0.5]["v12_best"] == pytest.approx(-0.55)
        assert by_target[-0.5]["v12_first"] == pytest.approx(0.1)
        assert by_target[-0.3]["v12_best"] == pytest.approx(-0.35)


class TestBestOfNPracticalMode:
    """k_fe_verify=K: only K (of N) candidates should ever reach the real
    FE solver - the whole point of the 'practical' mode is bounding FE cost."""

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
        # grid and get skipped - but it must never exceed the K budget.
        assert result["n_fe_calls_total"] <= 3 * 4
        assert result["n_fe_calls_total"] > 0

    def test_k_fe_verify_uses_surrogate_ranking_not_all_n_samples(
        self, tmp_path, monkeypatch,
    ):
        """With k_fe_verify < n_samples, evaluate_density_field must be
        called close to K times (K from the ranked loop, +1 for the
        separate `v12_first` re-scoring of candidate 0 that best_of_n()
        always does) rather than N times - regression guard against
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


class TestBestOfNConditionDimFlags:
    """A6 (docs/PROJECT_PLAN.md Nhóm 1): trước fix, đường sweep test.npz
    (custom_condition=None) suy extended_condition=(condition_dim==6) - BỎ
    SÓT include_nu0 khi condition_dim ∈ {4,8}, tạo CVAEDataset condition_dim
    lệch với model -> crash concat trong Encoder/Decoder. condition_flags_
    from_dim() sửa lỗi này - test xác nhận sweep path chạy được với đủ 4
    condition_dim {2,4,6,8}, không chỉ path custom_condition (--v12/--v21)."""

    def test_condition_dim_4_nu0_only_sweep_does_not_crash(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.4, 0.2], nu_values=[0.25, 0.35])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))
        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path, condition_dim=4)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=2, n_samples=2, device="cpu", seed=1,
        )
        assert result["condition_dim"] == 4

    def test_condition_dim_8_extended_and_nu0_sweep_does_not_crash(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        # extended_condition=True cần param_names/params (void_size_frac) -
        # CVAEDataset đọc từ đây, xem dataset.py.
        n, n_seeds = 2, 1
        seed_classes = np.array(["seed0"], dtype=object)
        seed_onehot = np.zeros((n, n_seeds), dtype=np.float32)
        seed_onehot[:, 0] = 1.0
        params = np.stack([
            np.full(n, 0.4, dtype=np.float32),   # volfrac
            np.full(n, 3.0, dtype=np.float32),   # penal
            np.full(n, 1.5, dtype=np.float32),   # rmin
            np.full(n, 0.1, dtype=np.float32),   # move
            np.full(n, 0.35, dtype=np.float32),  # void_size_frac
        ], axis=1)
        np.savez(
            test_npz,
            images=np.random.default_rng(0).random((n, 64, 64)).astype(np.float32),
            v12=np.array([-0.4, 0.2], dtype=np.float32),
            v21=np.array([-0.4, 0.2], dtype=np.float32),
            volfrac_achieved=np.full(n, 0.4, dtype=np.float32),
            seed_onehot=seed_onehot,
            seed_classes=seed_classes,
            params=params,
            param_names=np.array(["volfrac", "penal", "rmin", "move", "void_size_frac"]),
            nu=np.array([0.25, 0.35], dtype=np.float32),
        )
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))
        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path, condition_dim=8)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=2, n_samples=2, device="cpu", seed=1,
        )
        assert result["condition_dim"] == 8

    def test_fe_verification_uses_per_condition_nu0_not_fixed_default(
        self, tmp_path, monkeypatch,
    ):
        """Bug đã sửa 2026-08-19 (phát hiện lúc chạy thí nghiệm A6 thật lần
        đầu, đo R2(FE) cho checkpoint --include-nu0 trên outputs/phase3_a4/):
        evaluate_density_field() luôn nhận FE_PARAMS['nu'] CỐ ĐỊNH (0.3, xem
        verify_fe.FE_PARAMS), bất kể target nu0 thật của condition khác 0.3
        - hình học ĐÚNG cho nu0 mục tiêu vẫn bị chấm R2(FE) SAI vì verify
        dưới vật liệu khác với vật liệu nó được thiết kế cho. nu0_col (cùng
        quy ước losses.py::real_physics_loss) phải override fe_params['nu']
        đúng theo từng condition trước khi gọi evaluate_density_field()."""
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.4, 0.2], nu_values=[0.25, 0.35])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))
        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6, nu=0.3)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        seen_nu = []

        def fake_evaluate_density_field(img_fe, fe_params):
            seen_nu.append(fe_params["nu"])
            return -0.4, -0.4, None

        monkeypatch.setattr(boe_mod, "evaluate_density_field", fake_evaluate_density_field)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path, condition_dim=4)

        boe_mod.best_of_n(
            str(ckpt_path), n_conditions=2, n_samples=2, device="cpu", seed=1,
        )

        # Cả 2 condition (nu0 thật = 0.25 và 0.35) đều PHẢI xuất hiện trong
        # các lần gọi evaluate_density_field() - KHÔNG được toàn bộ là 0.3
        # (giá trị FE_PARAMS mặc định, dấu hiệu bug chưa sửa). float32 (npz)
        # -> so sánh xấp xỉ, không so bằng tuyệt đối.
        assert any(v == pytest.approx(0.25, abs=1e-5) for v in seen_nu)
        assert any(v == pytest.approx(0.35, abs=1e-5) for v in seen_nu)
        assert not any(v == pytest.approx(0.3, abs=1e-5) for v in seen_nu)

    def test_condition_dim_2_sweep_unaffected_regression(self, tmp_path, monkeypatch):
        """Regression: condition_dim=2 (mặc định, không A6) vẫn hoạt động
        y hệt trước khi có condition_flags_from_dim()."""
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.4, 0.2])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path, condition_dim=2)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=2, n_samples=2, device="cpu", seed=1,
        )
        assert result["condition_dim"] == 2

    def test_data_dir_overrides_phase3_dir(self, tmp_path, monkeypatch):
        """--data-dir (mới, A6) phải trỏ CVAEDataset tới thư mục khác
        PHASE3_DIR - cần thiết vì outputs/phase3/ KHÔNG có field 'nu'."""
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        other_dir = tmp_path / "phase3_a4"
        other_dir.mkdir()
        _write_test_npz(other_dir / "test.npz", v12_values=[-0.4, 0.2],
                         nu_values=[0.25, 0.35])
        # PHASE3_DIR trỏ tới thư mục KHÔNG có test.npz - nếu code lỡ dùng
        # PHASE3_DIR thay vì data_dir, sẽ crash FileNotFoundError ngay.
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path / "does_not_exist"))
        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path, condition_dim=4)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=2, n_samples=2, device="cpu", seed=1,
            data_dir=str(other_dir),
        )
        assert result["condition_dim"] == 4


class TestRequireManufacturable:
    """Roadmap 6.2/6.3: --require-manufacturable should restrict the
    candidate pool (ranking + FE verification) to images that pass
    manufacturability.check_manufacturability(), while still reporting
    v12_first (single-shot baseline) unfiltered - see best_of_n_eval.py."""

    def test_filters_candidate_pool_to_manufacturable_only(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.4])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))
        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        # only even-indexed candidates (0, 2 of 4) count as "manufacturable"
        state = {"n": 0}

        def fake_check_manufacturability(img_bin, min_feature_px=2, periodicity_tol=0.1):
            idx = state["n"]
            state["n"] += 1
            return {"passes_all": idx % 2 == 0}

        monkeypatch.setattr(boe_mod, "check_manufacturability", fake_check_manufacturability)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=1, n_samples=4, device="cpu", seed=1,
            require_manufacturable=True,
        )

        assert result["per_condition"][0]["n_manufacturable"] == 2
        assert result["per_condition"][0]["frac_manufacturable"] == pytest.approx(0.5)
        # n_fe_calls_total only counts oracle-loop FE calls (not the
        # separately-recomputed, always-unfiltered v12_first) - candidate
        # pool restricted to 2 manufacturable candidates -> <= 2.
        assert 0 < result["n_fe_calls_total"] <= 2

    def test_disabled_by_default_uses_full_pool(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.4])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))
        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        def fake_check_manufacturability(img_bin, min_feature_px=2, periodicity_tol=0.1):
            return {"passes_all": False}  # nothing manufacturable

        monkeypatch.setattr(boe_mod, "check_manufacturability", fake_check_manufacturability)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=1, n_samples=4, device="cpu", seed=1,
        )  # require_manufacturable defaults to False

        # all 4 candidates still scored by real FE despite none "passing" -
        # the filter must be a no-op when the flag is off.
        assert result["n_fe_calls_total"] == 4


class TestCompositeScoring:
    """best_of_n() picks the winner by composite_score = w_accuracy*accuracy
    + w_manuf*manuf + w_aesthetic*aesthetic (see docstring) instead of pure
    argmin(|Δv12|). FE, manufacturability AND aesthetic scores are all
    stubbed so the composite math can be checked exactly, independent of
    what an untrained CVAE actually generates."""

    def test_manuf_weight_can_override_pure_accuracy_winner_when_isolated(
        self, tmp_path, monkeypatch,
    ):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.4])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))
        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        # candidate 0 is CLOSER to target -0.4 (pure-accuracy winner: |Δ|=0.01
        # vs 0.30) but NOT manufacturable; candidate 1 is manufacturable.
        fe_values = [-0.41, -0.1]
        fe_call_idx = {"n": 0}

        def fake_evaluate(img_fe, fe_params):
            idx = fe_call_idx["n"]
            fe_call_idx["n"] = idx + 1
            v = fe_values[idx] if idx < len(fe_values) else fe_values[0]
            return v, v, np.eye(3)

        manuf_call_idx = {"n": 0}

        def fake_check_manufacturability(img_bin, min_feature_px=2, periodicity_tol=0.1):
            idx = manuf_call_idx["n"]
            manuf_call_idx["n"] += 1
            ok = idx == 1  # only candidate 1 (index 1) passes
            return {"is_connected": ok, "min_feature_ok": ok, "periodic_ok": ok, "passes_all": ok}

        monkeypatch.setattr(boe_mod, "evaluate_density_field", fake_evaluate)
        monkeypatch.setattr(boe_mod, "check_manufacturability", fake_check_manufacturability)

        # w_manuf=1.0 isolated (w_accuracy=w_aesthetic=0) - proves manuf_score
        # actually drives selection, not just decoration on the report.
        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=1, n_samples=2, device="cpu", seed=1,
            w_accuracy=0.0, w_manuf=1.0, w_aesthetic=0.0,
        )

        c = result["per_condition"][0]
        assert c["v12_best"] == pytest.approx(-0.1)  # candidate 1, NOT the accuracy winner
        assert c["manuf_score"] == pytest.approx(1.0)
        assert c["composite_score"] == pytest.approx(1.0)

    def test_report_includes_component_scores(self, tmp_path, monkeypatch):
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.4])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))
        tiny_fe_params = dict(boe_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(boe_mod, "FE_PARAMS", tiny_fe_params)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=1, n_samples=3, device="cpu", seed=1,
        )

        c = result["per_condition"][0]
        for key in ("accuracy_score", "manuf_score", "aesthetic_score", "composite_score"):
            assert key in c
            assert 0.0 <= c[key] <= 1.0
        expected = (0.6 * c["accuracy_score"] + 0.3 * c["manuf_score"]
                    + 0.1 * c["aesthetic_score"])
        assert c["composite_score"] == pytest.approx(expected, abs=1e-6)
        assert result["w_accuracy"] == 0.6
        assert result["w_manuf"] == 0.3
        assert result["w_aesthetic"] == 0.1
        assert result["condition_dim"] == 2

    def test_default_weights_reduce_to_pure_accuracy_when_manuf_and_aesthetic_tied(
        self, tmp_path, monkeypatch,
    ):
        """If every candidate scores identically on manuf/aesthetic, the
        composite ranking must reduce to the same winner as pure accuracy
        (their contribution is a constant offset, doesn't change argmax)."""
        from pipeline.phase5_cvae import best_of_n_eval as boe_mod

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, v12_values=[-0.5])
        monkeypatch.setattr(boe_mod, "PHASE3_DIR", str(tmp_path))

        fe_values = [0.1, -0.55, -0.9]  # closest to target -0.5 is -0.55
        call_idx = {"n": 0}

        def fake_evaluate_density_field(img_fe, fe_params):
            idx = call_idx["n"]
            call_idx["n"] = idx + 1
            v = fe_values[idx] if idx < len(fe_values) else fe_values[0]
            return v, v, np.eye(3)

        monkeypatch.setattr(boe_mod, "evaluate_density_field", fake_evaluate_density_field)
        monkeypatch.setattr(
            boe_mod, "check_manufacturability",
            lambda img_bin, **kw: {"is_connected": True, "min_feature_ok": True,
                                    "periodic_ok": True, "passes_all": True},
        )
        monkeypatch.setattr(boe_mod, "compute_aesthetic_score", lambda img, img_bin=None: 0.5)

        ckpt_path = tmp_path / "cvae.pt"
        _write_cvae_checkpoint(ckpt_path)

        result = boe_mod.best_of_n(
            str(ckpt_path), n_conditions=1, n_samples=3, device="cpu", seed=123,
        )

        assert result["per_condition"][0]["v12_best"] == pytest.approx(-0.55)


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

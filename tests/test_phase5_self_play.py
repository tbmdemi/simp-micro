"""
Tests for pipeline/phase5_cvae/self_play.py - verify_round (the FE-based
checkpoint scoring function; the orchestration in run()/main() spawns real
training subprocesses and is intentionally out of scope for a unit test).

Imports are lazy inside each test - see tests/conftest.py docstring.
"""
import numpy as np
import pytest


def _write_test_npz(path, n_samples=16, n_seeds=2, v12_range=(-0.7, 0.3),
                     nu_values=None):
    """nu_values: nếu truyền vào (độ dài = n_samples), thêm field 'nu' - cần
    cho test condition_dim ∈ {4,8} (A6, CVAEDataset(include_nu0=True)),
    xem TestVerifyRoundConditionDimFlags."""
    rng = np.random.default_rng(0)
    seed_classes = np.array([f"seed{i}" for i in range(n_seeds)], dtype=object)
    seed_idx = rng.integers(0, n_seeds, size=n_samples)
    seed_onehot = np.zeros((n_samples, n_seeds), dtype=np.float32)
    seed_onehot[np.arange(n_samples), seed_idx] = 1.0
    extra = {}
    if nu_values is not None:
        extra["nu"] = np.array(nu_values, dtype=np.float32)
    np.savez(
        path,
        images=rng.random((n_samples, 64, 64)).astype(np.float32),
        v12=rng.uniform(*v12_range, size=n_samples).astype(np.float32),
        v21=rng.uniform(*v12_range, size=n_samples).astype(np.float32),
        volfrac_achieved=rng.uniform(0.2, 0.6, size=n_samples).astype(np.float32),
        seed_onehot=seed_onehot,
        seed_classes=seed_classes,
        **extra,
    )


class TestVerifyRound:
    def test_returns_expected_keys_and_finite_r2(self, tmp_path, monkeypatch, make_cvae_checkpoint):
        from pipeline.phase5_cvae import self_play as sp_mod

        tiny_fe_params = dict(sp_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(sp_mod, "FE_PARAMS", tiny_fe_params)

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, n_samples=16)
        monkeypatch.setattr(sp_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = make_cvae_checkpoint("cvae.pt", latent_dim=6)

        result = sp_mod.verify_round(
            str(ckpt_path), n_conditions=4, n_per_condition=2,
            device="cpu", seed=123,
        )

        for key in ("r2_fe_v12", "hit_rate", "n_auxetic_targets", "n_samples"):
            assert key in result
        assert result["n_samples"] > 0

    def test_same_seed_is_reproducible(self, tmp_path, monkeypatch, make_cvae_checkpoint):
        """verify_round MUST give identical results across repeated calls
        with the same seed - self-play's round-over-round comparison is
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

        ckpt_path = make_cvae_checkpoint("cvae.pt", latent_dim=6)

        result_a = sp_mod.verify_round(
            str(ckpt_path), n_conditions=3, n_per_condition=2,
            device="cpu", seed=7,
        )
        result_b = sp_mod.verify_round(
            str(ckpt_path), n_conditions=3, n_per_condition=2,
            device="cpu", seed=7,
        )

        assert result_a == result_b

    def test_different_seed_selects_different_conditions(self, tmp_path, monkeypatch, make_cvae_checkpoint):
        from pipeline.phase5_cvae import self_play as sp_mod

        tiny_fe_params = dict(sp_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(sp_mod, "FE_PARAMS", tiny_fe_params)

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, n_samples=40)
        monkeypatch.setattr(sp_mod, "PHASE3_DIR", str(tmp_path))

        ckpt_path = make_cvae_checkpoint("cvae.pt", latent_dim=6)

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


class TestVerifyRoundConditionDimFlags:
    """LIMITATIONS.md mục 22: trước bản vá này, verify_round() luôn dựng
    CVAEDataset() với condition_dim=2 mặc định bất kể checkpoint thật là gì
    (không đọc condition_dim từ checkpoint như best_of_n_eval.py đã làm từ
    A6/A7) - chạy trên checkpoint condition_dim ∈ {4,6,8} sẽ crash (shape
    mismatch lúc concat trong Encoder/Decoder) hoặc âm thầm sai nếu shape
    tình cờ khớp. Cùng root cause + cùng cách vá với
    TestBestOfNConditionDimFlags (tests/test_phase5_best_of_n_eval.py)."""

    def test_condition_dim_4_nu0_does_not_crash(
        self, tmp_path, monkeypatch, make_cvae_checkpoint,
    ):
        from pipeline.phase5_cvae import self_play as sp_mod

        tiny_fe_params = dict(sp_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(sp_mod, "FE_PARAMS", tiny_fe_params)

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, n_samples=16,
                         nu_values=np.linspace(0.22, 0.38, 16))

        ckpt_path = make_cvae_checkpoint(
            "cvae.pt", latent_dim=6, condition_dim=4,
        )

        result = sp_mod.verify_round(
            str(ckpt_path), n_conditions=4, n_per_condition=2,
            device="cpu", seed=1, data_dir=str(tmp_path),
        )
        assert result["n_samples"] > 0

    def test_fe_verification_uses_per_condition_nu0_not_fixed_default(
        self, tmp_path, monkeypatch, make_cvae_checkpoint,
    ):
        """Cùng bug đã sửa ở best_of_n_eval.py (A7, EXPERIMENT_LOG.md
        2026-08-19): evaluate_density_field() KHÔNG được nhận FE_PARAMS['nu']
        cố định cho mọi condition khi checkpoint include_nu0=True - phải
        override đúng theo ν0 thật của từng target (nu0_col)."""
        from pipeline.phase5_cvae import self_play as sp_mod

        tiny_fe_params = dict(sp_mod.FE_PARAMS, nelx=6, nely=6, nu=0.3)
        monkeypatch.setattr(sp_mod, "FE_PARAMS", tiny_fe_params)

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, n_samples=8, nu_values=[0.25, 0.35] * 4)

        seen_nu = []

        def fake_evaluate_density_field(img_fe, fe_params):
            seen_nu.append(fe_params["nu"])
            return -0.4, -0.4, None

        monkeypatch.setattr(sp_mod, "evaluate_density_field",
                            fake_evaluate_density_field)

        ckpt_path = make_cvae_checkpoint(
            "cvae.pt", latent_dim=6, condition_dim=4,
        )

        sp_mod.verify_round(
            str(ckpt_path), n_conditions=4, n_per_condition=1,
            device="cpu", seed=1, data_dir=str(tmp_path),
        )

        # Cả 2 giá trị ν0 thật (0.25 và 0.35) PHẢI xuất hiện - KHÔNG được
        # toàn bộ là 0.3 (FE_PARAMS mặc định, dấu hiệu bug chưa sửa).
        assert any(v == pytest.approx(0.25, abs=1e-5) for v in seen_nu)
        assert any(v == pytest.approx(0.35, abs=1e-5) for v in seen_nu)
        assert not any(v == pytest.approx(0.3, abs=1e-5) for v in seen_nu)

    def test_condition_dim_2_regression(
        self, tmp_path, monkeypatch, make_cvae_checkpoint,
    ):
        """Regression: condition_dim=2 (mặc định, không A6) vẫn hoạt động y
        hệt trước khi có bản vá này."""
        from pipeline.phase5_cvae import self_play as sp_mod

        tiny_fe_params = dict(sp_mod.FE_PARAMS, nelx=6, nely=6)
        monkeypatch.setattr(sp_mod, "FE_PARAMS", tiny_fe_params)

        test_npz = tmp_path / "test.npz"
        _write_test_npz(test_npz, n_samples=16)

        ckpt_path = make_cvae_checkpoint(
            "cvae.pt", latent_dim=6, condition_dim=2,
        )

        result = sp_mod.verify_round(
            str(ckpt_path), n_conditions=4, n_per_condition=2,
            device="cpu", seed=1, data_dir=str(tmp_path),
        )
        assert result["n_samples"] > 0

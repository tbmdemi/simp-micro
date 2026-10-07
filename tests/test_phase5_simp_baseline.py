"""Tests cho baseline SIMP chạy từ đầu (plan.md v3 P1.6a,
pipeline/phase5_cvae/benchmark_simp_baseline.py).

Lưới nhỏ 16×16 để chạy nhanh. Import lazily (landmine phase4/phase5 cùng tên
module trong sys.modules - module dùng bare import `from verify_fe ...`).
"""

import importlib

import numpy as np
import pytest


@pytest.fixture
def simp_bench():
    """Module benchmark SIMP, import lazily."""
    return importlib.import_module(
        "pipeline.phase5_cvae.benchmark_simp_baseline"
    )


def _perturbed_design(mod, n: int = 16) -> np.ndarray:
    """Seed hourglass + nhiễu nhỏ, để gradient không rơi vào vùng bão hòa."""
    x = mod.initial_design("hourglass", n, n)
    rng = np.random.default_rng(0)
    return np.clip(x + 0.05 * rng.standard_normal(x.size), 0.01, 1.0)


@pytest.mark.parametrize(
    "key,dkey",
    [
        ("c", "dc"),
        ("g11", "dg11"),
        ("g22", "dg22"),
        ("vol", "dvol"),
    ],
)
def test_gradients_match_finite_difference(simp_bench, key, dkey):
    """Gradient giải tích (qua S=Q⁻¹, projection, filter) khớp sai phân
    trung tâm - objective/ràng buộc sai gradient làm MMA hội tụ sai mà
    không báo lỗi."""
    prob = simp_bench.TargetProblem((-0.5, -0.4), nelx=16, nely=16)
    prob.beta = 4.0
    x = _perturbed_design(simp_bench)
    analytic = prob.evaluate(x)[dkey]
    rng = np.random.default_rng(1)
    for j in rng.choice(x.size, 5, replace=False):
        h = 1e-6
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        fd = (prob.evaluate(xp)[key] - prob.evaluate(xm)[key]) / (2 * h)
        assert analytic[j] == pytest.approx(fd, rel=1e-3, abs=1e-9)


def test_initial_design_respects_volume_bound(simp_bench):
    """Seed đặc hơn volfrac_max phải được co về khả thi (bug smoke-test:
    circle ở void 0,315 có thể tích ~0,92 → MMA thoát về ô gần đặc)."""
    vmax = simp_bench.SIMP_PARAMS["volfrac_max"]
    for seed in simp_bench.DEFAULT_SEEDS:
        x0 = simp_bench.initial_design(seed, 50, 50)
        # Dung sai 1e-3: clip về X_MIN sau khi co làm thể tích nhích lên
        # ~1e-4; ràng buộc thật đặt trên x̂ (MMA tự kéo về), không trên x0.
        assert x0.mean() <= vmax + 1e-3
        assert x0.min() >= simp_bench.X_MIN


def test_run_simp_target_reduces_objective(simp_bench, monkeypatch):
    """Vài eval MMA mỗi mức β phải giảm objective so với điểm khởi tạo,
    và kết quả có đủ trường verify + đếm FE đúng (tối ưu + 1 verify)."""
    monkeypatch.setattr(simp_bench, "BETAS", (1.0, 4.0))
    target = (-0.3, -0.3)
    prob = simp_bench.TargetProblem(target, nelx=16, nely=16)
    c0 = prob.evaluate(simp_bench.initial_design("hourglass", 16, 16))["c"]

    res = simp_bench.run_simp_target(target, "hourglass", 10, nelx=16, nely=16)
    assert res["raw_objective"] < c0
    assert 2 <= res["n_fe"] <= 2 * 10 + 2
    assert set(res) >= {"v12", "v21", "raw_v12", "manufacturable", "image"}
    assert np.asarray(res["image"]).shape == (16, 16)


def test_pair_err_handles_failed_verify(simp_bench):
    """FE verify lỗi (None) → sai số vô cùng, không bao giờ được chọn làm
    best-of-seeds."""
    assert simp_bench._pair_err(None, None, (-0.5, -0.5)) == float("inf")
    assert simp_bench._pair_err(-0.4, -0.6, (-0.5, -0.5)) == pytest.approx(0.2)

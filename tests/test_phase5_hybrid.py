"""Tests cho thí nghiệm lai cVAE → SIMP (plan.md v3 P1.7,
pipeline/phase5_cvae/benchmark_hybrid.py).

Import lazily (landmine phase4/phase5 cùng tên module trong sys.modules).
"""

import importlib

import numpy as np
import pytest


@pytest.fixture
def hyb():
    """Module benchmark lai, import lazily."""
    return importlib.import_module("pipeline.phase5_cvae.benchmark_hybrid")


def _cond(accept: bool) -> dict:
    """1 condition giả kiểu JSON refine có ảnh (64×64)."""
    base = np.zeros((64, 64), dtype=np.uint8)
    ref = np.ones((64, 64), dtype=np.uint8)
    return {
        "target_v12": -0.5,
        "target_v21": -0.4,
        "guarded_accept": accept,
        "baseline_image": base.tolist(),
        "refined_image": ref.tolist(),
        "baseline_v12": -0.1,
        "baseline_v21": -0.1,
        "refined_v12": -0.49,
        "refined_v21": -0.41,
        "baseline_manufacturable": False,
        "refined_manufacturable": True,
        "n_fe_calls_baseline": 30,
        "n_fe_calls_refine": 77,
    }


def test_cvae_start_follows_guarded_choice(hyb):
    """Điểm khởi tạo lai phải là đúng thiết kế cVAE cuối (guarded)."""
    s = hyb.cvae_start(_cond(True))
    assert s["design50"].shape == (50, 50) and s["design50"].min() == 1
    assert (s["v12"], s["manufacturable"], s["n_fe"]) == (-0.49, True, 107)
    s = hyb.cvae_start(_cond(False))
    assert s["design50"].max() == 0 and s["v12"] == -0.1


def test_feasible_x0_respects_bounds_and_volume(hyb):
    """Ảnh đặc → co về volfrac_max, mọi pixel trong [X_MIN, 1]."""
    x0 = hyb.feasible_x0(np.ones((50, 50)))
    assert x0.shape == (2500,)
    assert x0.mean() == pytest.approx(hyb.sb.SIMP_PARAMS["volfrac_max"])
    x0 = hyb.feasible_x0(np.zeros((50, 50)))
    assert x0.min() == pytest.approx(hyb.sb.X_MIN)


def test_select_guarded_keeps_better(hyb):
    """Giữ lai chỉ khi sai số cặp verify nhỏ hơn; lai lỗi FE → giữ cVAE."""
    t = (-0.5, -0.4)
    start = {
        "v12": -0.45,
        "v21": -0.4,
        "manufacturable": False,
        "E_x": 0.05,
        "E_y": 0.05,
    }
    better = {
        "v12": -0.5,
        "v21": -0.4,
        "manufacturable": True,
        "E_x": 0.06,
        "E_y": 0.06,
    }
    worse = dict(better, v12=-0.2)
    failed = dict(better, v12=None, v21=None)
    assert hyb.select_guarded(start, better, t)["accepted"]
    assert hyb.select_guarded(start, better, t)["manufacturable"]
    assert not hyb.select_guarded(start, worse, t)["accepted"]
    assert not hyb.select_guarded(start, failed, t)["accepted"]


def test_evaluate_criteria_thresholds(hyb):
    """Lai bằng SIMP về sai số, đủ chế tạo, rẻ, không suy biến → đạt cả 4;
    1 ô suy biến trên 50 (2%) → trượt tiêu chí 4."""
    targets = [(-0.5, -0.4)] * 50
    simp = [{"v12": -0.51, "v21": -0.4}] * 50
    final = [
        {
            "v12": -0.51,
            "v21": -0.4,
            "manufacturable": True,
            "E_x": 0.05,
            "E_y": 0.05,
            "accepted": True,
        }
    ] * 50
    ok = hyb.evaluate_criteria(final, simp, targets, [168] * 50)
    assert ok["all_pass"]
    bad = list(final)
    bad[0] = dict(final[0], E_x=1e-6)
    res = hyb.evaluate_criteria(bad, simp, targets, [168] * 50)
    assert not res["pass"]["c4_nondegenerate"] and not res["all_pass"]
    res = hyb.evaluate_criteria(final, simp, targets, [400] * 50)
    assert not res["pass"]["c3_cost"]


def test_run_simp_target_accepts_x0_and_betas(hyb, monkeypatch):
    """x0 + betas tùy chỉnh: dùng đúng điểm khởi tạo và số mức β."""
    sb = hyb.sb
    x0 = sb.initial_design("hourglass", 16, 16)
    res = sb.run_simp_target(
        (-0.3, -0.3), "cvae", 3, nelx=16, nely=16, x0=x0, betas=(8.0, 64.0)
    )
    assert res["seed"] == "cvae"
    assert res["n_fe"] <= 2 * 3 + 2

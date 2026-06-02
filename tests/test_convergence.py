"""
Tests for the new ConvergenceChecker (cửa sổ trượt + min_iter).
"""

import pytest
from simp.core.convergence import ConvergenceChecker


class TestConvergenceChecker:
    """Test suite for the new ConvergenceChecker (window-based)."""

    def test_default_init(self):
        cc = ConvergenceChecker()
        assert cc.tol_change == 0.01
        assert cc.tol_obj == 0.05
        assert cc.window_change == 5
        assert cc.window_obj == 20
        assert cc.min_iter == 10
        assert not cc.converged
        assert cc.obj_changes == []
        assert cc.change_history == []

    def test_custom_init(self):
        cc = ConvergenceChecker(
            tol_change=0.001, tol_obj=0.01,
            window_change=3, window_obj=10, min_iter=5,
        )
        assert cc.tol_change == 0.001
        assert cc.tol_obj == 0.01
        assert cc.window_change == 3
        assert cc.window_obj == 10
        assert cc.min_iter == 5

    def test_should_stop_max_iter(self):
        """Luôn dừng khi loop >= max_iter (ưu tiên cao nhất)."""
        cc = ConvergenceChecker(min_iter=0)
        assert cc.should_stop(1.0, 1.0, 0.0, 200, 200) is True

    def test_should_stop_below_min_iter(self):
        """Không dừng nếu loop < min_iter dù change=0."""
        cc = ConvergenceChecker(tol_change=0.01, min_iter=100)
        # change=0 -> design "hội tụ" nhưng chưa đủ vòng
        assert cc.should_stop(0.0, 1.0, 0.0, 5, 200) is False

    def test_design_convergence_window(self):
        """Hội tụ thiết kế sau window_change vòng liên tiếp."""
        cc = ConvergenceChecker(
            tol_change=0.01, min_iter=0,
            window_change=3, window_obj=200,
        )
        # Mô phỏng 3 vòng liên tiếp change < 0.01
        for i in range(3):
            stopped = cc.should_stop(0.005, float(i), float(i - 1) if i > 0 else 0.0, i, 200)
            if i < 2:
                assert not stopped, f"Không nên dừng ở vòng {i}"
        assert cc.should_stop(0.005, 3.0, 2.0, 3, 200) is True
        assert cc.converged is True

    def test_design_not_converged_yet(self):
        """Chưa đủ cửa sổ → không hội tụ."""
        cc = ConvergenceChecker(
            tol_change=0.01, min_iter=0,
            window_change=5, window_obj=200,
        )
        for i in range(4):
            assert not cc.should_stop(0.005, float(i), float(i - 1) if i > 0 else 0.0, i, 200)
        assert not cc.converged

    def test_design_change_above_tol_breaks_window(self):
        """Một vòng change > tol làm reset hiệu quả cửa sổ."""
        cc = ConvergenceChecker(
            tol_change=0.01, min_iter=0,
            window_change=3, window_obj=200,
        )
        # 2 vòng tốt
        cc.should_stop(0.005, 0.0, 0.0, 0, 200)
        cc.should_stop(0.005, 1.0, 0.0, 1, 200)
        # Vòng thứ 3: change lớn → phá vỡ cửa sổ
        cc.should_stop(0.5, 2.0, 1.0, 2, 200)
        # Vòng 4 dù tốt nhưng chưa đủ 3 liên tiếp
        assert not cc.should_stop(0.005, 3.0, 2.0, 3, 200)

    def test_objective_stability_convergence(self):
        """Objective ổn định + change nhỏ → hội tụ."""
        cc = ConvergenceChecker(
            tol_change=0.02, tol_obj=0.05, min_iter=0,
            window_change=3, window_obj=20,
        )
        # 20 vòng obj thay đổi nhỏ + 3 vòng change nhỏ
        for i in range(30):
            obj_val = 1.0 + (0.01 if i > 5 else 0.1)
            prev_obj = 1.0 + (0.01 if i > 6 else 0.1)
            change_val = 0.005 if i > 2 else 0.05
            stopped = cc.should_stop(change_val, obj_val, prev_obj, i, 200)
        assert stopped is True

    def test_inf_prev_obj(self):
        """prev_obj = inf không gây lỗi."""
        cc = ConvergenceChecker(min_iter=0)
        assert not cc.should_stop(0.5, 1.0, float('inf'), 0, 200)

    def test_reset(self):
        """Reset xóa toàn bộ trạng thái."""
        cc = ConvergenceChecker(min_iter=0, window_change=3, window_obj=3)
        for i in range(3):
            cc.should_stop(0.005, float(i), float(i - 1) if i > 0 else 0.0, i, 200)
        assert len(cc.change_history) > 0
        assert len(cc.obj_changes) > 0
        cc.reset()
        assert cc.change_history == []
        assert cc.obj_changes == []
        assert cc.converged is False

    def test_is_converged_flag(self):
        cc = ConvergenceChecker(min_iter=0, window_change=3, window_obj=200)
        for i in range(3):
            cc.should_stop(0.005, float(i), float(i - 1) if i > 0 else 0.0, i, 200)
        assert cc.is_converged() is True

    def test_record_objective_change_inf(self):
        """record_objective_change với inf prev_obj trả False."""
        cc = ConvergenceChecker()
        assert cc.record_objective_change(1.0, float('inf')) is False

    def test_no_premature_convergence(self):
        """Không hội tụ nếu min_iter chưa đạt dù design ổn."""
        cc = ConvergenceChecker(
            tol_change=0.01, min_iter=50,
            window_change=3, window_obj=200,
        )
        for i in range(49):
            stopped = cc.should_stop(0.005, float(i), float(i - 1) if i > 0 else 0.0, i, 200)
            assert not stopped, f"Không nên dừng trước min_iter (vòng {i})"
        # Vòng 50: đã đủ min_iter + cửa sổ → hội tụ
        assert cc.should_stop(0.005, 50.0, 49.0, 50, 200) is True
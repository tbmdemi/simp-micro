"""
Tests for ConvergenceChecker.
"""

import pytest
from simp.core.convergence import ConvergenceChecker


class TestConvergenceChecker:
    """Test suite for ConvergenceChecker."""

    def test_default_init(self):
        """Test default initialization."""
        cc = ConvergenceChecker()
        assert cc.tol_change == 0.01
        assert cc.tol_obj == 0.05
        assert cc.window_size == 20
        assert not cc.converged
        assert cc.obj_changes == []

    def test_custom_init(self):
        """Test custom initialization."""
        cc = ConvergenceChecker(tol_change=0.001, tol_obj=0.01, window_size=10)
        assert cc.tol_change == 0.001
        assert cc.tol_obj == 0.01
        assert cc.window_size == 10

    def test_check_design_change_converged(self):
        """Test design change below tolerance."""
        cc = ConvergenceChecker(tol_change=0.01)
        assert cc.check_design_change(0.005) is True

    def test_check_design_change_not_converged(self):
        """Test design change above tolerance."""
        cc = ConvergenceChecker(tol_change=0.01)
        assert cc.check_design_change(0.05) is False

    def test_check_design_change_equal(self):
        """Test design change equal to tolerance."""
        cc = ConvergenceChecker(tol_change=0.01)
        assert cc.check_design_change(0.01) is True

    def test_objective_stability_not_enough_data(self):
        """Test objective stability with insufficient data."""
        cc = ConvergenceChecker(tol_obj=0.05, window_size=5)
        assert cc.check_objective_stability(1.0, 1.05) is False
        assert len(cc.obj_changes) == 1

    def test_objective_stability_converged(self):
        """Test objective stability with full window."""
        cc = ConvergenceChecker(tol_obj=0.05, window_size=3)
        # Fill the window with small changes
        for i in range(3):
            cc.check_objective_stability(1.0 + i * 0.01, 1.0 + (i - 1) * 0.01)
        assert cc.converged is True

    def test_objective_stability_not_converged(self):
        """Test objective stability with large changes."""
        cc = ConvergenceChecker(tol_obj=0.05, window_size=3)
        for i in range(3):
            cc.check_objective_stability(1.0 + i * 0.1, 1.0 + (i - 1) * 0.1)
        assert cc.converged is False

    def test_should_stop_max_iter(self):
        """Test should_stop returns True at max iterations."""
        cc = ConvergenceChecker()
        assert cc.should_stop(1.0, 1.0, 0.0, 200, 200) is True

    def test_should_stop_design_change(self):
        """Test should_stop returns True when design converges."""
        cc = ConvergenceChecker(tol_change=0.01)
        assert cc.should_stop(0.005, 1.0, 0.0, 10, 200) is True
        assert cc.converged is True

    def test_should_stop_not_converged(self):
        """Test should_stop returns False when not converged."""
        cc = ConvergenceChecker(tol_change=0.01, tol_obj=0.05, window_size=20)
        assert cc.should_stop(0.5, 1.0, 0.0, 10, 200) is False
        assert cc.converged is False

    def test_reset(self):
        """Test reset clears state."""
        cc = ConvergenceChecker(window_size=3)
        for i in range(3):
            cc.check_objective_stability(1.0, 0.99)
        assert len(cc.obj_changes) == 3
        cc.reset()
        assert cc.obj_changes == []
        assert cc.converged is False

    def test_inf_prev_obj(self):
        """Test that inf prev_obj doesn't cause errors."""
        cc = ConvergenceChecker()
        assert cc.check_objective_stability(1.0, float('inf')) is False

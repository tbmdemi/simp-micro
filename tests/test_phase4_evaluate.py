"""
Tests for pipeline/phase4_surrogate/evaluate.py — r2_score.

Import is done lazily inside each test (not at module top) because
evaluate.py does `sys.path.insert(...)` + bare `from dataset import X` /
`from model import X` at import time; see tests/conftest.py's
`_isolate_pipeline_bare_imports` docstring for why that must not run at
collection time alongside phase5_cvae's same-named siblings.
"""
import numpy as np
import pytest


class TestR2Score:
    def test_perfect_fit(self):
        from pipeline.phase4_surrogate.evaluate import r2_score
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(y, y) == 1.0

    def test_mean_prediction_gives_r2_zero(self):
        from pipeline.phase4_surrogate.evaluate import r2_score
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert r2_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)

    def test_worse_than_mean_is_negative(self):
        from pipeline.phase4_surrogate.evaluate import r2_score
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([10.0, -10.0, 20.0, -20.0])
        assert r2_score(y_true, y_pred) < 0

    def test_zero_variance_target_returns_nan(self):
        from pipeline.phase4_surrogate.evaluate import r2_score
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 4.0, 6.0])
        assert np.isnan(r2_score(y_true, y_pred))

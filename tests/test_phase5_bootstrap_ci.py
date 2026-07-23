"""
Tests for pipeline/phase5_cvae/bootstrap_ci.py — adds confidence intervals on
top of best_of_n_eval.py's point-estimate R2/hit_rate, which are computed on
very small n (19-24 conditions, sometimes as few as 3 for the
--require-manufacturable N=1500 result). No bare sys.path/torch imports here
(pure numpy/json/math), so none of the phase4/phase5 bare-import landmine
concerns from tests/conftest.py apply — plain top-level imports are safe.
"""
import json
import math

import numpy as np
import pytest

from pipeline.phase5_cvae.bootstrap_ci import (
    r2_score, bootstrap_r2, wilson_ci, hit_rate_ci, analyze_file,
)


def _cond(target, best, is_auxetic=True):
    return {"target_v12": target, "v12_best": best, "is_auxetic_target": is_auxetic}


class TestR2Score:
    def test_perfect_prediction_gives_r2_one(self):
        targets = np.array([-0.5, -0.3, -0.1, 0.2])
        assert r2_score(targets, targets.copy()) == pytest.approx(1.0)

    def test_predicting_the_mean_gives_r2_zero(self):
        targets = np.array([-0.5, -0.3, -0.1, 0.2])
        preds = np.full_like(targets, targets.mean())
        assert r2_score(targets, preds) == pytest.approx(0.0)

    def test_constant_targets_returns_nan_instead_of_dividing_by_zero(self):
        targets = np.array([-0.4, -0.4, -0.4])
        preds = np.array([-0.3, -0.5, -0.4])
        assert math.isnan(r2_score(targets, preds))


class TestBootstrapR2:
    def test_point_estimate_matches_plain_r2_score(self):
        per_condition = [
            _cond(-0.5, -0.45), _cond(-0.3, -0.1), _cond(-0.1, 0.05),
            _cond(0.2, 0.1), _cond(-0.6, -0.7),
        ]
        targets = np.array([c["target_v12"] for c in per_condition])
        preds = np.array([c["v12_best"] for c in per_condition])
        expected = r2_score(targets, preds)

        stats = bootstrap_r2(per_condition, n_boot=500, seed=0)
        assert stats["r2_point_estimate"] == pytest.approx(expected)
        assert stats["n_conditions"] == 5

    def test_ci_bounds_bracket_point_estimate_for_noisy_but_correlated_data(self):
        rng = np.random.default_rng(42)
        targets = rng.uniform(-0.8, 0.2, size=20)
        preds = targets + rng.normal(0, 0.1, size=20)
        per_condition = [_cond(float(t), float(p)) for t, p in zip(targets, preds)]

        stats = bootstrap_r2(per_condition, n_boot=2000, seed=1)
        assert stats["r2_ci95_lo"] <= stats["r2_point_estimate"] <= stats["r2_ci95_hi"]

    def test_small_n_gives_wide_ci_reflecting_low_confidence(self):
        """Regression guard for the actual finding that motivated this script:
        n=19-24 conditions is small enough that the 95% CI should span a
        large fraction of the [-inf, 1] R2 range, not look artificially
        tight. This mirrors outputs/phase5/self_play/best_of_n_result.json
        (R2=0.5955 point estimate, CI observed to be roughly [0.0, 0.84])."""
        rng = np.random.default_rng(7)
        n = 19
        targets = rng.uniform(-0.8, -0.05, size=n)
        preds = targets + rng.normal(0, 0.15, size=n)
        per_condition = [_cond(float(t), float(p)) for t, p in zip(targets, preds)]

        stats = bootstrap_r2(per_condition, n_boot=5000, seed=2)
        ci_width = stats["r2_ci95_hi"] - stats["r2_ci95_lo"]
        assert ci_width > 0.3

    def test_same_seed_is_reproducible(self):
        per_condition = [_cond(-0.5, -0.45), _cond(-0.3, -0.1), _cond(-0.1, 0.05)]
        a = bootstrap_r2(per_condition, n_boot=300, seed=99)
        b = bootstrap_r2(per_condition, n_boot=300, seed=99)
        assert a["r2_ci95_lo"] == b["r2_ci95_lo"]
        assert a["r2_ci95_hi"] == b["r2_ci95_hi"]


class TestWilsonCI:
    def test_matches_known_wilson_interval_for_19_of_19(self):
        # Hand-checkable reference value (z=1.96, n=19, hits=19).
        lo, hi = wilson_ci(hits=19, n=19)
        assert hi == pytest.approx(1.0)
        assert lo == pytest.approx(0.8318, abs=1e-3)

    def test_all_hits_ci_is_not_degenerate(self):
        """The whole reason to use Wilson instead of a naive percentile
        bootstrap here: resampling an all-1s vector always yields all-1s, so
        a percentile bootstrap on a 100% hit rate collapses to [1.0, 1.0]
        regardless of n, which understates uncertainty at small n. Wilson
        must not do that."""
        lo, hi = wilson_ci(hits=19, n=19)
        assert lo < 1.0

    def test_fifty_fifty_at_large_n_is_roughly_symmetric(self):
        lo, hi = wilson_ci(hits=50, n=100)
        assert lo == pytest.approx(1 - hi, abs=0.02)

    def test_zero_n_returns_nan(self):
        lo, hi = wilson_ci(hits=0, n=0)
        assert math.isnan(lo) and math.isnan(hi)


class TestHitRateCI:
    def test_only_counts_auxetic_target_conditions(self):
        per_condition = [
            _cond(-0.5, -0.4, is_auxetic=True),   # hit
            _cond(-0.3, 0.1, is_auxetic=True),    # miss
            _cond(0.4, 0.5, is_auxetic=False),    # excluded from denominator
        ]
        stats = hit_rate_ci(per_condition)
        assert stats["n_auxetic_conditions"] == 2
        assert stats["hit_rate_point_estimate"] == pytest.approx(0.5)

    def test_perfect_hit_rate_ci_is_not_one_to_one(self):
        per_condition = [_cond(-0.1 * i - 0.05, -0.05, is_auxetic=True) for i in range(19)]
        stats = hit_rate_ci(per_condition)
        assert stats["hit_rate_point_estimate"] == pytest.approx(1.0)
        assert stats["hit_rate_wilson_ci95_lo"] < 1.0


class TestAnalyzeFile:
    def test_reads_real_best_of_n_json_shape(self, tmp_path):
        payload = {
            "n_conditions": 3,
            "per_condition": [
                _cond(-0.5, -0.45, is_auxetic=True),
                _cond(-0.3, -0.1, is_auxetic=True),
                _cond(0.4, 0.5, is_auxetic=False),
            ],
        }
        path = tmp_path / "result.json"
        path.write_text(json.dumps(payload))

        stats = analyze_file(str(path), n_boot=200, seed=0)
        assert stats["n_conditions"] == 3
        assert stats["n_auxetic_conditions"] == 2
        assert set(stats) >= {
            "r2_point_estimate", "r2_ci95_lo", "r2_ci95_hi",
            "hit_rate_point_estimate", "hit_rate_wilson_ci95_lo",
            "hit_rate_wilson_ci95_hi",
        }

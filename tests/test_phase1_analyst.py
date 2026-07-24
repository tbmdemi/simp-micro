"""
Tests for pipeline/phase1_screening/analyst.py — Spearman aggregation and
sample counting. `compute_correlations`'s `best_obj_value = df['obj_value'].min()`
is a direct regression guard for the historical bug logged in
EXPERIMENT_LOG.md ("Dùng `max` thay vì `min` trong aggregate_correlations.py"
— picking max selected the *worst* objective value instead of the best,
since every objective in this pipeline is a minimization).
"""
import json

import pandas as pd
import pytest

from pipeline.phase1_screening.analyst import (
    compute_correlations,
    count_success_converged,
    discover_seeds,
)


class TestComputeCorrelations:
    def test_best_obj_value_is_the_minimum_not_the_maximum(self):
        """Regression guard: all objectives here are minimized (more
        negative Q12 = better auxetic behavior), so 'best' must be min()."""
        df = pd.DataFrame({
            "volfrac": [0.3, 0.4, 0.5, 0.6, 0.7],
            "obj_value": [-0.2, -0.8, -0.1, -0.5, 0.1],
        })
        _, _, _, best, n_valid = compute_correlations(df, ["volfrac"], "auxetic")
        assert best == pytest.approx(-0.8)
        assert n_valid == 5

    def test_top3_sorted_by_absolute_correlation_descending(self):
        rng_df = pd.DataFrame({
            "strong": [1, 2, 3, 4, 5],
            "weak": [3, 1, 4, 1, 5],
            "obj_value": [1, 2, 3, 4, 5],  # perfectly monotonic with "strong"
        })
        corr_list, pval_list, top3, best, n_valid = compute_correlations(
            rng_df, ["strong", "weak"], "auxetic"
        )
        assert top3[0][0] == "strong"
        assert abs(top3[0][1]) >= abs(top3[1][1])
        assert len(corr_list) == 2 and len(pval_list) == 2

    def test_returns_plain_python_floats_not_numpy_scalars(self):
        df = pd.DataFrame({"volfrac": [0.3, 0.5, 0.7], "obj_value": [-0.1, -0.3, -0.5]})
        corr_list, pval_list, _, best, _ = compute_correlations(df, ["volfrac"], "auxetic")
        assert isinstance(corr_list[0], float)
        assert isinstance(pval_list[0], float)
        assert isinstance(best, float)


class TestCountSuccessConverged:
    def test_counts_from_csv_path_config(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        pd.DataFrame({
            "success": [True, True, False],
            "converged": [True, False, False],
        }).to_csv(csv_path, index=False)

        n_samples, n_success, n_converged = count_success_converged({"csv_path": str(csv_path)})
        assert (n_samples, n_success, n_converged) == (3, 2, 1)

    def test_counts_from_sample_dirs_config_via_metadata_json(self, tmp_path):
        dirs = []
        for i, (success, converged) in enumerate([(True, True), (True, False), (False, False)]):
            d = tmp_path / f"sample_{i}"
            d.mkdir()
            (d / "metadata.json").write_text(json.dumps({"success": success, "converged": converged}))
            dirs.append(str(d))

        n_samples, n_success, n_converged = count_success_converged({"sample_dirs": dirs})
        assert (n_samples, n_success, n_converged) == (3, 2, 1)

    def test_missing_metadata_json_defaults_to_success_and_converged(self, tmp_path):
        d = tmp_path / "sample_0"
        d.mkdir()
        n_samples, n_success, n_converged = count_success_converged({"sample_dirs": [str(d)]})
        assert (n_samples, n_success, n_converged) == (1, 1, 1)

    def test_unknown_config_shape_returns_zeros(self):
        assert count_success_converged({}) == (0, 0, 0)


class TestDiscoverSeeds:
    def test_lists_only_directories_alphabetically(self, tmp_path):
        (tmp_path / "circle").mkdir()
        (tmp_path / "square").mkdir()
        (tmp_path / "hexagonal").mkdir()
        (tmp_path / "some_file.json").write_text("{}")

        seeds = discover_seeds(str(tmp_path))
        assert seeds == ["circle", "hexagonal", "square"]

    def test_skips_underscore_prefixed_entries(self, tmp_path):
        (tmp_path / "circle").mkdir()
        (tmp_path / "_all_correlations_dir").mkdir()

        seeds = discover_seeds(str(tmp_path))
        assert seeds == ["circle"]

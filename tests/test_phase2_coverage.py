"""
Tests for pipeline/phase2_multi_batch/coverage.py — property-space
coverage analysis: sparse-region detection and the coverage_report used
by adaptive.py to decide stop/refine/expand.
"""
import numpy as np
import pytest

from pipeline.phase2_multi_batch.coverage import (
    _compute_spatial_coverage,
    _to_array,
    coverage_report,
    find_sparse_regions,
)


def _make_results(n, v12_range=(-0.8, -0.1), v21_range=(-0.8, -0.1),
                   obj_range=(-0.3, -0.05), seed=0, success=True):
    rng = np.random.default_rng(seed)
    return [
        {
            "success": success,
            "v12": float(rng.uniform(*v12_range)),
            "v21": float(rng.uniform(*v21_range)),
            "obj_value": float(rng.uniform(*obj_range)),
        }
        for _ in range(n)
    ]


class TestToArray:
    def test_drops_none_and_nan(self):
        results = [{"v12": -0.5}, {"v12": None}, {"v12": float("nan")}, {"v12": -0.3}]
        arr = _to_array(results, "v12")
        assert list(arr) == [-0.5, -0.3]

    def test_missing_field_treated_as_none(self):
        results = [{"v12": -0.5}, {"other": 1}]
        arr = _to_array(results, "v12")
        assert list(arr) == [-0.5]


class TestFindSparseRegions:
    def test_too_few_points_returns_note(self):
        results = _make_results(5)
        regions = find_sparse_regions(results)
        assert len(regions) == 1
        assert "note" in regions[0]

    def test_no_valid_data_returns_empty(self):
        results = [{"success": True} for _ in range(20)]
        assert find_sparse_regions(results) == []

    def test_uniform_coverage_has_few_or_no_sparse_bins(self):
        rng = np.random.default_rng(0)
        results = [
            {"v12": float(v12), "v21": float(v21)}
            for v12 in np.linspace(-0.8, -0.1, 8)
            for v21 in np.linspace(-0.8, -0.1, 8)
        ]
        regions = find_sparse_regions(results, n_regions=5, density_threshold=0.1)
        # A regular grid should not flag most bins as sparse relative to the max.
        assert len(regions) <= 5

    def test_clustered_data_flags_the_empty_side_as_sparse(self):
        # All points clustered in one corner -> bins near the opposite
        # corner should come back with very low counts.
        results = _make_results(60, v12_range=(-0.85, -0.75), v21_range=(-0.85, -0.75))
        # add a couple of far-away points so the bounds actually span a gap
        results += [{"v12": -0.15, "v21": -0.15, "success": True} for _ in range(2)]
        regions = find_sparse_regions(results, n_regions=10, density_threshold=0.5)
        assert len(regions) > 0
        assert all(r["n_points"] < 60 for r in regions)


class TestComputeSpatialCoverage:
    def test_fewer_than_two_points_gives_zero_coverage(self):
        frac, detail = _compute_spatial_coverage([{"success": True, "obj_value": -0.1, "v12": -0.5, "v21": -0.5}])
        assert frac == 0.0

    def test_full_grid_gives_full_coverage(self):
        results = [
            {"success": True, "obj_value": o, "v12": v12, "v21": v21}
            for v12 in np.linspace(-0.8, -0.1, 4)
            for v21 in np.linspace(-0.8, -0.1, 4)
            for o in [-0.2]
        ]
        frac, detail = _compute_spatial_coverage(results, dims=("v12", "v21"), n_bins_per_dim=4)
        assert frac == pytest.approx(1.0)
        assert detail["n_bins_occupied"] == detail["n_bins_total"]


class TestCoverageReport:
    def test_interpretation_thresholds(self):
        # High coverage: dense regular grid across full declared range.
        dense = [
            {"success": True, "obj_value": -0.2, "v12": v12, "v21": v21}
            for v12 in np.linspace(-0.8, -0.1, 10)
            for v21 in np.linspace(-0.8, -0.1, 10)
        ]
        report = coverage_report(dense, dims=("v12", "v21"))
        assert report["spatial_coverage_pct"] > 70
        assert "HIGH" in report["interpretation"]

    def test_no_valid_results_short_circuits(self):
        results = [{"success": False} for _ in range(10)]
        report = coverage_report(results)
        assert report["n_valid"] == 0
        assert report["sparsity"] == {}

    def test_success_rate_reflects_failures(self):
        results = _make_results(8, success=True) + [{"success": False} for _ in range(2)]
        report = coverage_report(results, dims=("v12", "v21"))
        assert report["n_total"] == 10
        assert report["success_rate_pct"] == 80.0

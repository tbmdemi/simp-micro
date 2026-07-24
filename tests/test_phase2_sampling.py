"""
Tests for pipeline/phase2_multi_batch/sampling.py — sample generation
strategies (Sobol/LHS/random) and the seed x objective factorial design.
"""
import pytest

from pipeline.phase2_multi_batch.params import SamplingStrategy
from pipeline.phase2_multi_batch.sampling import (
    _validate_ranges,
    generate_design,
    generate_samples,
)

PARAMS = {
    "volfrac": {"range": [0.3, 0.7]},
    "void_size_frac": {"range": [0.1, 0.4]},
}


class TestValidateRanges:
    def test_extracts_sorted_names_and_bounds(self):
        names, lb, ub = _validate_ranges(PARAMS)
        assert names == ["void_size_frac", "volfrac"]
        assert list(lb) == [0.1, 0.3]
        assert list(ub) == [0.4, 0.7]

    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError):
            _validate_ranges({"volfrac": {"range": [0.7, 0.3]}})

    def test_rejects_wrong_length_range(self):
        with pytest.raises(ValueError):
            _validate_ranges({"volfrac": {"range": [0.3]}})


class TestGenerateSamples:
    @pytest.mark.parametrize("strategy", ["sobol", "lhs", "optimized_lhs", "random"])
    def test_stays_within_bounds_for_every_strategy(self, strategy):
        samples = generate_samples(50, PARAMS, strategy=strategy, seed=42)
        assert set(samples) == set(PARAMS)
        for name, values in samples.items():
            lo, hi = PARAMS[name]["range"]
            assert len(values) == 50
            assert all(lo <= v <= hi for v in values)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            generate_samples(10, PARAMS, strategy="not-a-strategy")

    def test_n_less_than_one_raises(self):
        with pytest.raises(ValueError):
            generate_samples(0, PARAMS)

    def test_sobol_is_reproducible_with_same_seed(self):
        a = generate_samples(20, PARAMS, strategy="sobol", seed=7)
        b = generate_samples(20, PARAMS, strategy="sobol", seed=7)
        assert a == b

    def test_random_strategy_differs_across_seeds(self):
        a = generate_samples(20, PARAMS, strategy="random", seed=1)
        b = generate_samples(20, PARAMS, strategy="random", seed=2)
        assert a != b


class TestGenerateDesign:
    def test_shape_is_seeds_times_objectives_times_samples(self):
        param_ranges = {"volfrac": (0.3, 0.7)}
        df = generate_design(
            n_samples=5,
            param_ranges=param_ranges,
            strategy=SamplingStrategy.SOBOL,
            batch_id=1,
            seed_map=["circle", "square"],
            objective_map=["auxetic"],
        )
        assert len(df) == 2 * 1 * 5
        assert set(df.columns) >= {"seed", "objective", "batch_id", "volfrac"}
        assert set(df["seed"]) == {"circle", "square"}
        assert (df["batch_id"] == 1).all()

    def test_param_values_stay_within_declared_range(self):
        param_ranges = {"volfrac": (0.3, 0.7)}
        df = generate_design(
            n_samples=10,
            param_ranges=param_ranges,
            strategy=SamplingStrategy.OPTIMIZED_LHS,
            batch_id=1,
            seed_map=["circle"],
            objective_map=["auxetic"],
        )
        assert df["volfrac"].between(0.3, 0.7).all()

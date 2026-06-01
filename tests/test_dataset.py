"""
Tests for analysis.dataset module.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from analysis.dataset import (
    load_iteration_data,
    compute_convergence_metrics,
    classify_auxetic,
)


class TestLoadIterationData:
    """Test suite for load_iteration_data."""

    def test_load_valid_csv(self):
        """Test loading a valid CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            pd.DataFrame({
                'Iteration': [1, 2, 3],
                'Poisson_v12': [-0.5, -0.6, -0.7],
                'Poisson_v21': [-0.3, -0.4, -0.5],
                'Objective': [0.8, 0.7, 0.6],
                'MeanDensity': [0.4, 0.4, 0.4],
            }).to_csv(csv_path, index=False)

            df = load_iteration_data(csv_path)
            assert len(df) == 3
            assert list(df.columns) == [
                'Iteration', 'Poisson_v12', 'Poisson_v21',
                'Objective', 'MeanDensity',
            ]

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_iteration_data('/nonexistent/path.csv')

    def test_missing_columns(self):
        """Test that ValueError is raised for missing columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'bad.csv')
            pd.DataFrame({'A': [1], 'B': [2]}).to_csv(csv_path, index=False)
            with pytest.raises(ValueError):
                load_iteration_data(csv_path)


class TestComputeConvergenceMetrics:
    """Test suite for compute_convergence_metrics."""

    def test_basic_metrics(self):
        """Test basic convergence metrics computation."""
        df = pd.DataFrame({
            'Iteration': range(1, 31),
            'Poisson_v12': np.linspace(-0.5, -0.7, 30),
            'Poisson_v21': np.linspace(-0.3, -0.5, 30),
            'Objective': np.linspace(1.0, 0.5, 30),
            'MeanDensity': [0.4] * 30,
        })
        metrics = compute_convergence_metrics(df, window=10)
        assert metrics['n_iters'] == 30
        assert metrics['final_v12'] == pytest.approx(-0.7, abs=0.01)
        assert metrics['final_v21'] == pytest.approx(-0.5, abs=0.01)
        assert metrics['final_objective'] == pytest.approx(0.5, abs=0.01)

    def test_insufficient_data(self):
        """Test metrics with insufficient data."""
        df = pd.DataFrame({
            'Iteration': [1],
            'Poisson_v12': [-0.5],
            'Poisson_v21': [-0.3],
            'Objective': [0.8],
            'MeanDensity': [0.4],
        })
        metrics = compute_convergence_metrics(df)
        assert metrics['n_iters'] == 1
        assert np.isnan(metrics['final_v12'])

    def test_objective_stability(self):
        """Test objective stability detection."""
        df = pd.DataFrame({
            'Iteration': range(1, 31),
            'Poisson_v12': [-0.5] * 30,
            'Poisson_v21': [-0.3] * 30,
            'Objective': [1.0] + [0.5 + 0.001 * i for i in range(29)],
            'MeanDensity': [0.4] * 30,
        })
        metrics = compute_convergence_metrics(df, window=10)
        assert metrics['obj_stable'] is True


class TestClassifyAuxetic:
    """Test suite for classify_auxetic."""

    def test_auxetic_v12(self):
        """Test classification when v12 is negative."""
        assert classify_auxetic(-0.5, 0.3) == 'Auxetic'

    def test_auxetic_v21(self):
        """Test classification when v21 is negative."""
        assert classify_auxetic(0.3, -0.5) == 'Auxetic'

    def test_conventional(self):
        """Test classification when both are positive."""
        assert classify_auxetic(0.3, 0.4) == 'Conventional'

    def test_zero_threshold(self):
        """Test classification at zero threshold."""
        assert classify_auxetic(0.0, 0.3) == 'Conventional'
        assert classify_auxetic(-0.0, 0.3) == 'Conventional'

    def test_custom_threshold(self):
        """Test classification with custom threshold."""
        assert classify_auxetic(0.1, 0.3, threshold=0.2) == 'Auxetic'
        assert classify_auxetic(0.3, 0.4, threshold=0.2) == 'Conventional'

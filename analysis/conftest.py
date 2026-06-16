"""
Pytest configuration and fixtures for the SIMP analysis package.

Provides shared fixtures for dataset loading, temporary directories,
sample CSV / image data, and image metric computations.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_csv_dir() -> Generator[str, None, None]:
    """Create a temporary directory and return its path.

    The directory is cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_iteration_csv(tmp_csv_dir: str) -> str:
    """Create a CSV with sample SIMP iteration data and return its path."""
    csv_path = os.path.join(tmp_csv_dir, 'iteration_data.csv')
    pd.DataFrame({
        'Iteration': list(range(1, 31)),
        'Poisson_v12': np.linspace(-0.5, -0.7, 30),
        'Poisson_v21': np.linspace(-0.3, -0.5, 30),
        'Objective': np.linspace(1.0, 0.5, 30),
        'MeanDensity': [0.4] * 30,
    }).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_iteration_csv_with_alias(tmp_csv_dir: str) -> str:
    """Create a CSV with *Volume_Fraction* instead of MeanDensity."""
    csv_path = os.path.join(tmp_csv_dir, 'iteration_data_alias.csv')
    pd.DataFrame({
        'Iteration': list(range(1, 31)),
        'Poisson_v12': np.linspace(-0.5, -0.7, 30),
        'Poisson_v21': np.linspace(-0.3, -0.5, 30),
        'Objective': np.linspace(1.0, 0.5, 30),
        'Volume_Fraction': [0.4] * 30,
    }).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_binary_image(tmp_csv_dir: str) -> str:
    """Create a small binary-ish PNG for image metric tests and return its path."""
    from PIL import Image

    size = 64
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[:size // 2, :] = 255       # top half white (solid)
    arr[size // 2:, :] = 0         # bottom half black (void)
    img_path = os.path.join(tmp_csv_dir, 'iteration_00030.png')
    Image.fromarray(arr, mode='L').save(img_path)
    return img_path

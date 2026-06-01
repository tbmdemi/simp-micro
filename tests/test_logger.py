"""
Tests for SimpLogger.
"""

import os
import tempfile
import pytest
from simp.io.logger import SimpLogger


class TestSimpLogger:
    """Test suite for SimpLogger."""

    def test_init_creates_directory(self):
        """Test that logger creates the output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'test_log')
            logger = SimpLogger(log_dir)
            assert os.path.exists(log_dir)

    def test_init_creates_csv(self):
        """Test that logger creates the CSV file with header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimpLogger(tmpdir)
            assert os.path.exists(logger.filename)
            with open(logger.filename, 'r') as f:
                header = f.readline().strip()
            assert 'Iteration' in header
            assert 'Poisson_v12' in header
            assert 'Objective' in header

    def test_log_appends_data(self):
        """Test that log() appends data to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimpLogger(tmpdir, buffer_size=1)
            logger.log(1, -0.5, -0.3, 0.8, 0.4)
            logger.flush()
            with open(logger.filename, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 2  # header + 1 data row
            assert '1,-0.5' in lines[1]

    def test_log_multiple_entries(self):
        """Test that multiple log entries are recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimpLogger(tmpdir, buffer_size=5)
            for i in range(5):
                logger.log(i + 1, -0.5 + i * 0.1, -0.3 + i * 0.1, 0.8, 0.4)
            logger.flush()
            assert len(logger.iterations) == 5
            assert len(logger.objectives) == 5

    def test_log_initial(self):
        """Test that log_initial writes NaN values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimpLogger(tmpdir, buffer_size=1)
            logger.log_initial(0, 0.5)
            logger.flush()
            with open(logger.filename, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert 'nan' in lines[1].lower()

    def test_flush_writes_buffer(self):
        """Test that flush writes buffered data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimpLogger(tmpdir, buffer_size=10)
            logger.log(1, -0.5, -0.3, 0.8, 0.4)
            # Data should be in buffer, not on disk yet
            logger.flush()
            with open(logger.filename, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 2

    def test_n_iters_property(self):
        """Test that n_iters returns correct count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimpLogger(tmpdir)
            assert logger.n_iters == 0
            logger.log(1, -0.5, -0.3, 0.8, 0.4)
            assert logger.n_iters == 1
            logger.log(2, -0.6, -0.4, 0.7, 0.4)
            assert logger.n_iters == 2

    def test_in_memory_arrays(self):
        """Test that in-memory arrays are populated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SimpLogger(tmpdir)
            logger.log(1, -0.5, -0.3, 0.8, 0.4)
            assert logger.iterations == [1]
            assert logger.poisson_ratios_v12 == [-0.5]
            assert logger.poisson_ratios_v21 == [-0.3]
            assert logger.objectives == [0.8]
            assert logger.volume_fractions == [0.4]

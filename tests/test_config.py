"""
Tests for SimpConfig validation.
"""

import pytest
from simp.config import SimpConfig


class TestSimpConfig:
    """Test suite for SimpConfig dataclass."""

    def test_default_config(self):
        """Test that default config creates without error."""
        cfg = SimpConfig()
        assert cfg.nelx == 100
        assert cfg.nely == 100
        assert cfg.volfrac == 0.4
        assert cfg.penal == 3.0
        assert cfg.objective_type == 'auxetic'

    def test_valid_config(self):
        """Test that valid parameters pass validation."""
        cfg = SimpConfig(nelx=60, nely=40, volfrac=0.3, penal=2.0)
        assert cfg.nelx == 60
        assert cfg.nely == 40

    def test_invalid_nelx(self):
        """Test that non-positive nelx raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(nelx=0)

    def test_invalid_nely(self):
        """Test that non-positive nely raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(nely=-1)

    def test_invalid_volfrac_zero(self):
        """Test that volfrac <= 0 raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(volfrac=0)

    def test_invalid_volfrac_over_one(self):
        """Test that volfrac > 1 raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(volfrac=1.5)

    def test_invalid_penal(self):
        """Test that penal < 1 raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(penal=0.5)

    def test_invalid_rmin(self):
        """Test that rmin <= 0 raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(rmin=0)

    def test_invalid_ft(self):
        """Test that ft not in (1, 2) raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(ft=3)

    def test_invalid_objective_type(self):
        """Test that invalid objective_type raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(objective_type='third')

    def test_invalid_max_iter(self):
        """Test that max_iter <= 0 raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(max_iter=0)

    def test_invalid_move(self):
        """Test that move <= 0 raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(move=0)

    def test_invalid_move_over_one(self):
        """Test that move > 1 raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(move=2.0)

    def test_invalid_save_every(self):
        """Test that save_every <= 0 raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(save_every=0)

    def test_invalid_scale_factor(self):
        """Test that scale_factor < 1 raises AssertionError."""
        with pytest.raises(AssertionError):
            SimpConfig(scale_factor=0)

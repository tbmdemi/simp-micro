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

    @pytest.mark.parametrize("kwargs", [
        {"nelx": 0}, {"nely": -1}, {"volfrac": 0}, {"volfrac": 1.5},
        {"penal": 0.5}, {"rmin": 0}, {"ft": 3}, {"objective_type": "third"},
        {"max_iter": 0}, {"move": 0}, {"move": 2.0}, {"save_every": 0},
        {"scale_factor": 0},
    ], ids=[
        "nelx<=0", "nely<=0", "volfrac<=0", "volfrac>1", "penal<1", "rmin<=0",
        "ft not in (1,2)", "objective_type invalid", "max_iter<=0", "move<=0",
        "move>1", "save_every<=0", "scale_factor<1",
    ])
    def test_invalid_params_raise(self, kwargs):
        """Mỗi tham số vật lý/số học ngoài khoảng hợp lệ phải raise
        AssertionError (SimpConfig.validate(), gọi tự động qua __post_init__)."""
        with pytest.raises(AssertionError):
            SimpConfig(**kwargs)

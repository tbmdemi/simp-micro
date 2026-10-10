"""
Tests for simp/io/logger.py::save_csv - bộ ghi `iteration_data.csv` thật
của run_simp()/run_simp_mma() (lớp SimpLogger cũ không dùng ở đâu nên đã
xóa 2026-10-10 cùng test của nó).
"""

import numpy as np

from simp.io.logger import save_csv

HEADER = "Iteration,Poisson_v12,Poisson_v21,Objective,Volume_Fraction"


def _history(n=3):
    """Lịch sử n vòng lặp có NaN ở vòng 0 (như run_simp ghi vòng khởi tạo)."""
    return {
        "iteration": np.arange(n),
        "v12": np.array([np.nan] + [-0.25] * (n - 1)),
        "v21": np.array([np.nan] + [-0.125] * (n - 1)),
        "objective": np.linspace(1.0, 0.5, n),
        "volume": np.full(n, 0.4),
    }


def test_creates_directory_and_returns_path(tmp_path):
    out_dir = tmp_path / "nested" / "run"
    path = save_csv(str(out_dir), _history())
    assert path == str(out_dir / "iteration_data.csv")
    assert (out_dir / "iteration_data.csv").is_file()


def test_header_and_row_format(tmp_path):
    lines = open(save_csv(str(tmp_path), _history())).read().splitlines()
    assert lines[0] == HEADER
    assert len(lines) == 4
    assert lines[1] == "0,nan,nan,1.000000,0.400000"
    assert lines[2] == "1,-0.250000,-0.125000,0.750000,0.400000"


def test_overwrites_existing_file(tmp_path):
    save_csv(str(tmp_path), _history(5))
    lines = open(save_csv(str(tmp_path), _history(2))).read().splitlines()
    assert len(lines) == 3

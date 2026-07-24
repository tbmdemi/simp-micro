"""
Tests for pipeline/phase3_dataset/scan_dataset.py — locating the final
iteration PNG and reading the final volume fraction for each SIMP sample
directory. These are filesystem-facing helpers, tested against tmp_path
fixtures instead of the real (gitignored) outputs/multi_batch/ tree.
"""
import pandas as pd

from pipeline.phase3_dataset.scan_dataset import find_final_iteration_png, final_volfrac


class TestFindFinalIterationPng:
    def test_picks_highest_iteration_number(self, tmp_path):
        for n in (5, 20, 134, 7):
            (tmp_path / f"iteration_{n:05d}.png").write_bytes(b"fake")
        result = find_final_iteration_png(str(tmp_path))
        assert result.endswith("iteration_00134.png")

    def test_returns_none_when_no_png_present(self, tmp_path):
        assert find_final_iteration_png(str(tmp_path)) is None

    def test_ignores_unrelated_files(self, tmp_path):
        (tmp_path / "iteration_00010.png").write_bytes(b"fake")
        (tmp_path / "metadata.json").write_text("{}")
        (tmp_path / "iteration_data.csv").write_text("a,b\n1,2\n")
        result = find_final_iteration_png(str(tmp_path))
        assert result.endswith("iteration_00010.png")


class TestFinalVolfrac:
    def test_reads_last_row_of_csv(self, tmp_path):
        csv_path = tmp_path / "iteration_data.csv"
        pd.DataFrame({
            "Iteration": [1, 2, 3],
            "Volume_Fraction": [0.5, 0.45, 0.401],
        }).to_csv(csv_path, index=False)
        assert final_volfrac(str(tmp_path)) == 0.401

    def test_missing_csv_returns_none(self, tmp_path):
        assert final_volfrac(str(tmp_path)) is None

    def test_empty_csv_returns_none(self, tmp_path):
        csv_path = tmp_path / "iteration_data.csv"
        pd.DataFrame({"Iteration": [], "Volume_Fraction": []}).to_csv(csv_path, index=False)
        assert final_volfrac(str(tmp_path)) is None

    def test_malformed_csv_returns_none_not_raise(self, tmp_path):
        csv_path = tmp_path / "iteration_data.csv"
        csv_path.write_text("this is not,a,valid\ncsv file at all\n\"unterminated")
        assert final_volfrac(str(tmp_path)) is None

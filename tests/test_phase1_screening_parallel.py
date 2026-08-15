"""
Tests for pipeline/phase1_screening/screening_parallel.py::save_results.
"""
import csv

import pandas as pd

from pipeline.phase1_screening.screening_parallel import save_results


def _fake_analysis(param_names):
    return {
        "param_names": param_names,
        "correlations": [None for _ in param_names],
        "p_values": [None for _ in param_names],
        "top_3": [(param_names[0], None, None)],
        "n_valid": 1,
    }


class TestSaveResultsCsvRoundtrip:
    """Bug đã sửa 2026-08-15: save_results() dùng `r['v12'] or ''` (và tương
    tự cho v21/obj_value/n_iters/elapsed_time) - vì 0.0 là falsy trong
    Python, mẫu có giá trị HỢP LỆ = 0.0 bị ghi thành chuỗi rỗng trong CSV.
    analyst.py sau đó đọc CSV bằng pd.to_numeric(errors='coerce') +
    dropna(), nên mẫu này bị âm thầm loại khỏi phân tích Spearman - trong
    khi file .json song song (dict gốc, không qua " or ''") vẫn đúng, tạo
    ra lệch dữ liệu CSV/JSON không ai phát hiện được."""

    def test_zero_v12_is_not_dropped_from_csv(self, tmp_path):
        objective = "auxetic"
        results = [
            {
                "sample_id": 0, "success": True,
                "v12": 0.0,  # đúng giá trị landmine - falsy nhưng HỢP LỆ
                "v21": 0.0,
                "obj_value": 0.0,
                "n_iters": 0,
                "converged": True,
                "elapsed_time": 0.0,
                "params": {},
                "error": "",
            },
        ]
        from pipeline.params import get_active_params
        param_names = get_active_params(objective)
        save_results(
            results, _fake_analysis(param_names), str(tmp_path),
            objective, "circle",
        )

        csv_path = tmp_path / f"phase1_circle_{objective}.csv"
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        # Trước fix: cột này đọc lên là NaN (rỗng) thay vì 0.0, và
        # analyst.py's dropna() sẽ loại hẳn dòng này khỏi phân tích.
        assert df.loc[0, "v12"] == 0.0
        assert df.loc[0, "v21"] == 0.0
        assert df.loc[0, "obj_value"] == 0.0
        assert df.loc[0, "n_iters"] == 0
        assert df.loc[0, "elapsed_time"] == 0.0
        assert not df["v12"].isna().any()

    def test_missing_value_still_written_as_empty(self, tmp_path):
        """None (thật sự thiếu, vd mẫu FAILED trước khi tính được v12) vẫn
        phải để trống - không được biến thành "None"/"nan" literal."""
        objective = "auxetic"
        results = [
            {
                "sample_id": 1, "success": False,
                "v12": None, "v21": None, "obj_value": None, "n_iters": None,
                "converged": False, "elapsed_time": None,
                "params": {}, "error": "FE solve failed",
            },
        ]
        from pipeline.params import get_active_params
        param_names = get_active_params(objective)
        save_results(
            results, _fake_analysis(param_names), str(tmp_path),
            objective, "circle",
        )

        csv_path = tmp_path / f"phase1_circle_{objective}.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["v12"] == ""
        assert rows[0]["elapsed_time"] == ""

        df = pd.read_csv(csv_path)
        assert pd.isna(df.loc[0, "v12"])

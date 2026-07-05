"""
ANOVA (Analysis of Variance)
=============================
One-way + two-way interaction ANOVA trên dữ liệu Phase 1.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.stats import f_oneway


def _quantize_to_levels(
    series: pd.Series,
    n_levels: int = 5,
) -> pd.Series:
    """Lượng tử hóa biến liên tục thành n_levels category."""
    # Nếu ít unique value hơn n_levels, giữ nguyên
    unique_vals = series.nunique()
    if unique_vals <= n_levels:
        return series.astype('category').cat.codes
    return pd.qcut(series, q=n_levels, labels=False, duplicates='drop')


def compute_oneway_anova(
    df: pd.DataFrame,
    param_cols: List[str],
    obj_col: str = 'obj_value',
    n_levels: int = 5,
) -> Dict:
    """Tính one-way ANOVA cho từng tham số.

    Args:
        df: DataFrame chứa dữ liệu.
        param_cols: Danh sách cột tham số.
        obj_col: Tên cột objective.
        n_levels: Số mức để lượng tử hóa.

    Returns:
        Dict {param_name: {'F': float, 'p': float, 'n_groups': int}}.
    """
    results = {}
    for col in param_cols:
        df_temp = df[[col, obj_col]].dropna().copy()
        if len(df_temp) < 10:
            results[col] = {'F': None, 'p': None, 'n_groups': 0}
            continue

        groups_cat = _quantize_to_levels(df_temp[col], n_levels=n_levels)
        df_temp['_cat'] = groups_cat

        groups = [
            df_temp.loc[df_temp['_cat'] == lev, obj_col].values
            for lev in sorted(df_temp['_cat'].unique())
        ]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            results[col] = {'F': None, 'p': None, 'n_groups': len(groups)}
            continue

        f_stat, p_val = f_oneway(*groups)
        results[col] = {
            'F': float(f_stat),
            'p': float(p_val),
            'n_groups': len(groups),
        }

    return results


def compute_anova_from_csv(
    csv_path: str,
    param_cols: List[str],
    obj_col: str = 'obj_value',
    success_only: bool = True,
    n_levels: int = 5,
) -> Dict:
    """Đọc CSV và chạy ANOVA.

    Args:
        csv_path: Đường dẫn file CSV Phase 1.
        param_cols: Danh sách cột tham số.
        obj_col: Tên cột objective value.
        success_only: Chỉ lấy mẫu thành công.
        n_levels: Số mức cho lượng tử hóa.

    Returns:
        Dict kết quả ANOVA với keys: 'oneway', 'n_samples'.
    """
    df = pd.read_csv(csv_path)

    if success_only and 'success' in df.columns:
        df = df[df['success'] == True].copy()

    df = df.dropna(subset=[obj_col])

    oneway = compute_oneway_anova(df, param_cols, obj_col, n_levels=n_levels)

    return {
        'oneway': oneway,
        'n_samples': len(df),
    }
"""
Standardized Regression Coefficients (SRC)
===========================================
Tính hệ số hồi quy chuẩn hóa từ dữ liệu Phase 1 screening.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def compute_src(
    X: np.ndarray,
    y: np.ndarray,
    param_names: List[str],
) -> Dict:
    """Tính Standardized Regression Coefficients (SRC).

    Args:
        X: Ma trận thiết kế (N, n_params).
        y: Vector output (N,).
        param_names: Tên các tham số.

    Returns:
        Dict với keys: 'coef', 'r2', 'r2_adjusted', 'param_names', 'n_samples'.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    reg = LinearRegression()
    reg.fit(X_scaled, y_scaled)

    n = X.shape[0]
    p = X.shape[1]
    r2 = reg.score(X_scaled, y_scaled)
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    return {
        'coef': dict(zip(param_names, reg.coef_)),
        'r2': r2,
        'r2_adjusted': r2_adj,
        'param_names': param_names,
        'n_samples': n,
    }


def compute_src_from_csv(
    csv_path: str,
    param_cols: List[str],
    obj_col: str = 'obj_value',
    success_only: bool = True,
) -> Dict:
    """Đọc CSV và tính SRC.

    Args:
        csv_path: Đường dẫn file CSV Phase 1.
        param_cols: Danh sách tên cột tham số.
        obj_col: Tên cột objective value.
        success_only: Chỉ lấy các mẫu thành công.

    Returns:
        Dict kết quả SRC.
    """
    df = pd.read_csv(csv_path)

    if success_only and 'success' in df.columns:
        df = df[df['success'] == True].copy()

    # Lọc các dòng có obj_value không null
    df = df.dropna(subset=[obj_col])

    if len(df) < 5:
        return {
            'coef': {p: None for p in param_cols},
            'r2': None,
            'r2_adjusted': None,
            'param_names': param_cols,
            'n_samples': len(df),
            'warning': 'Quá ít mẫu để hồi quy',
        }

    X = df[param_cols].values.astype(float)
    y = df[obj_col].values.astype(float)

    return compute_src(X, y, param_cols)
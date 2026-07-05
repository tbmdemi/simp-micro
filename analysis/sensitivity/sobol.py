"""
Sobol Sensitivity Indices
=========================
Tính Sobol indices bậc 1 và tổng (ST) thông qua surrogate model
(Gaussian Process) trained trên dữ liệu Phase 1 thay vì Saltelli sampling
trên toy model.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


def _gpr_surrogate(
    X: np.ndarray,
    y: np.ndarray,
) -> GaussianProcessRegressor:
    """Train Gaussian Process surrogate model.

    Args:
        X: Input matrix (N, n_params).
        y: Output vector (N,).

    Returns:
        Trained GPR model.
    """
    # Chuẩn hóa đầu vào cho GPR ổn định
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xs = x_scaler.fit_transform(X)
    ys = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = (
        ConstantKernel(1.0) *
        RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2)) +
        WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        random_state=42,
        normalize_y=False,
    )
    gpr.fit(Xs, ys)

    # Lưu scalers để dùng sau
    gpr._x_scaler = x_scaler
    gpr._y_scaler = y_scaler

    return gpr


def _gpr_predict(gpr: GaussianProcessRegressor, X: np.ndarray) -> np.ndarray:
    """Predict using GPR with unscaled input."""
    Xs = gpr._x_scaler.transform(X)
    ys = gpr.predict(Xs)
    # Đưa về thang đo gốc
    return gpr._y_scaler.inverse_transform(ys.reshape(-1, 1)).ravel()


def compute_sobol_from_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    param_names: List[str],
    bounds: List[Tuple[float, float]],
    n_mc: int = 2 ** 12,
) -> Dict:
    """Tính Sobol indices thông qua GPR surrogate + Monte Carlo integration.

    Args:
        X: Dữ liệu huấn luyện (N_samples, n_params).
        y: Output tương ứng (N_samples,).
        param_names: Tên tham số.
        bounds: Danh sách (min, max) cho mỗi tham số.
        n_mc: Số mẫu Monte Carlo cho tích phân Sobol.

    Returns:
        Dict với keys: 'S1', 'ST', 'S2', 'param_names', 'n_train', 'r2_surrogate'.
    """
    n_params = X.shape[1]

    # Train surrogate
    gpr = _gpr_surrogate(X, y)

    # Đánh giá chất lượng surrogate (in-sample R²)
    y_pred = _gpr_predict(gpr, X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_surr = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # LHS sampling trong [0,1]^d
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    A = sampler.random(n=n_mc)  # A ~ [0,1]
    B = sampler.random(n=n_mc)  # B ~ [0,1]

    # Scale về miền thực
    A_scaled = lb + A * (ub - lb)
    B_scaled = lb + B * (ub - lb)

    fA = _gpr_predict(gpr, A_scaled)
    fB = _gpr_predict(gpr, B_scaled)
    f0 = np.mean(fA)

    V = np.var(fA, ddof=0)

    S1 = np.zeros(n_params)
    ST = np.zeros(n_params)
    S2 = np.zeros((n_params, n_params))

    # Tạo ma trận AB_i: cột i từ B, còn lại từ A
    for i in range(n_params):
        # First-order: V(E[Y|X_i])
        AB_i = A_scaled.copy()
        AB_i[:, i] = B_scaled[:, i]
        fAB_i = _gpr_predict(gpr, AB_i)
        S1[i] = (np.mean(fB * (fAB_i - fA)) / V) if V > 1e-12 else 0.0

        # Total-order: 1 - V(E[Y|X_{-i}])/V(Y)
        BA_i = B_scaled.copy()
        BA_i[:, i] = A_scaled[:, i]
        fBA_i = _gpr_predict(gpr, BA_i)
        ST[i] = (1.0 - np.mean(fA * fBA_i - f0**2) / V) if V > 1e-12 else 0.0

    # Second-order (nếu số params ít, <= 6)
    if n_params <= 6:
        for i in range(n_params):
            for j in range(i + 1, n_params):
                AB_ij = A_scaled.copy()
                AB_ij[:, i] = B_scaled[:, i]
                AB_ij[:, j] = B_scaled[:, j]
                fAB_ij = _gpr_predict(gpr, AB_ij)
                # S2 = V(E[Y|X_i,X_j]) - S1_i - S1_j
                term = np.mean(fB * fAB_ij - f0**2) / V - S1[i] - S1[j]
                S2[i, j] = max(0.0, term)
                S2[j, i] = S2[i, j]

    return {
        'S1': dict(zip(param_names, S1.tolist())),
        'ST': dict(zip(param_names, ST.tolist())),
        'S2': S2,
        'param_names': param_names,
        'n_train': X.shape[0],
        'n_mc': n_mc,
        'r2_surrogate': r2_surr,
    }


def compute_sobol_from_csv(
    csv_path: str,
    param_cols: List[str],
    bounds: List[Tuple[float, float]],
    obj_col: str = 'obj_value',
    success_only: bool = True,
    n_mc: int = 2 ** 12,
) -> Dict:
    """Đọc CSV và tính Sobol indices.

    Args:
        csv_path: Đường dẫn file CSV Phase 1.
        param_cols: Danh sách tên cột tham số.
        bounds: Danh sách (min, max) cho từng tham số.
        obj_col: Tên cột objective value.
        success_only: Chỉ lấy mẫu thành công.
        n_mc: Số mẫu Monte Carlo.

    Returns:
        Dict kết quả Sobol.
    """
    df = pd.read_csv(csv_path)

    if success_only and 'success' in df.columns:
        df = df[df['success'] == True].copy()

    df = df.dropna(subset=[obj_col])

    if len(df) < 10:
        return {
            'S1': {p: None for p in param_cols},
            'ST': {p: None for p in param_cols},
            'S2': None,
            'param_names': param_cols,
            'n_train': len(df),
            'n_mc': 0,
            'r2_surrogate': None,
            'warning': 'Quá ít mẫu để xây dựng surrogate',
        }

    X = df[param_cols].values.astype(float)
    y = df[obj_col].values.astype(float)

    return compute_sobol_from_surrogate(X, y, param_cols, bounds, n_mc=n_mc)
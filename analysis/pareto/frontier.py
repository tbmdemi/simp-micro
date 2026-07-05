"""
Pareto Frontier Computation
============================
Tìm Pareto front và tính hypervolume từ dữ liệu multi-objective.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def is_pareto_efficient(
    costs: np.ndarray,
    maximize: bool = False,
) -> np.ndarray:
    """Tìm các điểm Pareto-efficient.

    Args:
        costs: Ma trận (n_samples, n_objectives) - mỗi cột là một objective.
        maximize: True nếu các objective cần maximize (mặc định minimize).

    Returns:
        Boolean mask (n_samples,) - True cho các điểm trên Pareto front.
    """
    if not maximize:
        costs = -costs  # Đảo dấu để minimize thành maximize

    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not is_efficient[i]:
            continue
        # Loại bỏ các điểm bị dominates bởi i
        is_efficient[i + 1:] = np.any(
            costs[i + 1:] > costs[i], axis=1
        )
        # Giữ lại i nếu không bị dominates bởi {i+1, ...}
        is_efficient[i] = np.any(costs[i] > costs[i + 1:], axis=0).all()

    return is_efficient


def compute_pareto_front(
    df: pd.DataFrame,
    obj_cols: List[str],
    maximize: bool = False,
    success_col: str = 'success',
) -> Dict:
    """Tìm Pareto front từ DataFrame.

    Args:
        df: DataFrame chứa dữ liệu.
        obj_cols: Danh sách cột objective (2 hoặc 3 objectives).
        maximize: True nếu cần maximize.
        success_col: Tên cột success flag (None để bỏ qua).

    Returns:
        Dict với keys:
          - 'frontier': DataFrame các điểm Pareto.
          - 'mask': Boolean mask trên toàn bộ df.
          - 'params': Tham số của các điểm Pareto.
          - 'n_frontier': Số điểm trên front.
          - 'n_total': Tổng số điểm.
    """
    if success_col and success_col in df.columns:
        df = df[df[success_col] == True].copy()

    df = df.dropna(subset=obj_cols)

    if len(df) == 0:
        return {
            'frontier': pd.DataFrame(),
            'mask': np.array([], dtype=bool),
            'n_frontier': 0,
            'n_total': 0,
        }

    costs = df[obj_cols].values.astype(float)
    if not maximize:
        costs = -costs  # minimize → maximize

    mask = is_pareto_efficient(costs, maximize=True)
    frontier_df = df[mask].copy().sort_values(obj_cols[0])

    # Tách params và objectives
    param_cols = [c for c in df.columns if c not in obj_cols
                  and c not in ('success', 'converged', 'error', 'sample_id')]

    return {
        'frontier': frontier_df,
        'mask': mask,
        'params': param_cols,
        'n_frontier': int(mask.sum()),
        'n_total': len(df),
    }


def compute_hypervolume(
    frontier: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
) -> float:
    """Tính hypervolume của Pareto front (2D).

    Args:
        frontier: Mảng (n_points, 2) các điểm Pareto (đã minimize).
        reference_point: Điểm tham chiếu (nếu None, lấy max*1.1).

    Returns:
        Giá trị hypervolume.
    """
    if frontier.shape[0] == 0:
        return 0.0

    # Sắp xếp theo objective 0
    sorted_idx = np.argsort(frontier[:, 0])
    frontier = frontier[sorted_idx]

    if reference_point is None:
        reference_point = frontier.max(axis=0) * 1.1

    # Hypervolume = tổng diện tích hình chữ nhật
    hv = 0.0
    prev_x = frontier[0, 0]
    for i in range(frontier.shape[0] - 1):
        x_left = frontier[i, 0]
        x_right = frontier[i + 1, 0]
        y_top = frontier[i, 1]
        hv += (x_right - x_left) * (reference_point[1] - y_top)
    # Điểm cuối cùng
    hv += (reference_point[0] - frontier[-1, 0]) * (
        reference_point[1] - frontier[-1, 1]
    )

    return hv
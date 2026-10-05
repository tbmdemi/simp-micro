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

    LƯU Ý (bug đã sửa 2026-08-20): bản trước dùng vòng lặp có tác dụng phụ
    kiểu "loại dần costs[i+1:]" - sai ở BIÊN i=n_points-1 (điểm CUỐI CÙNG
    trong mảng): costs[i+1:] rỗng, `np.any(..., axis=0).all()` trên mảng
    rỗng cho ra False (đúng ngữ nghĩa numpy - "any" trên tập rỗng là False,
    nhưng .all() của False lại là False, trong khi ngữ nghĩa ĐÚNG cần ở đây
    là "không có điểm nào phía sau lấn át tôi -> tôi hiệu quả -> True"). Hệ
    quả: điểm CUỐI CÙNG của mảng đầu vào luôn bị đánh dấu sai là KHÔNG
    Pareto-efficient, bất kể nó có thực sự bị lấn át hay không - kể cả khi
    đó là điểm tốt nhất tuyệt đối, hoặc khi mảng chỉ có 1 điểm (luôn hiệu quả
    tầm thường). Phát hiện khi Nhóm 3.2 (PROJECT_PLAN.md) dùng hàm này lặp
    lại (loại bỏ từng tầng Pareto) - bug khiến điểm cuối cùng còn lại không
    bao giờ được xếp hạng, gây vòng lặp vô hạn. Module chưa từng chạy thật
    cho kết quả nào đã công bố (không có output/pareto/* trên đĩa) - không
    có claim khoa học nào đã công bố bị ảnh hưởng.

    Thay bằng thuật toán O(n²) tường minh, không có tác dụng phụ theo thứ
    tự: với mỗi điểm i, kiểm tra TRỰC TIẾP có tồn tại điểm j≠i lấn át i hay
    không (j lấn át i <=> costs[j]>=costs[i] ở MỌI trục VÀ costs[j]>costs[i]
    ở ÍT NHẤT 1 trục) - không phụ thuộc thứ tự duyệt, không có trường hợp
    biên mảng rỗng."""
    if not maximize:
        costs = -costs  # Đảo dấu để minimize thành maximize

    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        dominates_i = np.all(costs >= costs[i], axis=1) & np.any(
            costs > costs[i], axis=1
        )
        dominates_i[i] = False  # không tự so sánh với chính mình
        if dominates_i.any():
            is_efficient[i] = False

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
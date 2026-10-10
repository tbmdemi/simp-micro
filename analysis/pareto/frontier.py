"""
Pareto Frontier Computation
============================
Tìm các điểm Pareto-efficient (dùng bởi `docs/paper1/scripts/c5_pareto_nofp.py`
và `notebooks/03_cheap_physical_properties.ipynb`).
"""

import numpy as np


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

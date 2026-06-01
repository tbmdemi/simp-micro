"""
Hàm mục tiêu loại thứ nhất cho tối ưu hóa hình dạng SIMP.

Tối đa hóa thành phần Q₁₂ (shear coupling) của ten-xơ độ cứng
đồng nhất hóa, với cơ chế suy giảm (beta decay) để ổn định hội tụ.

Công thức (theo MATLAB First_Obj):
    c = Q₁₂ - β^loop * (Q₁₁ + Q₂₂)
    dc = dQ₁₂ - β^loop * (dQ₁₁ + dQ₂₂)
"""

import numpy as np


def compute_first_objective(
    Q: np.ndarray,
    dQ: np.ndarray,
    iteration: int,
    beta: float = 0.8,
):
    """Tính hàm mục tiêu loại thứ nhất và đạo hàm của nó.

    Hàm mục tiêu: c = Q₁₂ - β^loop * (Q₁₁ + Q₂₂)
    nhằm tối đa hóa độ cứng cắt Q₁₂ đồng thời hạn chế
    độ cứng dọc trục Q₁₁, Q₂₂ thông qua cơ chế suy giảm.

    Args:
        Q: Ten-xơ độ cứng đồng nhất hóa (3×3).
        dQ: Đạo hàm của Q theo mật độ phần tử (3×3×nely×nelx).
        iteration: Số vòng lặp hiện tại.
        beta: Hệ số suy giảm (mặc định 0.8).

    Returns:
        Bộ (c, dc) với:
            c : Giá trị hàm mục tiêu (vô hướng).
            dc: Mảng (nely, nelx) đạo hàm của hàm mục tiêu.
    """
    decay = beta ** iteration

    # Hàm mục tiêu: c = Q₁₂ - β^loop * (Q₁₁ + Q₂₂)
    c = Q[0, 1] - decay * (Q[0, 0] + Q[1, 1])

    # Đạo hàm: dc = dQ₁₂ - β^loop * (dQ₁₁ + dQ₂₂)
    dc = dQ[0, 1, :, :] - decay * (dQ[0, 0, :, :] + dQ[1, 1, :, :])

    return c, dc

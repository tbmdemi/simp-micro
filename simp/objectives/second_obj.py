"""
Hàm mục tiêu loại thứ hai cho tối ưu hóa hình dạng SIMP.

Tối đa hóa thành phần Q₁₂ (shear coupling) của ten-xơ độ cứng
đồng nhất hóa, với ràng buộc phạt đối với độ cứng dọc trục.

Công thức (theo MATLAB Second_Obj):
    c = Q₁₂
    Với ràng buộc: Q₁₁ ≥ δ và Q₂₂ ≥ δ (δ = 0.1 * volfrac * E₀)
    Nếu vi phạm: c += penalty * (δ - Q)²
"""

import numpy as np


def compute_second_objective(
    Q: np.ndarray,
    dQ: np.ndarray,
    iteration: int,
    volfrac: float,
    E0: float,
    beta_second: float = 1.0,
):
    """Tính hàm mục tiêu loại thứ hai và đạo hàm của nó.

    Hàm mục tiêu: c = Q₁₂ (tối đa hóa độ cứng cắt)
    Với ràng buộc phạt: Q₁₁ ≥ δ và Q₂₂ ≥ δ
    trong đó δ = 0.1 * volfrac * E₀.

    Args:
        Q: Ten-xơ độ cứng đồng nhất hóa (3×3).
        dQ: Đạo hàm của Q theo mật độ phần tử (3×3×nely×nelx).
        iteration: Số vòng lặp hiện tại.
        volfrac: Tỉ lệ thể tích yêu cầu.
        E0: Modul đàn hồi Young của vật liệu đặc.

    Returns:
        Bộ (c, dc) với:
            c : Giá trị hàm mục tiêu (vô hướng).
            dc: Mảng (nely, nelx) đạo hàm của hàm mục tiêu.
    """
    # Ngưỡng ràng buộc độ cứng dọc trục
    delta = 0.1 * volfrac * E0
    penalty = beta_second  # Hệ số phạt

    # Hàm mục tiêu cơ bản: tối đa hóa Q₁₂
    c = Q[0, 1]
    dc = dQ[0, 1, :, :].copy()

    # Ràng buộc phạt cho Q₁₁ và Q₂₂
    if Q[0, 0] < delta:
        c += penalty * (delta - Q[0, 0]) ** 2
        dc += -2 * penalty * (delta - Q[0, 0]) * dQ[0, 0, :, :]

    if Q[1, 1] < delta:
        c += penalty * (delta - Q[1, 1]) ** 2
        dc += -2 * penalty * (delta - Q[1, 1]) * dQ[1, 1, :, :]

    return c, dc

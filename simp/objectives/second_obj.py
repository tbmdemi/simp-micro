"""
Hàm mục tiêu loại thứ hai cho tối ưu hóa hình dạng SIMP.

Tối đa hóa thành phần Q₁₂ (shear coupling) của ten-xơ độ cứng
đồng nhất hóa, với ràng buộc phạt đối với độ cứng dọc trục.

Công thức (theo MATLAB Second_Obj, topK_Hourglass_New_obj.m):
    c = Q₁₂
    - 20 iteration đầu: c -= (1 - 0.02*loop) * (Q₁₁ + Q₂₂)
    - Ràng buộc: Q₁₁ ≥ δ và Q₂₂ ≥ δ (δ = 0.1 * volfrac * E₀)
    - Nếu vi phạm: c += penalty * (δ - Q)²
"""

import numpy as np


def compute_second_objective(
    Q: np.ndarray,
    dQ: np.ndarray,
    iteration: int,
    volfrac: float,
    E0: float,
    beta_second: float = 100.0,
):
    """Tính hàm mục tiêu loại thứ hai và đạo hàm của nó.

    Hoàn toàn khớp MATLAB topK_Hourglass_New_obj.m:
    - Hàm mục tiêu: c = Q₁₂ (tối đa hóa độ cứng cắt)
    - 20 iteration đầu: giảm dần ảnh hưởng của Q₁₁+Q₂₂
    - Ràng buộc phạt: Q₁₁ ≥ δ và Q₂₂ ≥ δ
    - Hệ số phạt penalty = beta_second (mặc định 100.0)

    Args:
        Q: Ten-xơ độ cứng đồng nhất hóa (3×3).
        dQ: Đạo hàm của Q theo mật độ phần tử (3×3×nely×nelx).
        iteration: Số vòng lặp hiện tại (1-indexed).
        volfrac: Tỉ lệ thể tích yêu cầu.
        E0: Modul đàn hồi Young của vật liệu đặc.
        beta_second: Hệ số phạt stiffness (mặc định 100.0, khớp MATLAB).

    Returns:
        Bộ (c, dc) với:
            c : Giá trị hàm mục tiêu (vô hướng).
            dc: Mảng (nely, nelx) đạo hàm của hàm mục tiêu.
    """
    # Ngưỡng ràng buộc độ cứng dọc trục
    delta = 0.1 * volfrac * E0
    penalty = beta_second  # Hệ số phạt, mặc định 100.0 (khớp MATLAB)

    # === MATLAB: c = Q(1,2) ===
    c = Q[0, 1]
    dc = dQ[0, 1, :, :].copy()

    # === MATLAB: first-20-iter scaling ===
    # scale_factor = 1 - 0.02 * loop, loop = 1..20
    # c = c - scale_factor * (Q(1,1) + Q(2,2))
    # dc = dc - scale_factor * (dQ{1,1} + dQ{2,2})
    if iteration <= 20:
        scale_factor = 1.0 - 0.02 * iteration
        c -= scale_factor * (Q[0, 0] + Q[1, 1])
        dc -= scale_factor * (dQ[0, 0, :, :] + dQ[1, 1, :, :])

    # === MATLAB: penalty constraints ===
    # if Q(1,1) < delta: c += penalty * (delta - Q(1,1))^2
    # if Q(2,2) < delta: c += penalty * (delta - Q(2,2))^2
    if Q[0, 0] < delta:
        c += penalty * (delta - Q[0, 0]) ** 2
        dc += -2 * penalty * (delta - Q[0, 0]) * dQ[0, 0, :, :]

    if Q[1, 1] < delta:
        c += penalty * (delta - Q[1, 1]) ** 2
        dc += -2 * penalty * (delta - Q[1, 1]) * dQ[1, 1, :, :]

    return c, dc

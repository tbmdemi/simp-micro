"""
Hàm mục tiêu auxetic dùng Q12 (shear coupling) thay vì ν12.

Lý do: gradient dν12/dx ≈ 0 (identity toán học: dQ12/Q12 ≈ dQ22/Q22),
nên ν12 không thể dùng cho topology optimization.

Giải pháp: tối thiểu hóa trực tiếp Q12.
  - Q12 < 0 → auxetic (hệ số Poisson âm)
  - Q12 > 0 → conventional
  - Q12 càng âm → càng auxetic

Cơ chế phạt stiffness: tránh collapse bằng cách phạt khi Q11 hoặc Q22
xuống dưới ngưỡng δ = 0.1 * volfrac * E0.
"""

import numpy as np


def compute_auxetic_q12_objective(
    Q: np.ndarray,
    dQ: np.ndarray,
    volfrac: float,
    E0: float,
    beta: float = 1.0,
) -> tuple:
    """Tối thiểu hóa Q12 (→ auxetic) với ràng buộc phạt stiffness.

    Hàm mục tiêu (dạng minimization cho OC):
        c = Q12 + penalty_terms
    Trong đó penalty kích hoạt khi Q11 < δ hoặc Q22 < δ.

    Args:
        Q: Ten-xơ độ cứng đồng nhất hóa (3×3).
        dQ: Đạo hàm Q theo mật độ (3×3×nely×nelx).
        volfrac: Tỉ lệ thể tích.
        E0: Modul đàn hồi Young.
        beta: Hệ số phạt stiffness.

    Returns:
        (c, dc) với:
            c : Giá trị hàm mục tiêu (vô hướng).
            dc: Mảng (nely, nelx) đạo hàm.
    """
    delta = 0.1 * volfrac * E0
    penalty = beta

    # Mục tiêu chính: tối thiểu hóa Q12 (âm = auxetic)
    c = Q[0, 1]
    dc = dQ[0, 1, :, :].copy()

    # Phạt stiffness: tránh collapse
    if Q[0, 0] < delta:
        c += penalty * (delta - Q[0, 0]) ** 2
        dc += -2 * penalty * (delta - Q[0, 0]) * dQ[0, 0, :, :]

    if Q[1, 1] < delta:
        c += penalty * (delta - Q[1, 1]) ** 2
        dc += -2 * penalty * (delta - Q[1, 1]) * dQ[1, 1, :, :]

    return c, dc

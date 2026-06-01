"""
Hàm mục tiêu auxetic cho tối ưu hóa hình dạng SIMP.

Tối đa hóa hệ số Poisson âm (ν₁₂) để tạo vật liệu auxetic
có tính chất giãn nở âm dưới tác dụng của tải trọng.

Công thức:
    c = ν₁₂ = -Q₁₂ / Q₂₂
    dc = -(dQ₁₂ * Q₂₂ - Q₁₂ * dQ₂₂) / Q₂₂²
"""

import numpy as np


def compute_auxetic_objective(
    Q: np.ndarray,
    dQ: np.ndarray,
) -> tuple:
    """Tính hàm mục tiêu auxetic và đạo hàm của nó.

    Hàm mục tiêu là hệ số Poisson ν₁₂ = -Q₁₂ / Q₂₂.
    Tối ưu hóa nhằm tối thiểu hóa ν₁₂ (càng âm càng tốt).
    Độ nhạy được tính bằng quy tắc đạo hàm thương số.

    Args:
        Q: Ten-xơ độ cứng đồng nhất hóa (3×3).
        dQ: Đạo hàm của Q theo mật độ phần tử (3×3×nely×nelx).

    Returns:
        Bộ (c, dc) với:
            c : Giá trị hàm mục tiêu (ν₁₂, vô hướng).
            dc: Mảng (nely, nelx) đạo hàm của hàm mục tiêu.
    """
    eps = 1e-12

    # Trích xuất thành phần ten-xơ
    Q12 = Q[0, 1]
    Q22 = Q[1, 1]

    # Hệ số Poisson ν₁₂ = -Q₁₂ / Q₂₂
    nu12 = -Q12 / (Q22 + eps)
    c = nu12

    # Độ nhạy: đạo hàm thương số
    d_nu12 = -(dQ[0, 1] * Q22 - Q12 * dQ[1, 1]) / (Q22**2 + eps)
    dc = d_nu12

    return c, dc

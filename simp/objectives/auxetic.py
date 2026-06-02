"""
Hàm mục tiêu auxetic cho tối ưu hóa hình dạng SIMP.

Tối thiểu hóa hệ số Poisson ν₁₂ để tạo vật liệu auxetic
(ν₁₂ < 0, giãn nở âm dưới tác dụng của tải trọng).

Công thức:
    ν₁₂ = Q₁₂ / Q₂₂               (từ compliance tensor S = Q⁻¹)
    c   = ν₁₂                       → minimize để ν₁₂ càng âm càng tốt
    dc  = (dQ₁₂ * Q₂₂ - Q₁₂ * dQ₂₂) / Q₂₂²

Note (2026-02-06): Sửa lỗi công thức. Trước đây dùng ν₁₂ = -Q₁₂/Q₂₂
dẫn đến tối ưu tìm ν₁₂ → +1 (vật liệu dương cực đại), sai mục tiêu auxetic.
"""

import numpy as np


def compute_auxetic_objective(
    Q: np.ndarray,
    dQ: np.ndarray,
) -> tuple:
    """Tính hàm mục tiêu auxetic và đạo hàm của nó.

    Hàm mục tiêu là hệ số Poisson ν₁₂ = Q₁₂ / Q₂₂.
    Tối ưu hóa nhằm tối thiểu hóa ν₁₂ (giá trị âm = auxetic).
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

    # Công thức đúng: ν₁₂ = Q₁₂ / Q₂₂ (xem derivation trong docstring)
    nu12 = Q12 / (Q22 + eps)
    c = nu12  # minimize → ν₁₂ càng âm (auxetic) càng tốt

    # Độ nhạy: đạo hàm thương số d(Q₁₂/Q₂₂)/dx
    d_nu12 = (dQ[0, 1] * Q22 - Q12 * dQ[1, 1]) / (Q22**2 + eps)
    dc = d_nu12

    return c, dc

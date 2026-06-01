"""
Mẫu seed chữ thập vuông nhỏ cho tối ưu hóa hình dạng SIMP.

Tạo trường mật độ ban đầu với một chữ thập vuông nhỏ ở tâm ô cơ sở.
"""

import numpy as np


def small_square_cross_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Tạo mẫu chữ thập vuông nhỏ.

    Args:
        nelx: Số phần tử theo phương x.
        nely: Số phần tử theo phương y.
        void_size_frac: Tỉ lệ kích thước lỗ rỗng (so với cạnh ngắn nhất).
        rotation_deg: Góc xoay mẫu seed (độ).

    Returns:
        Mảng (nely, nelx) mật độ ban đầu.
    """
    x = np.ones((nely, nelx))
    cx, cy = nelx / 2, nely / 2
    
    # Kích thước ô vuông s dựa trên void_size_frac
    # Tổng chiều dài chữ thập là 3s, nên s = void_size_frac * min / 3
    s = void_size_frac * min(nelx, nely) / 3
    
    # Góc xoay (radian)
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            # Chuyển đổi tọa độ xoay
            dx, dy = i - cx, j - cy
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t
            
            # Kiểm tra nếu điểm nằm trong thanh ngang hoặc thanh dọc của chữ thập
            in_horizontal = (abs(ny) < s / 2) and (abs(nx) < 3 * s / 2)
            in_vertical = (abs(ny) < 3 * s / 2) and (abs(nx) < s / 2)
            
            if in_horizontal or in_vertical:
                x[j, i] = 0.0

    return x

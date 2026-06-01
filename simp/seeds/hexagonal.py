"""
Mẫu seed lỗ lục giác cho tối ưu hóa hình dạng SIMP.

Tạo trường mật độ ban đầu với một lỗ lục giác ở tâm ô cơ sở.
"""

import numpy as np


def hexagonal_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Tạo mẫu lỗ lục giác.

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
    
    # Bán kính lục giác dựa trên void_size_frac
    r = void_size_frac * min(nelx, nely) / 2
    
    # Góc xoay (radian)
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            # Chuyển đổi tọa độ xoay
            dx, dy = i - cx, j - cy
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t
            
            # Chuyển sang tọa độ lục giác chuẩn hóa
            qx = nx / r
            qy = ny / r
            
            # Kiểm tra nếu điểm nằm trong lục giác
            if (abs(qx) < 1 and abs(qy) < np.sqrt(3) / 2 and
                abs(qx) + abs(qy) / np.sqrt(3) < 1):
                x[j, i] = 0.0

    return x

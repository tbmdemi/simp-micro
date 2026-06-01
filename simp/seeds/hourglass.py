"""
Mẫu seed hình đồng hồ cát cho tối ưu hóa hình dạng SIMP.

Tạo trường mật độ ban đầu với hai lỗ hình tam giác
tạo thành hình đồng hồ cát.
"""

import numpy as np


def hourglass_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Tạo mẫu hình đồng hồ cát.

    Args:
        nelx: Số phần tử theo phương x.
        nely: Số phần tử theo phương y.
        void_size_frac: Tỉ lệ kích thước lỗ rỗng (so với chiều rộng).
        rotation_deg: Góc xoay mẫu seed (độ).

    Returns:
        Mảng (nely, nelx) mật độ ban đầu.
    """
    x = np.ones((nely, nelx))
    cx, cy = nelx / 2, nely / 2
    
    # Hệ số dốc dựa trên void_size_frac
    # Chiều rộng đáy = void_size_frac * nelx
    # Chiều cao tam giác = nely / 2
    # Slope = (void_size_frac * nelx / 2) / (nely / 2) = void_size_frac * nelx / nely
    slope = void_size_frac * nelx / nely
    
    # Góc xoay (radian)
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            # Chuyển đổi tọa độ xoay
            dx, dy = i - cx, j - cy
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t
            
            # Hai hình tam giác: trên và dưới (trong hệ tọa độ xoay)
            in_upper = (ny > 0) and (abs(nx) < ny * slope)
            in_lower = (ny < 0) and (abs(nx) < -ny * slope)
            
            if in_upper or in_lower:
                x[j, i] = 0.0

    return x

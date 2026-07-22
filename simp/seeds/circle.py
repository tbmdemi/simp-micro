"""
Mẫu seed: lỗ tròn đơn ở tâm ô cơ sở cho tối ưu hóa hình dạng SIMP.
"""

import numpy as np


def circle_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Trường mật độ ban đầu (nely, nelx) với một lỗ tròn ở tâm, bán kính = void_size_frac * min(nelx, nely) / 2."""
    x = np.ones((nely, nelx))
    cx, cy = nelx / 2, nely / 2
    r = void_size_frac * min(nelx, nely) / 2

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            # Xoay tọa độ quanh tâm trước khi kiểm tra biên lỗ
            dx, dy = i - cx, j - cy
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t

            if nx**2 + ny**2 < r**2:
                x[j, i] = 0.0

    return x

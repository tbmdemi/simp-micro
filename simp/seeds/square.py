"""
Mẫu seed: lỗ vuông đơn ở tâm ô cơ sở cho tối ưu hóa hình dạng SIMP.
"""

import numpy as np


def square_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Trường mật độ ban đầu (nely, nelx) với một lỗ vuông ở tâm."""
    x = np.ones((nely, nelx))
    cx, cy = nelx / 2, nely / 2
    side = void_size_frac * min(nelx, nely)
    half = side / 2

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            dx, dy = i - cx, j - cy
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t

            if abs(nx) < half and abs(ny) < half:
                x[j, i] = 0.0

    return x

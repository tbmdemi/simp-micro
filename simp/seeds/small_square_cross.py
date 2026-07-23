"""
Mẫu seed: chữ thập vuông nhỏ ở tâm ô cơ sở cho tối ưu hóa hình dạng SIMP.
"""

import numpy as np


def small_square_cross_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Trường mật độ ban đầu (nely, nelx) với chữ thập tâm; tổng chiều dài chữ thập = 3s nên s = void_size_frac*min/3."""
    x = np.ones((nely, nelx))
    cx, cy = nelx / 2, nely / 2
    s = void_size_frac * min(nelx, nely) / 3

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            dx, dy = i - cx, j - cy
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t

            in_horizontal = (abs(ny) < s / 2) and (abs(nx) < 3 * s / 2)
            in_vertical = (abs(ny) < 3 * s / 2) and (abs(nx) < s / 2)

            if in_horizontal or in_vertical:
                x[j, i] = 0.0

    return x

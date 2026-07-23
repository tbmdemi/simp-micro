"""
Mẫu seed: lỗ hình chữ thập ở tâm ô cơ sở cho tối ưu hóa hình dạng SIMP.
"""

import numpy as np


def cross_rectangular_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Trường mật độ ban đầu (nely, nelx) với lỗ chữ thập tâm; bề ngang w = h/2."""
    x = np.ones((nely, nelx))
    cx, cy = nelx / 2, nely / 2
    h = void_size_frac * min(nelx, nely)
    w = h / 2

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            dx, dy = i - cx, j - cy
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t

            in_horizontal = (abs(ny) < w / 2) and (abs(nx) < h / 2)
            in_vertical = (abs(ny) < h / 2) and (abs(nx) < w / 2)

            if in_horizontal or in_vertical:
                x[j, i] = 0.0

    return x

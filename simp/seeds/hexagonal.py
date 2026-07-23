"""
Mẫu seed: lỗ lục giác ở tâm ô cơ sở cho tối ưu hóa hình dạng SIMP.
"""

import numpy as np


def hexagonal_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Trường mật độ ban đầu (nely, nelx) với lỗ lục giác đều ở tâm."""
    x = np.ones((nely, nelx))
    cx, cy = nelx / 2, nely / 2
    r = void_size_frac * min(nelx, nely) / 2

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            dx, dy = i - cx, j - cy
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t

            # Chuẩn hóa theo bán kính rồi test biên lục giác đều
            # (|qx|<1, |qy|<sqrt(3)/2, |qx|+|qy|/sqrt(3)<1)
            qx = nx / r
            qy = ny / r

            if (abs(qx) < 1 and abs(qy) < np.sqrt(3) / 2 and
                abs(qx) + abs(qy) / np.sqrt(3) < 1):
                x[j, i] = 0.0

    return x

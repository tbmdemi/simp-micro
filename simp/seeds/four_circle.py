"""
Mẫu seed: bốn lỗ tròn bố trí đối xứng quanh tâm ô cơ sở.
"""

import numpy as np


def four_circle_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Trường mật độ ban đầu (nely, nelx) với 4 lỗ tròn, tâm lệch min(nelx,nely)/4 theo mỗi trục."""
    x = np.ones((nely, nelx))
    cx, cy = nelx / 2, nely / 2
    r = void_size_frac * min(nelx, nely) / 4
    offset = min(nelx, nely) / 4

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            dx, dy = i - cx, j - cy
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t

            if ((nx - offset)**2 + (ny - offset)**2 < r**2 or
                (nx + offset)**2 + (ny - offset)**2 < r**2 or
                (nx - offset)**2 + (ny + offset)**2 < r**2 or
                (nx + offset)**2 + (ny + offset)**2 < r**2):
                x[j, i] = 0.0

    return x

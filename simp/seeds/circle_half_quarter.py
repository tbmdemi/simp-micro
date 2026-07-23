"""
Mẫu seed: lỗ tròn ở tâm cộng bốn lỗ 1/4 hình tròn ở các góc ô cơ sở.
"""

import numpy as np


def circle_half_quarter_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Trường mật độ ban đầu (nely, nelx): lỗ tròn tâm + 4 lỗ 1/4 hình tròn ở góc."""
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

            # Lỗ tâm, rồi 4 lỗ góc (mỗi elif ứng với một góc phần tư,
            # tâm lỗ dịch (+-cx, +-cy) để nằm đúng góc ô)
            if nx**2 + ny**2 < r**2:
                x[j, i] = 0.0
            elif (nx < 0 and ny < 0 and
                  (nx + cx)**2 + (ny + cy)**2 < r**2):
                x[j, i] = 0.0
            elif (nx > 0 and ny < 0 and
                  (nx - cx)**2 + (ny + cy)**2 < r**2):
                x[j, i] = 0.0
            elif (nx < 0 and ny > 0 and
                  (nx + cx)**2 + (ny - cy)**2 < r**2):
                x[j, i] = 0.0
            elif (nx > 0 and ny > 0 and
                  (nx - cx)**2 + (ny - cy)**2 < r**2):
                x[j, i] = 0.0

    return x

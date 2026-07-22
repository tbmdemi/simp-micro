"""
Mẫu seed: chín lỗ tròn bố trí lưới 3x3 cho tối ưu hóa hình dạng SIMP.
"""

import numpy as np


def nine_circle_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Trường mật độ ban đầu (nely, nelx) với 9 lỗ tròn lưới 3x3; r chia /6 vì 3 lỗ chiếm trọn chiều rộng."""
    x = np.ones((nely, nelx))
    cx_dom, cy_dom = nelx / 2, nely / 2

    r = void_size_frac * min(nelx, nely) / 6

    rel_centers = []
    for ci in range(3):
        for cj in range(3):
            rel_cx = (ci + 0.5) * nelx / 3 - cx_dom
            rel_cy = (cj + 0.5) * nely / 3 - cy_dom
            rel_centers.append((rel_cx, rel_cy))

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            dx, dy = i - cx_dom, j - cy_dom
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t

            for rcx, rcy in rel_centers:
                if (nx - rcx)**2 + (ny - rcy)**2 < r**2:
                    x[j, i] = 0.0
                    break

    return x

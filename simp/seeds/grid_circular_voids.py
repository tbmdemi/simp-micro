"""
Mẫu seed: lưới lỗ tròn đều đặn phủ toàn bộ ô cơ sở.
"""

import numpy as np


def grid_circular_voids_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Trường mật độ ban đầu (nely, nelx) với lưới lỗ tròn, void_size_frac tính theo khoảng cách lưới (không phải cạnh ô)."""
    x = np.ones((nely, nelx))
    cx_dom, cy_dom = nelx / 2, nely / 2

    # Tối thiểu 3x3 lỗ; mật độ lưới ~1 lỗ/20 phần tử
    n_holes_x = max(3, nelx // 20)
    n_holes_y = max(3, nely // 20)

    r = void_size_frac * (nelx / n_holes_x) / 2

    rel_centers = []
    for hi in range(n_holes_x):
        for hj in range(n_holes_y):
            rel_cx = (hi + 0.5) * nelx / n_holes_x - cx_dom
            rel_cy = (hj + 0.5) * nely / n_holes_y - cy_dom
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

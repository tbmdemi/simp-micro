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
            # Xoay tọa độ quanh tâm trước khi kiểm tra biên lỗ.
            # Bug đã sửa 2026-08-15: phần tử i chiếm khoảng [i, i+1] trong hệ
            # tọa độ nút (build_dof_mesh), TÂM thật của nó ở i+0.5, không phải
            # i - trước đây dùng i-cx (thiếu +0.5) khiến seed lệch nửa pixel,
            # phá đối xứng gương dự định (circle_seed(40,40,0.5,0) không đối
            # xứng trái-phải, 38/1600 pixel lệch). CHỈ áp dụng cho code sinh
            # dữ liệu MỚI - dataset 57k mẫu hiện có đã sinh bằng công thức cũ,
            # không rebuild lại (xem docs/LIMITATIONS.md).
            dx, dy = (i + 0.5) - cx, (j + 0.5) - cy
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t

            if nx**2 + ny**2 < r**2:
                x[j, i] = 0.0

    return x

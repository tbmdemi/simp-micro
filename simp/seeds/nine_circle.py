"""
Mẫu seed chín lỗ tròn cho tối ưu hóa hình dạng SIMP.

Tạo trường mật độ ban đầu với chín lỗ tròn bố trí lưới 3×3.
"""

import numpy as np


def nine_circle_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Tạo mẫu chín lỗ tròn (lưới 3×3).

    Args:
        nelx: Số phần tử theo phương x.
        nely: Số phần tử theo phương y.
        void_size_frac: Tỉ lệ kích thước lỗ rỗng (so với cạnh ngắn nhất).
        rotation_deg: Góc xoay mẫu seed (độ).

    Returns:
        Mảng (nely, nelx) mật độ ban đầu.
    """
    x = np.ones((nely, nelx))
    cx_dom, cy_dom = nelx / 2, nely / 2
    
    # Bán kính dựa trên void_size_frac (3 lỗ tròn theo chiều rộng)
    r = void_size_frac * min(nelx, nely) / 6
    
    # Chín tâm lỗ tròn bố trí lưới 3×3 (tọa độ tương đối so với tâm domain)
    rel_centers = []
    for ci in range(3):
        for cj in range(3):
            rel_cx = (ci + 0.5) * nelx / 3 - cx_dom
            rel_cy = (cj + 0.5) * nely / 3 - cy_dom
            rel_centers.append((rel_cx, rel_cy))
    
    # Góc xoay (radian)
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            # Chuyển đổi tọa độ xoay quanh tâm domain
            dx, dy = i - cx_dom, j - cy_dom
            nx = dx * cos_t - dy * sin_t
            ny = dx * sin_t + dy * cos_t
            
            for rcx, rcy in rel_centers:
                if (nx - rcx)**2 + (ny - rcy)**2 < r**2:
                    x[j, i] = 0.0
                    break

    return x

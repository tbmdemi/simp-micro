"""
Mẫu seed lưới lỗ tròn cho tối ưu hóa hình dạng SIMP.

Tạo trường mật độ ban đầu với lưới các lỗ tròn đều đặn.
"""

import numpy as np


def grid_circular_voids_seed(nelx: int, nely: int, void_size_frac: float, rotation_deg: float) -> np.ndarray:
    """Tạo mẫu lưới lỗ tròn.

    Args:
        nelx: Số phần tử theo phương x.
        nely: Số phần tử theo phương y.
        void_size_frac: Tỉ lệ kích thước lỗ rỗng (so với khoảng cách giữa các lỗ).
        rotation_deg: Góc xoay mẫu seed (độ).

    Returns:
        Mảng (nely, nelx) mật độ ban đầu.
    """
    x = np.ones((nely, nelx))
    cx_dom, cy_dom = nelx / 2, nely / 2
    
    n_holes_x = max(3, nelx // 20)
    n_holes_y = max(3, nely // 20)
    
    # Bán kính dựa trên void_size_frac và khoảng cách lưới
    # Đường kính = void_size_frac * (nelx / n_holes_x)
    r = void_size_frac * (nelx / n_holes_x) / 2
    
    # Danh sách các tâm lỗ tròn (tọa độ tương đối so với tâm domain)
    rel_centers = []
    for hi in range(n_holes_x):
        for hj in range(n_holes_y):
            rel_cx = (hi + 0.5) * nelx / n_holes_x - cx_dom
            rel_cy = (hj + 0.5) * nely / n_holes_y - cy_dom
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

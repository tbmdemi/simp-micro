"""
Mẫu seed hình đồng hồ cát (MATLAB-style) cho tối ưu hóa hình dạng SIMP:
vùng lõm "soft void" (mật độ volfrac/2) hình đồng hồ cát, góc nghiêng 50°,
trên nền đặc mật độ volfrac.
"""

import numpy as np

def hourglass_seed(nelx: int, nely: int, volfrac: float, rotation_deg: float = 0.0) -> np.ndarray:
    """Trường mật độ ban đầu (nely, nelx): nền = volfrac, vùng đồng hồ cát = volfrac/2 (soft void, không phải 0)."""
    x = np.full((nely, nelx), volfrac)

    # Tham số hình học cố định, khớp bản gốc MATLAB
    hourglass_width = nelx / 14     # bề rộng tại eo
    hourglass_height = nely / 5     # chiều cao từ tâm đến đỉnh/đáy
    center_x = nelx / 2
    center_y = nely / 2
    taper_angle_deg = 50            # góc nghiêng từ eo ra ngoài
    taper_slope = np.tan(np.radians(taper_angle_deg))

    theta = np.radians(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            dx, dy = i - center_x, j - center_y
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t

            vertical_dist = abs(ry)
            # bề rộng nở dần từ eo ra theo taper_slope
            taper_width = hourglass_width + taper_slope * vertical_dist
            if abs(rx) < taper_width and vertical_dist < hourglass_height:
                x[j, i] = volfrac / 2

    return x

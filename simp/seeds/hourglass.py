"""
Mẫu seed hình đồng hồ cát cho tối ưu hóa hình dạng SIMP.

Tạo trường mật độ ban đầu với hai lỗ hình tam giác
tạo thành hình đồng hồ cát.
"""

import numpy as np

def hourglass_seed(nelx: int, nely: int, volfrac: float, rotation_deg: float = 0.0) -> np.ndarray:
    """Tạo mẫu hình đồng hồ cát (MATLAB-style).

    Tạo trường mật độ với vùng lõm hình đồng hồ cát có góc nghiêng 50°.
    Vùng lõm có mật độ volfrac/2, nền có mật độ volfrac.
    Hỗ trợ rotation qua phép xoay tọa độ (như các seed khác).

    Args:
        nelx: Số phần tử theo phương x.
        nely: Số phần tử theo phương y.
        volfrac: Tỉ lệ thể tích (dùng cho mật độ nền và vùng lõm).
        rotation_deg: Góc xoay mẫu seed (độ, mặc định 0).

    Returns:
        Mảng (nely, nelx) mật độ ban đầu.
    """
    # Mật độ nền = volfrac (giống MATLAB)
    x = np.full((nely, nelx), volfrac)

    # Tham số hình học cố định (giống MATLAB)
    hourglass_width = nelx / 14     # Bề rộng tại eo
    hourglass_height = nely / 5     # Chiều cao từ tâm đến đỉnh/đáy
    center_x = nelx / 2
    center_y = nely / 2
    taper_angle_deg = 50            # Góc nghiêng từ eo ra ngoài
    taper_slope = np.tan(np.radians(taper_angle_deg))

    # Ma trận xoay
    theta = np.radians(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    for i in range(nelx):
        for j in range(nely):
            # Chuyển về tọa độ tâm
            dx, dy = i - center_x, j - center_y
            # Xoay tọa độ (ngược chiều kim đồng hồ)
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t

            vertical_dist = abs(ry)
            # Bề rộng tại độ cao vertical_dist (nở rộng từ eo ra)
            taper_width = hourglass_width + taper_slope * vertical_dist
            # Nếu nằm trong vùng đồng hồ cát
            if abs(rx) < taper_width and vertical_dist < hourglass_height:
                x[j, i] = volfrac / 2  # "soft void" (giống MATLAB)

    return x

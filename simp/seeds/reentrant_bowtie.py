"""
Mẫu seed reentrant bowtie (nơ bướm) - cơ chế chuẩn cho auxetic honeycomb.

4 thanh chéo nối từ tâm ra 4 góc theo góc reentrant (< 90 độ so với
phương ngang), mô phỏng cơ chế "gập-xoay" (rotating hinge) đặc trưng của
reentrant honeycomb (Lakes, 1987; Gibson & Ashby).

Khác biệt cốt lõi so với các seed khác: nền là VOID, chỉ thanh chéo là
SOLID (hourglass_seed thì ngược lại - nền đặc, lõm cục bộ). Seed nền-đặc
thường hội tụ về nghiệm gần isotropic (v12~nu vật liệu nền) vì đã kẹt gần
local optimum đẳng hướng; seed nền-thưa có cơ chế gập-xoay sẵn nên giữ
được tính auxetic tốt hơn qua tối ưu. Đã kiểm chứng: kết hợp volfrac~0.5,
seed này cho v12 gần 0 hơn đáng kể so với seed nền-đặc (sau khi sửa lỗi
U0+U trong runner.py).
"""

import numpy as np


def reentrant_bowtie_seed(
    nelx: int,
    nely: int,
    volfrac: float,
    rotation_deg: float = 0.0,
    reentrant_angle_deg: float = 60.0,
    strut_width_frac: float = 0.12,
) -> np.ndarray:
    """Reentrant bowtie: 4 thanh chéo dạng chữ X co góc reentrant, nền void.

    reentrant_angle_deg: góc thanh chéo so với phương ngang, < 90 độ tạo hình
        "nơ bướm" lõm reentrant. Điển hình 50-70 độ; 60 độ tốt nhất trong thử
        nghiệm sơ bộ.
    strut_width_frac: độ dày thanh chéo, phần trăm của min(nelx, nely).
        Điển hình 0.08-0.15.
    """
    # Nền = void nhẹ (khác hourglass_seed dùng nền đặc = volfrac)
    x = np.full((nely, nelx), max(0.01, volfrac * 0.15))

    center_x = nelx / 2.0
    center_y = nely / 2.0
    R = min(nelx, nely) * 0.46  # bán kính thanh chéo, gần chạm biên cell

    theta = np.radians(rotation_deg)
    alpha = np.radians(reentrant_angle_deg)
    strut_hw = strut_width_frac * min(nelx, nely) / 2.0  # nửa độ dày thanh

    # 4 hướng thanh chéo tạo hình bowtie reentrant (đối xứng qua tâm)
    directions = [alpha, np.pi - alpha, np.pi + alpha, -alpha]

    ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
    dx = (ii - center_x).astype(float)
    dy = (jj - center_y).astype(float)

    # Áp dụng rotation toàn cục
    rx = dx * np.cos(theta) - dy * np.sin(theta)
    ry = dx * np.sin(theta) + dy * np.cos(theta)

    solid_mask = np.zeros((nely, nelx), dtype=bool)
    for d in directions:
        ux, uy = np.cos(d), np.sin(d)
        # Khoảng cách vuông góc từ điểm đến đường thẳng qua tâm hướng (ux,uy)
        perp_dist = np.abs(rx * (-uy) + ry * ux)
        # Khoảng cách dọc theo hướng thanh (giới hạn chiều dài <= R)
        along_dist = rx * ux + ry * uy
        strut = (perp_dist < strut_hw) & (along_dist >= 0) & (along_dist <= R)
        solid_mask |= strut

    x[solid_mask] = min(1.0, volfrac * 2.2)

    return x
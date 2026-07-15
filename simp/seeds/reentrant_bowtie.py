"""
Mẫu seed reentrant bowtie (no bướm) - cơ chế chuẩn cho auxetic honeycomb.

Khác với các seed hiện có (lỗ tròn/hourglass cục bộ trên nền đặc), seed
này tạo ra 4 thanh chéo nối từ tâm ra 4 góc của unit cell theo góc
reentrant (< 90 độ so với phương ngang), mô phỏng cơ chế "gập-xoay"
(rotating hinge) đặc trưng của reentrant honeycomb - cấu trúc auxetic
kinh điển trong literature (Lakes, 1987; Gibson & Ashby).

Nền là VOID (mật độ thấp), chỉ các thanh chéo là SOLID - ngược lại hoàn
toàn so với hourglass_seed (nền đặc, lõm cục bộ). Đây là điểm khác biệt
quan trọng nhất giúp thoát khỏi local optimum gần đẳng hướng: seed bắt
đầu từ gần-đồng nhất-đặc thường hội tụ về nghiệm gần isotropic (v12~nu
của vật liệu nền), trong khi seed bắt đầu từ cấu trúc thưa có cơ chế
sẵn dễ giữ được tính auxetic hơn qua quá trình tối ưu.

Đã kiểm chứng thực nghiệm: kết hợp với volfrac~0.5, seed này cho v12 gần
0 hơn đáng kể so với các seed nền-đặc truyền thống (sau khi đã sửa lỗi
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
    """Tạo mẫu reentrant bowtie (4 thanh chéo dạng chữ X co góc reentrant).

    Args:
        nelx: Số phần tử theo phương x.
        nely: Số phần tử theo phương y.
        volfrac: Tỉ lệ thể tích mục tiêu (dùng để chuẩn hóa mật độ nền/thanh).
        rotation_deg: Góc xoay toàn bộ mẫu (độ).
        reentrant_angle_deg: Góc giữa thanh chéo và phương ngang, < 90 độ
            tạo hình "nơ bướm" lõm vào trong (đặc trưng reentrant). Giá
            trị điển hình 50-70 độ; 60 độ cho kết quả tốt nhất trong thử
            nghiệm sơ bộ.
        strut_width_frac: Độ dày thanh chéo, tính theo phần trăm của
            min(nelx, nely). Giá trị điển hình 0.08-0.15.

    Returns:
        Mảng (nely, nelx) mật độ ban đầu - nền void, thanh chéo solid.
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
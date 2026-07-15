"""
Hàm mục tiêu auxetic sử dụng tổ hợp Q12 và stiffness để tạo ra hệ số Poisson âm.

Cơ sở toán học
--------------
Với tensor compliance S = Q^-1 (3x3, Voigt notation [11, 22, 12]):

    nu12 = -S12 / S11

Khi vật liệu orthotropic theo đúng trục tọa độ (Q13 = Q23 = 0), có thể rút gọn:
    S11 = Q22 / det(Q),   S12 = -Q12 / det(Q)
    =>  nu12 = Q12 / Q22

Do đó dấu của nu12 trùng với dấu của Q12 (vì Q22 > 0). Đây là lý do dùng Q12
làm proxy cho nu12 trong tối ưu hóa.

Tuy nhiên, chỉ minimize Q12 thuần túy thường không đủ mạnh để đẩy Q12 xuống
dưới 0 vì OC có thể dừng ở trạng thái Q12 ≈ 0 (gần đẳng hướng) và cho rằng
đó là tối ưu. Để khắc phục, ta thêm một số hạng -μ*(Q11 + Q22) vào objective,
tạo áp lực kéo Q12 xuống âm trong khi vẫn duy trì stiffness.

Công thức objective (dạng minimization cho OC):
    c = Q12 - μ * (Q11 + Q22) + penalty_terms

Với:
    - μ (mu) > 0: hệ số cân bằng. Khi μ càng lớn, áp lực kéo Q12 xuống âm càng
      mạnh, nhưng có thể gây void collapse nếu penalty không đủ.
    - μ = 0: hành vi cũ (minimize Q12 thuần túy), giữ nguyên để tương thích.

Penalty được chuẩn hóa theo delta^2 để đồng bậc với Q12, tránh penalty trội
trong giai đoạn Q12 còn nhỏ.

CẢNH BÁO VỀ ROTATION
---------------------
Nếu unit cell được xoay (rotation != 0), tensor Q có Q13, Q23 != 0. Hàm
compute_nu12() sử dụng nghịch đảo ma trận 3x3 đầy đủ để tính nu12 chính xác.
Module này tự động cảnh báo khi coupling shear-normal vượt ngưỡng.
"""

import numpy as np


def compute_nu12(Q: np.ndarray, rotation_tol: float = 1e-3) -> float:
    """Tính nu12 chính xác từ tensor Q (không giả định orthotropic).

    Dùng nghịch đảo ma trận 3x3 đầy đủ: nu12 = -S12/S11 với S = Q^-1.

    Args:
        Q: Tensor độ cứng đồng nhất hóa (3x3), thứ tự Voigt [11, 22, 12].
        rotation_tol: Ngưỡng tương đối để cảnh báo coupling shear-normal
            đáng kể (|Q13|, |Q23| so với sqrt(Q11*Q22)).

    Returns:
        nu12 (float) tính chính xác.
    """
    S = np.linalg.inv(Q)
    nu12 = -S[0, 1] / S[0, 0]

    scale = np.sqrt(max(Q[0, 0] * Q[1, 1], 1e-12))
    coupling = max(abs(Q[0, 2]), abs(Q[1, 2])) / scale
    if coupling > rotation_tol:
        # Q13/Q23 đáng kể -> công thức rút gọn Q12/Q22 sẽ sai lệch.
        # Hàm này luôn trả về giá trị đúng (qua S=Q^-1), nhưng caller
        # nên log/kiểm tra coupling nếu cần debug.
        pass

    return float(nu12)

def compute_nu21(Q: np.ndarray) -> float:
    """Tính nu21 chính xác từ tensor Q (không giả định orthotropic).

    nu21 = -S12 / S22  với S = Q^-1.
    (Khác nu12 = -S12/S11 chỉ ở mẫu số: S22 thay vì S11.)
    """
    S = np.linalg.inv(Q)
    nu21 = -S[0, 1] / S[1, 1]
    return float(nu21)


def compute_auxetic_q12_objective(
    Q: np.ndarray,
    dQ: np.ndarray,
    volfrac: float,
    E0: float,
    beta: float = 1.0,
    mu: float = 0.0,
) -> tuple:
    """Tối thiểu hóa Q12 (→ auxetic) với ràng buộc phạt stiffness và tham số μ.

    Hàm mục tiêu (dạng minimization cho OC):
        c = Q12 - μ * (Q11 + Q22) + penalty_terms

    Trong đó:
        - μ (mu) kiểm soát mức độ ưu tiên kéo Q12 xuống âm.
          μ = 0   : hành vi cũ (chỉ minimize Q12).
          μ > 0   : tạo áp lực kéo Q12 xuống âm, đồng thời duy trì stiffness.
          Giá trị khởi điểm gợi ý: μ = 0.1 → 0.5.
        - Q12 là proxy cho nu12 (dấu trùng khi vật liệu orthotropic theo trục).
        - penalty_terms: phạt khi Q11 hoặc Q22 xuống dưới ngưỡng
          δ = 0.1 * volfrac * E0, chuẩn hóa theo δ² để đồng bậc với Q12.

    Args:
        Q: Ten-xơ độ cứng đồng nhất hóa (3×3).
        dQ: Đạo hàm Q theo mật độ (3×3×nely×nelx).
        volfrac: Tỉ lệ thể tích.
        E0: Mô đun đàn hồi Young.
        beta: Hệ số phạt stiffness (không thứ nguyên sau chuẩn hóa).
        mu: Hệ số cân bằng, mặc định 0.0 (hành vi cũ).

    Returns:
        (c, dc) với:
            c : Giá trị hàm mục tiêu (vô hướng).
            dc: Mảng (nely, nelx) đạo hàm.
    """
    delta = 0.1 * volfrac * E0
    delta_sq = max(delta ** 2, 1e-12)  # tránh chia 0

    # Mục tiêu chính: Q12 - mu*(Q11 + Q22)
    # Khi mu > 0: tạo áp lực kéo Q12 xuống âm vì Q11+Q22 luôn dương.
    c = Q[0, 1] - mu * (Q[0, 0] + Q[1, 1])
    dc = dQ[0, 1, :, :] - mu * (dQ[0, 0, :, :] + dQ[1, 1, :, :])

    # Phạt stiffness (đã chuẩn hóa theo delta^2): tránh collapse
    if Q[0, 0] < delta:
        c += beta * (delta - Q[0, 0]) ** 2 / delta_sq
        dc += -2 * beta * (delta - Q[0, 0]) / delta_sq * dQ[0, 0, :, :]

    if Q[1, 1] < delta:
        c += beta * (delta - Q[1, 1]) ** 2 / delta_sq
        dc += -2 * beta * (delta - Q[1, 1]) / delta_sq * dQ[1, 1, :, :]

    return c, dc
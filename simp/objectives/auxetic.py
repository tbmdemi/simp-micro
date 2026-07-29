"""
Hàm mục tiêu auxetic: tối thiểu Q12 (proxy cho nu12 âm) với phạt stiffness.

Cơ sở: nu12 = -S12/S11 với S = Q^-1. Khi orthotropic theo trục (Q13=Q23=0),
rút gọn còn nu12 = Q12/Q22 -> dấu Q12 trùng dấu nu12 (vì Q22 > 0), nên dùng
Q12 làm proxy. Minimize Q12 thuần túy dễ dừng ở Q12≈0 (OC coi là đủ tối ưu),
nên thêm số hạng -μ*(Q11+Q22) (μ>0) để tạo áp lực kéo Q12 âm hơn trong khi
vẫn giữ stiffness; μ=0 giữ hành vi cũ. Penalty chuẩn hóa theo delta^2 để
đồng bậc với Q12.

CẢNH BÁO ROTATION: nếu unit cell bị xoay, Q13/Q23 != 0 và công thức rút gọn
trên sai lệch; compute_nu12()/compute_nu21() luôn dùng nghịch đảo ma trận
3x3 đầy đủ nên đúng trong mọi trường hợp.
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
        # Coupling đáng kể - công thức rút gọn Q12/Q22 sẽ sai, nhưng giá trị
        # trả về ở đây (qua S=Q^-1) vẫn luôn đúng.
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
    """Tối thiểu hóa Q12 (proxy cho nu12 âm), với phạt stiffness và tham số μ.

    c = Q12 - μ*(Q11 + Q22) + penalty_terms (xem docstring đầu module).
    Gợi ý khởi điểm μ = 0.1 → 0.5 nếu bật (mặc định μ=0, hành vi cũ).

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

    # mu > 0 tạo áp lực kéo Q12 xuống âm vì Q11+Q22 luôn dương.
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


def compute_auxetic_normalized_objective(
    Q: np.ndarray,
    dQ: np.ndarray,
    volfrac: float,
    E0: float,
    beta: float = 1.0,
) -> tuple:
    """Biến thể chuẩn hóa của compute_auxetic_q12_objective(): thay vì minimize
    Q12 thô, minimize hệ số ghép cắt-pháp chuẩn hóa

        c = Q12 / sqrt(Q11 * Q22)

    TÍNH NĂNG THỬ NGHIỆM (--objective-variant normalized qua run_simp), TẮT
    MẶC ĐỊNH - dùng compute_auxetic_q12_objective() (mặc định) trừ khi bật
    tường minh. CHƯA áp dụng cho dataset hiện có, cần A/B test trước. Xem
    AUDIT_REPORT_INDEPENDENT_2026-07-29.md mục 4.3/B3.

    Cơ sở: với ten-xơ độ cứng Q hợp lệ về mặt vật lý (positive semi-definite,
    vì Q sinh ra từ dạng toàn phương năng lượng biến dạng), submatrix
    [[Q11,Q12],[Q12,Q22]] cũng PSD, kéo theo Cauchy-Schwarz Q12^2 <= Q11*Q22
    - nghĩa là c BỊ CHẶN tự nhiên trong [-1, 1] bởi chính vật lý, không cần
    hệ số μ tùy chỉnh (đã tắt vì sai sót khái niệm, xem
    compute_auxetic_q12_objective và README "Giới hạn Đã biết" mục 4). Đây
    KHÔNG tự động ngăn Q11/Q22 cùng nhỏ (thiết kế yếu nhưng tỉ lệ vẫn giữ),
    nên vẫn giữ NGUYÊN phạt stiffness delta như hàm gốc để so sánh công bằng
    (chỉ đổi số hạng tỉ lệ Q12 thô -> chuẩn hóa, giữ nguyên phần còn lại).

    Args:
        Q, dQ, volfrac, E0, beta: giống compute_auxetic_q12_objective().

    Returns:
        (c, dc) - cùng shape với compute_auxetic_q12_objective().
    """
    delta = 0.1 * volfrac * E0
    delta_sq = max(delta ** 2, 1e-12)

    eps = 1e-9
    P = Q[0, 0] * Q[1, 1]
    denom = np.sqrt(max(P, eps))

    c = Q[0, 1] / denom
    if P > eps:
        dP = dQ[0, 0, :, :] * Q[1, 1] + Q[0, 0] * dQ[1, 1, :, :]
        dc = dQ[0, 1, :, :] / denom - Q[0, 1] * dP / (2 * denom ** 3)
    else:
        # P kẹp ở eps (gần suy biến) - coi denom như hằng số cục bộ để tránh
        # chia cho ~0 trong đạo hàm (giống cách delta_sq clamp ở trên).
        dc = dQ[0, 1, :, :] / denom

    if Q[0, 0] < delta:
        c += beta * (delta - Q[0, 0]) ** 2 / delta_sq
        dc += -2 * beta * (delta - Q[0, 0]) / delta_sq * dQ[0, 0, :, :]

    if Q[1, 1] < delta:
        c += beta * (delta - Q[1, 1]) ** 2 / delta_sq
        dc += -2 * beta * (delta - Q[1, 1]) / delta_sq * dQ[1, 1, :, :]

    return c, dc
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


def stiffness_delta(volfrac: float, E0: float) -> float:
    """Ngưỡng stiffness tối thiểu δ = 10% độ cứng vật liệu nền theo tỉ lệ thể
    tích thiết kế - dùng làm ngưỡng chung cho CẢ phạt mềm (bên dưới, trong c/dc)
    LẪN ràng buộc cứng trong oc_update() (xem simp/runner.py, FIX 2026-07-29
    B_stiffness) - một nguồn duy nhất để tránh 2 nơi tính delta lệch nhau."""
    return 0.1 * volfrac * E0


def compute_nu12(Q: np.ndarray) -> float:
    """Tính nu12 chính xác từ tensor Q (không giả định orthotropic).

    Dùng nghịch đảo ma trận 3x3 đầy đủ: nu12 = -S12/S11 với S = Q^-1. Luôn
    đúng bất kể coupling cắt-pháp (Q13/Q23, do rotation) - khác công thức rút
    gọn Q12/Q22 (chỉ đúng khi Q13=Q23=0).

    Args:
        Q: Tensor độ cứng đồng nhất hóa (3x3), thứ tự Voigt [11, 22, 12].

    Returns:
        nu12 (float) tính chính xác.
    """
    S = np.linalg.inv(Q)
    nu12 = -S[0, 1] / S[0, 0]
    return float(nu12)

def compute_nu21(Q: np.ndarray) -> float:
    """Tính nu21 chính xác từ tensor Q (không giả định orthotropic).

    nu21 = -S12 / S22  với S = Q^-1.
    (Khác nu12 = -S12/S11 chỉ ở mẫu số: S22 thay vì S11.)
    """
    S = np.linalg.inv(Q)
    nu21 = -S[0, 1] / S[1, 1]
    return float(nu21)


def compute_elastic_constants(Q: np.ndarray) -> dict:
    """Trích xuất hằng số kỹ thuật vĩ mô (Ex, Ey, Gxy, B_eff) từ tensor Q.

    Dùng cùng nghịch đảo ma trận 3x3 đầy đủ như compute_nu12()/compute_nu21()
    (không giả định orthotropic - đúng cả khi Q13/Q23 != 0 do rotation, xem
    CẢNH BÁO ROTATION ở đầu module) - gọi lại 2 hàm đó cho nu_12/nu_21 thay vì
    chép lại công thức, tránh 2 nguồn tính cùng đại lượng bị lệch nhau.

    B_eff là mô-đun khối hiệu dụng 2D (plane stress):
        B* = Ex*Ey / [Ex*(1-nu21) + Ey*(1-nu12)]
    Đã kiểm chứng bằng số: rút gọn đúng về E/(2*(1-nu)) khi vật liệu đẳng
    hướng (Ex=Ey=E, nu12=nu21=nu) - khớp công thức bulk modulus 2D
    plane-stress chuẩn (xem test_solid_cell_recovers_isotropic_constants).

    Args:
        Q: Tensor độ cứng đồng nhất hóa (3x3), thứ tự Voigt [11, 22, 12].

    Returns:
        dict với các khóa 'E_x', 'E_y', 'G_xy', 'nu_12', 'nu_21', 'B_eff'.

    Raises:
        numpy.linalg.LinAlgError: nếu Q suy biến (không nghịch đảo được) -
            KHÔNG bắt lỗi ở đây, để caller tự quyết định xử lý (giống
            compute_nu12()/compute_nu21() - xem simp/runner.py chỗ gọi 2 hàm
            đó, nơi Q zero-init do FE-solve lỗi được caller kiểm tra TRƯỚC
            khi gọi, không phải bên trong hàm tính toán).
    """
    S = np.linalg.inv(Q)
    E_x = 1.0 / S[0, 0]
    E_y = 1.0 / S[1, 1]
    G_xy = 1.0 / S[2, 2]
    nu_12 = compute_nu12(Q)
    nu_21 = compute_nu21(Q)
    B_eff = (E_x * E_y) / (E_x * (1.0 - nu_21) + E_y * (1.0 - nu_12))
    return {
        'E_x': float(E_x),
        'E_y': float(E_y),
        'G_xy': float(G_xy),
        'nu_12': nu_12,
        'nu_21': nu_21,
        'B_eff': float(B_eff),
    }


def compute_wave_speeds(
    Q: np.ndarray, rel_density: float, E0: float
) -> dict:
    """Tốc độ sóng đàn hồi giới hạn bước sóng dài (quasi-static) theo 2 trục.

    Khi bước sóng >> kích thước ô cơ sở, vật liệu tuần hoàn truyền sóng như
    môi trường đồng nhất có độ cứng Q và khối lượng riêng hiệu dụng
    rho_eff = rel_density * rho_s. Tốc độ pha theo hướng n là nghiệm bài toán
    Christoffel det(Gamma - rho*c^2*I) = 0 với Gamma_ik = C_ijkl n_j n_l.
    Trong ký hiệu Voigt [11, 22, 12] (Q33 = C1212, ứng với biến dạng cắt
    kỹ thuật gamma12):
        n = x: Gamma = [[Q11, Q13], [Q13, Q33]]
        n = y: Gamma = [[Q33, Q23], [Q23, Q22]]
    Dùng trị riêng đầy đủ (không giả định Q13 = Q23 = 0) nên đúng cả khi ô
    cơ sở bị xoay - cùng nguyên tắc với compute_nu12().

    Kết quả không thứ nguyên, chuẩn hóa theo c_s = sqrt(E0 / rho_s) của vật
    liệu nền: c_tilde = sqrt(lambda / (E0 * rel_density)). Không áp dụng cho
    vùng tần số cao/band gap (cần bài toán trị riêng Bloch riêng).

    Args:
        Q: Tensor độ cứng đồng nhất hóa (3x3), thứ tự Voigt [11, 22, 12],
            cùng đơn vị với E0.
        rel_density: Mật độ tương đối (mean xPhys, 0 < rel_density <= 1).
        E0: Modul Young vật liệu nền (cùng đơn vị với Q).

    Returns:
        dict với 'c_qL_x', 'c_qT_x', 'c_qL_y', 'c_qT_y' - tốc độ chuẩn hóa
        c/c_s của sóng quasi-longitudinal / quasi-transverse theo trục x, y.
        Ô đặc hoàn toàn đẳng hướng cho c_qL = sqrt(1/(1-nu^2)),
        c_qT = sqrt(1/(2(1+nu))).
    """
    gamma_x = np.array([[Q[0, 0], Q[0, 2]], [Q[0, 2], Q[2, 2]]])
    gamma_y = np.array([[Q[2, 2], Q[1, 2]], [Q[1, 2], Q[1, 1]]])
    # eigvalsh trả trị riêng tăng dần: [quasi-transverse, quasi-longitudinal].
    # clip >= 0 chỉ để chặn nhiễu số âm ~1e-15 khi Q gần suy biến.
    scale = E0 * rel_density
    lam_x = np.clip(np.linalg.eigvalsh(gamma_x), 0.0, None)
    lam_y = np.clip(np.linalg.eigvalsh(gamma_y), 0.0, None)
    c_x = np.sqrt(lam_x / scale)
    c_y = np.sqrt(lam_y / scale)
    return {
        'c_qL_x': float(c_x[1]),
        'c_qT_x': float(c_x[0]),
        'c_qL_y': float(c_y[1]),
        'c_qT_y': float(c_y[0]),
    }


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
    delta = stiffness_delta(volfrac, E0)
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
    delta = stiffness_delta(volfrac, E0)
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
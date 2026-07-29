"""
Cập nhật theo tiêu chí tối ưu (OC) cho tối ưu hóa hình dạng SIMP.

Thực hiện thuật toán cập nhật OC (Optimality Criteria) cổ điển
để cập nhật biến thiết kế dựa trên độ nhạy và ràng buộc thể tích.
"""

import numpy as np
from scipy.sparse import csr_matrix

from .filter import apply_heaviside_projection

# X_MIN: sàn dưới của biến thiết kế x - khớp giá trị chuẩn tham chiếu Sigmund
# (2001) 99-line MATLAB (`xnew = max(0.001, ...)`), KHÔNG phải 0.0. Lý do:
# dc/dx (dQ) ∝ x^(penal-1) (xem homogenization/compute.py) -> TẠI x=0 CHÍNH
# XÁC, độ nhạy = 0 (với penal>1), khiến x*ratio giữ nguyên 0 MÃI MÃI (cập
# nhật nhân, không phải cộng) - một trạng thái "hấp thụ" toán học không thể
# thoát ra dù có thêm bất kỳ ràng buộc/phạt nào khác (đã kiểm chứng: bug FIX
# 2026-07-29 dùng x_min=0.0 khiến `hexagonal`/`reentrant_bowtie` sụp về vol~0
# vĩnh viễn chỉ trong ~10 vòng lặp đầu, xem EXPERIMENT_LOG.md). Dùng 0.001
# giữ độ nhạy khác 0 tại "gần-void", cho phép phần tử hồi phục nếu cần.
X_MIN = 0.001


def oc_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    volfrac: float,
    move: float,
    H,
    Hs,
    ft: int,
    Q: np.ndarray | None = None,
    delta: float | None = None,
    use_sqrt: bool = False,
    projection: str | None = None,
    beta_proj: float = 8.0,
    eta_proj: float = 0.5,
):
    """Cập nhật biến thiết kế dùng tiêu chí tối ưu (OC).

    Thực hiện cập nhật OC với tìm kiếm nhị phân trên hệ số Lagrange
    để thỏa mãn ràng buộc thể tích.

    Hỗ trợ thêm ràng buộc stiffness (Q₁₁ ≥ δ, Q₂₂ ≥ δ) dùng cho auxetic objective
    (giống MATLAB topK_Hourglass.m).

    Args:
        x: Mảng (nely, nelx) biến thiết kế hiện tại.
        dc: Mảng (nely, nelx) độ nhạy hàm mục tiêu.
        dv: Mảng (nely, nelx) độ nhạy thể tích.
        volfrac: Tỉ lệ thể tích yêu cầu.
        move: Giới hạn thay đổi cho phép mỗi vòng lặp.
        H: Ma trận lọc thưa.
        Hs: Vector tổng trọng số lọc.
        ft: Loại bộ lọc (1=độ nhạy, 2=mật độ).
        Q: Ten-xơ độ cứng đồng nhất hóa (3×3, optional). Dùng khi có ràng buộc stiffness.
        delta: Ngưỡng stiffness tối thiểu (optional). Yêu cầu Q nếu delta được cung cấp.
        use_sqrt: Nếu True, dùng x * sqrt(-dc/(dv*lmid)) (Sigmund 2001 heuristic).
                   Nếu False, dùng x * (-dc/(dv*lmid)) (MATLAB reference).
                   Mặc định False để khớp MATLAB.
        projection: None (mặc định) hoặc 'heaviside'. Nếu 'heaviside', ràng buộc
            thể tích trong bisection nhắm vào mean(x̂) = mean(apply_heaviside_
            projection(xPhys, beta_proj, eta_proj)), không phải mean(xPhys) thô
            (xPhys ở đây là x̃, trường đã lọc nhưng chưa qua projection).
        beta_proj, eta_proj: Tham số Heaviside projection (chỉ dùng khi
            projection='heaviside'), xem core/filter.py::apply_heaviside_projection().

    Returns:
        Bộ (xnew, xPhys) với:
            xnew : Mảng (nely, nelx) biến thiết kế mới (chưa lọc).
            xPhys: Mảng (nely, nelx) mật độ x̃ đã lọc (CHƯA projection - caller
                tự áp projection để có x̂, giống hành vi cũ).
    """
    nely, nelx = x.shape
    l1 = 0.0
    l2 = 1e9

    # Xác định có ràng buộc stiffness hay không
    has_stiffness_constraint = (Q is not None) and (delta is not None)

    # Tìm kiếm nhị phân cho hệ số Lagrange
    # Lặp tối đa 100 lần hoặc đến khi |mean(xPhys) - volfrac| < 1e-6
    for _ in range(100):
        lmid = (l1 + l2) / 2

        # Quy tắc cập nhật OC (xem use_sqrt ở docstring)
        ratio = np.maximum(0.0, -dc / (dv * lmid + 1e-15))
        if use_sqrt:
            ratio = np.sqrt(ratio)
        xnew = np.maximum(
            X_MIN,
            np.maximum(
                x - move,
                np.minimum(
                    1.0,
                    np.minimum(
                        x + move,
                        x * ratio,
                    ),
                ),
            ),
        )

        # Áp dụng bộ lọc mật độ
        if ft == 1:
            xPhys = xnew.copy()
        elif ft == 2:
            xPhys_flat = H @ xnew.flatten('F') / Hs
            xPhys = np.reshape(xPhys_flat, (nely, nelx), order='F')

        # Q được evaluate tại x cũ, không phải xnew - approximation chuẩn của
        # OC update (Sigmund 2001, Andreassen 2011), chấp nhận được với move
        # limit nhỏ (0.05-0.2).
        # FIX (xem AUDIT_REPORT_INDEPENDENT_2026-07-29.md mục B1): khi có
        # projection, ràng buộc thể tích phải nhắm vào mean(x̂) (SAU projection,
        # trường thật sự dùng ở FE/chế tạo), không phải mean(x̃) (TRƯỚC
        # projection) - vì projection không bảo toàn thể tích tuyệt đối, dùng
        # x̃ để bisection gây lệch volfrac có hệ thống (đo được trong pilot).
        if projection == 'heaviside':
            x_hat = apply_heaviside_projection(xPhys, beta_proj, eta_proj)
            vol = np.mean(x_hat)
        else:
            vol = np.mean(xPhys)

        # MATLAB-style: mean(xPhys) > volfrac && Q(1,1) >= delta && Q(2,2) >= delta
        if has_stiffness_constraint:
            stiff_ok = (Q[0, 0] >= delta) and (Q[1, 1] >= delta)
        else:
            stiff_ok = True

        if vol > volfrac and stiff_ok:
            l1 = lmid
        else:
            l2 = lmid

        # Dừng sớm nếu Lagrange multiplier đã đạt độ chính xác cao
        if abs(vol - volfrac) < 1e-6 or (l2 - l1) < 1e-12:
            break

    return xnew, xPhys

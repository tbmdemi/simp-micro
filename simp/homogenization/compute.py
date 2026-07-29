"""
Tính toán đồng nhất hóa cho vật liệu tuần hoàn.

Thực hiện phương pháp đồng nhất hóa dựa trên năng lượng
để tính ten-xơ độ cứng đồng nhất hóa và đạo hàm của nó
theo mật độ phần tử.
"""

import numpy as np


def compute_homogenized_tensor(
    U: np.ndarray,
    U0: np.ndarray,
    xPhys: np.ndarray,
    KE: np.ndarray,
    edofMat: np.ndarray,
    penal: float,
    E0: float,
    Emin: float,
    rho0: float = 1.0,
):
    """Tính ten-xơ độ cứng đồng nhất hóa và đạo hàm của nó.

    Energy-based homogenization, dùng TỔNG chuyển vị u = u0 + fluctuation
    (không phải fluctuation riêng) theo Andreassen et al. (2014), eq (6):
        Q_ij = 1/|Ω| Σ_e (u_e^i)^T * k_e * (u_e^j),  k_e = E_penal[e] * KE / E0.
    Caller (runner.py) phải cộng U0 + U (fluctuation) trước khi gọi hàm này.

    Args:
        U: Ma trận chuyển vị tổng (ndof, 3) cho 3 trường hợp tải.
        U0: Ma trận chuyển vị biến dạng đơn vị (ndof, 3).
        xPhys: Mảng (nely, nelx) mật độ vật lý.
        KE: Ma trận độ cứng phần tử (8×8).
        edofMat: Ma trận (nelx*nely, 8) ánh xạ phần tử → bậc tự do.
        penal: Số mũ phạt SIMP.
        E0: Modul đàn hồi Young của vật liệu đặc.
        Emin: Modul đàn hồi Young của lỗ rỗng.

    Returns:
        Bộ (Q, dQ, Ue) với:
            Q : Ten-xơ độ cứng đồng nhất hóa (3×3).
            dQ: Đạo hàm của Q theo mật độ (3×3×nely×nelx).
            Ue: Chuyển vị phần tử (nelx*nely, 8, 3) - trường tổng.
    """
    nely, nelx = xPhys.shape
    nele = nelx * nely

    # Chuyển edofMat về chỉ số 0-based
    edofMat_0 = edofMat - 1

    # Trích xuất chuyển vị phần tử từ U (tổng, xem docstring)
    Ue = np.zeros((nele, 8, 3))
    for i in range(nele):
        for j in range(3):
            Ue[i, :, j] = U[edofMat_0[i, :], j]

    # Tính độ cứng vật liệu cho mỗi phần tử (vector hóa)
    # E(x) = Emin + (rho0 * x^penal) * (E0 - Emin)
    x_flat = xPhys.flatten('F')
    E_penal = Emin + (rho0 * x_flat ** penal) * (E0 - Emin)

    # BUG FIX (2026-07-29): k_e PHẢI chia cho E0 (KE đã chứa sẵn E0 - xem
    # Material._compute_element_stiffness(), D = E0/(1-nu^2)*[...]). Thiếu
    # phép chia này khiến Q bị nhân đúp E0 (Q ~ E0^2 thay vì tuyến tính theo
    # E0). Đồng thời KHÔNG chia cho (nelx*nely): U0 (solver.py::solve_fe)
    # dùng tọa độ chuẩn hóa [0,1] (x_coord=i/nelx, y_coord=j/nely) nên miền
    # đã có diện tích |Ω|=1 - chia thêm cho nelx*nely tạo hệ số dư 1/nele.
    # Kết hợp 2 lỗi: Q_cũ = Q_đúng * E0/nele (đã kiểm chứng bằng số: ô cơ sở
    # đặc hoàn toàn (x=1, penal=1) phải cho Q=D chính xác - xem
    # tests/test_core_smoke.py::test_homogenized_tensor_matches_input_material_for_solid_cell).
    # KHÔNG ảnh hưởng ν12/ν21 (bất biến với hệ số nhân đồng đều qua S=Q^-1),
    # chỉ ảnh hưởng các nơi dùng Q làm giá trị vật lý tuyệt đối (vd ngưỡng
    # delta trong simp/objectives/auxetic.py).
    k_e = E_penal / E0

    Q = np.einsum('e,emi,mn,enj->ij', k_e, Ue, KE, Ue)

    # dQ_ij/dx_e = (1/|Ω|) * d(k_e)/dx_e * (Ue^i)^T * KE * (Ue^j), |Ω|=1
    dk_e = (rho0 * penal * x_flat ** (penal - 1) * (E0 - Emin)) / E0
    dQ_flat = np.einsum('e,emi,mn,enj->eij', dk_e, Ue, KE, Ue)
    # BUG FIX (2026-07-24): chỉ số phần tử e được xây dựng theo order='F'
    # (khớp x_flat = xPhys.flatten('F') ở trên, và toàn bộ codebase - xem
    # solve_fe()). reshape() mặc định dùng order='C', khiến dQ[:,:,i,j] bị
    # HOÁN VỊ (với lưới vuông nelx=nely, tương đương transpose dQ[:,:,j,i])
    # so với pixel thật - đã xác nhận bằng finite-difference (xem
    # tests/test_phase5_real_physics.py::TestGradientCorrectness). Với
    # density field đối xứng qua đường chéo thì vô hại (transpose không đổi
    # gì), nhưng SAI với field bất đối xứng (hourglass, hexagonal,
    # reentrant_bowtie trong số 11 seed hiện có) - dc/oc_update() nhận
    # sensitivity map sai hướng suốt cả quá trình tối ưu.
    dQ = dQ_flat.transpose(1, 2, 0).reshape(3, 3, nely, nelx, order='F')

    return Q, dQ, Ue

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

    # k_e = E_penal[e] * KE; KHÔNG chia cho E0 (KE đã chứa E0 sẵn, xem
    # Material). Do đó Q ~ 1/E0 giống U, nhưng delta = 0.1*volfrac*E0 trong
    # objective cũng tỉ lệ E0 nên tự cân bằng.
    k_e = E_penal

    Q = np.einsum('e,emi,mn,enj->ij', k_e, Ue, KE, Ue) / (nelx * nely)

    # dQ_ij/dx_e = (1/|Ω|) * d(k_e)/dx_e * (Ue^i)^T * KE * (Ue^j)
    dk_e = rho0 * penal * x_flat ** (penal - 1) * (E0 - Emin)
    dQ_flat = np.einsum('e,emi,mn,enj->eij', dk_e, Ue, KE, Ue) / (nelx * nely)
    dQ = dQ_flat.transpose(1, 2, 0).reshape(3, 3, nely, nelx)

    return Q, dQ, Ue

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
):
    """Tính ten-xơ độ cứng đồng nhất hóa và đạo hàm của nó.

    Sử dụng phương pháp đồng nhất hóa dựa trên năng lượng
    (energy-based homogenization) cho vật liệu tuần hoàn.
    Công thức sử dụng trường dao động (fluctuation field)
    χ = u - u⁰, trong đó u⁰ là chuyển vị biến dạng đơn vị.

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

    # Trích xuất chuyển vị phần tử (tổng và biến dạng đơn vị)
    Ue = np.zeros((nele, 8, 3))
    U0e = np.zeros((nele, 8, 3))
    for i in range(nele):
        for j in range(3):
            Ue[i, :, j] = U[edofMat_0[i, :], j]
            U0e[i, :, j] = U0[edofMat_0[i, :], j]

    # Trường dao động (fluctuation field): χ = u - u⁰
    chi = Ue - U0e

    # Tính độ cứng vật liệu cho mỗi phần tử (vector hóa)
    # E(x) = Emin + x^penal * (E0 - Emin)
    x_flat = xPhys.flatten('F')
    E = Emin + x_flat ** penal * (E0 - Emin)

    # Tính ten-xơ độ cứng đồng nhất hóa Q (vector hóa bằng einsum)
    # Q_ij = (1/|Ω|) * Σ_e E_e * (χ_e^(i)ᵀ * KE * χ_e^(j))
    # 'e' : element, 'm'/'n' : DOF, 'i'/'j' : load case
    Q = np.einsum('e,emi,mn,enj->ij', E, chi, KE, chi) / (nelx * nely)

    # Tính đạo hàm của Q theo mật độ phần tử dQ (vector hóa bằng einsum)
    # dQ_ij/dx_e = (1/|Ω|) * dE/dx_e * (χ_e^(i)ᵀ * KE * χ_e^(j))
    # dE/dx = penal * x^(penal-1) * (E0 - Emin)
    dE = penal * x_flat ** (penal - 1) * (E0 - Emin)
    
    # Kết quả einsum là (nele, 3, 3), sau đó chuyển thành (3, 3, nely, nelx)
    dQ_flat = np.einsum('e,emi,mn,enj->eij', dE, chi, KE, chi) / (nelx * nely)
    dQ = dQ_flat.transpose(1, 2, 0).reshape(3, 3, nely, nelx)

    return Q, dQ, Ue

"""
Bộ giải phần tử hữu hạn thưa cho tối ưu hóa hình dạng SIMP.

Giải hệ phương trình FE với ma trận độ cứng thưa và
điều kiện biên tuần hoàn (PBC) cho phân tích ô cơ sở.
"""

import logging

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, splu, cg, LinearOperator

# Hằng số local cho eps (tránh magic number)
_EPS_SOLVER = 1e-9


def solve_fe(
    xPhys: np.ndarray,
    KE: np.ndarray,
    iK: np.ndarray,
    jK: np.ndarray,
    pbc: csr_matrix,
    penal: float,
    E0: float,
    Emin: float,
    rho0: float = 1.0,
):
    """Giải bài toán FE với ràng buộc PBC.

    Lắp ráp ma trận độ cứng toàn cục từ ma trận độ cứng phần tử
    và trường mật độ, sau đó giải với ràng buộc PBC.

    Args:
        xPhys: Mảng (nely, nelx) mật độ vật lý.
        KE: Ma trận độ cứng phần tử (8×8).
        iK: Vector chỉ số hàng cho lắp ráp K thưa.
        jK: Vector chỉ số cột cho lắp ráp K thưa.
        pbc: Ma trận ràng buộc PBC thưa.
        penal: Số mũ phạt SIMP.
        E0: Modul đàn hồi Young của vật liệu đặc.
        Emin: Modul đàn hồi Young của lỗ rỗng (tránh ma trận suy biến).

    Returns:
        Bộ (U, U0) với:
            U : Ma trận chuyển vị (ndof, 3) thỏa mãn ràng buộc PBC.
            U0: Ma trận chuyển vị biến dạng đơn vị (ndof, 3).
    """
    nelx, nely = xPhys.shape[1], xPhys.shape[0]
    nele = nelx * nely
    ndof = pbc.shape[0]

    # Lắp ráp ma trận độ cứng toàn cục
    # xPhys: (nely, nelx) -> chuyển thành vector (nele,) theo thứ tự Fortran
    xPhys_vec = xPhys.flatten('F')  # (nele,)
    # KE: (8,8) -> (64,)
    KE_vec = KE.flatten()  # (64,)
    # Mỗi phần tử có KE riêng: (64, 1) * (1, nele) -> (64, nele)
    # Công thức SIMP: E(x) = Emin + x^penal * (E0 - Emin)
    # Hỗ trợ rho0 scaling (MATLAB Second_Obj: rho0=7850, E0=1)
    # E_penal = Emin + (rho0 * x^penal) * (E0 - Emin)
    xe = rho0 * xPhys_vec ** penal
    sK = KE_vec[:, np.newaxis] * (Emin + xe[np.newaxis, :] * (E0 - Emin))
    sK = sK.flatten('F')

    K_global = coo_matrix(
        (sK, (iK, jK)),
        shape=(ndof, ndof),
    ).tocsr()

    # Symmetrize K để tránh sai số số học (giống MATLAB: K = (K+K')/2)
    K_global = (K_global + K_global.T) * 0.5

    # PBC là ma trận chiếu (ndof, n_master)
    # K_pbc = PBC^T @ K_global @ PBC
    K_pbc = pbc.T @ K_global @ pbc

    # Vế phải: tải trọng đơn vị cho đồng nhất hóa
    n_cases = 3
    ndof_reduced = K_pbc.shape[0]
    nnx = nelx + 1
    nny = nely + 1

    # Xây dựng chuyển vị biến dạng đơn vị cho mỗi nút
    # U⁰(x,y) = ε⁰ · [x, y]ᵀ
    # 
    # FIX (2026-06-06): Dùng nodenrs-based DOF indexing thay vì idx-based
    # để tương thích với edofMat (dùng node number).
    # Lấy nodenrs từ tham số hoặc rebuild
    try:
        from .fem import build_dof_mesh
        _dummy_nodenrs, _, _, _, _ = build_dof_mesh(nelx, nely)
    except ImportError:
        _dummy_nodenrs = np.reshape(
            np.arange(1, (1 + nelx) * (1 + nely) + 1),
            (1 + nely, 1 + nelx), order='F')
    
    U0 = np.zeros((ndof, n_cases))
    for j in range(nny):
        for i in range(nnx):
            node = _dummy_nodenrs[j, i]  # 1-based node number
            x_coord = i / nelx  # chuẩn hóa [0, 1]
            y_coord = j / nely  # chuẩn hóa [0, 1]
            dof_u = 2 * (node - 1)
            dof_v = 2 * (node - 1) + 1
            # ε_xx = 1
            U0[dof_u, 0] = x_coord
            # ε_yy = 1
            U0[dof_v, 1] = y_coord
            # γ_xy = 1
            U0[dof_u, 2] = y_coord / 2
            U0[dof_v, 2] = x_coord / 2

    # Tải trọng: F = -PBC^T @ (K_global @ U0)
    # FIX (2026-06-06): Sửa sign - nghiệm là fluctuation χ thỏa K@χ = -K@U0
    F = -pbc.T @ (K_global @ U0)

    # Cố định 2 bậc tự do đầu tiên (u, v của nút 0) để loại bỏ chuyển vị cứng
    fixed_dofs = [0, 1]
    free_dofs = np.setdiff1d(np.arange(ndof_reduced), fixed_dofs)

    K_pbc_free = K_pbc[free_dofs, :][:, free_dofs]
    F_free = F[free_dofs, :]

    # Giải cho mỗi trường hợp tải — factorize LU MỘT LẦN (K_pbc_free giống
    # nhau cho cả 3 case, chỉ RHS khác) thay vì spsolve() riêng lẻ 3 lần
    # (mỗi lần tự phân rã LU lại từ đầu → lãng phí ~3x chi phí factorization).
    U_reduced_free = np.zeros((len(free_dofs), n_cases))
    try:
        lu = splu(K_pbc_free.tocsc())
        U_reduced_free = lu.solve(F_free)

        if np.any(np.linalg.norm(U_reduced_free, axis=0) > 1e6):
            raise np.linalg.LinAlgError("Matrix is nearly singular (large displacement detected)")

    except Exception:
        diag_K = K_pbc_free.diagonal()
        diag_K[diag_K == 0] = _EPS_SOLVER
        M_inv = LinearOperator((K_pbc_free.shape), matvec=lambda x: x / diag_K)

        for i in range(n_cases):
            U_reduced_free[:, i], info = cg(K_pbc_free, F_free[:, i], tol=1e-6, maxiter=10000, M=M_inv)
            if info > 0:
                logging.getLogger(__name__).warning("CG failed to converge for case %d after %d iterations", i, info)

    # Khôi phục vector đầy đủ
    U_reduced = np.zeros((ndof_reduced, n_cases))
    U_reduced[free_dofs, :] = U_reduced_free

    # Chiếu trở lại không gian bậc tự do đầy đủ
    U = pbc @ U_reduced

    return U, U0

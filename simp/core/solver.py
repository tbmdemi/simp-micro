"""
Bộ giải phần tử hữu hạn thưa cho tối ưu hóa hình dạng SIMP.

Giải hệ phương trình FE với ma trận độ cứng thưa và
điều kiện biên tuần hoàn (PBC) cho phân tích ô cơ sở.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

# Ngưỡng chuẩn chuyển vị coi là hệ gần suy biến (thiết kế rời/rỗng).
_MAX_DISPLACEMENT_NORM = 1e6


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
    ndof = pbc.shape[0]

    # Lắp ráp ma trận độ cứng toàn cục. E_penal = Emin + (rho0*x^penal)*(E0-Emin)
    # - rho0 hỗ trợ scaling kiểu MATLAB Second_Obj (rho0=7850, E0=1).
    xPhys_vec = xPhys.flatten('F')  # (nele,)
    KE_vec = KE.flatten()  # (64,)
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

    # Xây dựng chuyển vị biến dạng đơn vị cho mỗi nút: U⁰(x,y) = ε⁰ · [x, y]ᵀ
    # FIX (2026-06-06): dùng nodenrs-based DOF indexing (không phải idx-based)
    # để tương thích với edofMat (dùng node number).
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
    # FIX (2026-06-06): sign đúng phải âm - nghiệm là fluctuation χ thỏa K@χ = -K@U0
    F = -pbc.T @ (K_global @ U0)

    # Cố định 2 bậc tự do đầu tiên (u, v của nút 0) để loại bỏ chuyển vị cứng
    fixed_dofs = [0, 1]
    free_dofs = np.setdiff1d(np.arange(ndof_reduced), fixed_dofs)

    K_pbc_free = K_pbc[free_dofs, :][:, free_dofs]
    F_free = F[free_dofs, :]

    # Giải cho mỗi trường hợp tải - factorize LU MỘT LẦN (K_pbc_free giống
    # nhau cho cả 3 case, chỉ RHS khác) thay vì spsolve() riêng lẻ 3 lần
    # (mỗi lần tự phân rã LU lại từ đầu → lãng phí ~3x chi phí factorization).
    # Hệ suy biến -> báo lỗi rõ ràng; nơi gọi coi là FE thất bại (NaN /
    # đếm lỗi). Nhánh dự phòng CG cũ đã xóa 2026-10-10: gọi cg(tol=) mà SciPy
    # >= 1.14 không còn tham số này nên luôn ném TypeError (kích hoạt 0/700
    # thiết kế thật đã đo) - hành vi thực tế vẫn là "thất bại", nay tường minh.
    lu = splu(K_pbc_free.tocsc())
    U_reduced_free = lu.solve(F_free)
    if np.any(np.linalg.norm(U_reduced_free, axis=0) > _MAX_DISPLACEMENT_NORM):
        raise np.linalg.LinAlgError(
            "Ma trận độ cứng gần suy biến (chuyển vị > "
            f"{_MAX_DISPLACEMENT_NORM:g}) - thiết kế rời hoặc rỗng"
        )

    # Khôi phục vector đầy đủ
    U_reduced = np.zeros((ndof_reduced, n_cases))
    U_reduced[free_dofs, :] = U_reduced_free

    # Chiếu trở lại không gian bậc tự do đầy đủ
    U = pbc @ U_reduced

    return U, U0

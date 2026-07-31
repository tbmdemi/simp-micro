"""
Phase 8.1 - skfem_homogenization.py
============================================================
Implementation FE/homogenization ĐỘC LẬP (thư viện `scikit-fem`, không tái
dùng simp/core/pbc.py hay simp/homogenization/compute.py) để đối chiếu với
engine nội bộ - xem README §Giới hạn Đã biết #15 và EXPERIMENT_LOG.md mục
2026-07-31 cho lý do/thiết kế/kết quả đầy đủ.

Công thức: D=E/(1-ν²)·[[1,ν,0],[ν,1,0],[0,0,(1-ν)/2]] (plane stress),
E(x)=Emin+x^penal(E0-Emin) (SIMP), U0=ε·[x,y] (3 macro-strain Voigt xx/yy/xy),
K·χ=-P^T·K·U0 (periodic reduction P), Q_ij=U_i^T·K·U_j, S=Q^-1,
nu12=-S01/S00, nu21=-S01/S11.

Cổng kiểm chứng BẮT BUỘC trước khi tin bất kỳ so sánh nào: ô đặc hoàn toàn
(x=1, penal=1) phải cho Q == D dạng đóng - xem tests/test_skfem_homogenization.py.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from skfem import MeshQuad, Basis, ElementVector, ElementQuad1, ElementQuad0, BilinearForm
from skfem.helpers import ddot, trace, sym_grad, eye
from skfem.models.elasticity import plane_stress


def build_mesh_and_basis(nelx: int, nely: int):
    m = MeshQuad.init_tensor(np.linspace(0, 1, nelx + 1), np.linspace(0, 1, nely + 1))
    basis = Basis(m, ElementVector(ElementQuad1()))
    basis0 = basis.with_element(ElementQuad0())  # cùng quadrature với basis - bắt buộc
    return m, basis, basis0


def element_grid_index_map(m: MeshQuad, nelx: int, nely: int):
    """element skfem -> ô lưới (i,j) qua trọng tâm - không giả định thứ tự
    đánh số nội bộ của thư viện."""
    centroids = m.p[:, m.t].mean(axis=1)
    i_idx = np.clip(np.floor(centroids[0] * nelx).astype(int), 0, nelx - 1)
    j_idx = np.clip(np.floor(centroids[1] * nely).astype(int), 0, nely - 1)
    return i_idx, j_idx


def density_to_element_array(xPhys: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray):
    """xPhys shape (nely, nelx), quy ước hàng=y cột=x giống toàn codebase."""
    return xPhys[j_idx, i_idx]


def simp_modulus(x_elem: np.ndarray, E0: float, Emin: float, penal: float):
    return Emin + x_elem ** penal * (E0 - Emin)


def _elasticity_weakform():
    def C(T, lam, mu):
        return 2.0 * mu * T + lam * eye(trace(T), T.shape[0])

    @BilinearForm
    def weakform(u, v, w):
        return ddot(C(sym_grad(u), w["lam"], w["mu"]), sym_grad(v))
    return weakform


def assemble_K(basis: Basis, basis0: Basis, E_elem: np.ndarray, nu: float):
    lam_e, mu_e = plane_stress(E_elem, nu)
    lam_field = basis0.interpolate(lam_e)
    mu_field = basis0.interpolate(mu_e)
    return _elasticity_weakform().assemble(basis, lam=lam_field, mu=mu_field)


def build_periodic_reduction(m: MeshQuad, basis: Basis, tol: float = 1e-8):
    """Ghép DOF biên tuần hoàn (trái<->phải, dưới<->trên, góc nối về góc
    dưới-trái). Trả về (P, pin_dofs): u_full = P @ u_reduced, u tại
    pin_dofs luôn = 0 (loại bỏ rigid-body translation)."""
    x, y = m.p[0], m.p[1]
    nodal = basis.dofs.nodal_dofs  # (2, n_nodes)
    n_nodes = nodal.shape[1]

    on_left = np.isclose(x, 0, atol=tol)
    on_right = np.isclose(x, 1, atol=tol)
    on_bottom = np.isclose(y, 0, atol=tol)
    on_top = np.isclose(y, 1, atol=tol)

    corner_bl = int(np.where(on_left & on_bottom)[0][0])
    corner_br = int(np.where(on_right & on_bottom)[0][0])
    corner_tl = int(np.where(on_left & on_top)[0][0])
    corner_tr = int(np.where(on_right & on_top)[0][0])

    node_master = np.arange(n_nodes)

    left_nodes = np.where(on_left & ~on_bottom & ~on_top)[0]
    right_nodes = np.where(on_right & ~on_bottom & ~on_top)[0]
    left_sorted = left_nodes[np.argsort(y[left_nodes])]
    right_sorted = right_nodes[np.argsort(y[right_nodes])]
    assert len(left_sorted) == len(right_sorted), "số node cạnh trái/phải không khớp"
    node_master[right_sorted] = left_sorted

    bottom_nodes = np.where(on_bottom & ~on_left & ~on_right)[0]
    top_nodes = np.where(on_top & ~on_left & ~on_right)[0]
    bottom_sorted = bottom_nodes[np.argsort(x[bottom_nodes])]
    top_sorted = top_nodes[np.argsort(x[top_nodes])]
    assert len(bottom_sorted) == len(top_sorted), "số node cạnh dưới/trên không khớp"
    node_master[top_sorted] = bottom_sorted

    node_master[corner_br] = corner_bl
    node_master[corner_tl] = corner_bl
    node_master[corner_tr] = corner_bl

    ndof = basis.N
    dof_master = np.arange(ndof)
    for comp in range(2):
        dof_master[nodal[comp]] = nodal[comp, node_master]

    pin_dofs = nodal[:, corner_bl]

    rows, cols = [], []
    col_of_master = {}
    n_master = 0
    for d in range(ndof):
        md = int(dof_master[d])
        if md in pin_dofs:
            continue
        if md not in col_of_master:
            col_of_master[md] = n_master
            n_master += 1
        rows.append(d)
        cols.append(col_of_master[md])
    data = np.ones(len(rows))
    P = sp.csr_matrix((data, (rows, cols)), shape=(ndof, n_master))
    return P, pin_dofs


def macro_strain_u0(m: MeshQuad, basis: Basis, case: int):
    """case 0/1/2 = Voigt xx/yy/xy (xy tách đối xứng u=y/2, v=x/2)."""
    x, y = m.p[0], m.p[1]
    nodal = basis.dofs.nodal_dofs
    U0 = np.zeros(basis.N)
    if case == 0:
        U0[nodal[0]] = x
    elif case == 1:
        U0[nodal[1]] = y
    elif case == 2:
        U0[nodal[0]] = y / 2.0
        U0[nodal[1]] = x / 2.0
    else:
        raise ValueError(f"case phải trong {{0,1,2}}, nhận {case}")
    return U0


def solve_fluctuation(K, P: sp.csr_matrix, U0_full: np.ndarray):
    """Giải nhiễu loạn tuần hoàn chi, trả về U_total = U0 + chi."""
    K_red = (P.T @ K @ P).tocsc()
    F_red = -(P.T @ (K @ U0_full))
    chi_red = spla.spsolve(K_red, F_red)
    chi_full = P @ chi_red
    return U0_full + chi_full


def compute_q_skfem(K, U_totals):
    """Q_ij = U_i^T . K . U_j - KHÔNG chia lại cho E0 (khác cách viết
    trong compute.py, vốn dùng 1 KE cố định theo E0 rồi nhân hệ số E/E0
    mỗi phần tử; ở đây K đã lắp ráp bằng modulus THẬT per-element nên
    không cần chia lại - chia thêm sẽ sai 1 hệ số E0, bắt được qua cổng
    kiểm chứng ô đặc, xem tests/test_skfem_homogenization.py)."""
    Q = np.zeros((3, 3))
    KU = [K @ U_totals[j] for j in range(3)]
    for i in range(3):
        for j in range(3):
            Q[i, j] = float(U_totals[i] @ KU[j])
    return Q


def compute_nu12_nu21_skfem(Q: np.ndarray):
    S = np.linalg.inv(Q)
    nu12 = -S[0, 1] / S[0, 0]
    nu21 = -S[0, 1] / S[1, 1]
    return float(nu12), float(nu21)


def evaluate_density_field_skfem(xPhys: np.ndarray, E0: float = 199.0,
                                  Emin: float = 1e-9, nu: float = 0.3,
                                  penal: float = 3.0):
    """Tương đương verify_fe.py::evaluate_density_field() nhưng qua engine
    scikit-fem độc lập. Trả về (nu12, nu21, Q)."""
    nely, nelx = xPhys.shape
    m, basis, basis0 = build_mesh_and_basis(nelx, nely)
    i_idx, j_idx = element_grid_index_map(m, nelx, nely)
    x_elem = density_to_element_array(xPhys, i_idx, j_idx)
    E_elem = simp_modulus(x_elem, E0, Emin, penal)
    K = assemble_K(basis, basis0, E_elem, nu)
    P, _pin_dofs = build_periodic_reduction(m, basis)

    U_totals = []
    for case in range(3):
        U0 = macro_strain_u0(m, basis, case)
        U_totals.append(solve_fluctuation(K, P, U0))

    Q = compute_q_skfem(K, U_totals)
    nu12, nu21 = compute_nu12_nu21_skfem(Q)
    return nu12, nu21, Q

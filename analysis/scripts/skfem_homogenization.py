"""
Phase 8.1 - skfem_homogenization.py
============================================================
Kiểm chứng ĐỘC LẬP engine FE/homogenization nội bộ
(simp/core/solver.py + simp/homogenization/compute.py) - xem README §Giới hạn
Đã biết #15: R²=0,999 hiện tại chỉ "nhất quán nội bộ" (cùng 1 engine cho cả
train lẫn eval), chưa từng đối chiếu với 1 implementation FE khác.

Module này giải LẠI đúng bài toán homogenization tuần hoàn 2D (Q4 bilinear,
plane-stress, SIMP, 3 macro-strain đơn vị Voigt [xx,yy,xy]) bằng thư viện
`scikit-fem` (mesh/basis/assembly/solve dùng thẳng thư viện), nhưng periodic
BC + tải trọng macro-strain + công thức homogenize Q đều tự viết MỚI - KHÔNG
tái dùng simp/core/pbc.py hay simp/homogenization/compute.py, vì tái dùng lại
sẽ chỉ so sánh phần lắp ráp FE cơ bản, không bắt được lỗi trong đúng phần đã
từng có bug nghiêm trọng nhất dự án (A1 Q-scaling, hoán vị dQ, dấu RHS - đều
nằm trong phần homogenization-specific, không phải FE assembly cơ bản).

Công thức tham chiếu (đọc trực tiếp từ simp/materials/isotropic.py,
simp/core/pbc.py, simp/core/solver.py, simp/homogenization/compute.py,
simp/objectives/auxetic.py, pipeline/phase5_cvae/verify_fe.py):
  - D = E/(1-ν²)·[[1,ν,0],[ν,1,0],[0,0,(1-ν)/2]] (plane stress)
  - E(x) = Emin + x^penal·(E0-Emin)  (SIMP, rho0=1.0)
  - U0 = ε·[x,y] (case1: u=x,v=0; case2: u=0,v=y; case3: u=y/2,v=x/2)
  - K·χ_reduced = -P^T·K·U0  (periodic reduction P, pin 1 góc để bỏ rigid
    translation), U_total = U0 + P@χ_reduced
  - Q_ij = U_total_i^T · K_thực · U_total_j  (K_thực lắp ráp bằng modulus
    SIMP THẬT per-element - KHÔNG chia lại cho E0, xem docstring
    compute_q_skfem() giải thích vì sao KHÁC 1 chi tiết implementation so
    với công thức compute.py gốc dù toán học tương đương)
  - S = Q^-1 (nghịch đảo đầy đủ 3x3), nu12 = -S01/S00, nu21 = -S01/S11

Cổng kiểm chứng BẮT BUỘC trước khi tin bất kỳ so sánh nào: ô đặc hoàn toàn
(x=1 mọi nơi, penal=1) phải cho Q == D dạng đóng (rtol=1e-6) - xem
tests/test_skfem_homogenization.py, mirror đúng
tests/test_core_smoke.py::TestHomogenization::
test_homogenized_tensor_matches_input_material_for_solid_cell.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from skfem import MeshQuad, Basis, ElementVector, ElementQuad1, ElementQuad0, BilinearForm
from skfem.helpers import ddot, trace, sym_grad, eye
from skfem.models.elasticity import plane_stress


def build_mesh_and_basis(nelx: int, nely: int):
    """Lưới Q4 đều trên [0,1]x[0,1] - ô đơn vị chuẩn hoá, khớp quy ước
    U0=eps*[x,y] với x,y trong [0,1] của simp/core/solver.py."""
    m = MeshQuad.init_tensor(np.linspace(0, 1, nelx + 1), np.linspace(0, 1, nely + 1))
    basis = Basis(m, ElementVector(ElementQuad1()))
    basis0 = basis.with_element(ElementQuad0())  # cùng quadrature với basis - BẮT BUỘC
    return m, basis, basis0


def element_grid_index_map(m: MeshQuad, nelx: int, nely: int):
    """Ánh xạ element index của skfem -> ô lưới (i=cột/x, j=hàng/y) bằng
    trọng tâm phần tử - KHÔNG giả định thứ tự nội bộ của skfem, khớp trực
    tiếp toạ độ để đảm bảo đúng bất kể quy ước đánh số của thư viện."""
    centroids = m.p[:, m.t].mean(axis=1)  # (2, nelements)
    i_idx = np.clip(np.floor(centroids[0] * nelx).astype(int), 0, nelx - 1)
    j_idx = np.clip(np.floor(centroids[1] * nely).astype(int), 0, nely - 1)
    return i_idx, j_idx


def density_to_element_array(xPhys: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray):
    """xPhys shape (nely, nelx) (quy ước hàng=y, cột=x, giống toàn bộ
    codebase) -> mảng theo đúng thứ tự element của skfem."""
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
    """Ghép DOF biên tuần hoàn (trái<->phải, dưới<->trên, mọi góc nối về
    góc dưới-trái) - tự viết mới trên DOF layout của skfem, KHÔNG tái dùng
    simp/core/pbc.py. Trả về (P, pin_dofs): P (ndof, n_master) 0/1 sao cho
    u_full = P @ u_reduced (u tại pin_dofs LUÔN = 0, đúng ý nghĩa loại bỏ
    rigid-body translation bằng cách cố định góc dưới-trái)."""
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
    """U0 = eps^case . [x,y] toàn bộ ndof - case 0/1/2 = Voigt xx/yy/xy,
    xy tách đối xứng u=y/2, v=x/2 (đúng simp/core/solver.py)."""
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
    """Giải chi (nhiễu loạn tuần hoàn) cho 1 macro-strain case, trả về
    U_total = U0 + chi (đầy đủ ndof)."""
    K_red = (P.T @ K @ P).tocsc()
    F_red = -(P.T @ (K @ U0_full))
    chi_red = spla.spsolve(K_red, F_red)
    chi_full = P @ chi_red
    return U0_full + chi_full


def compute_q_skfem(K, U_totals):
    """Q_ij = U_total_i^T . K_thực . U_total_j - KHÔNG chia lại cho E0.

    Lưu ý quan trọng (khác 1 chi tiết implementation so với công thức gốc
    trong compute.py, dù kết quả toán học tương đương): compute.py dùng 1
    ma trận KE CỐ ĐỊNH tham chiếu theo E0, nhân thêm hệ số k_e=E_penal/E0
    mỗi phần tử trong vòng einsum - nên công thức gốc CẦN chia lại cho E0.
    Ở đây `K` đã được lắp ráp trực tiếp bằng modulus THỰC per-element
    (assemble_K() dùng E_elem = SIMP(x) thật, không phải E0 tham chiếu) -
    vì KE(E) tuyến tính theo E (KE(E) = (E/E0)*KE(E0)), 2 cách này cho
    CÙNG 1 K_thực = Σ_e KE(E_penal,e), nên KHÔNG cần chia lại cho E0 nữa -
    chia thêm sẽ sai 1 lần (bug đã bắt được qua chính cổng kiểm chứng ô đặc
    ở tests/test_skfem_homogenization.py: kết quả lệch đúng 1 hệ số E0)."""
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
    """Hàm cấp cao nhất - tương đương
    pipeline/phase5_cvae/verify_fe.py::evaluate_density_field() nhưng chạy
    hoàn toàn qua engine scikit-fem độc lập. Trả về (nu12, nu21, Q)."""
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

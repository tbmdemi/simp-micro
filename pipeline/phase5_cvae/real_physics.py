"""
Phase 5 - real_physics.py
=====================
Differentiable-physics layer: thay vì học qua surrogate CNN đông cứng (dễ bị
decoder "đánh lừa" - xem losses.py, README §5 "surrogate exploitation"), lớp
này chạy FE + homogenization THẬT (simp/core/solver.py, simp/homogenization/
compute.py) trong chính vòng lặp backward của cVAE, dùng ĐẠO HÀM GIẢI TÍCH
sẵn có (không phải autodiff số hay adjoint solve riêng).

Cơ sở toán học: compute_homogenized_tensor() đã trả về dQ_ij/dx_e - đạo hàm
CHÍNH XÁC của tensor độ cứng đồng nhất hóa Q theo từng pixel mật độ, nhờ tính
tự-adjoint (self-adjoint) của bài toán năng lượng (công thức sensitivity
chuẩn kiểu 99-line SIMP - xem compute.py). Module này chỉ cần lan truyền đạo
hàm đó qua phép nghịch đảo ma trận 3x3 (S = Q^-1) và công thức nu12 = -S01/S00
bằng quy tắc chuỗi giải tích (không xấp xỉ số):

    dS/dx = -S @ (dQ/dx) @ S               (đạo hàm nghịch đảo ma trận)
    d(nu12)/dx = -(dS01*S00 - S01*dS00) / S00^2   (quy tắc thương)
    d(nu21)/dx = -(dS01*S11 - S01*dS11) / S11^2

QUAN TRỌNG: forward() đã tính dQ (cần cho chính giá trị Q/nu12) nên backward()
KHÔNG cần giải FE lại - chỉ là các phép nhân ma trận nhỏ, gần như miễn phí.
Chi phí thật nằm ở forward() (solve_fe + compute_homogenized_tensor, ~50-100ms/
mẫu trên lưới 50x50, xem benchmark trong test). Đã KIỂM CHỨNG bằng finite-
difference (xem tests/test_phase5_real_physics.py::TestGradientCorrectness) -
sai số tương đối < 1e-4 so với perturbation số, xác nhận công thức đúng.
"""
import logging
import multiprocessing as mp

import numpy as np
import torch

from simp.materials.isotropic import Material
from simp.core.fem import build_dof_mesh
from simp.core.pbc import build_pbc
from simp.core.solver import solve_fe
from simp.homogenization.compute import compute_homogenized_tensor

# Cache mesh theo (nelx, nely, E0, Emin, nu) - các thành phần này KHÔNG phụ
# thuộc density nên tính 1 lần, dùng lại cho mọi mẫu/mọi step trong training.
_MESH_CACHE = {}


def _get_mesh(nelx: int, nely: int, E0: float, Emin: float, nu: float):
    key = (nelx, nely, E0, Emin, nu)
    if key not in _MESH_CACHE:
        material = Material(E0=E0, Emin=Emin, nu=nu)
        nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(nelx, nely)
        pbc = build_pbc(nelx, nely, nodenrs)
        _MESH_CACHE[key] = (material, edofMat, iK, jK, pbc)
    return _MESH_CACHE[key]


def solve_nu_with_grad(
    xPhys: np.ndarray,
    penal: float,
    E0: float = 199.0,
    Emin: float = 1e-9,
    nu: float = 0.3,
    rho0: float = 1.0,
):
    """1 lần FE-solve + homogenization THẬT trên xPhys (nely, nelx) trong
    [0,1], trả về (v12, v21, d_v12/dx, d_v21/dx) - giá trị VÀ đạo hàm giải
    tích chính xác trong CÙNG 1 lần giải (không cần giải thêm cho backward).

    Returns:
        v12, v21 : float.
        d_v12, d_v21 : np.ndarray (nely, nelx) - đạo hàm theo từng pixel.

    Không raise: nếu FE-solve hoặc nghịch đảo Q thất bại (density gần suy
    biến - vd gần như toàn rỗng/toàn đặc lúc decoder mới khởi tạo, ngẫu
    nhiên - xem VERIFICATION_GUIDE.md § Phần 2 "Training không ổn định"),
    trả về v12=v21=0.0 và gradient 0 cho mẫu đó thay vì crash cả batch -
    cùng triết lý với try/except trong simp/runner.py's OC loop (ở đó thay
    dc bằng penalty lớn; ở đây trả gradient 0 vì đây là loss có target cụ
    thể, không phải tối ưu tự do - đóng góp 0 vào backward an toàn hơn
    đóng góp giá trị rác)."""
    nely, nelx = xPhys.shape
    material, edofMat, iK, jK, pbc = _get_mesh(nelx, nely, E0, Emin, nu)

    try:
        U, U0 = solve_fe(xPhys, material.KE, iK, jK, pbc, penal, E0, Emin, rho0=rho0)
        U_total = U0 + U
        Q, dQ, _ = compute_homogenized_tensor(
            U_total, U0, xPhys, material.KE, edofMat, penal, E0, Emin, rho0=rho0
        )

        S = np.linalg.inv(Q)
        v12 = -S[0, 1] / S[0, 0]
        v21 = -S[0, 1] / S[1, 1]

        # dS/dx_pixel = -S @ dQ[:,:,i,j] @ S, vector hóa trên toàn lưới pixel.
        dS = -np.einsum('ik,klpq,lm->impq', S, dQ, S)

        d_v12 = -(dS[0, 1] * S[0, 0] - S[0, 1] * dS[0, 0]) / (S[0, 0] ** 2)
        d_v21 = -(dS[0, 1] * S[1, 1] - S[0, 1] * dS[1, 1]) / (S[1, 1] ** 2)

        if not (np.isfinite(v12) and np.isfinite(v21)
                and np.all(np.isfinite(d_v12)) and np.all(np.isfinite(d_v21))):
            raise FloatingPointError("v12/v21/gradient không hữu hạn (Q gần suy biến)")

        return float(v12), float(v21), d_v12, d_v21
    except Exception as e:
        logging.getLogger(__name__).warning(
            "solve_nu_with_grad thất bại (%s) - trả v12=v21=0, gradient=0 cho mẫu này "
            "thay vì crash cả batch.", e
        )
        zeros = np.zeros((nely, nelx))
        return 0.0, 0.0, zeros, zeros


def _solve_worker(args):
    """Hàm top-level (bắt buộc để pickle được cho multiprocessing.Pool) -
    unwrap args rồi gọi solve_nu_with_grad(). Trên Linux (start method
    'fork', mặc định), tiến trình con kế thừa _MESH_CACHE đã có sẵn của
    tiến trình cha qua copy-on-write, nên không tốn thêm chi phí dựng mesh."""
    xPhys, penal, E0, Emin, nu, rho0 = args
    return solve_nu_with_grad(xPhys, penal, E0=E0, Emin=Emin, nu=nu, rho0=rho0)


# Pool tái sử dụng giữa các batch/step (tạo tiến trình con 1 lần, tránh chi
# phí spawn lặp lại mỗi forward()) - cache theo n_workers, dọn bằng
# shutdown_pool() khi cần (vd cuối script train.py, hoặc đổi n_workers).
_POOL_CACHE = {}


def _get_pool(n_workers: int):
    if n_workers not in _POOL_CACHE:
        _POOL_CACHE[n_workers] = mp.Pool(n_workers)
    return _POOL_CACHE[n_workers]


def shutdown_pool():
    """Đóng mọi worker pool đã tạo - gọi khi kết thúc training để giải
    phóng tiến trình con sạch sẽ (không bắt buộc, pool cũng tự dọn khi
    tiến trình chính thoát, nhưng gọi tường minh tránh cảnh báo treo)."""
    for pool in _POOL_CACHE.values():
        pool.close()
        pool.join()
    _POOL_CACHE.clear()


class RealPhysicsNu(torch.autograd.Function):
    """torch.autograd.Function bọc FE-solve THẬT làm 1 lớp khả vi trong
    mạng: forward nhận ảnh mật độ (B, nely, nelx) trong [0,1] (đã tách khỏi
    graph, numpy), trả (B, 2) = [v12, v21]; backward trả gradient GIẢI TÍCH
    (không xấp xỉ) đã tính sẵn ở forward - xem solve_nu_with_grad().

    Batch xử lý TUẦN TỰ mặc định (n_workers=0); truyền n_workers>0 để chạy
    song song qua multiprocessing.Pool (mỗi mẫu độc lập, ~86ms/mẫu tuần tự
    trên lưới 50x50 - xem benchmark trong tests/test_phase5_real_physics.py,
    n_workers=12 giảm gần tuyến tính theo số core)."""

    @staticmethod
    def forward(ctx, density_grid: torch.Tensor, penal: float, E0: float, Emin: float, nu: float, rho0: float, n_workers: int = 0):
        device = density_grid.device
        dtype = density_grid.dtype
        batch = density_grid.detach().cpu().numpy().astype(np.float64)  # (B, nely, nelx)
        B = batch.shape[0]
        clipped = np.clip(batch, 0.0, 1.0)

        v_out = np.zeros((B, 2), dtype=np.float64)
        grads = np.zeros((B, 2) + batch.shape[1:], dtype=np.float64)

        if n_workers > 0:
            pool = _get_pool(n_workers)
            args = [(clipped[i], penal, E0, Emin, nu, rho0) for i in range(B)]
            results = pool.map(_solve_worker, args)
        else:
            results = [solve_nu_with_grad(clipped[i], penal, E0=E0, Emin=Emin, nu=nu, rho0=rho0)
                       for i in range(B)]

        for i, (v12, v21, d_v12, d_v21) in enumerate(results):
            v_out[i, 0], v_out[i, 1] = v12, v21
            grads[i, 0], grads[i, 1] = d_v12, d_v21

        ctx.grads = torch.from_numpy(grads).to(device=device, dtype=dtype)  # (B, 2, nely, nelx)
        return torch.from_numpy(v_out).to(device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (B, 2). Chain rule: d(loss)/d(density) =
        # sum_k grad_output[:,k] * d(v_k)/d(density) - không cần giải FE lại.
        grad_density = torch.einsum('bk,bkij->bij', grad_output, ctx.grads)
        return grad_density, None, None, None, None, None, None

"""
Bộ điều phối vòng lặp tối ưu hóa SIMP (phiên bản đơn giản).

Phối hợp: FE -> đồng nhất hóa -> hàm mục tiêu -> lọc -> OC.
Dùng dict params thay cho SimpConfig.

LỊCH SỬ FIX QUAN TRỌNG (xem CHANGELOG.md để biết chi tiết đầy đủ):
  - v12/v21 giờ dùng compute_nu12()/compute_nu21() (nghịch đảo ma trận Q
    đầy đủ), thay vì công thức rút gọn Q12/Q22 chỉ đúng khi rotation=0.
  - FIX QUAN TRỌNG NHẤT: solve_fe() trả về U là trường DAO ĐỘNG (fluctuation)
    chi, nghiệm của K@chi = -K@U0 - không phải tổng chuyển vị. Trước đây
    compute_homogenized_tensor() nhận nhầm chi làm tổng chuyển vị, khiến Q
    luôn phản ánh gần đúng vật liệu nền đẳng hướng (v12 ~ nu) bất kể
    topology. Đã sửa: cộng lại U_total = U0 + chi trước khi tính Q.
"""

import json
import math
import os
import shutil
import subprocess
import time

import numpy as np

from .materials.isotropic import Material
from .core.fem import build_dof_mesh
from .core.filter import build_filter, apply_filter, apply_sensitivity_filter
from .core.pbc import build_pbc
from .core.solver import solve_fe
from .core.oc import oc_update
from .core.convergence import ConvergenceChecker
from .homogenization.compute import compute_homogenized_tensor
from .objectives.auxetic import compute_auxetic_q12_objective, compute_nu12, compute_nu21
from .io.logger import save_csv

# Ánh xạ tên seed -> hàm sinh mật độ ban đầu
SEED_MAP = {}
for _name in [
    'circle', 'square', 'hourglass', 'four_circle',
    'hexagonal', 'nine_circle', 'cross_rectangular',
    'grid_circular_voids', 'small_square_cross', 'circle_half_quarter',
    'reentrant_bowtie',
]:
    mod = __import__(f'simp.seeds.{_name}', fromlist=[f'{_name}_seed'])
    SEED_MAP[_name] = getattr(mod, f'{_name}_seed')


def run_simp(params: dict) -> dict:
    """Chạy vòng lặp tối ưu hóa hình dạng SIMP.

    Args:
        params: Từ điển tham số với các key:
            nelx, nely, volfrac, penal, rmin, ft,
            E0, Emin, nu, move, max_iter,
            tol_change, tol_obj, window_size,
            seed (tên seed), objective ('auxetic'),
            void_size_frac, rotation_deg, beta, mu,
            output_dir, save_every, scale_factor.

    Returns:
        Từ điển kết quả:
            xPhys, Q, n_iters, converged, v12, v21, objective,
            output_dir, elapsed_time, history.
    """
    def _get_git_hash() -> str:
        try:
            return subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return 'unknown'

    # --- Trích xuất tham số ---
    nelx = params['nelx']
    nely = params['nely']
    volfrac = params['volfrac']
    penal = params.get('penal', 3.0)
    rmin = params.get('rmin', 3.0)
    ft = params.get('ft', 2)
    E0 = params.get('E0', 199.0)
    Emin = params.get('Emin', 1e-9)
    nu = params.get('nu', 0.3)
    move = params.get('move', 0.1)
    max_iter = params.get('max_iter', 200)
    tol_change = params.get('tol_change', 0.01)
    tol_obj = params.get('tol_obj', 0.05)
    window_size = params.get('window_size', 20)
    seed_name = params.get('seed', 'circle')
    obj_type = params.get('objective', 'auxetic')
    verbose = params.get('verbose', True)
    void_size_frac = params.get('void_size_frac', 0.4)
    rotation_deg = params.get('rotation_deg', 0.0)
    beta = params.get('beta', 1.0)
    # mu: mặc định 0.0 (khuyến nghị GIỮ 0.0). Số hạng mu*(Q11+Q22) trong
    # objective KHÔNG tạo áp lực trực tiếp lên Q12 - nó thưởng độ cứng,
    # điều không liên quan tới việc kéo Q12 xuống âm. Đã kiểm chứng thực
    # nghiệm: mu > 0 không cải thiện (thậm chí có xu hướng làm Q12 dương
    # hơn). Cần thiết kế lại cơ chế này trước khi bật lại mu > 0.
    mu = params.get('mu', 0.0)
    rho0 = params.get('rho0', 1.0)
    save_every = params.get('save_every', 1)
    scale_factor = params.get('scale_factor', 1)

    # Thư mục đầu ra
    output_dir = params.get('output_dir') or f'outputs/simp_results_{seed_name}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Thiết lập ---
    material = Material(E0=E0, Emin=Emin, nu=nu)
    nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(nelx, nely)
    H, Hs = build_filter(nelx, nely, rmin)
    pbc = build_pbc(nelx, nely, nodenrs)
    conv_checker = ConvergenceChecker(
        tol_change=tol_change,
        tol_obj=tol_obj,
        window_change=max(3, int(window_size / 4)),
        window_obj=window_size,
        min_iter=10,
    )

    # Seed ban đầu
    # Một số seed (hourglass, square) cần volfrac thay vì void_size_frac
    seed_fn = SEED_MAP.get(seed_name)
    if seed_fn is not None:
        if seed_name == 'hourglass':
            x = seed_fn(nelx, nely, volfrac, rotation_deg)
        else:
            x = seed_fn(nelx, nely, void_size_frac, rotation_deg)
    else:
        from .seeds.circle import circle_seed
        x = circle_seed(nelx, nely, void_size_frac, rotation_deg)

    xPhys = x.copy()
    from .io.visualizer import save_density_image
    save_density_image(xPhys, output_dir, 0, scale_factor)

    # --- Vòng lặp ---
    change = 1.0
    loop = 0
    prev_obj = float('inf')
    converged = False
    c = float('nan')
    Q = np.zeros((3, 3))
    v12 = float('nan')
    v21 = float('nan')

    history = {
        'iteration': [0],
        'v12': [float('nan')],
        'v21': [float('nan')],
        'objective': [float('nan')],
        'volume': [float(np.mean(xPhys))],
    }

    metadata = {
        'git_hash': _get_git_hash(),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'version': '1.4.0',
        'params': {k: v for k, v in params.items() if k != 'output_dir'},
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    t_start = time.time()

    while loop < max_iter:
        loop += 1

        try:
            # FE: trả về U là trường DAO ĐỘNG (fluctuation) chi thỏa
            # K@chi = -K@U0, KHÔNG phải tổng chuyển vị (xem docstring
            # solve_fe trong core/solver.py).
            U, U0 = solve_fe(xPhys, material.KE, iK, jK, pbc, penal, E0, Emin, rho0=rho0)

            # Đồng nhất hóa: công thức Q_ij = 1/|Omega| Sum_e (u_e^i)^T k_e (u_e^j)
            # (Andreassen et al. 2014, eq. 6) yêu cầu TỔNG chuyển vị u = u0 + chi.
            # Cộng lại U0 trước khi dùng - thiếu bước này khiến Q luôn lệch về
            # phía vật liệu nền đẳng hướng bất kể topology thực tế.
            U_total = U0 + U
            Q, dQ, _ = compute_homogenized_tensor(
                U_total, U0, xPhys, material.KE, edofMat, penal, E0, Emin, rho0=rho0,
            )

            # Hàm mục tiêu
            c, dc = compute_auxetic_q12_objective(Q, dQ, volfrac, E0, beta=beta, mu=mu)

            if math.isnan(c) or np.isnan(np.sum(xPhys)):
                print(f'[STOP] NaN tại lần lặp {loop}')
                break

        except Exception as e:
            print(f'[ERROR] Loop {loop}: {e}')
            c = 1e12
            dc = -np.ones((nely, nelx)) * 1e6
            if not hasattr(run_simp, '_err_count'):
                run_simp._err_count = 0
            run_simp._err_count += 1
            if run_simp._err_count >= 5:
                print('[STOP] Quá 5 lỗi liên tiếp, dừng pipeline')
                break

        # Hệ số Poisson tính chính xác qua nghịch đảo ma trận đầy đủ
        # (S = Q^-1), đúng cho cả trường hợp có rotation (Q13, Q23 != 0).
        v12 = compute_nu12(Q)
        v21 = compute_nu21(Q)

        # Kiểm tra hội tụ
        if conv_checker.should_stop(change, c, prev_obj, loop, max_iter):
            converged = conv_checker.converged
            if converged and verbose:
                print(f'[DONE] Hội tụ tại lần lặp {loop}')
            break
        prev_obj = c

        # Lọc độ nhạy
        dv = np.ones((nely, nelx))
        dc = apply_sensitivity_filter(dc, x, H, Hs, ft)
        if ft == 2:
            dv = apply_filter(dv, H, Hs)

        # OC
        xnew, xPhys = oc_update(x, dc, dv, volfrac, move, H, Hs, ft)
        change = np.max(np.abs(xnew - x))
        x = xnew

        history['iteration'].append(loop)
        history['v12'].append(v12)
        history['v21'].append(v21)
        history['objective'].append(c)
        history['volume'].append(float(np.mean(xPhys)))

        if loop % save_every == 0:
            from .io.visualizer import save_density_image
            save_density_image(xPhys, output_dir, loop, scale_factor)

        print(f'Loop:{loop:4d}  obj:{c:+.4e}  vol:{np.mean(xPhys):.3f}  '
                f'chg:{change:.3f}  v12:{v12:.4f}  v21:{v21:.4f}') if verbose else None

        if hasattr(run_simp, '_err_count'):
            run_simp._err_count = 0

    # --- Kết thúc ---
    t_elapsed = time.time() - t_start

    save_csv(output_dir, history)

    if loop % save_every != 0:
        save_density_image(xPhys, output_dir, loop, scale_factor)

    print(f'Hoàn thành {loop} loops ({t_elapsed:.1f}s)')
    print(f'  obj={c:.4f}  v12={v12:.4f}  v21={v21:.4f}  vol={np.mean(xPhys):.3f}')

    return {
        'xPhys': xPhys,
        'Q': Q,
        'n_iters': loop,
        'converged': converged,
        'v12': v12,
        'v21': v21,
        'objective': c,
        'output_dir': output_dir,
        'elapsed_time': t_elapsed,
        'history': history,
    }
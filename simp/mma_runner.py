"""
Bộ điều phối vòng lặp tối ưu hóa SIMP dùng MMA (Method of Moving Asymptotes,
Svanberg 1987) qua nlopt.LD_MMA, thay cho core/oc.py::oc_update() (OC).

TẠI SAO: xem EXPERIMENT_LOG.md mục 2026-07-30. `oc_update_dual()` (OC hai-
multiplier tự viết, nhánh OC-2-multilayer) được implement để thay cổng nhị
phân stiff_ok của oc_update() gốc (đã xác nhận gây dao động/sụp cấu trúc khi
ràng buộc stiffness vi phạm) - nhưng THẤT BẠI: bisection multiplier trần trụi
không có cơ chế trust-region, nên khi mục tiêu tuyến tính hóa của 1 ràng
buộc vượt quá những gì đạt được trong 1 bước move-limit, multiplier bị đẩy
tới trần và "xóa sổ" ảnh hưởng của multiplier kia - dù đã thử chặn mục tiêu
về mức khả thi, vẫn thất bại vì trường độ nhạy stiffness rộng khắp domain
(không cục bộ). MMA giải ĐÚNG vấn đề này bằng "moving asymptotes" - biên độ
xấp xỉ lồi tự thích ứng theo lịch sử hội tụ (mở rộng nếu dao động, thu hẹp
nếu đơn điệu), cung cấp cơ chế trust-region ngầm mà OC/bisection không có,
và xử lý NHIỀU ràng buộc non-linear tổng quát bằng KKT/đối ngẫu chuẩn thay
vì heuristic gate/nested-bisection tự chế.

KHÁC BIỆT KIẾN TRÚC quan trọng so với runner.py::run_simp(): OC được gọi 1
LẦN MỖI vòng lặp SIMP (runner.py tự quản lý `while loop < max_iter`, tự giải
FE mỗi vòng RỒI gọi oc_update() lấy xnew). nlopt.LD_MMA thì NGƯỢC LẠI - nó tự
quản lý TOÀN BỘ vòng lặp tối ưu bên trong 1 lệnh gọi opt.optimize(), gọi lại
callback objective/constraint (mỗi callback tự giải FE+homogenization THẬT
tại điểm x nlopt đưa ra) nhiều lần cho tới khi hội tụ hoặc hết maxeval. Vì
vậy run_simp_mma() không dùng chung vòng lặp while với run_simp() - nó đóng
gói toàn bộ forward-pipeline (filter -> FE -> homogenization -> objective ->
sensitivity-filter) thành callback cho nlopt, và 2 ràng buộc stiffness
(Q11>=delta, Q22>=delta) được truyền RIÊNG BIỆT cho nlopt (không gộp thành 1
"thành phần ràng buộc chặt hơn" như oc_update_dual() phải làm để tương thích
kiến trúc nested-bisection) - MMA tự xử lý cả 2 đồng thời đúng kiểu KKT.

TÍNH NĂNG THỬ NGHIỆM, TẮT MẶC ĐỊNH (dùng qua stiffness_mode/optimizer='mma'
tường minh): chỉ hỗ trợ ft=2 (density filter), projection=None (KHÔNG hỗ trợ
Heaviside projection - out of scope cho lần triển khai đầu), penal CỐ ĐỊNH
(không continuation - nlopt tự quản lý toàn bộ vòng lặp bên trong 1 lệnh gọi
nên không có "vòng lặp SIMP" bên ngoài để continuation theo). Yêu cầu package
`nlopt` (xem requirements.txt).
"""
import json
import os
import shutil
import subprocess
import time

import numpy as np
import nlopt

from .materials.isotropic import Material
from .core.fem import build_dof_mesh
from .core.filter import build_filter, apply_filter, apply_sensitivity_filter
from .core.pbc import build_pbc
from .core.solver import solve_fe
from .core.oc import X_MIN
from .homogenization.compute import compute_homogenized_tensor
from .objectives.auxetic import (
    compute_auxetic_q12_objective, compute_auxetic_normalized_objective,
    compute_nu12, compute_nu21, stiffness_delta,
)
from .io.logger import save_csv
from .io.visualizer import save_density_image
from .runner import SEED_MAP


def run_simp_mma(params: dict) -> dict:
    """Chạy vòng lặp tối ưu hóa hình dạng SIMP dùng nlopt.LD_MMA.

    Args: cùng dict params với run_simp() (xem runner.py), nhưng chỉ đọc các
        key liên quan trực tiếp tới vật lý/mục tiêu - các tính năng thử
        nghiệm chỉ có ý nghĩa trong kiến trúc vòng lặp OC (move_min,
        penal_init, projection, enforce_connectivity, use_sqrt, ...) KHÔNG
        được hỗ trợ (bị bỏ qua nếu có mặt, trừ projection - raise lỗi rõ
        ràng nếu bật, xem docstring module).

    Returns: cùng cấu trúc dict với run_simp() (xPhys, Q, n_iters, converged,
        v12, v21, objective, output_dir, elapsed_time, history).
    """
    def _get_git_hash() -> str:
        try:
            return subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return 'unknown'

    nelx = params['nelx']
    nely = params['nely']
    volfrac = params['volfrac']
    penal = params.get('penal', 3.0)
    rmin = params.get('rmin', 3.0)
    ft = params.get('ft', 2)
    if ft != 2:
        raise ValueError("run_simp_mma() chỉ hỗ trợ ft=2 (density filter).")
    if params.get('projection') is not None:
        raise ValueError(
            "run_simp_mma() chưa hỗ trợ projection (TÍNH NĂNG THỬ NGHIỆM, "
            "xem docstring module)."
        )
    E0 = params.get('E0', 199.0)
    Emin = params.get('Emin', 1e-9)
    nu = params.get('nu', 0.3)
    max_iter = params.get('max_iter', 200)
    seed_name = params.get('seed', 'circle')
    verbose = params.get('verbose', True)
    void_size_frac = params.get('void_size_frac', 0.4)
    rotation_deg = params.get('rotation_deg', 0.0)
    beta = params.get('beta', 1.0)
    mu = params.get('mu', 0.0)
    objective_variant = params.get('objective_variant', 'q12')
    save_every = params.get('save_every', 1)
    scale_factor = params.get('scale_factor', 1)
    ftol_rel = params.get('mma_ftol_rel', 1e-5)
    xtol_rel = params.get('mma_xtol_rel', 1e-5)
    stiffness_tol = params.get('mma_stiffness_tol', 1e-3)
    volume_tol = params.get('mma_volume_tol', 1e-6)

    output_dir = params.get('output_dir') or f'outputs/simp_results_{seed_name}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Thiết lập (giống run_simp(), không đổi) ---
    material = Material(E0=E0, Emin=Emin, nu=nu)
    nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(nelx, nely)
    H, Hs = build_filter(nelx, nely, rmin)
    pbc = build_pbc(nelx, nely, nodenrs)
    nele = nelx * nely
    delta = stiffness_delta(volfrac, E0)
    # d(xPhys)/dx cố định suốt (bộ lọc mật độ tuyến tính, không phụ thuộc x)
    # - dùng cho gradient ràng buộc volume, KHÁC dc/dg (phải re-filter mỗi
    # vòng vì apply_sensitivity_filter với ft=1 phụ thuộc x, dù ft=2 ở đây
    # thì apply_filter==apply_sensitivity_filter nên fixed field cũng đúng).
    dv_filtered = apply_filter(np.ones((nely, nelx)), H, Hs)

    seed_fn = SEED_MAP.get(seed_name)
    if seed_fn is not None:
        if seed_name == 'hourglass':
            x0 = seed_fn(nelx, nely, volfrac, rotation_deg)
        else:
            x0 = seed_fn(nelx, nely, void_size_frac, rotation_deg)
    else:
        from .seeds.circle import circle_seed
        x0 = circle_seed(nelx, nely, void_size_frac, rotation_deg)

    # Seed thô có pixel = 0.0 tuyệt đối (vùng void) - dưới sàn X_MIN=0.001,
    # nlopt.optimize() từ chối điểm khởi tạo vi phạm bound (invalid_argument)
    # khác oc_update() (chỉ áp X_MIN cho xnew SAU vòng lặp đầu, không đòi hỏi
    # x0 hợp lệ). Clip trước khi đưa vào nlopt.
    x0 = np.clip(x0, X_MIN, 1.0)
    save_density_image(x0, output_dir, 0, scale_factor)

    history = {
        'iteration': [0], 'v12': [float('nan')], 'v21': [float('nan')],
        'objective': [float('nan')], 'volume': [float(np.mean(x0))],
    }
    state = {'loop': 0, 'cache_key': None, 'cache': None, 'last_x_flat': x0.flatten('F').copy()}

    def _evaluate(x_flat: np.ndarray) -> dict:
        """Chạy 1 lần forward-pipeline đầy đủ tại x_flat, cache theo x - nlopt
        thường gọi objective RỒI từng constraint ở CÙNG 1 x trong 1 vòng
        outer MMA, nên cache tránh giải FE lặp lại 3 lần/vòng."""
        key = x_flat.tobytes()
        if key == state['cache_key']:
            return state['cache']

        x2d = x_flat.reshape((nely, nelx), order='F')
        xPhys_flat = H @ x2d.flatten('F') / Hs
        xPhys = np.reshape(xPhys_flat, (nely, nelx), order='F')

        U, U0 = solve_fe(xPhys, material.KE, iK, jK, pbc, penal, E0, Emin)
        Q, dQ, _ = compute_homogenized_tensor(
            U0 + U, U0, xPhys, material.KE, edofMat, penal, E0, Emin,
        )
        if objective_variant == 'normalized':
            c, dc = compute_auxetic_normalized_objective(Q, dQ, volfrac, E0, beta=beta)
        else:
            c, dc = compute_auxetic_q12_objective(Q, dQ, volfrac, E0, beta=beta, mu=mu)
        dc_filtered = apply_sensitivity_filter(dc, x2d, H, Hs, ft)
        v12 = compute_nu12(Q)
        v21 = compute_nu21(Q)

        state['loop'] += 1
        loop = state['loop']
        state['last_x_flat'] = x_flat.copy()
        history['iteration'].append(loop)
        history['v12'].append(v12)
        history['v21'].append(v21)
        history['objective'].append(c)
        history['volume'].append(float(np.mean(xPhys)))
        if loop % save_every == 0:
            save_density_image(xPhys, output_dir, loop, scale_factor)
        if verbose:
            print(f'[MMA] eval:{loop:4d}  obj:{c:+.4e}  vol:{np.mean(xPhys):.3f}  '
                  f'v12:{v12:.4f}  v21:{v21:.4f}')

        result = dict(c=c, dc=dc_filtered, Q=Q, dQ=dQ, xPhys=xPhys, v12=v12, v21=v21)
        state['cache_key'] = key
        state['cache'] = result
        return result

    def objective_cb(x_flat, grad):
        r = _evaluate(x_flat)
        if grad.size > 0:
            grad[:] = r['dc'].flatten('F')
        return float(r['c'])

    def make_stiffness_cb(idx: int):
        def cb(x_flat, grad):
            r = _evaluate(x_flat)
            Q, dQ = r['Q'], r['dQ']
            g = delta - Q[idx, idx]
            if grad.size > 0:
                x2d = x_flat.reshape((nely, nelx), order='F')
                dg_filtered = apply_sensitivity_filter(dQ[idx, idx], x2d, H, Hs, ft)
                # g = delta - Q_ii -> dg/dx = -dQ_ii/dx (khác dg trong
                # oc_update_dual, ở đó dg là dQ_ii/dx THÔ vì công thức OC tự
                # xử lý dấu riêng - ở đây nlopt cần gradient ĐÚNG của g theo
                # đúng quy ước g(x)<=0 chuẩn, nên đảo dấu tường minh tại đây).
                grad[:] = -dg_filtered.flatten('F')
            return float(g)
        return cb

    def volume_cb(x_flat, grad):
        r = _evaluate(x_flat)
        if grad.size > 0:
            grad[:] = (dv_filtered / nele).flatten('F')
        return float(np.mean(r['xPhys']) - volfrac)

    opt = nlopt.opt(nlopt.LD_MMA, nele)
    opt.set_lower_bounds(X_MIN)
    opt.set_upper_bounds(1.0)
    opt.set_min_objective(objective_cb)
    opt.add_inequality_constraint(volume_cb, volume_tol)
    opt.add_inequality_constraint(make_stiffness_cb(0), stiffness_tol)
    opt.add_inequality_constraint(make_stiffness_cb(1), stiffness_tol)
    opt.set_maxeval(max_iter)
    opt.set_ftol_rel(ftol_rel)
    opt.set_xtol_rel(xtol_rel)
    mma_initial_step = params.get('mma_initial_step', None)
    if mma_initial_step is not None:
        opt.set_initial_step(mma_initial_step)

    x_init_flat = x0.flatten('F')
    t_start = time.time()
    result_code = None
    try:
        x_opt_flat = opt.optimize(x_init_flat)
        result_code = opt.last_optimize_result()
    except Exception as e:
        # nlopt raises (vd RoundoffLimited) thay vì trả về khi thất bại số
        # học - rơi về điểm cuối cùng đã evaluate thành công thay vì crash
        # toàn bộ pipeline (giống hành vi try/except trong run_simp()).
        if verbose:
            print(f'[MMA] [WARN] nlopt.optimize() raise {type(e).__name__}: {e} '
                  f'- dùng điểm cuối đã evaluate.')
        x_opt_flat = state['last_x_flat']
        result_code = -999
    t_elapsed = time.time() - t_start

    final = _evaluate(x_opt_flat)
    loop = state['loop']
    # converged=True chỉ khi nlopt tự báo hội tụ THẬT (không phải chạm
    # maxeval) VÀ số eval < max_iter - cùng quy ước "n_iters<max_iter đáng
    # tin hơn cờ converged" đã có trong run_simp()/ConvergenceChecker, xem
    # EXPERIMENT_LOG.md ("converged=True không đảm bảo hội tụ thật").
    converged = (
        result_code in (nlopt.SUCCESS, nlopt.FTOL_REACHED, nlopt.XTOL_REACHED)
        and loop < max_iter
    )

    save_csv(output_dir, history)
    metadata = {
        'git_hash': _get_git_hash(),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'version': '1.4.0-mma',
        'params': {k: v for k, v in params.items() if k != 'output_dir'},
        'nlopt_result_code': int(result_code),
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f'Hoàn thành {loop} evals ({t_elapsed:.1f}s, nlopt_code={result_code})')
        print(f"  obj={final['c']:.4f}  v12={final['v12']:.4f}  v21={final['v21']:.4f}  "
              f"vol={np.mean(final['xPhys']):.3f}")

    return {
        'xPhys': final['xPhys'],
        'Q': final['Q'],
        'n_iters': loop,
        'converged': converged,
        'v12': final['v12'],
        'v21': final['v21'],
        'objective': final['c'],
        'output_dir': output_dir,
        'elapsed_time': t_elapsed,
        'history': history,
    }

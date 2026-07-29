"""
Bộ điều phối vòng lặp tối ưu hóa SIMP (phiên bản đơn giản).

Phối hợp: FE -> đồng nhất hóa -> hàm mục tiêu -> lọc -> OC. Dùng dict params
thay cho SimpConfig.

FIX QUAN TRỌNG (xem CHANGELOG.md): solve_fe() trả về fluctuation chi (nghiệm
K@chi=-K@U0), không phải tổng chuyển vị. compute_homogenized_tensor() từng
nhận nhầm chi làm tổng chuyển vị, khiến Q luôn lệch về vật liệu nền đẳng
hướng bất kể topology - đã sửa bằng U_total = U0 + chi trước khi tính Q.
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
from .core.filter import (
    build_filter, apply_filter, apply_sensitivity_filter,
    apply_heaviside_projection, heaviside_projection_derivative,
)
from .core.pbc import build_pbc
from .core.solver import solve_fe
from .core.oc import oc_update
from .core.convergence import ConvergenceChecker
from .homogenization.compute import compute_homogenized_tensor
from .objectives.auxetic import (
    compute_auxetic_q12_objective, compute_auxetic_normalized_objective,
    compute_nu12, compute_nu21, stiffness_delta,
)
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


def move_at_iteration(move: float, move_min, loop: int, max_iter: int) -> float:
    """move tại vòng lặp `loop` (1-indexed) khi bật giảm dần tuyến tính.

    move_min=None: trả về `move` không đổi (hành vi mặc định/cũ).
    move_min đặt: giảm tuyến tính move -> move_min khi loop đi từ 1 -> max_iter
    (loop=1 trả về move, loop=max_iter trả về move_min, kẹp trong [1,max_iter]).
    """
    if move_min is None:
        return move
    if max_iter <= 1:
        return move_min
    frac = min(max(loop - 1, 0), max_iter - 1) / (max_iter - 1)
    return move - (move - move_min) * frac


def penal_at_iteration(penal: float, penal_init, loop: int, max_iter: int,
                        penal_step_every: int | None = None) -> float:
    """penal tại vòng lặp `loop` (1-indexed) khi bật continuation.

    penal_init=None: trả về `penal` không đổi (hành vi mặc định/cũ).
    penal_init đặt: tăng penal_init -> penal khi loop đi từ 1 -> max_iter
    (loop=1 trả về penal_init, loop=max_iter trả về penal, kẹp trong [1,max_iter]).
    Continuation kinh điển (Sigmund/Bendsoe): bắt đầu penal thấp (bài toán lồi
    hơn, tránh local minima sớm), tăng dần về penal cuối để ép nhị phân hóa.

    penal_step_every=None (mặc định): tuyến tính liên tục mỗi vòng lặp.
    penal_step_every=N: bậc thang kinh điển - giữ nguyên penal trong N vòng
    rồi mới nhảy (như 99-line Sigmund: giữ cố định đến khi ổn định rồi tăng),
    thay vì đổi "định luật vật liệu" liên tục mỗi vòng. Thử nghiệm 2026-07-26:
    continuous linear gây osc_score bùng nổ trên reentrant_bowtie (yield
    46%->26%) - nghi ngờ OC không kịp gần-cân-bằng trước khi penal đổi tiếp.
    """
    if penal_init is None:
        return penal
    if max_iter <= 1:
        return penal
    effective_loop = loop
    if penal_step_every is not None and penal_step_every > 0:
        effective_loop = (loop // penal_step_every) * penal_step_every
    effective_loop = min(max(effective_loop - 1, 0), max_iter - 1)
    frac = effective_loop / (max_iter - 1)
    return penal_init + (penal - penal_init) * frac


def nudge_disconnected_islands(x: np.ndarray, xPhys: np.ndarray,
                                min_feature_px: int = 2,
                                min_relative_size: float = 0.15,
                                floor: float = 0.01) -> np.ndarray:
    """Hạ mật độ các mảnh vật liệu NHIỄU (nhỏ) trong xPhys xuống `floor` trên
    biến thiết kế x tại đúng vị trí đó - heuristic "nudge" giữa các vòng lặp
    cho enforce_connectivity (xem run_simp()).

    FIX 2026-07-29 (regression B2, xem EXPERIMENT_LOG.md): bản đầu tiên dìm
    TẤT CẢ mảnh không phải mảnh lớn nhất, bất kể kích thước - trace thực tế
    trên seed `four_circle` cho thấy điều này dìm luôn cả các "cánh" cấu trúc
    thật (vd 132px = 51% kích thước mảnh lớn nhất tại thời điểm đó, KHÔNG
    phải nhiễu) đang trong quá trình hình thành cầu nối với thân chính, tạo
    hiệu ứng "khóa cứng" winner-take-last: mảnh lớn nhất được ưu ái phình to
    liên tục, còn các cánh bị dìm lặp lại mỗi connectivity_every vòng trước
    khi kịp vượt ngưỡng 0.5 để tái liên thông - kết quả RỜI RẠC HƠN (10->19
    mảnh) thay vì liên thông hơn. Sửa: chỉ dìm mảnh thực sự nhỏ (nhiễu) -
    dưới min_feature_px pixel TUYỆT ĐỐI, HOẶC dưới min_relative_size (mặc
    định 15%) kích thước mảnh lớn nhất - để mảnh cấu trúc đang hình thành có
    cơ hội tự nhiên phát triển/liên thông thay vì bị diệt non.

    Viết lại độc lập với pipeline/phase5_cvae/manufacturability.py::
    check_connectivity() (cùng logic 8-connectivity + ndimage.label) vì
    simp/ không phụ thuộc pipeline/ (hướng dependency ngược lại). LƯU Ý:
    min_feature_px ở đây là ngưỡng SỐ PIXEL của cả mảnh (area), khác với
    check_connectivity() (đó là bán kính erosion đo bề rộng nét) - cùng tên
    tham số nhưng khác đơn vị/ý nghĩa, không dùng chung logic.

    Args:
        x: Biến thiết kế hiện tại (nely, nelx) - SẼ bị sửa tại chỗ các pixel
            thuộc đảo nhiễu.
        xPhys: Mật độ vật lý hiện tại (nely, nelx) - dùng để XÁC ĐỊNH đảo
            (binarize > 0.5), nhưng bản thân không bị sửa trực tiếp ở đây
            (sẽ được filter/projection tính lại từ x đã nudge ở vòng sau).
        min_feature_px: Ngưỡng số pixel TUYỆT ĐỐI - mảnh có ít hơn ngần này
            pixel luôn bị coi là nhiễu, bất kể tỷ lệ so với mảnh lớn nhất.
        min_relative_size: Ngưỡng TƯƠNG ĐỐI (phần của kích thước mảnh lớn
            nhất) - mảnh nhỏ hơn ngưỡng này (và lớn hơn min_feature_px) vẫn
            bị coi là nhiễu; mảnh lớn hơn được coi là cấu trúc thật, GIỮ
            NGUYÊN (không nudge) để có cơ hội tự liên thông.
        floor: Giá trị x hạ xuống tại các pixel thuộc đảo nhiễu.

    Returns:
        x đã nudge (cùng mảng, sửa tại chỗ - trả về cho tiện dùng theo chuỗi).
    """
    from scipy import ndimage
    solid = xPhys > 0.5
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labels, n_components = ndimage.label(solid, structure=structure)
    if n_components <= 1:
        return x
    sizes = ndimage.sum(solid, labels, index=range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    largest_size = sizes[largest_label - 1]
    noise_threshold = max(min_feature_px, min_relative_size * largest_size)
    noise_labels = [
        i + 1 for i, s in enumerate(sizes)
        if (i + 1) != largest_label and s < noise_threshold
    ]
    if not noise_labels:
        return x
    island_mask = np.isin(labels, noise_labels)
    x[island_mask] = np.minimum(x[island_mask], floor)
    return x


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
    # move_min (mặc định None = TẮT, hành vi cũ - move cố định suốt vòng lặp):
    # nếu đặt, `move` giảm tuyến tính từ giá trị ở trên xuống move_min qua
    # max_iter vòng lặp (xem _move_at_iteration bên dưới). Thử nghiệm giảm
    # oscillation/limit-cycle cuối vòng lặp (xem osc_score,
    # analysis/scripts/audit_label_stability.py) - move lớn cố định suốt
    # quá trình là nguyên nhân kinh điển gây dao động quanh cực trị thay vì
    # hội tụ êm khi OC đã gần đáp án. CHƯA áp dụng cho dataset hiện có -
    # chỉ là tính năng thử nghiệm, cần pilot trước khi cân nhắc rebuild.
    move_min = params.get('move_min', None)
    # penal_init (mặc định None = TẮT, hành vi cũ - penal cố định suốt vòng
    # lặp): nếu đặt, `penal` tăng tuyến tính từ penal_init lên giá trị `penal`
    # ở trên qua max_iter vòng lặp (continuation kinh điển - xem
    # penal_at_iteration). Thử nghiệm giảm local-minima/oscillation đầu vòng
    # lặp, bổ trợ move_min (tác động cuối vòng lặp). CHƯA áp dụng cho dataset
    # hiện có - chỉ là tính năng thử nghiệm, cần pilot trước.
    penal_init = params.get('penal_init', None)
    penal_step_every = params.get('penal_step_every', None)
    # use_sqrt (mặc định False = hành vi cũ, x*ratio): bật để dùng damping
    # exponent kinh điển eta=0.5 (x*sqrt(ratio), Bendsoe 1995/Sigmund 2001
    # 99-line, Andreassen 2011 88-line) - chuẩn ngành để ổn định vòng lặp OC,
    # xem oc_update() cho công thức đầy đủ. Thử nghiệm cho vấn đề dao động
    # (osc_score) độc lập với move_min/penal_init ở trên.
    use_sqrt = params.get('use_sqrt', False)
    # projection (mặc định None = TẮT, hành vi cũ - xPhys = x̃ filtered thẳng):
    # 'heaviside' bật robust/minimum-length-scale projection (Wang-Lazarov-
    # Sigmund 2011) qua core/filter.py::apply_heaviside_projection(). Xem
    # AUDIT_REPORT_INDEPENDENT_2026-07-29.md mục 4.1/B1. CHỈ hỗ trợ ft=2
    # (density filter) - ft=1 không có bước "xPhys = filter(x)" trung gian để
    # chiếu lên. CHƯA áp dụng cho dataset hiện có - tính năng thử nghiệm, cần
    # pilot trên vài seed trước khi cân nhắc dùng rộng rãi.
    # FIX 2026-07-29 (B1, xem AUDIT_REPORT_INDEPENDENT_2026-07-29.md): trước
    # đây ràng buộc thể tích trong oc_update() bisection trên mean(x̃) (TRƯỚC
    # projection) thay vì mean(x̂) (SAU projection), gây lệch volfrac có hệ
    # thống đo được trong pilot (vd volfrac=0.45 -> vol thật 0.469). Đã sửa:
    # oc_update() giờ nhận beta_proj/eta_proj và tự bisection trên mean(x̂).
    projection = params.get('projection', None)
    beta_proj = params.get('beta_proj', 8.0)
    eta_proj = params.get('eta', 0.5)
    # beta_proj_init (mặc định None = TẮT, beta_proj cố định suốt vòng lặp):
    # nếu đặt, beta_proj tăng tuyến tính từ beta_proj_init lên beta_proj qua
    # max_iter vòng lặp (continuation kinh điển cho Heaviside projection,
    # Wang-Lazarov-Sigmund 2011) - projection gần identity lúc đầu (bài toán
    # lồi hơn, tránh local minima sớm khi thể tích/hình dạng còn đang định
    # hình), sắc dần về beta_proj cuối. Cùng pattern với penal_init ở trên.
    beta_proj_init = params.get('beta_proj_init', None)
    # enforce_connectivity (mặc định False = TẮT, hành vi cũ): mỗi
    # connectivity_every vòng lặp, kiểm tra liên thông trên xPhys hiện tại
    # (analogue của pipeline/phase5_cvae/manufacturability.py::check_connectivity(),
    # viết lại độc lập ở đây vì simp/ không phụ thuộc pipeline/), NẾU có >1
    # mảnh vật liệu rời rạc thì "nudge" (KHÔNG PHẢI ràng buộc Lagrange chặt)
    # - hạ mật độ pixel thuộc các mảnh KHÔNG PHẢI mảnh lớn nhất xuống gần 0
    # trên biến thiết kế x (trước filter/projection), để các vòng OC tiếp
    # theo có xu hướng loại bỏ đảo đó thay vì giữ nguyên. Đây là heuristic
    # thực dụng giữa các vòng lặp, KHÔNG sửa gradient toán học của objective.
    # Xem AUDIT_REPORT_INDEPENDENT_2026-07-29.md mục 4.2/B2. CHƯA áp dụng cho
    # dataset hiện có - cần pilot trên seed hay bị rời rạc trước
    # (grid_circular_voids/nine_circle/four_circle theo audit).
    enforce_connectivity = params.get('enforce_connectivity', False)
    connectivity_every = params.get('connectivity_every', 10)
    connectivity_min_px = params.get('connectivity_min_feature_px', 2)
    # FIX 2026-07-29 (regression B2 four_circle, xem nudge_disconnected_islands()
    # và EXPERIMENT_LOG.md): ngưỡng tương đối để phân biệt mảnh cấu trúc thật
    # (giữ nguyên) với mảnh nhiễu thật sự nhỏ (mới nudge).
    connectivity_min_relative_size = params.get('connectivity_min_relative_size', 0.15)
    connectivity_floor = params.get('connectivity_floor', 0.01)
    # objective_variant (mặc định 'q12' = hành vi cũ, minimize Q12 thô):
    # 'normalized' dùng compute_auxetic_normalized_objective() (minimize
    # Q12/sqrt(Q11*Q22), bị chặn tự nhiên [-1,1] bởi PSD của Q - xem
    # objectives/auxetic.py). CHƯA áp dụng cho dataset hiện có, cần A/B test.
    # Xem AUDIT_REPORT_INDEPENDENT_2026-07-29.md mục 4.3/B3.
    objective_variant = params.get('objective_variant', 'q12')
    # FIX 2026-07-29 (phát hiện SAU bug-fix A1 Q-scaling, xem EXPERIMENT_LOG.md):
    # oc_update() đã có sẵn tham số Q/delta để enforce ràng buộc CỨNG
    # (Q11>=delta, Q22>=delta) trong bisection - đúng thiết kế tham chiếu gốc
    # (MATLAB topK_Hourglass.m, xem docstring oc.py) - nhưng run_simp() trước
    # đây KHÔNG BAO GIỜ truyền Q/delta vào lời gọi oc_update() (orphaned, độc
    # lập với bug A1). Trước khi sửa A1, bug scaling Q vô tình khiến phạt MỀM
    # trong objectives/auxetic.py (so Q với cùng delta) LUÔN kích hoạt mạnh,
    # che giấu việc thiếu ràng buộc CỨNG này. Sau khi sửa A1 (Q về đúng scale
    # thật), phạt mềm một mình không đủ mạnh - kiểm chứng trực tiếp: hexagonal
    # sụp cấu trúc hoàn toàn (vol~0.005) ở 2/6 điều kiện thử trong dải volfrac
    # production (0.50-0.58). Sửa: nối lại ràng buộc cứng đã có sẵn - dùng
    # ĐÚNG delta (stiffness_delta()) chia sẻ với phạt mềm, Q lấy từ vòng lặp
    # HIỆN TẠI (evaluate tại x cũ, đúng quy ước đã ghi chú sẵn trong oc.py).
    if projection == 'heaviside' and ft != 2:
        raise ValueError("projection='heaviside' chỉ hỗ trợ ft=2 (density filter).")
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
    # mu mặc định 0.0 - KHUYẾN NGHỊ GIỮ NGUYÊN: số hạng mu*(Q11+Q22) chỉ
    # thưởng độ cứng, không tạo áp lực lên Q12 - thực nghiệm cho thấy mu > 0
    # không cải thiện (thậm chí Q12 dương hơn). Cần thiết kế lại trước khi bật.
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
    # x_tilde: trường ĐÃ QUA FILTER nhưng CHƯA QUA PROJECTION - khi
    # projection TẮT, x_tilde luôn == xPhys (tương thích ngược hoàn toàn).
    # Ở vòng lặp 1, seed chưa qua filter/projection nào (giống hành vi cũ).
    x_tilde = xPhys.copy()
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
        'move': [move],
        'penal': [penal if penal_init is None else penal_init],
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
        penal_t = penal_at_iteration(penal, penal_init, loop, max_iter,
                                      penal_step_every=penal_step_every)
        # beta_proj_t: continuation tuyến tính beta_proj_init -> beta_proj
        # (tái dùng đúng công thức penal_at_iteration, chỉ khác tên tham số).
        beta_proj_t = (penal_at_iteration(beta_proj, beta_proj_init, loop, max_iter)
                       if projection == 'heaviside' else beta_proj)

        try:
            # solve_fe trả về U là fluctuation chi (K@chi=-K@U0), KHÔNG phải
            # tổng chuyển vị (xem docstring module ở trên).
            U, U0 = solve_fe(xPhys, material.KE, iK, jK, pbc, penal_t, E0, Emin, rho0=rho0)

            # Đồng nhất hóa (Andreassen 2014 eq.6) cần TỔNG chuyển vị u=u0+chi.
            U_total = U0 + U
            Q, dQ, _ = compute_homogenized_tensor(
                U_total, U0, xPhys, material.KE, edofMat, penal_t, E0, Emin, rho0=rho0,
            )

            # Hàm mục tiêu
            if objective_variant == 'normalized':
                c, dc = compute_auxetic_normalized_objective(Q, dQ, volfrac, E0, beta=beta)
            else:
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

        # dc hiện là d(obj)/d(xPhys) = d(obj)/d(x̂) (xPhys vừa dùng ở FE/
        # homogenization/objective phía trên). Nếu có projection, lan truyền
        # ngược qua dx̂/dx̃ TRƯỚC KHI đưa vào apply_sensitivity_filter() (hàm
        # đó xử lý bước dx̃/dx, giữ nguyên không đổi).
        if projection == 'heaviside':
            dc = dc * heaviside_projection_derivative(x_tilde, beta_proj_t, eta_proj)

        # Lọc độ nhạy
        dv = np.ones((nely, nelx))
        dc = apply_sensitivity_filter(dc, x, H, Hs, ft)
        if ft == 2:
            dv = apply_filter(dv, H, Hs)

        # OC
        move_t = move_at_iteration(move, move_min, loop, max_iter)
        delta_t = stiffness_delta(volfrac, E0)
        xnew, x_tilde = oc_update(x, dc, dv, volfrac, move_t, H, Hs, ft, use_sqrt=use_sqrt,
                                   Q=Q, delta=delta_t,
                                   projection=projection, beta_proj=beta_proj_t, eta_proj=eta_proj)
        change = np.max(np.abs(xnew - x))
        x = xnew
        xPhys = apply_heaviside_projection(x_tilde, beta_proj_t, eta_proj) if projection == 'heaviside' else x_tilde

        if enforce_connectivity and loop % connectivity_every == 0:
            x = nudge_disconnected_islands(
                x, xPhys, min_feature_px=connectivity_min_px,
                min_relative_size=connectivity_min_relative_size, floor=connectivity_floor,
            )

        history['iteration'].append(loop)
        history['v12'].append(v12)
        history['v21'].append(v21)
        history['objective'].append(c)
        history['volume'].append(float(np.mean(xPhys)))
        history['move'].append(move_t)
        history['penal'].append(penal_t)

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
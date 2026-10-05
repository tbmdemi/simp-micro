"""
Định nghĩa không gian tham số cho từng mục tiêu tối ưu hóa.
"""

from typing import Dict, Tuple, List

# ──────────────────────────────────────────────
#  Định nghĩa khoảng tham số (min, max)
# ──────────────────────────────────────────────
PARAM_SPACE: Dict[str, Tuple[float, float]] = {
    'volfrac':       (0.45, 0.70),   # thu hẹp, tập trung vào vùng cao
    'penal':         (2.0, 5.0),     # giữ rộng để khảo sát
    'rmin':          (1.0, 2.5),     # thu hẹp, tránh rmin cao
    'move':          (0.05, 0.25),   # tinh chỉnh nhẹ
    'void_size_frac': (0.25, 0.55),  # mở rộng lên cao hơn
    # Giai đoạn A (vary vật liệu nền, docs/PROJECT_PLAN.md Nhóm 1 A1/A3):
    # dải hẹp 0.2-0.4 để kiểm tra hội tụ FE trước khi mở rộng - nu=0.5 làm
    # (1-nu**2)=0 trong ma trận D (Material._compute_element_stiffness),
    # nu gần -1 cũng phân kỳ, nên KHÔNG mở dải sát biên vật lý (-1, 0.5)
    # (validate ở Material.__init__) khi chưa xác nhận FE hội tụ ổn.
    'nu':            (0.2, 0.4),
    # rotation_deg đã được cố định trong FIXED_PARAMS
}

# Tham số cố định (không đổi trong Phase 1 screening).
# LƯU Ý: chỉ dùng bởi pipeline/phase1_screening/screening_parallel.py. pipeline/phase2_multi_batch/runner.py
# (Phase 2 - nơi sinh 7,920 mẫu dùng cho Phase 3/4/5) có DEFAULT_FIXED riêng,
# hardcode nelx=nely=50 độc lập với file này - đổi nelx/nely ở đây KHÔNG ảnh
# hưởng độ phân giải dataset thật (đã xác nhận outputs/phase3 ở lưới 50x50,
# xem pipeline/phase5_cvae/verify_fe.py). TƯƠNG TỰ cho 'nu': đã chuyển lên
# PARAM_SPACE ở đây (Phase 1 screening), nhưng KHÔNG tự động lan xuống dataset
# thật - phải propagate riêng ở pipeline/phase2_multi_batch/params.py
# (ACTIVE_PARAMETERS) VÀ analysis/scripts/generate_production_batch.py (script
# thực sự đã sinh outputs/phase3/*.npz, độc lập với phase2_multi_batch/, xem
# docs/PIPELINE.md mục "2026-07-25 - Rebuild v2").
FIXED_PARAMS = {
    'nelx': 80,           # tăng độ phân giải (chỉ áp dụng cho Phase 1 screening)
    'nely': 80,
    'ft': 2,
    'E0': 199.0,
    'Emin': 1e-9,
    'max_iter': 200,      # tăng lên 200 để hội tụ tốt hơn với lưới mịn
    'tol_change': 0.01,
    'tol_obj': 0.05,
    'window_size': 20,
    'save_every': 9999,   # không lưu ảnh trung gian
    'scale_factor': 1,
    'mu': 0.0,
    'beta': 3.0,          
    'rotation_deg': 0.0,  
}


def get_active_params(objective: str = 'auxetic') -> List[str]:
    """Trả về danh sách tham số được vary.

    Args:
        objective: (unused, kept for backward compatibility)

    Returns:
        Danh sách tên tham số.
    """
    return list(PARAM_SPACE.keys())


def get_param_bounds(objective: str) -> List[Tuple[float, float]]:
    """Trả về danh sách (min, max) cho các tham số active."""
    return [PARAM_SPACE[p] for p in get_active_params(objective)]


SEEDS: List[str] = [
    'circle',
    'square',
    'hourglass',
    'four_circle',
    'hexagonal',
    'nine_circle',
    'cross_rectangular',
    'grid_circular_voids',
    'small_square_cross',
    'circle_half_quarter',
    'reentrant_bowtie',
]

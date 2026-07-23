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
    # rotation_deg đã được cố định trong FIXED_PARAMS
}

# Tham số cố định (không đổi trong Phase 1 screening).
# LƯU Ý: chỉ dùng bởi phase1_screening_parallel.py. pipeline/multi_batch/runner.py
# (Phase 2 - nơi sinh 7,920 mẫu dùng cho Phase 3/4/5) có DEFAULT_FIXED riêng,
# hardcode nelx=nely=50 độc lập với file này - đổi nelx/nely ở đây KHÔNG ảnh
# hưởng độ phân giải dataset thật (đã xác nhận outputs/phase3 ở lưới 50x50,
# xem pipeline/phase5_cvae/verify_fe.py).
FIXED_PARAMS = {
    'nelx': 80,           # tăng độ phân giải (chỉ áp dụng cho Phase 1 screening)
    'nely': 80,
    'ft': 2,
    'E0': 199.0,
    'Emin': 1e-9,
    'nu': 0.3,
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

"""
Định nghĩa không gian tham số cho từng mục tiêu tối ưu hóa.
"""

from typing import Dict, Tuple, List

# ──────────────────────────────────────────────
#  Định nghĩa khoảng tham số (min, max)
# ──────────────────────────────────────────────
PARAM_SPACE: Dict[str, Tuple[float, float]] = {
    'volfrac':       (0.25, 0.65),
    'penal':         (1.0, 5.0),
    'rmin':          (1.0, 6.0),
    'move':          (0.05, 0.3),
    'void_size_frac': (0.15, 0.45),
    'rotation_deg':   (0.0, 90.0),
    'beta':          (0.3, 1.5),     # chỉ dùng cho 'first'
    'beta_second':   (0.5, 2.5),     # chỉ dùng cho 'second'
}

# Tham số cố định (không thay đổi trong screening)
FIXED_PARAMS = {
    'nelx': 50,
    'nely': 50,
    'ft': 2,
    'E0': 199.0,
    'Emin': 1e-9,
    'nu': 0.3,
    'max_iter': 150,
    'tol_change': 0.01,
    'tol_obj': 0.05,
    'window_size': 20,
    'save_every': 9999,   # không lưu ảnh trung gian (chỉ lưu vòng đầu & cuối)
    'scale_factor': 1,
}


def get_active_params(objective: str) -> List[str]:
    """Trả về danh sách tham số được vary cho từng mục tiêu.

    Args:
        objective: 'auxetic', 'first', hoặc 'second'.

    Returns:
        Danh sách tên tham số.
    """
    base = ['volfrac', 'penal', 'rmin', 'move', 'void_size_frac', 'rotation_deg']
    if objective == 'first':
        return base + ['beta']
    elif objective == 'second':
        return base + ['beta_second']
    else:  # auxetic
        return base


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
]

OBJECTIVES: List[str] = ['auxetic', 'first', 'second']
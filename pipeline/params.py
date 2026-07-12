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


def get_active_params(objective: str = 'auxetic') -> List[str]:
    """Trả về danh sách tham số được vary.

    Args:
        objective: (unused, kept for backward compatibility)

    Returns:
        Danh sách tên tham số.
    """
    return ['volfrac', 'penal', 'rmin', 'move', 'void_size_frac', 'rotation_deg']


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

OBJECTIVES: List[str] = ['auxetic']

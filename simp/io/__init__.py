"""
Gói io — các công cụ nhập/xuất cho tối ưu hóa hình dạng SIMP.

Bao gồm:
    - save_csv:     Ghi dữ liệu vòng lặp ra file CSV.
    - save_density_image: Lưu ảnh trường mật độ dùng matplotlib.
"""

from .logger import save_csv

__all__ = ['save_csv', 'save_density_image']


def save_density_image(*args, **kwargs):
    """Lazy-load visualizer to avoid importing matplotlib at package import time."""
    from .visualizer import save_density_image as _save_density_image
    return _save_density_image(*args, **kwargs)
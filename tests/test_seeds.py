"""
Tests for simp/seeds/*.py - initial density field generators.

Bug đã sửa 2026-08-15: tất cả 11 seed dùng `i - cx` (element index thô) làm
tọa độ thay vì `(i + 0.5) - cx` (tâm THẬT của phần tử i, theo quy ước
build_dof_mesh - phần tử i chiếm [i, i+1] trong hệ tọa độ nút) - lệch nửa
pixel phá đối xứng gương dự định của mọi seed đối xứng qua tâm. CHỈ áp dụng
cho code sinh dữ liệu MỚI, KHÔNG rebuild dataset 57k mẫu hiện có (xem
docs/LIMITATIONS.md mục 20/21). Test này bảo vệ chống regression về lại bug
cũ, không phải kiểm tra hình học chi tiết của từng seed.
"""
import numpy as np
import pytest

from simp.seeds.circle import circle_seed
from simp.seeds.circle_half_quarter import circle_half_quarter_seed
from simp.seeds.cross_rectangular import cross_rectangular_seed
from simp.seeds.four_circle import four_circle_seed
from simp.seeds.grid_circular_voids import grid_circular_voids_seed
from simp.seeds.hexagonal import hexagonal_seed
from simp.seeds.hourglass import hourglass_seed
from simp.seeds.nine_circle import nine_circle_seed
from simp.seeds.reentrant_bowtie import reentrant_bowtie_seed
from simp.seeds.small_square_cross import small_square_cross_seed
from simp.seeds.square import square_seed

# (seed_fn, tham số thứ 3) - hourglass/reentrant_bowtie nhận volfrac, 9 seed
# còn lại nhận void_size_frac (xem simp/runner.py::_VOLFRAC_SEEDS).
SEED_FNS = [
    (circle_seed, 0.5),
    (circle_half_quarter_seed, 0.5),
    (cross_rectangular_seed, 0.4),
    (four_circle_seed, 0.3),
    (grid_circular_voids_seed, 0.3),
    (hexagonal_seed, 0.4),
    (hourglass_seed, 0.5),
    (nine_circle_seed, 0.3),
    (reentrant_bowtie_seed, 0.5),
    (small_square_cross_seed, 0.4),
    (square_seed, 0.4),
]


@pytest.mark.parametrize("seed_fn,arg", SEED_FNS, ids=[fn.__name__ for fn, _ in SEED_FNS])
class TestSeedSymmetryAtZeroRotation:
    """rotation_deg=0.0, kích thước chẵn (40x40) -> mọi seed phải đối xứng
    gương trái-phải VÀ trên-dưới qua tâm hình học (nelx/2, nely/2)."""

    def test_left_right_symmetric(self, seed_fn, arg):
        x = seed_fn(40, 40, arg, 0.0)
        assert np.array_equal(x, x[:, ::-1]), (
            f"{seed_fn.__name__} không đối xứng trái-phải ở rotation_deg=0 - "
            f"{np.sum(x != x[:, ::-1])} pixel lệch (bug half-pixel offset?)"
        )

    def test_up_down_symmetric(self, seed_fn, arg):
        x = seed_fn(40, 40, arg, 0.0)
        assert np.array_equal(x, x[::-1, :]), (
            f"{seed_fn.__name__} không đối xứng trên-dưới ở rotation_deg=0 - "
            f"{np.sum(x != x[::-1, :])} pixel lệch (bug half-pixel offset?)"
        )

    def test_returns_correct_shape_and_range(self, seed_fn, arg):
        x = seed_fn(40, 40, arg, 0.0)
        assert x.shape == (40, 40)
        assert x.min() >= 0.0 and x.max() <= 1.0

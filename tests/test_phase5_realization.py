"""Tests cho pipeline/phase5_cvae/realization.py (plan.md mục N1).

Hiện thực hóa phải là phép dịch hình học thật: dịch nguyên phần tử khớp
torch.roll, không dịch + không làm mượt + cùng lưới là đồng nhất. Lệch ở
đây nghĩa là objective N1 phạt một thứ khác "đặt lưới khác chỗ".
"""

import numpy as np
import torch

from pipeline.phase5_cvae.realization import filtered_design, realize_shifted


def _binary(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return torch.tensor((rng.random((1, 1, n, n)) > 0.5), dtype=torch.float64)


def test_zero_shift_same_grid_is_identity_on_binary():
    x = _binary()
    out = realize_shifted(x, 1e4, 12, 12, [(0.0, 0.0)], sigma=0.0)
    assert torch.equal((out > 0.5).double(), x)


def test_integer_shift_equals_periodic_roll():
    """Dịch 1 phần tử = lấy mẫu tại tâm phần tử kế tiếp (tuần hoàn)."""
    x = _binary(seed=1)
    out = realize_shifted(x, 1e4, 12, 12, [(1.0, 2.0)], sigma=0.0)
    ref = torch.roll(x, shifts=(-1, -2), dims=(-2, -1))
    assert torch.equal((out > 0.5).double(), ref)


def test_output_shape_and_gradient():
    x = torch.rand(1, 1, 16, 16, dtype=torch.float64, requires_grad=True)
    out = realize_shifted(x, 4.0, 10, 10, [(0.0, 0.5), (0.25, 0.75)])
    assert out.shape == (2, 1, 10, 10)
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_half_pixel_shift_flips_corner_contact():
    """Bản lề 1 nút (2 pixel chéo) nhạy với dịch nửa phần tử - đúng thứ N1
    muốn phạt; khối dày 4×4 thì không đổi diện tích đáng kể."""
    hinge = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    hinge[..., 3, 3] = 1
    hinge[..., 4, 4] = 1
    a = realize_shifted(hinge, 1e4, 8, 8, [(0.0, 0.0), (0.5, 0.5)], 0.5)
    assert not torch.equal(a[0] > 0.5, a[1] > 0.5)


def test_eta_dilates_and_erodes():
    """eta < 0,5 cho bản giãn (nhiều vật liệu hơn), > 0,5 cho bản co - đúng
    quy ước robust formulation."""
    x = torch.zeros(1, 1, 16, 16, dtype=torch.float64)
    x[..., 4:12, 4:12] = 1
    kw = dict(beta=1e4, nely=16, nelx=16, shifts=[(0.0, 0.0)], sigma=1.0)
    dil = (realize_shifted(x, eta=0.25, **kw) > 0.5).sum()
    nom = (realize_shifted(x, eta=0.5, **kw) > 0.5).sum()
    ero = (realize_shifted(x, eta=0.75, **kw) > 0.5).sum()
    assert dil > nom > ero


def test_filtered_design_hard_threshold_matches_large_beta():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 16, 16, dtype=torch.float64)
    hard = filtered_design(x, None, 10, 10, 1.0)
    soft = filtered_design(x, 1e6, 10, 10, 1.0)
    assert hard.shape == (1, 1, 10, 10)
    assert torch.equal(hard, (soft > 0.5).double())


def test_filtered_design_eta_order_and_gradient():
    x = torch.zeros(1, 1, 16, 16, dtype=torch.float64)
    x[..., 4:12, 4:12] = 1
    dil = filtered_design(x, None, 16, 16, 1.0, 0.25).sum()
    nom = filtered_design(x, None, 16, 16, 1.0, 0.5).sum()
    ero = filtered_design(x, None, 16, 16, 1.0, 0.75).sum()
    assert dil > nom > ero
    y = torch.rand(1, 1, 16, 16, dtype=torch.float64, requires_grad=True)
    filtered_design(y, 8.0, 10, 10, 1.0).sum().backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0

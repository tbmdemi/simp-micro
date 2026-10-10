"""Tests cho pipeline/phase5_cvae/manuf_penalty.py (plan.md mục K1).

Phạt khả vi phải khớp ĐÚNG thứ verify đo trên ảnh nhị phân: chạm góc khớp
manufacturability.count_corner_contacts, nét mảnh bằng 0 đúng khi
check_connectivity không đánh trượt vì min_feature. Lệch ở đây nghĩa là
refine phạt một thứ khác thứ được kiểm.
"""

import numpy as np
import pytest
import torch

from pipeline.phase5_cvae.manuf_penalty import (
    corner_contact_penalty,
    thin_feature_penalty,
)
from pipeline.phase5_cvae.manufacturability import (
    check_connectivity,
    count_corner_contacts,
)


def _diag_pair():
    """2 khối 3×3 chỉ chạm nhau tại 1 góc (bản lề 1 nút)."""
    img = np.zeros((10, 10), dtype=np.float32)
    img[1:4, 1:4] = 1
    img[4:7, 4:7] = 1
    return img


def test_corner_penalty_matches_numpy_count_on_binary():
    rng = np.random.default_rng(0)
    for _ in range(5):
        img = (rng.random((16, 16)) > 0.5).astype(np.float32)
        got = corner_contact_penalty(torch.tensor(img)).item()
        assert got == pytest.approx(count_corner_contacts(img) / img.size)


def test_corner_penalty_counts_contact_across_periodic_seam():
    """Chạm góc qua đường nối giữa các ô lát cũng là bản lề thật."""
    img = np.zeros((6, 6), dtype=np.float32)
    img[5, 5] = 1
    img[0, 0] = 1
    assert count_corner_contacts(img) == 1
    assert corner_contact_penalty(torch.tensor(img)).item() > 0


def test_connectivity_4_splits_corner_contact_but_8_does_not():
    img = _diag_pair()
    assert check_connectivity(img, connectivity=8)["n_components"] == 1
    assert check_connectivity(img, connectivity=4)["n_components"] == 2
    assert not check_connectivity(img, connectivity=4)["manufacturable"]


def test_connectivity_rejects_invalid_value():
    with pytest.raises(ValueError):
        check_connectivity(_diag_pair(), connectivity=6)


def test_thin_penalty_only_corners_of_thick_block_but_whole_thin_strut():
    """Opening chữ thập chỉ gọt 4 góc lồi của khối dày (phạt nhẹ, đẩy về bo
    góc) nhưng xóa trọn nét rộng 1 px."""
    thick = np.zeros((12, 12), dtype=np.float32)
    thick[2:9, 2:9] = 1
    base = thin_feature_penalty(torch.tensor(thick)).item()
    assert base == pytest.approx(4 / thick.size)
    thin = thick.copy()
    thin[5, 9:12] = 1  # nét rộng 1 px, dài 3 px gắn vào khối
    added = thin_feature_penalty(torch.tensor(thin)).item() - base
    # Pixel chân nét (sát khối) được dilation từ khối giữ lại -> mất 2/3.
    assert added == pytest.approx(2 / thick.size)


def test_thin_penalty_flags_component_that_check_rejects():
    """Mảnh rộng 2 px biến mất sau erosion -> check đánh trượt, phạt > 0."""
    img = np.zeros((12, 12), dtype=np.float32)
    img[3:5, 1:11] = 1
    assert not check_connectivity(img)["min_feature_ok"]
    assert thin_feature_penalty(torch.tensor(img)).item() > 0


def test_penalties_have_gradient():
    x = torch.full((1, 1, 8, 8), 0.5, requires_grad=True)
    (corner_contact_penalty(x) + thin_feature_penalty(x)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

"""Tests cho pipeline/phase5_cvae/heaviside.py - các phép hậu xử lý bản torch
khả vi phải khớp ĐÚNG bản numpy/PIL mà bước verify dùng (plan.md v3 P1.1e);
lệch ở đây nghĩa là refine lại tối ưu một thứ khác thứ được kiểm chứng."""

import numpy as np
import pytest
import torch

from pipeline.phase5_cvae.heaviside import (
    force_periodic_torch,
    heaviside_projection_torch,
    resize_nearest_torch,
)


@pytest.mark.parametrize("beta", [0.5, 4.0, 64.0])
def test_heaviside_matches_simp_core_numpy(beta):
    from simp.core.filter import apply_heaviside_projection

    x = np.random.default_rng(0).random((8, 8))
    got = heaviside_projection_torch(torch.tensor(x), beta).numpy()
    np.testing.assert_allclose(
        got, apply_heaviside_projection(x, beta), rtol=1e-12, atol=1e-12
    )


def test_heaviside_large_beta_approaches_threshold_step():
    x = torch.tensor([0.1, 0.45, 0.55, 0.9])
    got = heaviside_projection_torch(x, 500.0)
    np.testing.assert_allclose(got.numpy(), [0, 0, 1, 1], atol=1e-6)


def test_heaviside_has_gradient():
    x = torch.full((3,), 0.5, requires_grad=True)
    heaviside_projection_torch(x, 8.0).sum().backward()
    assert torch.all(x.grad > 0)


def test_force_periodic_matches_numpy():
    from pipeline.phase5_cvae.manufacturability import force_periodic

    img = np.random.default_rng(1).random((64, 64)).astype(np.float32)
    got = force_periodic_torch(torch.tensor(img)[None, None])[0, 0].numpy()
    np.testing.assert_allclose(got, force_periodic(img), rtol=1e-6)


def test_resize_nearest_matches_verify_pil_resize():
    """Ảnh nhị phân 64x64 -> lưới FE 50x50 phải trùng pixel-by-pixel với
    verify_fe.resize_to_fe_grid (PIL NEAREST)."""
    from pipeline.phase5_cvae.verify_fe import resize_to_fe_grid

    img = (np.random.default_rng(2).random((64, 64)) > 0.5).astype(np.float32)
    got = resize_nearest_torch(torch.tensor(img)[None, None], 50, 50)
    np.testing.assert_array_equal(
        got[0, 0].numpy(), resize_to_fe_grid(img, 50, 50)
    )


@pytest.mark.parametrize("shape", [(64, 50), (50, 64), (300, 50)])
def test_resize_index_matches_pil_for_other_sizes(shape):
    """Bảng chỉ số lấy từ PIL phải đúng cả khi phóng to và khi nguồn > 256
    (vượt giới hạn ảnh 8-bit)."""
    from PIL import Image

    src, dst = shape
    img = np.random.default_rng(3).integers(0, 255, (1, src)).astype(np.uint8)
    ref = np.asarray(Image.fromarray(img).resize((dst, 1), Image.NEAREST))
    got = resize_nearest_torch(torch.tensor(img.astype(np.float32)), 1, dst)
    np.testing.assert_array_equal(got.numpy(), ref.astype(np.float32))


def test_project_for_fe_matches_verify_pipeline_at_large_beta():
    """β lớn: project_for_fe(ảnh) phải trùng force_periodic -> nhị phân hóa
    -> resize PIL mà verify dùng (trừ pixel sát ngưỡng 0,5)."""
    from pipeline.phase5_cvae.heaviside import project_for_fe
    from pipeline.phase5_cvae.manufacturability import force_periodic
    from pipeline.phase5_cvae.verify_fe import resize_to_fe_grid

    rng = np.random.default_rng(4)
    img = rng.random((64, 64)).astype(np.float32)
    img[np.abs(img - 0.5) < 0.02] = 0.9  # bỏ pixel sát ngưỡng
    ref = resize_to_fe_grid(
        (force_periodic(img) > 0.5).astype(np.float32), 50, 50
    )
    got = project_for_fe(
        torch.tensor(img)[None, None], 1e4, 50, 50, periodic=True
    )
    np.testing.assert_allclose(got[0, 0].numpy(), ref, atol=1e-4)


def test_prior_loss_uses_projection_when_beta_given(monkeypatch):
    """real_physics_prior_loss(projection_beta=...) phải đưa ảnh đã chiếu
    đúng lưới FE vào real_physics_loss."""
    import pipeline.phase5_cvae.losses as losses

    seen = {}

    def fake_rpl(density, condition, fe_params, **kw):
        seen["shape"] = tuple(density.shape)
        return density.mean()

    monkeypatch.setattr(losses, "real_physics_loss", fake_rpl)
    dec = (
        lambda z, c: torch.sigmoid(z[:, :1])
        .view(-1, 1, 1, 1)
        .expand(-1, 1, 8, 8)
    )
    fe = dict(nelx=6, nely=5)
    losses.real_physics_prior_loss(
        dec, 3, torch.zeros(2, 2), fe, projection_beta=4.0, periodic=True
    )
    assert seen["shape"] == (2, 1, 5, 6)
    losses.real_physics_prior_loss(dec, 3, torch.zeros(2, 2), fe)
    assert seen["shape"] == (2, 1, 8, 8)

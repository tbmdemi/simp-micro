"""Kiểm chứng real_physics.py: đạo hàm GIẢI TÍCH (không autodiff số) của
nu12/nu21 theo pixel mật độ, xem docstring pipeline/phase5_cvae/real_physics.py.

TestGradientCorrectness là bài test QUAN TRỌNG NHẤT của cả module: so sánh
d_v12/d_v21 giải tích với finite-difference (central difference) trên nhiều
pixel/nhiều density field ngẫu nhiên - nếu công thức quy tắc chuỗi (đạo hàm
nghịch đảo ma trận + quy tắc thương) sai dấu hoặc sai hệ số, test này PHẢI
fail. Không mock, không xấp xỉ - dùng đúng solve_fe/compute_homogenized_tensor
thật.
"""
import numpy as np
import pytest
import torch

from pipeline.phase5_cvae.real_physics import (
    solve_nu_with_grad, RealPhysicsNu, _get_mesh,
)


FE_KW = dict(penal=3.0, E0=199.0, Emin=1e-9, nu=0.3, rho0=1.0)


def _make_density(nely, nelx, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.15, 0.85, size=(nely, nelx)).astype(np.float64)


class TestGradientCorrectness:
    """So sánh đạo hàm giải tích vs finite-difference trung tâm."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_v12_gradient_matches_finite_difference(self, seed):
        nely, nelx = 8, 8
        xPhys = _make_density(nely, nelx, seed)
        v12_0, v21_0, d_v12, d_v21 = solve_nu_with_grad(xPhys, **FE_KW)

        eps = 1e-4
        rng = np.random.default_rng(seed + 100)
        pixels = [(int(i), int(j)) for i, j in
                  zip(rng.integers(0, nely, 5), rng.integers(0, nelx, 5))]

        for (i, j) in pixels:
            xp = xPhys.copy(); xp[i, j] += eps
            xm = xPhys.copy(); xm[i, j] -= eps
            v12_p, _, _, _ = solve_nu_with_grad(xp, **FE_KW)
            v12_m, _, _, _ = solve_nu_with_grad(xm, **FE_KW)
            numeric = (v12_p - v12_m) / (2 * eps)
            analytic = d_v12[i, j]
            # So sánh tuyệt đối (giá trị đạo hàm thường O(0.1-10)) + tương đối
            assert numeric == pytest.approx(analytic, abs=2e-3, rel=1e-2), (
                f"pixel=({i},{j}) numeric={numeric:.6f} analytic={analytic:.6f}"
            )

    @pytest.mark.parametrize("seed", [0, 1])
    def test_v21_gradient_matches_finite_difference(self, seed):
        nely, nelx = 8, 8
        xPhys = _make_density(nely, nelx, seed)
        v12_0, v21_0, d_v12, d_v21 = solve_nu_with_grad(xPhys, **FE_KW)

        eps = 1e-4
        rng = np.random.default_rng(seed + 200)
        pixels = [(int(i), int(j)) for i, j in
                  zip(rng.integers(0, nely, 4), rng.integers(0, nelx, 4))]

        for (i, j) in pixels:
            xp = xPhys.copy(); xp[i, j] += eps
            xm = xPhys.copy(); xm[i, j] -= eps
            _, v21_p, _, _ = solve_nu_with_grad(xp, **FE_KW)
            _, v21_m, _, _ = solve_nu_with_grad(xm, **FE_KW)
            numeric = (v21_p - v21_m) / (2 * eps)
            analytic = d_v21[i, j]
            assert numeric == pytest.approx(analytic, abs=2e-3, rel=1e-2), (
                f"pixel=({i},{j}) numeric={numeric:.6f} analytic={analytic:.6f}"
            )

    def test_gradient_matches_saved_v12_value(self):
        """v12 trả về từ solve_nu_with_grad phải khớp compute_nu12(Q) độc lập
        (regression đơn giản, không liên quan FD, nhưng bắt lỗi sign/API)."""
        from simp.objectives.auxetic import compute_nu12
        from simp.core.solver import solve_fe
        from simp.homogenization.compute import compute_homogenized_tensor

        xPhys = _make_density(6, 6, 42)
        material, edofMat, iK, jK, pbc = _get_mesh(6, 6, FE_KW["E0"], FE_KW["Emin"], FE_KW["nu"])
        U, U0 = solve_fe(xPhys, material.KE, iK, jK, pbc, FE_KW["penal"], FE_KW["E0"], FE_KW["Emin"], rho0=1.0)
        Q, dQ, _ = compute_homogenized_tensor(U0 + U, U0, xPhys, material.KE, edofMat, FE_KW["penal"], FE_KW["E0"], FE_KW["Emin"], rho0=1.0)
        expected_v12 = compute_nu12(Q)

        v12, v21, d_v12, d_v21 = solve_nu_with_grad(xPhys, **FE_KW)
        assert v12 == pytest.approx(expected_v12, abs=1e-9)


class TestSolveNuWithGradErrorHandling:
    """solve_nu_with_grad() KHÔNG được raise ra ngoài, kể cả khi FE-solve/
    nghịch đảo Q thất bại (density gần suy biến) - phải trả fallback an
    toàn (0.0, 0.0, gradient 0) thay vì crash cả batch training. Xem
    docstring hàm + VERIFICATION_GUIDE.md § Phần 2."""

    def test_returns_safe_fallback_when_solve_fe_raises(self, monkeypatch):
        import pipeline.phase5_cvae.real_physics as rp

        def _boom(*a, **kw):
            raise RuntimeError("giả lập solve_fe thất bại")

        monkeypatch.setattr(rp, "solve_fe", _boom)
        xPhys = np.full((6, 6), 0.5)
        v12, v21, d_v12, d_v21 = rp.solve_nu_with_grad(xPhys, **FE_KW)

        assert v12 == 0.0
        assert v21 == 0.0
        np.testing.assert_array_equal(d_v12, np.zeros((6, 6)))
        np.testing.assert_array_equal(d_v21, np.zeros((6, 6)))

    def test_returns_safe_fallback_when_q_is_singular(self, monkeypatch):
        """Mô phỏng Q suy biến (nghịch đảo thất bại) mà không cần density
        thật gây suy biến - patch compute_homogenized_tensor trả Q toàn 0."""
        import pipeline.phase5_cvae.real_physics as rp

        def _fake_homogenized(*a, **kw):
            return np.zeros((3, 3)), np.zeros((3, 3, 6, 6)), None

        monkeypatch.setattr(rp, "compute_homogenized_tensor", _fake_homogenized)
        xPhys = np.full((6, 6), 0.5)
        v12, v21, d_v12, d_v21 = rp.solve_nu_with_grad(xPhys, **FE_KW)

        assert v12 == 0.0
        assert v21 == 0.0
        assert np.all(d_v12 == 0.0)
        assert np.all(d_v21 == 0.0)

    def test_normal_case_still_works_after_error_handling_added(self):
        """Regression guard: try/except mới thêm không được nuốt nhầm
        trường hợp thành công bình thường."""
        xPhys = _make_density(6, 6, 0)
        v12, v21, d_v12, d_v21 = solve_nu_with_grad(xPhys, **FE_KW)
        assert np.isfinite(v12) and np.isfinite(v21)
        assert not np.all(d_v12 == 0.0)


class TestRealPhysicsNuAutogradFunction:
    """torch.autograd.Function wrapper - kiểm tra forward value + backward
    gradient khớp solve_nu_with_grad() trực tiếp (không lặp lại FD, vì đã
    kiểm ở TestGradientCorrectness - ở đây chỉ kiểm 'ống dẫn' torch đúng)."""

    def test_forward_matches_direct_call(self):
        nely, nelx = 6, 6
        xPhys = _make_density(nely, nelx, 7)
        v12_direct, v21_direct, _, _ = solve_nu_with_grad(xPhys, **FE_KW)

        density_t = torch.tensor(xPhys, dtype=torch.float32).unsqueeze(0)  # (1, nely, nelx)
        out = RealPhysicsNu.apply(density_t, FE_KW["penal"], FE_KW["E0"], FE_KW["Emin"], FE_KW["nu"], FE_KW["rho0"])
        assert out.shape == (1, 2)
        assert out[0, 0].item() == pytest.approx(v12_direct, abs=1e-4)
        assert out[0, 1].item() == pytest.approx(v21_direct, abs=1e-4)

    def test_backward_gradient_matches_direct_call(self):
        nely, nelx = 6, 6
        xPhys = _make_density(nely, nelx, 9)
        _, _, d_v12_direct, d_v21_direct = solve_nu_with_grad(xPhys, **FE_KW)

        density_t = torch.tensor(xPhys[None], dtype=torch.float32, requires_grad=True)
        out = RealPhysicsNu.apply(density_t, FE_KW["penal"], FE_KW["E0"], FE_KW["Emin"], FE_KW["nu"], FE_KW["rho0"])
        # loss = v12 (grad_output = [1, 0]) -> gradient phải khớp d_v12
        loss = out[0, 0]
        loss.backward()
        np.testing.assert_allclose(
            density_t.grad[0].numpy(), d_v12_direct, atol=1e-4, rtol=1e-3
        )

    def test_backward_combined_v12_v21_gradient(self):
        nely, nelx = 6, 6
        xPhys = _make_density(nely, nelx, 11)
        _, _, d_v12_direct, d_v21_direct = solve_nu_with_grad(xPhys, **FE_KW)

        density_t = torch.tensor(xPhys[None], dtype=torch.float32, requires_grad=True)
        out = RealPhysicsNu.apply(density_t, FE_KW["penal"], FE_KW["E0"], FE_KW["Emin"], FE_KW["nu"], FE_KW["rho0"])
        loss = out[0, 0] + 2.0 * out[0, 1]
        loss.backward()
        expected = d_v12_direct + 2.0 * d_v21_direct
        np.testing.assert_allclose(density_t.grad[0].numpy(), expected, atol=1e-4, rtol=1e-3)

    def test_batch_processing_independent_samples(self):
        nely, nelx = 6, 6
        x0 = _make_density(nely, nelx, 1)
        x1 = _make_density(nely, nelx, 2)
        v12_0, v21_0, _, _ = solve_nu_with_grad(x0, **FE_KW)
        v12_1, v21_1, _, _ = solve_nu_with_grad(x1, **FE_KW)

        batch = torch.tensor(np.stack([x0, x1]), dtype=torch.float32)
        out = RealPhysicsNu.apply(batch, FE_KW["penal"], FE_KW["E0"], FE_KW["Emin"], FE_KW["nu"], FE_KW["rho0"])
        assert out[0, 0].item() == pytest.approx(v12_0, abs=1e-4)
        assert out[1, 0].item() == pytest.approx(v12_1, abs=1e-4)
        assert out[0, 1].item() == pytest.approx(v21_0, abs=1e-4)
        assert out[1, 1].item() == pytest.approx(v21_1, abs=1e-4)

    def test_gradient_flows_through_upstream_op(self):
        """Xác nhận gradient chảy được qua 1 phép toán torch phía trước (mô
        phỏng decoder), không chỉ tới thẳng input FE-solve."""
        nely, nelx = 6, 6
        raw = torch.tensor(_make_density(nely, nelx, 3)[None], dtype=torch.float32, requires_grad=True)
        density_t = torch.sigmoid(raw)  # phép toán trung gian có đạo hàm khác 1
        out = RealPhysicsNu.apply(density_t, FE_KW["penal"], FE_KW["E0"], FE_KW["Emin"], FE_KW["nu"], FE_KW["rho0"])
        loss = out[0, 0]
        loss.backward()
        assert raw.grad is not None
        assert torch.all(torch.isfinite(raw.grad))
        assert raw.grad.abs().sum().item() > 0

    def test_multiprocessing_matches_serial(self):
        """n_workers>0 phải cho đúng cùng kết quả với đường tuần tự
        (n_workers=0) - chỉ khác cách thực thi, không khác công thức."""
        nely, nelx = 6, 6
        x0 = _make_density(nely, nelx, 21)
        x1 = _make_density(nely, nelx, 22)
        batch = torch.tensor(np.stack([x0, x1]), dtype=torch.float32, requires_grad=True)

        out_serial = RealPhysicsNu.apply(
            batch, FE_KW["penal"], FE_KW["E0"], FE_KW["Emin"], FE_KW["nu"], FE_KW["rho0"], 0
        )
        loss_serial = out_serial[:, 0].sum()
        loss_serial.backward()
        grad_serial = batch.grad.clone()
        batch.grad = None

        out_parallel = RealPhysicsNu.apply(
            batch, FE_KW["penal"], FE_KW["E0"], FE_KW["Emin"], FE_KW["nu"], FE_KW["rho0"], 2
        )
        loss_parallel = out_parallel[:, 0].sum()
        loss_parallel.backward()
        grad_parallel = batch.grad.clone()

        np.testing.assert_allclose(out_serial.detach().numpy(), out_parallel.detach().numpy(), atol=1e-9)
        np.testing.assert_allclose(grad_serial.numpy(), grad_parallel.numpy(), atol=1e-9)

        from pipeline.phase5_cvae.real_physics import shutdown_pool
        shutdown_pool()

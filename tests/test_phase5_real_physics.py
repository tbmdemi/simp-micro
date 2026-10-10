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
    solve_nu_with_grad, RealPhysicsNu, _get_mesh, _get_mesh_topology,
    _MESH_TOPOLOGY_CACHE,
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

    @pytest.mark.filterwarnings(
        "ignore:This process \\(pid=.*\\) is multi-threaded, use of fork\\(\\) "
        "may lead to deadlocks in the child.:DeprecationWarning"
    )
    def test_multiprocessing_matches_serial(self):
        """n_workers>0 phải cho đúng cùng kết quả với đường tuần tự
        (n_workers=0) - chỉ khác cách thực thi, không khác công thức.

        Warning fork()-deadlock từ pytest+torch multi-threaded được filter
        có chủ đích: đây là cảnh báo lành tính trong test (pool tồn tại
        trong đúng 1 lệnh gọi ngắn), đổi sang 'spawn' toàn cục sẽ chậm hơn
        và cần mọi thứ picklable - không đáng đánh đổi cho 1 warning test.
        """
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


class TestPerSampleMaterial:
    """A5 (docs/archive/PROJECT_PLAN.md Nhóm 1): E0/nu giờ chấp nhận mảng
    per-sample
    thay vì chỉ scalar áp cho cả batch - kiểm tra giá trị/gradient khớp đúng
    với việc gọi solve_nu_with_grad() riêng lẻ từng sample với nu/E0 khác
    nhau, và mesh TOPOLOGY vẫn dùng chung (không cache lại theo vật liệu)."""

    def test_forward_matches_per_sample_direct_call_different_nu(self):
        nely, nelx = 6, 6
        x0 = _make_density(nely, nelx, 30)
        x1 = _make_density(nely, nelx, 31)
        nu0, nu1 = 0.2, 0.4
        v12_0, v21_0, _, _ = solve_nu_with_grad(x0, penal=3.0, E0=199.0, Emin=1e-9, nu=nu0, rho0=1.0)
        v12_1, v21_1, _, _ = solve_nu_with_grad(x1, penal=3.0, E0=199.0, Emin=1e-9, nu=nu1, rho0=1.0)

        batch = torch.tensor(np.stack([x0, x1]), dtype=torch.float32)
        out = RealPhysicsNu.apply(batch, 3.0, 199.0, 1e-9, [nu0, nu1], 1.0)
        assert out[0, 0].item() == pytest.approx(v12_0, abs=1e-4)
        assert out[1, 0].item() == pytest.approx(v12_1, abs=1e-4)
        assert out[0, 1].item() == pytest.approx(v21_0, abs=1e-4)
        assert out[1, 1].item() == pytest.approx(v21_1, abs=1e-4)

    def test_forward_matches_per_sample_direct_call_different_e0(self):
        nely, nelx = 6, 6
        x0 = _make_density(nely, nelx, 32)
        x1 = _make_density(nely, nelx, 33)
        E0_0, E0_1 = 150.0, 250.0
        v12_0, v21_0, _, _ = solve_nu_with_grad(x0, penal=3.0, E0=E0_0, Emin=1e-9, nu=0.3, rho0=1.0)
        v12_1, v21_1, _, _ = solve_nu_with_grad(x1, penal=3.0, E0=E0_1, Emin=1e-9, nu=0.3, rho0=1.0)

        batch = torch.tensor(np.stack([x0, x1]), dtype=torch.float32)
        out = RealPhysicsNu.apply(batch, 3.0, np.array([E0_0, E0_1]), 1e-9, 0.3, 1.0)
        assert out[0, 0].item() == pytest.approx(v12_0, abs=1e-4)
        assert out[1, 0].item() == pytest.approx(v12_1, abs=1e-4)
        assert out[0, 1].item() == pytest.approx(v21_0, abs=1e-4)
        assert out[1, 1].item() == pytest.approx(v21_1, abs=1e-4)

    def test_backward_gradient_correct_per_sample_nu(self):
        """Mỗi sample phải nhận đúng gradient của CHÍNH nu của nó, không bị
        lẫn/broadcast nhầm từ sample khác."""
        nely, nelx = 6, 6
        x0 = _make_density(nely, nelx, 34)
        x1 = _make_density(nely, nelx, 35)
        nu0, nu1 = 0.15, 0.35
        _, _, d_v12_0, _ = solve_nu_with_grad(x0, penal=3.0, E0=199.0, Emin=1e-9, nu=nu0, rho0=1.0)
        _, _, d_v12_1, _ = solve_nu_with_grad(x1, penal=3.0, E0=199.0, Emin=1e-9, nu=nu1, rho0=1.0)

        batch = torch.tensor(np.stack([x0, x1]), dtype=torch.float32, requires_grad=True)
        out = RealPhysicsNu.apply(batch, 3.0, 199.0, 1e-9, torch.tensor([nu0, nu1]), 1.0)
        loss = out[:, 0].sum()
        loss.backward()

        np.testing.assert_allclose(batch.grad[0].numpy(), d_v12_0, atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(batch.grad[1].numpy(), d_v12_1, atol=1e-4, rtol=1e-3)

    def test_scalar_nu_still_broadcasts_to_whole_batch(self):
        """Regression: nu scalar (đường cũ, tương thích ngược) vẫn phải áp
        dụng đồng nhất cho mọi sample trong batch."""
        nely, nelx = 6, 6
        x0 = _make_density(nely, nelx, 36)
        x1 = _make_density(nely, nelx, 37)
        v12_0, v21_0, _, _ = solve_nu_with_grad(x0, **FE_KW)
        v12_1, v21_1, _, _ = solve_nu_with_grad(x1, **FE_KW)

        batch = torch.tensor(np.stack([x0, x1]), dtype=torch.float32)
        out = RealPhysicsNu.apply(batch, FE_KW["penal"], FE_KW["E0"], FE_KW["Emin"], FE_KW["nu"], FE_KW["rho0"])
        assert out[0, 0].item() == pytest.approx(v12_0, abs=1e-4)
        assert out[1, 0].item() == pytest.approx(v12_1, abs=1e-4)
        assert out[0, 1].item() == pytest.approx(v21_0, abs=1e-4)
        assert out[1, 1].item() == pytest.approx(v21_1, abs=1e-4)

    @pytest.mark.filterwarnings(
        "ignore:This process \\(pid=.*\\) is multi-threaded, use of fork\\(\\) "
        "may lead to deadlocks in the child.:DeprecationWarning"
    )
    def test_multiprocessing_matches_serial_with_per_sample_nu(self):
        """n_workers>0 phải cho đúng kết quả như tuần tự KỂ CẢ khi mỗi
        sample có nu riêng (không chỉ trường hợp scalar đã test ở trên)."""
        nely, nelx = 6, 6
        x0 = _make_density(nely, nelx, 38)
        x1 = _make_density(nely, nelx, 39)
        batch = torch.tensor(np.stack([x0, x1]), dtype=torch.float32, requires_grad=True)
        nu_per_sample = [0.18, 0.42]

        out_serial = RealPhysicsNu.apply(batch, 3.0, 199.0, 1e-9, nu_per_sample, 1.0, 0)
        loss_serial = out_serial[:, 0].sum()
        loss_serial.backward()
        grad_serial = batch.grad.clone()
        batch.grad = None

        out_parallel = RealPhysicsNu.apply(batch, 3.0, 199.0, 1e-9, nu_per_sample, 1.0, 2)
        loss_parallel = out_parallel[:, 0].sum()
        loss_parallel.backward()
        grad_parallel = batch.grad.clone()

        np.testing.assert_allclose(out_serial.detach().numpy(), out_parallel.detach().numpy(), atol=1e-9)
        np.testing.assert_allclose(grad_serial.numpy(), grad_parallel.numpy(), atol=1e-9)

        from pipeline.phase5_cvae.real_physics import shutdown_pool
        shutdown_pool()

    def test_mesh_topology_cache_shared_across_different_nu(self):
        """_get_mesh với 2 giá trị nu khác nhau trên cùng (nelx,nely) không
        được tạo 2 entry topology riêng - phần đắt (edofMat/iK/jK/pbc) phải
        dùng chung, chỉ Material là dựng riêng (xem comment real_physics.py)."""
        nelx, nely = 9, 9
        _MESH_TOPOLOGY_CACHE.pop((nelx, nely), None)

        mat_a, edofMat_a, iK_a, jK_a, pbc_a = _get_mesh(nelx, nely, 199.0, 1e-9, 0.2)
        n_entries_after_first = len(_MESH_TOPOLOGY_CACHE)
        mat_b, edofMat_b, iK_b, jK_b, pbc_b = _get_mesh(nelx, nely, 199.0, 1e-9, 0.4)
        n_entries_after_second = len(_MESH_TOPOLOGY_CACHE)

        assert n_entries_after_second == n_entries_after_first
        assert edofMat_a is edofMat_b
        assert iK_a is iK_b
        assert jK_a is jK_b
        assert pbc_a is pbc_b
        # Material phải khác nhau thật (nu khác nhau -> KE khác nhau).
        assert mat_a.nu == pytest.approx(0.2)
        assert mat_b.nu == pytest.approx(0.4)
        assert not np.allclose(mat_a.KE, mat_b.KE)

    def test_get_mesh_topology_direct(self):
        """_get_mesh_topology() (hàm mới) trả đúng 4 thành phần, cache theo
        (nelx, nely) độc lập với vật liệu."""
        nelx, nely = 7, 7
        _MESH_TOPOLOGY_CACHE.pop((nelx, nely), None)
        edofMat, iK, jK, pbc = _get_mesh_topology(nelx, nely)
        assert edofMat.shape[0] == nelx * nely
        edofMat2, iK2, jK2, pbc2 = _get_mesh_topology(nelx, nely)
        assert edofMat2 is edofMat


class TestSolveElasticWithGrad:
    """solve_elastic_with_grad(): đạo hàm giải tích của E_x/E_y/G_xy/B_eff
    theo pixel (qua dS = -S dQ S) phải khớp finite-difference trung tâm,
    và giá trị phải khớp compute_elastic_constants() độc lập - cùng tinh
    thần TestGradientCorrectness cho nu12/nu21."""

    @pytest.mark.parametrize("key", ["E_x", "E_y", "G_xy", "B_eff"])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_gradient_matches_finite_difference(self, key, seed):
        from pipeline.phase5_cvae.real_physics import solve_elastic_with_grad

        nely, nelx = 8, 8
        xPhys = _make_density(nely, nelx, seed)
        _, grads = solve_elastic_with_grad(xPhys, **FE_KW)

        eps = 1e-4
        rng = np.random.default_rng(seed + 300)
        pixels = [(int(i), int(j)) for i, j in
                  zip(rng.integers(0, nely, 4), rng.integers(0, nelx, 4))]
        for (i, j) in pixels:
            xp = xPhys.copy(); xp[i, j] += eps
            xm = xPhys.copy(); xm[i, j] -= eps
            vp, _ = solve_elastic_with_grad(xp, **FE_KW)
            vm, _ = solve_elastic_with_grad(xm, **FE_KW)
            numeric = (vp[key] - vm[key]) / (2 * eps)
            analytic = grads[key][i, j]
            # Mô-đun có đơn vị E0 (~199) nên đạo hàm cỡ O(1-100): ngưỡng
            # tương đối là chính, tuyệt đối chỉ chặn pixel đạo hàm ~0.
            assert numeric == pytest.approx(analytic, abs=1e-2, rel=1e-2), (
                f"{key} pixel=({i},{j}) numeric={numeric:.6f} "
                f"analytic={analytic:.6f}"
            )

    def test_values_match_compute_elastic_constants(self):
        """B_eff dạng compliance 1/(S00+S11+2S01) phải trùng công thức
        Ex*Ey/[...] của compute_elastic_constants() - kiểm chứng tương
        đương đại số trên density bất đối xứng thật."""
        from pipeline.phase5_cvae.real_physics import solve_elastic_with_grad
        from simp.objectives.auxetic import compute_elastic_constants
        from simp.core.solver import solve_fe
        from simp.homogenization.compute import compute_homogenized_tensor

        xPhys = _make_density(8, 8, 7)
        values, _ = solve_elastic_with_grad(xPhys, **FE_KW)

        material, edofMat, iK, jK, pbc = _get_mesh(
            8, 8, FE_KW["E0"], FE_KW["Emin"], FE_KW["nu"]
        )
        U, U0 = solve_fe(xPhys, material.KE, iK, jK, pbc, FE_KW["penal"],
                         FE_KW["E0"], FE_KW["Emin"], rho0=1.0)
        Q, _, _ = compute_homogenized_tensor(
            U0 + U, U0, xPhys, material.KE, edofMat, FE_KW["penal"],
            FE_KW["E0"], FE_KW["Emin"], rho0=1.0,
        )
        expected = compute_elastic_constants(Q)
        assert values["E_x"] == pytest.approx(expected["E_x"], rel=1e-10)
        assert values["E_y"] == pytest.approx(expected["E_y"], rel=1e-10)
        assert values["G_xy"] == pytest.approx(expected["G_xy"], rel=1e-10)
        assert values["B_eff"] == pytest.approx(expected["B_eff"], rel=1e-10)
        assert values["v12"] == pytest.approx(expected["nu_12"], rel=1e-10)
        assert values["v21"] == pytest.approx(expected["nu_21"], rel=1e-10)

    def test_nu_slice_matches_solve_nu_with_grad(self):
        """solve_nu_with_grad() giờ là lát cắt của solve_elastic_with_grad()
        - giá trị và gradient phải trùng khít (regression cho refactor)."""
        from pipeline.phase5_cvae.real_physics import solve_elastic_with_grad

        xPhys = _make_density(6, 6, 3)
        values, grads = solve_elastic_with_grad(xPhys, **FE_KW)
        v12, v21, d_v12, d_v21 = solve_nu_with_grad(xPhys, **FE_KW)
        assert v12 == values["v12"] and v21 == values["v21"]
        np.testing.assert_array_equal(d_v12, grads["v12"])
        np.testing.assert_array_equal(d_v21, grads["v21"])

    def test_returns_safe_fallback_when_q_is_singular(self, monkeypatch):
        import pipeline.phase5_cvae.real_physics as rp

        def _fake_homogenized(*a, **kw):
            return np.zeros((3, 3)), np.zeros((3, 3, 6, 6)), None

        monkeypatch.setattr(
            rp, "compute_homogenized_tensor", _fake_homogenized
        )
        values, grads = rp.solve_elastic_with_grad(
            np.full((6, 6), 0.5), **FE_KW
        )
        assert set(values) == set(rp.ELASTIC_KEYS)
        assert all(v == 0.0 for v in values.values())
        assert all(np.all(g == 0.0) for g in grads.values())

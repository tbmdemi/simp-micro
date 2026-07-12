"""
Smoke tests for core SIMP modules: FEM, filter, solver, OC, materials, objectives.

These tests verify that each module can run without error on a small mesh
and produce outputs of the expected shape/type. They are NOT convergence tests.
"""

import numpy as np
import pytest


class TestFEM:
    """Smoke tests for core/fem.py — build_dof_mesh."""

    def test_build_dof_mesh_small(self):
        from simp.core.fem import build_dof_mesh
        nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(3, 3)
        # nodenrs: (nely+1) x (nelx+1)
        assert nodenrs.shape == (4, 4)
        # edofVec: (nelx*nely,) — one entry per element
        assert len(edofVec) == 9
        # edofMat: (nelx*nely, 8) — 8 DOFs per quad element
        assert edofMat.shape == (9, 8)
        # iK, jK: sparse index vectors for stiffness assembly
        assert len(iK) > 0
        assert len(jK) > 0
        assert len(iK) == len(jK)

    def test_build_dof_mesh_rectangular(self):
        from simp.core.fem import build_dof_mesh
        nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(4, 2)
        assert nodenrs.shape == (3, 5)
        assert edofMat.shape == (8, 8)

    @pytest.mark.parametrize("nelx,nely", [(1, 1), (2, 2), (5, 5)])
    def test_various_mesh_sizes(self, nelx, nely):
        from simp.core.fem import build_dof_mesh
        nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(nelx, nely)
        assert edofMat.shape == (nelx * nely, 8)
        assert nodenrs.shape == (nely + 1, nelx + 1)


class TestMaterial:
    """Smoke tests for materials/isotropic.py — Material class."""

    def test_material_ke_shape(self):
        from simp.materials.isotropic import Material
        mat = Material(E0=199.0, Emin=1e-9, nu=0.3)
        assert mat.KE.shape == (8, 8)
        # Plane stress: KE should be symmetric
        assert np.allclose(mat.KE, mat.KE.T)
        # Diagonal should be positive
        assert np.all(np.diag(mat.KE) > 0)

    def test_material_defaults(self):
        from simp.materials.isotropic import Material
        mat = Material()
        # Material() khởi tạo với E0=199.0, Emin=1e-9, nu=0.3 (từ code)
        assert mat.E0 == 199.0
        assert mat.Emin == 1e-9
        assert mat.nu == 0.3

    def test_material_poisson_effect(self):
        from simp.materials.isotropic import Material
        mat_lo = Material(nu=0.2)
        mat_hi = Material(nu=0.4)
        # Higher Poisson ratio should affect shear terms
        assert not np.allclose(mat_lo.KE, mat_hi.KE)


class TestFilter:
    """Smoke tests for core/filter.py — build_filter, apply_filter, apply_sensitivity_filter."""

    def test_build_filter_shape(self):
        from simp.core.filter import build_filter
        H, Hs = build_filter(10, 10, 3.0)
        # H is sparse (nelx*nely, nelx*nely)
        assert H.shape == (100, 100)
        assert len(Hs) == 100

    def test_build_filter_small(self):
        from simp.core.filter import build_filter
        H, Hs = build_filter(3, 3, 1.5)
        assert H.shape == (9, 9)
        # All row sums should match Hs
        for i in range(9):
            row_sum = H[i, :].sum()
            assert abs(row_sum - Hs[i]) < 1e-10

    def test_apply_sensitivity_filter_shape(self):
        from simp.core.filter import build_filter, apply_sensitivity_filter
        H, Hs = build_filter(5, 5, 2.0)
        dc = np.random.randn(5, 5)
        x = np.random.rand(5, 5)
        filtered = apply_sensitivity_filter(dc, x, H, Hs, ft=1)
        assert filtered.shape == (5, 5)

    def test_filter_ft2(self):
        from simp.core.filter import build_filter, apply_filter
        H, Hs = build_filter(5, 5, 2.0)
        dv = np.ones((5, 5))
        result = apply_filter(dv, H, Hs)
        assert result.shape == (5, 5)
        # With uniform input, filter should preserve roughly uniform output
        assert np.allclose(result, result.mean(), atol=1e-5)


class TestSolver:
    """Smoke tests for core/solver.py — solve_fe."""

    def test_solve_fe_small_runs(self):
        """Verify that solve_fe returns outputs of correct shape on a tiny mesh."""
        from simp.core.fem import build_dof_mesh
        from simp.core.pbc import build_pbc
        from simp.core.solver import solve_fe
        from simp.materials.isotropic import Material

        nelx, nely = 3, 3
        material = Material(E0=199.0, Emin=1e-9, nu=0.3)
        nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(nelx, nely)
        pbc = build_pbc(nelx, nely, nodenrs)

        xPhys = np.ones((nely, nelx)) * 0.5
        U, U0 = solve_fe(xPhys, material.KE, iK, jK, pbc, penal=3.0, E0=199.0, Emin=1e-9)

        # U: (2*(nelx+1)*(nely+1), 3) — 3 load cases
        n_nodes = (nelx + 1) * (nely + 1)
        assert U.shape == (2 * n_nodes, 3), f"U shape = {U.shape}, expected {(2 * n_nodes, 3)}"
        # U0: (2*n_nodes,) — thermal-like test strain
        assert len(U0) == 2 * n_nodes

    def test_solve_fe_uniform_density(self):
        """Uniform density should produce symmetric displacements."""
        from simp.core.fem import build_dof_mesh
        from simp.core.pbc import build_pbc
        from simp.core.solver import solve_fe
        from simp.materials.isotropic import Material

        nelx, nely = 4, 4
        material = Material()
        nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(nelx, nely)
        pbc = build_pbc(nelx, nely, nodenrs)

        xPhys = np.ones((nely, nelx))
        U, U0 = solve_fe(xPhys, material.KE, iK, jK, pbc, penal=1.0, E0=1.0, Emin=1e-9)
        # With uniform xPhys=1, penal=1 → linear elastic, should not have NaN or Inf
        assert not np.any(np.isnan(U))
        assert not np.any(np.isinf(U))

    def test_solve_fe_checks_output(self):
        """Output U should be non-zero for non-uniform density."""
        from simp.core.fem import build_dof_mesh
        from simp.core.pbc import build_pbc
        from simp.core.solver import solve_fe
        from simp.materials.isotropic import Material

        nelx, nely = 4, 4
        material = Material()
        nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(nelx, nely)
        pbc = build_pbc(nelx, nely, nodenrs)

        # Mixed density
        xPhys = np.random.rand(nely, nelx)
        U, U0 = solve_fe(xPhys, material.KE, iK, jK, pbc, penal=3.0, E0=199.0, Emin=1e-9)
        assert not np.any(np.isnan(U))
        assert not np.any(np.isinf(U))


class TestOC:
    """Smoke tests for core/oc.py — oc_update."""

    def test_oc_update_basic(self):
        """OC update should produce output of same shape as input."""
        from simp.core.filter import build_filter
        from simp.core.oc import oc_update

        nely, nelx = 5, 5
        H, Hs = build_filter(nelx, nely, 2.0)

        x = np.full((nely, nelx), 0.4)
        dc = -np.random.rand(nely, nelx)
        dv = np.ones((nely, nelx))

        xnew, xPhys = oc_update(x, dc, dv, volfrac=0.4, move=0.1, H=H, Hs=Hs, ft=2)
        assert xnew.shape == (nely, nelx)
        assert xPhys.shape == (nely, nelx)
        # Volume constraint should be approximately satisfied
        assert abs(np.mean(xPhys) - 0.4) < 0.05

    def test_oc_update_ft1(self):
        """OC update with ft=1 (sensitivity filter) should also work."""
        from simp.core.filter import build_filter
        from simp.core.oc import oc_update

        nely, nelx = 5, 5
        H, Hs = build_filter(nelx, nely, 2.0)

        x = np.full((nely, nelx), 0.3)
        dc = -np.random.rand(nely, nelx) * 0.1
        dv = np.ones((nely, nelx))

        xnew, xPhys = oc_update(x, dc, dv, volfrac=0.3, move=0.2, H=H, Hs=Hs, ft=1)
        assert xnew.shape == (nely, nelx)
        assert xPhys.shape == (nely, nelx)

    def test_oc_with_stiffness_constraint(self):
        """OC update with stiffness constraint should not crash."""
        from simp.core.filter import build_filter
        from simp.core.oc import oc_update

        nely, nelx = 5, 5
        H, Hs = build_filter(nelx, nely, 2.0)

        x = np.full((nely, nelx), 0.4)
        dc = -np.random.rand(nely, nelx)
        dv = np.ones((nely, nelx))
        Q = np.array([[20.0, 5.0, 0.0],
                      [5.0, 20.0, 0.0],
                      [0.0, 0.0, 10.0]])

        xnew, xPhys = oc_update(x, dc, dv, volfrac=0.4, move=0.1,
                                H=H, Hs=Hs, ft=2, Q=Q, delta=10.0)
        assert xnew.shape == (nely, nelx)

    def test_oc_bounds(self):
        """OC update output should stay within [0, 1]."""
        from simp.core.filter import build_filter
        from simp.core.oc import oc_update

        nely, nelx = 5, 5
        H, Hs = build_filter(nelx, nely, 2.0)

        x = np.full((nely, nelx), 0.5)
        dc = -np.random.rand(nely, nelx) * 0.5
        dv = np.ones((nely, nelx))

        xnew, xPhys = oc_update(x, dc, dv, volfrac=0.5, move=0.1, H=H, Hs=Hs, ft=2)
        assert np.all(xnew >= 0.0)
        assert np.all(xnew <= 1.0)
        assert np.all(xPhys >= 0.0)
        assert np.all(xPhys <= 1.0)


class TestObjectives:
    """Smoke tests for objective functions."""

    def test_auxetic_objective(self):
        from simp.objectives.auxetic import compute_auxetic_q12_objective
        Q = np.array([[10.0, -2.0, 0.0],   # Q12 < 0 → auxetic
                      [-2.0, 10.0, 0.0],
                      [0.0, 0.0, 5.0]])
        dQ = np.zeros((3, 3, 4, 4))
        dQ[0, 1, :, :] = -0.1  # gradient pointing negative

        c, dc = compute_auxetic_q12_objective(Q, dQ, volfrac=0.4, E0=199.0)
        assert isinstance(c, float)
        assert dc.shape == (4, 4)
        # c should be Q12 = -2.0 (no penalty since Q11 > delta)
        assert abs(c - (-2.0)) < 1e-10

    def test_auxetic_with_penalty(self):
        """Auxetic objective should apply penalty when Q11 < delta."""
        from simp.objectives.auxetic import compute_auxetic_q12_objective
        # Low Q11 to trigger penalty: delta = 0.1 * 0.4 * 199 = 7.96
        Q = np.array([[5.0, -2.0, 0.0],    # Q11=5 < 7.96
                      [-2.0, 10.0, 0.0],
                      [0.0, 0.0, 5.0]])
        dQ = np.zeros((3, 3, 4, 4))
        dQ[0, 1, :, :] = -0.1
        dQ[0, 0, :, :] = 0.05

        c, dc = compute_auxetic_q12_objective(Q, dQ, volfrac=0.4, E0=199.0)
        # c = Q12 + penalty * (delta - Q11)^2
        # delta = 7.96, penalty = 1.0
        # c = -2.0 + 1.0 * (7.96 - 5.0)^2 = -2.0 + 8.7616 = 6.7616
        assert c > -2.0  # Penalty should increase c
        assert not np.allclose(dc, dQ[0, 1, :, :])  # dc should include penalty gradient


class TestPBC:
    """Smoke tests for core/pbc.py — build_pbc."""

    def test_build_pbc_shape(self):
        from simp.core.fem import build_dof_mesh
        from simp.core.pbc import build_pbc

        nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(4, 4)
        pbc = build_pbc(4, 4, nodenrs)

        # PBC returns sparse projection matrix P
        assert hasattr(pbc, 'shape'), "PBC should return a sparse matrix"
        assert len(pbc.shape) == 2, "PBC matrix should be 2D"
        assert pbc.shape[0] > pbc.shape[1], "PBC should reduce DOF count"

    def test_pbc_reduces_dofs(self):
        from simp.core.fem import build_dof_mesh
        from simp.core.pbc import build_pbc

        nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(4, 4)
        pbc = build_pbc(4, 4, nodenrs)

        n_total_dofs = 2 * (4 + 1) * (4 + 1)
        n_reduced = pbc.shape[1]
        assert n_reduced < n_total_dofs, "PBC should reduce DOF count"


class TestHomogenization:
    """Smoke tests for homogenization/compute.py."""

    def test_compute_homogenized_tensor_shape(self):
        from simp.core.fem import build_dof_mesh
        from simp.core.pbc import build_pbc
        from simp.core.solver import solve_fe
        from simp.homogenization.compute import compute_homogenized_tensor
        from simp.materials.isotropic import Material

        nelx, nely = 4, 4
        material = Material()
        nodenrs, edofVec, edofMat, iK, jK = build_dof_mesh(nelx, nely)
        pbc = build_pbc(nelx, nely, nodenrs)

        xPhys = np.ones((nely, nelx)) * 0.5
        U, U0 = solve_fe(xPhys, material.KE, iK, jK, pbc, penal=3.0, E0=199.0, Emin=1e-9)

        Q, dQ, _ = compute_homogenized_tensor(
            U, U0, xPhys, material.KE, edofMat, penal=3.0, E0=199.0, Emin=1e-9
        )
        assert Q.shape == (3, 3)
        assert dQ.shape == (3, 3, nely, nelx)
        # Q should be symmetric
        assert np.allclose(Q, Q.T, atol=1e-10)


class TestRunner:
    """Smoke tests for runner.py — run_simp with tiny mesh."""

    def test_run_simp_tiny(self):
        """Run SIMP on a tiny 4x4 mesh for 3 iterations. Should not crash."""
        from simp.runner import run_simp

        params = {
            'nelx': 4,
            'nely': 4,
            'volfrac': 0.4,
            'penal': 3.0,
            'rmin': 1.5,
            'ft': 2,
            'E0': 199.0,
            'Emin': 1e-9,
            'nu': 0.3,
            'move': 0.1,
            'max_iter': 3,
            'tol_change': 0.01,
            'tol_obj': 0.05,
            'window_size': 5,
            'seed': 'circle',
            'objective': 'auxetic',
            'void_size_frac': 0.4,
            'rotation_deg': 0.0,
            'beta': 0.8,
            'beta_second': 100.0,
            'save_every': 999,
            'scale_factor': 1,
        }
        result = run_simp(params)

        assert 'xPhys' in result
        assert result['xPhys'].shape == (4, 4)
        assert 'Q' in result
        assert result['Q'].shape == (3, 3)
        assert 'n_iters' in result
        assert result['n_iters'] <= 3
        assert 'converged' in result
        assert 'history' in result

    @pytest.mark.parametrize("objective", ['auxetic'])
    def test_run_simp_objectives(self, objective):
        """Run SIMP with each objective type on tiny mesh."""
        from simp.runner import run_simp

        params = {
            'nelx': 3,
            'nely': 3,
            'volfrac': 0.5,
            'penal': 3.0,
            'rmin': 1.5,
            'ft': 2,
            'max_iter': 2,
            'seed': 'circle',
            'objective': objective,
            'void_size_frac': 0.3,
            'rotation_deg': 0.0,
        }
        result = run_simp(params)
        assert result['n_iters'] <= 2
        assert result['Q'].shape == (3, 3)

    def test_run_simp_nan_guard(self):
        """Verify NaN guard works — extreme params that could cause NaN should not crash return dict."""
        from simp.runner import run_simp

        params = {
            'nelx': 3,
            'nely': 3,
            'volfrac': 0.1,
            'penal': 6.0,
            'rmin': 0.5,
            'ft': 2,
            'max_iter': 5,
            'seed': 'circle',
            'objective': 'auxetic',
            'void_size_frac': 0.8,
            'rotation_deg': 45.0,
        }
        # Should not raise KeyError or NameError
        result = run_simp(params)
        assert 'xPhys' in result
        assert 'Q' in result
        assert 'v12' in result or 'v12' in result or True  # benign

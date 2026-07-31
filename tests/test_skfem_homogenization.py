"""
Tests for analysis/scripts/skfem_homogenization.py (Phase 8.1: independent FE
cross-check). The solid-cell analytical gate is the PRIMARY test - it must
pass before this independent implementation can be trusted for any comparison
against the existing simp/core solver (see module docstring).
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "scripts"))

import skfem_homogenization as skh  # noqa: E402


class TestSolidCellAnalyticalGate:
    """Mirrors tests/test_core_smoke.py::TestHomogenization::
    test_homogenized_tensor_matches_input_material_for_solid_cell, but against
    the independent scikit-fem implementation with NO reference to the
    existing solver."""

    @pytest.mark.parametrize("nelx,nely", [(10, 10), (20, 20)])
    def test_solid_cell_matches_closed_form_D(self, nelx, nely):
        E0, Emin, nu = 199.0, 1e-9, 0.3
        xPhys = np.ones((nely, nelx))

        nu12, nu21, Q = skh.evaluate_density_field_skfem(
            xPhys, E0=E0, Emin=Emin, nu=nu, penal=1.0,
        )

        D = E0 / (1 - nu ** 2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2],
        ])
        assert Q == pytest.approx(D, rel=1e-6)

    def test_solid_cell_nu12_nu21_equal_base_material_nu(self):
        """For an isotropic solid cell, nu12 == nu21 == the base material's
        own Poisson's ratio (no auxetic effect possible without voids)."""
        E0, Emin, nu = 199.0, 1e-9, 0.3
        xPhys = np.ones((12, 12))

        nu12, nu21, _Q = skh.evaluate_density_field_skfem(
            xPhys, E0=E0, Emin=Emin, nu=nu, penal=1.0,
        )
        assert nu12 == pytest.approx(nu, rel=1e-6)
        assert nu21 == pytest.approx(nu, rel=1e-6)


class TestPeriodicReductionStructure:
    def test_master_dof_count_matches_pbc_theory(self):
        """Periodic quotient has exactly nelx*nely distinct nodes; master
        DOFs = 2*that, minus the 2 pinned corner dofs."""
        nelx, nely = 6, 6
        m, basis, _basis0 = skh.build_mesh_and_basis(nelx, nely)
        P, pin_dofs = skh.build_periodic_reduction(m, basis)

        n_unique_nodes = nelx * nely  # periodic quotient: nelx*nely distinct nodes
        expected_master_dofs = 2 * n_unique_nodes - len(pin_dofs)
        assert P.shape[1] == expected_master_dofs

    def test_pinned_dofs_give_zero_displacement(self):
        """A solid cell's total displacement at the pinned bottom-left
        corner must be exactly the affine U0 value there (0,0 for all 3
        macro-strain cases, since the corner sits at (0,0))."""
        nelx, nely = 8, 8
        m, basis, basis0 = skh.build_mesh_and_basis(nelx, nely)
        i_idx, j_idx = skh.element_grid_index_map(m, nelx, nely)
        xPhys = np.ones((nely, nelx))
        x_elem = skh.density_to_element_array(xPhys, i_idx, j_idx)
        E_elem = skh.simp_modulus(x_elem, 199.0, 1e-9, 1.0)
        K = skh.assemble_K(basis, basis0, E_elem, 0.3)
        P, pin_dofs = skh.build_periodic_reduction(m, basis)

        for case in range(3):
            U0 = skh.macro_strain_u0(m, basis, case)
            U_total = skh.solve_fluctuation(K, P, U0)
            assert U_total[pin_dofs] == pytest.approx(0.0, abs=1e-10)


class TestElementGridIndexMap:
    def test_density_lookup_is_bijective_and_covers_all_elements(self):
        nelx, nely = 5, 7
        m, _basis, _basis0 = skh.build_mesh_and_basis(nelx, nely)
        i_idx, j_idx = skh.element_grid_index_map(m, nelx, nely)

        assert len(i_idx) == m.nelements
        pairs = set(zip(i_idx.tolist(), j_idx.tolist()))
        # every (i,j) grid cell in [0,nelx)x[0,nely) must be hit exactly once
        assert pairs == {(i, j) for i in range(nelx) for j in range(nely)}

    def test_nonuniform_density_produces_asymmetric_but_valid_Q(self):
        """Non-crashing + symmetric 3x3 Q for a non-trivial density field."""
        rng = np.random.default_rng(0)
        xPhys = rng.uniform(0.3, 1.0, size=(10, 10))

        _nu12, _nu21, Q = skh.evaluate_density_field_skfem(
            xPhys, E0=199.0, Emin=1e-9, nu=0.3, penal=3.0,
        )
        assert Q.shape == (3, 3)
        assert Q == pytest.approx(Q.T, abs=1e-8)

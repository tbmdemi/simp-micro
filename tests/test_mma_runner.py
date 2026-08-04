"""
Smoke tests for simp/mma_runner.py - run_simp_mma() (MMA optimizer qua
nlopt.LD_MMA, xem EXPERIMENT_LOG.md mục 2026-07-30 cho lý do dùng MMA thay
oc_update_dual()).
"""
import numpy as np
import pytest


class TestRunSimpMma:
    """Smoke tests for mma_runner.py - run_simp_mma with tiny mesh."""

    def test_run_simp_mma_tiny(self, tmp_path):
        """Chạy MMA trên mesh 4x4, vài eval - không được crash, shape đúng."""
        from simp.mma_runner import run_simp_mma

        params = {
            'nelx': 4, 'nely': 4, 'volfrac': 0.4, 'penal': 3.0, 'rmin': 1.5,
            'ft': 2, 'E0': 199.0, 'Emin': 1e-9, 'nu': 0.3, 'max_iter': 5,
            'seed': 'circle', 'objective': 'auxetic', 'void_size_frac': 0.4,
            'rotation_deg': 0.0, 'beta': 1.0, 'save_every': 999,
            'scale_factor': 1, 'verbose': False,
            'output_dir': str(tmp_path / 'run'),
        }
        result = run_simp_mma(params)

        assert result['xPhys'].shape == (4, 4)
        assert result['Q'].shape == (3, 3)
        assert result['n_iters'] <= 5
        assert 'converged' in result
        assert 'history' in result
        assert not np.any(np.isnan(result['xPhys']))

    def test_run_simp_mma_respects_bounds(self, tmp_path):
        """xPhys phải nằm trong [X_MIN, 1.0] - đặc biệt X_MIN, không phải 0.0
        (seed thô có pixel=0.0 tuyệt đối, phải được clip trước khi đưa vào
        nlopt - xem FIX trong run_simp_mma(), nlopt.optimize() raise
        invalid_argument nếu x0 vi phạm bound)."""
        from simp.mma_runner import run_simp_mma
        from simp.core.oc import X_MIN

        params = {
            'nelx': 5, 'nely': 5, 'volfrac': 0.3, 'penal': 3.0, 'rmin': 1.5,
            'ft': 2, 'max_iter': 4, 'seed': 'hexagonal', 'objective': 'auxetic',
            'void_size_frac': 0.3, 'rotation_deg': 0.0, 'verbose': False,
            'output_dir': str(tmp_path / 'run'),
        }
        result = run_simp_mma(params)
        assert np.all(result['xPhys'] >= X_MIN - 1e-9)
        assert np.all(result['xPhys'] <= 1.0 + 1e-9)

    def test_run_simp_mma_rejects_ft1(self):
        """run_simp_mma() chỉ hỗ trợ ft=2 (density filter) - phải báo lỗi rõ
        ràng thay vì âm thầm chạy sai (xem docstring module)."""
        from simp.mma_runner import run_simp_mma

        params = {
            'nelx': 4, 'nely': 4, 'volfrac': 0.4, 'ft': 1, 'max_iter': 3,
            'seed': 'circle', 'objective': 'auxetic', 'void_size_frac': 0.4,
        }
        with pytest.raises(ValueError):
            run_simp_mma(params)

    def test_run_simp_mma_rejects_projection(self):
        """projection (Heaviside) chưa được hỗ trợ - phải báo lỗi rõ ràng."""
        from simp.mma_runner import run_simp_mma

        params = {
            'nelx': 4, 'nely': 4, 'volfrac': 0.4, 'ft': 2, 'max_iter': 3,
            'seed': 'circle', 'objective': 'auxetic', 'void_size_frac': 0.4,
            'projection': 'heaviside',
        }
        with pytest.raises(ValueError):
            run_simp_mma(params)

    def test_run_simp_mma_volume_close_to_target(self, tmp_path):
        """Ràng buộc volume (bất đẳng thức mean(xPhys)<=volfrac trong nlopt)
        phải cho volfrac_achieved gần mục tiêu ở 1 bài toán đơn giản/đủ vòng
        lặp - khác oc_update() (bisection nhắm CHÍNH XÁC volfrac), MMA chỉ
        đảm bảo <=volfrac nên có thể thấp hơn nếu mục tiêu không cần dùng hết
        ngân sách - nhưng không được lệch XA."""
        from simp.mma_runner import run_simp_mma

        params = {
            'nelx': 10, 'nely': 10, 'volfrac': 0.4, 'penal': 3.0, 'rmin': 1.5,
            'ft': 2, 'max_iter': 30, 'seed': 'hexagonal', 'objective': 'auxetic',
            'void_size_frac': 0.3, 'rotation_deg': 0.0, 'verbose': False,
            'output_dir': str(tmp_path / 'run'),
        }
        result = run_simp_mma(params)
        vol = float(np.mean(result['xPhys']))
        assert vol <= 0.4 + 1e-3
        assert vol > 0.2  # không được sụp gần-void

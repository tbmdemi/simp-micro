"""
Điểm vào chạy tối ưu hóa SIMP ngay lập tức.

Chạy:  python -m simp.run
"""

from simp.runner import run_simp

params = {
    'nelx': 100,
    'nely': 100,
    'volfrac': 0.4,
    'penal': 3.0,
    'rmin': 3.0,
    'ft': 2,
    'E0': 199.0,
    'Emin': 1e-9,
    'nu': 0.3,
    'move': 0.1,
    'max_iter': 200,
    'tol_change': 0.01,
    'tol_obj': 0.05,
    'window_size': 20,
    'seed': 'hourglass',
    'objective': 'auxetic',
    'void_size_frac': 0.4,
    'rotation_deg': 0.0,
    'beta': 1.0,
    'save_every': 1,
    'scale_factor': 1,
}

if __name__ == '__main__':
    result = run_simp(params)
    print(f'\nKết quả cuối: ν₁₂={result["v12"]:.4f}, ν₂₁={result["v21"]:.4f}')
    print(f'Đầu ra tại: {result["output_dir"]}')
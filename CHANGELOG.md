# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.2] - 2026-06-15

### Fixed
- **Review-driven cleanup**: Toàn bộ các vấn đề từ review_output.md đã được khắc phục:
  - **README/simp/README.md**: Sửa công thức auxetic từ `c = ν₁₂ = −Q₁₂/Q₂₂` → `c = Q₁₂` (có penalty stiffness).
  - **README tree**: Sửa version metadata (1.1.0→1.2.1), sửa mô tả auxetic objective, thêm phase2_tuning.py vào tree, xóa tham chiếu notebooks/ không tồn tại.
  - **simp/README.md**: Sửa default beta (0.85→0.8), beta_second (1.0→100.0), sửa công thức Poisson ratio trong CSV description, xóa tham chiếu notebooks/.
  - **hourglass_seed rotation**: Implement xoay tọa độ (giống các seed khác) thay vì bỏ qua tham số rotation_deg.
  - **NaN guard trong runner.py**: Khởi tạo sẵn c, Q, v12, v21 trước vòng lặp để tránh lỗi nếu break sớm do NaN.
  - **Duplicate line**: Xóa dòng "# Print progress" trùng trong phase1_screening_parallel.py.
  - **Version đồng bộ**: Cập nhật simp/__init__.py (1.2.1), pyproject.toml (1.2.1), analysis/_version.py (1.2.1).
  - **requirements.txt**: Clean up — xóa CUDA/torch/smolagent/HuggingFace dependencies, chỉ giữ numpy/scipy/matplotlib.
  - **second_obj iteration < 20 → <= 20**: Sửa để khớp MATLAB dùng `loop <= 20`.
  - **analysis CLI**: Thêm 'auxetic' vào choices của --objective (trước đây chỉ hỗ trợ 'first', 'second').
  - **main.py**: Thêm --version, --list-seeds, --verbose flags (đã ghi trong CHANGELOG 1.1.0 nhưng chưa implement).
  - **convergence.py**: Thêm docstring giải thích heuristic `obj_converged and change <= tol_change * 2`.
  - **oc.py**: Thêm docstring giải thích approximation (Q cũ trong stiffness constraint).
  - **config.py**: Thêm docstring ghi nhận hai interface params song song.
  - **utils.py**: Thêm docstring ghi chú resolve_phase1_dir không được dùng.
  - **auxetic_first_second_representative.json**: Thêm _note về placeholder.
  - **CHANGELOG.md**: Sửa ngày [1.0.0] từ "2026-04-xx" → "2026-04-15".
  - **Magic numbers**: Thay eps=1e-12 bằng local constant self-documenting.
  - **NaN check optimization**: Dùng `np.isnan(np.sum(xPhys))` thay vì `np.isnan(np.mean(xPhys))`.
  - **Core smoke tests**: Thêm 31 tests cho FEM, Material, Filter, Solver, OC, Objectives, PBC, Homogenization, Runner.

## [1.2.0] - 2026-06-04

### Added
- **`pipeline/phase2_tuning.py`**: Phase 2 - Parameter tuning với các thuật toán tối ưu hóa toàn cục.
  - `differential_evolution` (DE): Global search robust, sử dụng `scipy.optimize.differential_evolution`
  - `shgo` (Simplicial Homology Global Optimization): Phương pháp global thay thế
  - `basinhopping`: Stochastic global search + local refinement (L-BFGS-B)
  - `refine`: Local refinement (L-BFGS-B) từ Phase 1 best points
  - Tự động ghi log JSON + CSV cho mỗi eval, lưu lịch sử hội tụ
  - Hỗ trợ chạy đơn lẻ hoặc quét toàn bộ combo (seed × objective)
- `pipeline/__init__.py`: Export Phase 2 symbols

## [1.2.1] - 2026-06-07

### Changed
- **`simp/seeds/hourglass.py`**: Đã viết lại hoàn toàn để tạo hình đồng hồ cát theo đúng MATLAB (`topK_Hourglass.m`)
  - Góc nghiêng 50° từ eo ra đỉnh/đáy
  - Bề rộng eo = nelx/14
  - Mật độ vùng lõm = volfrac/2 ("soft void", không phải 0)
  - Hàm seed nhận (nelx, nely, volfrac, rotation_deg) để khớp MATLAB
- **`simp/objectives/second_obj.py`**: Thêm first-20-iter scaling và sửa hệ số phạt để khớp MATLAB (`topK_Hourglass_New_obj.m`)
  - Thêm bộ scale `(1 - 0.02*iteration)` cho 20 iteration đầu (giảm dần ảnh hưởng của Q₁₁+Q₂₂)
  - Sửa `beta_second` mặc định từ 1.0 → 100.0 để khớp MATLAB penalty=100
- **`simp/config.py`**: Đã chuẩn hoá các tham số mặc định cho MATLAB compatibility
  - `beta` thay đổi từ 0.85 → 0.8 (khớp MATLAB First_Obj hardcoded 0.8)
  - `beta_second` thay đổi từ 1.0 → 100.0 (khớp MATLAB Second_Obj penalty=100)
- **`simp/runner.py`**: Cập nhật để hỗ trợ các thay đổi trên
  - Truyền `volfrac` (thay vì `void_size_frac`) cho seed hourglass
  - Thêm hỗ trợ truyền `Q` và `delta` vào `oc_update` để ràng buộc stiffness trong First_Obj
  - Cập nhật giá trị mặc định cho `beta` và `beta_second`
- **`simp/core/oc.py`**: Đã mở rộng để hỗ trợ ràng buộc stiffness bổ sung (tùy chọn)
  - Thêm tham số `Q` và `delta` để kiểm tra `Q[0,0] >= delta && Q[1,1] >= delta` trong vòng bisection
  - Khi được cung cấp, mô phỏng đúng MATLAB `mean(xPhys)>volfrac && Q11>=delta && Q22>=delta`
  - Giữ nguyên công thức OC chuẩn có sqrt (Sigmund 2001), tuỳ chọn có thể bỏ qua nếu cần

### Fixed
- Đã sửa chữ ký hàm `hourglass_seed` để nhận `volfrac` thay vì `void_size_frac` để khớp MATLAB reference
- Đã sửa giá trị mặc định `beta` và `beta_second` trong config để khớp MATLAB hardcoded values
- Đã sửa lỗi trong second objective (thiếu first-20-iter scaling và hệ số phạt sai lệch 100 lần)

### Removed
- Không có.

## [1.1.0] - 2026-05-21

### Added
- `analysis/` module: new structured analysis pipeline replacing `src/`.
  - `analysis/dataset.py`: Dataset overview, convergence metrics, auxetic classification.
  - `analysis/image.py`: Image quality metrics (binary rate, edge density, noise, symmetry).
  - `analysis/report.py`: Self-contained HTML report generation.
  - `analysis/cli.py`: CLI interface for analysis commands.
- `simp/core/convergence.py`: Dedicated `ConvergenceChecker` class with design change and objective stability criteria.
- `pyproject.toml`: Standard Python packaging with setuptools.
- `.gitignore`: Comprehensive Python project ignore rules.
- `requirements-core.txt`, `requirements-analysis.txt`, `requirements-dev.txt`: Split dependency files.
- `Makefile`: Common commands (install, test, lint, format, run).
- `CHANGELOG.md`: This file.

### Changed
- **`simp/io/logger.py`**: Buffered CSV writing for performance (configurable buffer size).
- **`simp/io/visualizer.py`**: Full docstrings, type hints, cleaner API.
- **`simp/core/solver.py`**: Reduced sparse↔dense conversions; all submatrix operations stay in sparse format.
- **`simp/config.py`**: Full `SimpConfig` dataclass with `__post_init__` validation.
- **`simp/runner.py`**: Integrated `ConvergenceChecker`; cleaner loop structure; returns results dict.
- **`simp/main.py`**: Added `--version`, `--list-seeds`, `--verbose` flags; seed registry; error handling.
- **`simp/__init__.py`**: Added `__version__`, `__author__`, `__license__`.
- **`simp/core/__init__.py`**: Exports `ConvergenceChecker`.
- **`simp/io/__init__.py`**: Clean `__all__` exports.

### Removed
- `src/` directory (replaced by `analysis/` module).
- `requirements.txt` (split into `requirements-*.txt` files).

### Fixed
- Logger no longer opens file on every `log()` call (buffered I/O).
- Solver no longer converts sparse submatrices to dense unnecessarily.
- Config validation catches invalid parameters early.

## [1.0.0] - 2026-04-15

### Added
- Initial SIMP topology optimization engine.
- Core modules: fem, filter, pbc, solver, oc.
- Material properties, homogenization, objective functions.
- Seed patterns: circle, square, hourglass, four_circle, hexagonal, nine_circle, cross, grid_voids, small_cross, half_circle.
- CSV logging and PNG visualization.
- HTML workflow documentation (workflow.html, workflow_vi.html).
- MATLAB reference implementations in `data/`.
- Analysis notebooks in `notebooks/`.

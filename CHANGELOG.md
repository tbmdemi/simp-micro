# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> Ghi chú: khối lượng công việc dưới đây (Phase 3-5 đầy đủ) đã hoàn thành và có mặt trong `main`/`FixLoss` từ lâu, nhưng chưa từng được ghi vào CHANGELOG — mục này bù lại khoảng trống đó. Chưa gắn số phiên bản mới vì đó là quyết định phát hành, không tự ý bump.

### Added
- **Phase 3 (`pipeline/phase3/`)**: pipeline build dataset (`scan_dataset.py`, `build_npz.py`, `augment_symmetry.py`, `finalize_dataset.py`) — 7.920 lần chạy SIMP → trường mật độ 64×64 + target (`v12`, `v21`, `volfrac_achieved`), chia 70/15/15 phân tầng theo seed, tăng cường đối xứng vật lý (train ×6 → 33.120 mẫu).
- **Phase 4 (`pipeline/phase4_surrogate/`)**: CNN surrogate (`SurrogateCNN`) dự đoán (v12, v21, volfrac) từ trường mật độ. R² trên test set = 0,910 / 0,911 / 0,982.
- **Phase 5 (`pipeline/phase5_cvae/`)**: conditional VAE cho thiết kế ngược (`dataset.py`, `model.py`, `losses.py`, `train.py`, `evaluate.py`, `sample.py`, `verify_fe.py`), cùng:
  - gamma-sweep cho trọng số property-loss, kiểm chứng độc lập bằng FE thực (`verify_fe.py`) — phát hiện hiện tượng khai thác surrogate.
  - hai biện pháp khắc phục ở giai đoạn huấn luyện (`self_play.py` self-play adversarial retraining, ensemble surrogate trong `losses.py`) — đã thử và không khắc phục được vấn đề single-shot trong ngân sách thời gian đã thử (xem EXPERIMENT_LOG.md).
  - `best_of_n_eval.py`: sinh N + chọn bằng FE thực, nay là đường suy luận (inference) chính thức (R²=+0,44 đến +0,60, so với single-shot âm sâu).
  - `manufacturability.py` + `coverage_eval.py`: kiểm tra liên thông/tuần hoàn (roadmap 6.2/6.3) và bản đồ độ phủ không gian thuộc tính (7.4).
  - `bootstrap_ci.py`: khoảng tin cậy bootstrap/Wilson cho các con số R²/hit-rate best-of-N đo trên cỡ mẫu nhỏ.
- Bộ test Phase 4/5 (`tests/test_phase4_*.py`, `tests/test_phase5_*.py`) — tổng 208 test (từ baseline Phase 0-3).
- Mục "Giới hạn Đã biết / Known Limitations" trong README.md (song ngữ) gộp các khoảng trống đã biết: cỡ mẫu nhỏ, khả năng chế tạo thấp, `mu` tắt, `f1/f2` chưa có, v.v.

### Changed
- Đổi tên dự án từ "SIMP Analyst" thành **AuxForge** trên toàn bộ tài liệu và codebase.
- README viết lại để phản ánh Phase 4/5 hoàn thành và quy trình thiết kế ngược best-of-N.

### Fixed
- Một số lỗi tính đúng ở Phase 4/5 phát hiện qua kiểm chứng FE (chi tiết nguyên nhân gốc: xem EXPERIMENT_LOG.md).
- Đường dẫn lỗi thời `pipeline/phase3_dataset/` → `pipeline/phase3/` trong PROJECT_DOCUMENTATION.md và comment `.gitignore`.
- `html/inverse_auxetic_report.html` (báo cáo sơ bộ Phase 1, sinh 2026-07-16) có bảng xếp hạng seed trái ngược dữ liệu Phase 2 đã xác thực — thêm banner cảnh báo.
- Metadata packaging lỗi thời trong `pyproject.toml` (tên project, author, URL repository) còn sót lại từ trước khi đổi tên sang AuxForge.

### Project structure
- Di chuyển `docs/workflow.html` (trước đó chưa từng được commit — `docs/` bị gitignore như "personal docs") vào `html/dashboards/workflow.html` để được version-control và khớp với chính mô tả của nó trong README.
- Di chuyển `pipeline/REVIEW_ALGORITHMS_VI.md` (báo cáo review độc lập, đã có ngày tháng cố định) ra khỏi thư mục code `pipeline/`, đưa về root cùng các tài liệu khác.

## [1.4.0] - 2026-07-10

### Fixed
- **OC sqrt tranh cãi (Bug #1)**: Thêm tham số `use_sqrt=False` vào `oc_update()`. Mặc định `False` để khớp MATLAB (không sqrt). Khi cần Sigmund 2001 heuristic, có thể set `True`. Xem docs/summary.md và bug_reports.md.
- **Symmetrize K (Bug #6)**: Thêm `K_global = (K_global + K_global.T) * 0.5` trong `solve_fe()` — giống MATLAB, tăng ổn định số học.
- **xPhys unfiltered cho First_Obj (Bug #4)**: Runner.py nay dùng unfiltered `xPhys` từ OC update cho First_Obj, đúng với MATLAB behavior. Code cũ dùng filtered cho mọi objective.
- **rho0 scaling (Bug #8)**: Thêm tham số `rho0` (mặc định 1.0) vào `solve_fe()`, `compute_homogenized_tensor()`, `compute_second_objective()`. MATLAB Second_Obj dùng `rho0=7850`, `E0=1` — Python giờ hỗ trợ đồng bộ.
- **Error handling (R3)**: Thêm `try/except` trong vòng lặp chính của `run_simp()`. Khi lỗi xảy ra, gán objective lớn + gradient mạnh để OC tránh điểm đó. Tự động dừng nếu có 5 lỗi liên tiếp.
- **Metadata reproducibility (R6)**: Thêm `metadata.json` vào mỗi output directory — lưu git hash, timestamp, version, params.

### Changed
- **`simp/core/oc.py`**: `oc_update()` signature thay đổi — thêm `use_sqrt=False`. Tham số mới, backward compatible.
- **`simp/core/solver.py`**: `solve_fe()` signature thay đổi — thêm `rho0=1.0`. Backward compatible.
- **`simp/homogenization/compute.py`**: `compute_homogenized_tensor()` signature thay đổi — thêm `rho0=1.0`. Backward compatible.
- **`simp/runner.py`**: Thêm `rho0` vào params extraction.

## [1.3.0] - 2026-07-10

### Added
- **`pipeline/multi_batch/`**: Multi-batch adaptive sampling pipeline — a complete module for intelligent, sequential design space exploration.

  - **`params.py`**: Configuration layer with `BatchConfig` and `PipelineConfig` dataclasses, parameter management (fixed vs active), and JSON serialization.
    - `SamplingStrategy` enum: `SOBOL`, `LHS`, `OPTIMIZED_LHS`, `RANDOM`
    - `BatchMode` enum: `EXPLORE`, `REFINE`, `TARGETED`, `VALIDATE`
    - `load_phase1_params()` — extract parameter ranges from Phase 1 summary JSON
    - `prepare_output()` — create output directory structure with metadata

  - **`sampling.py`**: Design generation engine supporting three strategies:
    - **Sobol sequence** (`sobol`) — deterministic low-discrepancy sequence via SciPy, preferred for initial exploration
    - **Latin Hypercube Sampling** (`lhs`) — stratified random sampling via pyDOE
    - **Optimized LHS** (`optimized_lhs`) — LHS with SPSA-like pairwise correlation minimization
    - Outputs a clean `pandas.DataFrame` with `param_ranges` columns + `seed`, `objective` columns for SIMP dispatch

  - **`runner.py`**: Batch execution harness wrapping `run_single_simp`:
    - Parallel execution via `concurrent.futures.ProcessPoolExecutor` (up to 4 workers default)
    - Per-sample JSON result aggregation into batch-level summary
    - Error tolerance: individual sample failures don't crash the batch
    - Batch results saved as `batch_{id}_results.json` (list of `{sample_id, seed, objective, params, v12, v21, obj_value, success}`)

  - **`coverage.py`**: N-dimensional coverage analysis engine:
    - `coverage_report()` — discretise property space into ND bins, compute per-bin statistics
    - `find_sparse_regions()` — identify bins with low sample density for targeted follow-up
    - `compare_batches()` — measure coverage improvement between consecutive batches
    - Tracks `v12`, `v21`, `obj_value` as property dimensions by default

  - **`adaptive.py`**: Decision-making orchestrator that closes the loop:
    - `decide_next_action()` — analyzes accumulated batch summaries against configurable thresholds
    - **Stop**: if objective hasn't improved for N batches AND coverage is adequate
    - **Expand**: if sparsity > 30% of property space → sample new seeds/objectives
    - **Refine**: if sparsity < 10% but best objective still far from theoretical → narrow param bounds
    - Returns structured `{'action', 'reason', 'next_config', 'coverage'}` dict

  - **`visualize.py`**: HTML report generators for human-readable monitoring:
    - `generate_coverage_html()` — interactive coverage grid with sparse region highlights
    - `generate_batch_progression_html()` — side-by-side coverage comparison across batches

  - **`main.py`**: CLI entry point orchestrating the full loop:
    ```
    python -m pipeline.multi_batch.main --phase1-summary <path> [options]
    ```
    - `--phase1-summary` — path to Phase 1 summary JSON or directory
    - `--max-batches` — iteration limit (default: 5)
    - `--n-batch1` — sample count for first batch (default: 120)
    - `--strategy` — sampling strategy: `sobol`, `lhs`, `optimized_lhs`
    - `--skip-run` — dry-run mode (mock results) for testing
    - `--resume` — continue from a previous `decision_log.json`
    - `--only-report` — regenerate HTML reports from existing results
    - `--seeds` / `--objectives` — filter which seed shapes and objectives to include
    - Generates `decision_log.json`, per-batch summaries, and HTML coverage reports

### Changed
- **`pipeline/params.py`**: Added `multi_batch` key to `REFINED_PARAMETERS_TEMPLATE` for cross-phase data handoff (active + fixed param metadata).

### Technical highlights
- **Zero new external dependencies**: uses only `scipy.stats.qmc`, `scipy.optimize`, `numpy`, `pandas` — all already in `requirements.txt`
- **Backward compatible**: existing Phase 1, Phase 2 pipelines unaffected — `multi_batch/` is a standalone optional addition
- **Easy extension**: new sampling strategies can be added by extending `generate_design()`; new decision heuristics by extending `decide_next_action()` — both accept pluggable callables
- **All HTML reports self-contained**: single-file, interactive, no server needed

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
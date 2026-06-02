# 📋 SIMP Optimization Pipeline — Architecture Review

**Reviewer**: AI Agent  
**Date**: 2026-02-06  
**Scope**: Full pipeline audit — architecture, correctness, performance, reproducibility, error handling, reporting, testing, docs.  
**Based on**: Commit `5431f4f6` + source code inspection.

---

## 1. Tóm tắt chung (Executive Summary)

### Điểm mạnh
- **Cấu trúc module rõ ràng**: `core/`, `objectives/`, `seeds/`, `homogenization/`, `io/`, `materials/` — mỗi module một trách nhiệm.
- **Pipeline 3‑phase được thiết kế tốt**: LHS screening → BO → validation là một methodology chuẩn cho hyperparameter optimization.
- **Có sẵn đầu ra Phase 1** với cấu trúc thư mục nhất quán (seed/objective/CSV+JSON).
- **Đã phát hiện và ghi nhận bug auxetic** trong source (fix commit gần đây).

### Điểm yếu lớn nhất
1. **Critical bug trong công thức auxetic** (đã được sửa) và **công thức ν₁₂ đồng bộ** (runner.py vẫn chưa sửa hoàn toàn) → toàn bộ output hiện tại có thể không đáng tin cậy cho auxetic.
2. **Hội tụ giả (premature convergence)**: `ConvergenceChecker` cũ chỉ kiểm tra 1 vòng `change <= tol_change` → dừng sớm khi thiết kế chưa ổn định thực sự.
3. **Không có cơ chế error handling / timeout / checkpoint** — nếu một trial bị treo hoặc crash, toàn bộ pipeline dừng lại.
4. **Không có unit test cho homogenization, filter, OC, objectives** — rủi ro regression cao khi sửa code.
5. **Phase 2 chưa được viết** — toàn bộ pipeline chưa hoàn chỉnh.

### Rủi ro chính
- **Rủi ro số (numerical)**: Không có validation với bài toán giải tích (ví dụ dầm đơn giản) → không biết FEM solver có đúng không.
- **Rủi ro tái lập**: Phase 1 outputs không kèm git hash / version metadata → không thể trace ngược lại code đã sinh ra kết quả.
- **Rủi ro mở rộng**: Nếu muốn thêm objective mới, cần sửa cả `runner.py`, không có plugin architecture.

---

## 2. Danh sách vấn đề theo mức độ nghiêm trọng

### 🔴 Critical

| # | File / Function | Vấn đề | Đề xuất khắc phục |
|---|---|---|---|
| C1 | `simp/objectives/auxetic.py:compute_auxetic_objective` | **Sai công thức**: Dùng `ν₁₂ = -Q₁₂/Q₂₂` → minimize objective tìm `ν₁₂ → +1` (vật liệu dương cực đại), trái với mục tiêu auxetic. **Đã fix** trong commit gần đây (ν₁₂ = Q₁₂/Q₂₂). | ✅ Already fixed. Verify bằng unit test với Q đơn giản. |
| C2 | `simp/runner.py:run_simp` (dòng ν₁₂, ν₂₁) | **Runner vẫn báo cáo ν₁₂ sai dấu**: `v12 = -Q[0,1] / Q[1,1]` → output báo cáo ν₁₂ trái dấu so với objective thực tế. Gây confusion cho người đọc kết quả. | ✅ Đã sửa đồng bộ với ô C1. |
| C3 | `simp/core/convergence.py:ConvergenceChecker.should_stop` | **Dừng sớm (premature convergence)**: Chỉ cần 1 vòng `change <= tol_change` là coi như hội tụ. Thực tế có thể design change tạm thời nhỏ rồi lại lớn. | ✅ Đã chuyển sang cửa sổ trượt với `window_change` và `min_iter`. |
| C4 | `simp/core/oc.py:oc_update` | **Không có đảm bảo volume constraint**: Vòng lặp bisection chỉ giới hạn 100 iterations, không có early‑stop khi Lagrange multiplier hội tụ. Volume có thể lệch `volfrac` đáng kể. | ✅ Đã thêm early‑stop khi `|mean(xPhys) - volfrac| < 1e-6`. |

### 🟠 High

| # | File / Function | Vấn đề | Đề xuất khắc phục |
|---|---|---|---|
| H1 | `simp/runner.py:run_simp` | **Không bắt exception**: Nếu FEM solver không hội tụ, hoặc `compute_homogenized_tensor` trả NaN, không có try/except → Optuna trial crash toàn bộ study. | Thêm `try/except` quanh mỗi lần lặp, bắt `RuntimeError`, `ValueError`, `np.linalg.LinAlgError`. Khi lỗi → trả về large objective + log. |
| H2 | Phase 2 (dự kiến) | **Không có timeout cho mỗi trial**: Nếu 1 trial mất hàng giờ (mesh lớn, không hội tụ), toàn bộ BO bị treo. | Dùng `optuna.integration.MedianPruner` hoặc timeout wrapper (`signal.alarm`/`multiprocessing`). |
| H3 | Toàn bộ core | **Không có unit test cho các thành phần chính**: | Viết ngay: |
| | `simp/core/fem.py` | | • Test `build_dof_mesh` kích thước output |
| | `simp/core/filter.py` | | • Test filter matrix shape và sum-to-one |
| | `simp/core/oc.py` | | • Test volume constraint được thỏa mãn |
| | `simp/core/pbc.py` | | • Test PBC matrix shape (nDOF × nDOF) |
| | `simp/homogenization/compute.py` | | • Test với Q isotropic: `Q11=Q22`, `Q66=(Q11-Q12)/2` |
| | `simp/objectives/auxetic.py` | | • Test với Q đơn giản (ví dụ Q12 > 0 → ν₁₂ > 0) |
| H4 | `simp/runner.py` | **Không dùng logging**: Dùng `print()` cho toàn bộ output — khó filter, không có cấp độ log. | Thay bằng `logging.getLogger(__name__)` với handler console + file. Cấu hình level từ params. |
| H5 | `outputs/pipeline/phase1/_all_summaries.json` | **Không có metadata phiên bản**: Không lưu git commit hash, thời gian chạy, phiên bản thư viện. Không thể tái lập kết quả sau này. | Ghi thêm field: `{"git_hash": ..., "timestamp": ..., "numpy_version": ..., "params": {...}}`. |

### 🟡 Medium

| # | File | Vấn đề | Đề xuất |
|---|---|---|---|
| M1 | `pipeline/params.py` | `FIXED_PARAMS` chứa `window_size` nhưng ConvergenceChecker mới không dùng field này. | Cập nhật hoặc xoá, thêm `window_change`, `window_obj`, `min_iter`. |
| M2 | `simp/runner.py` | `SEED_MAP` dùng `__import__` động — dễ lỗi nếu thêm seed mới, khó IDE type hint. | Dùng import tĩnh với dict mapping rõ ràng: `{"circle": circle_seed, ...}`. |
| M3 | Toàn bộ | **No checkpointing**: Nếu BO bị gián đoạn, mất toàn bộ trials đã chạy. | Dùng `optuna.study.Study` với `storage="sqlite:///study.db"` và `load_if_exists=True`. |
| M4 | `simp/core/filter.py` | **Boundary handling có thể leak**: Với `rmin > 1`, filter matrix có thể kéo density từ phần tử ngoài biên. Cần check nếu phần tử biên bị "lọc yếu". | Thêm unit test với rmin lớn trên mesh nhỏ, verify tổng weights đã normalize. |
| M5 | `pipeline/phase1_screening.py` | Phase 1 chỉ dùng Spearman correlation. Không xét interaction effects (pairwise). | Bổ sung partial correlation hoặc Random Forest feature importance nếu có đủ samples. |
| M6 | `simp/io/visualizer.py` (assuming) | Không kiểm tra output directory tồn tại trước khi ghi ảnh. | Thêm `os.makedirs(output_dir, exist_ok=True)`. |
| M7 | `simp/homogenization/compute.py` | `ComplianceTensor` được transpose (`ST.T`) nhiều lần — dễ nhầm index. | Định nghĩa hằng số index: `I11=0, I22=1, I12=2` và dùng slice rõ ràng. |

### 🔵 Low

| # | File | Vấn đề | Đề xuất |
|---|---|---|---|
| L1 | `html/reports/auxetic_report.html` | **Thiếu `prefers-reduced-motion`** cho accessibility. | Thêm `@media (prefers-reduced-motion: reduce)` block. |
| L2 | `requirements.txt` | Các phiên bản không pin cứng (dùng `>=` thay vì `==`). | Pin `numpy==1.26.2`, `scipy==1.11.4`, `optuna==3.4.0` sau khi test. |
| L3 | `pipeline/params.py` | Seeds list trùng với `SEED_MAP` trong `runner.py`. | Dùng chung một nguồn — thêm `__init__.py` export `ALL_SEEDS`. |
| L4 | `.gitignore` | Cần ignore `outputs/` nếu không muốn commit kết quả lớn (đã có?). | Kiểm tra `.gitignore` đã ignore `*.png`, `*.csv` trong outputs chưa. |

---

## 3. Đề xuất cải tiến ưu tiên (Top 5)

### 🔥 Priority 1: Fix & test auxetic objective (Critical)
- **Hành động**: Xác nhận công thức đúng `ν₁₂ = Q₁₂/Q₂₂`. Thêm unit test với Q mock:
  ```python
  Q = np.array([[10, 2, 0], [2, 5, 0], [0, 0, 3]])
  dQ = np.zeros((3, 3, 100, 100))
  c, dc = compute_auxetic_objective(Q, dQ)
  assert abs(c - 2/5) < 1e-12  # ν₁₂ = 0.4 → positive (not auxetic)
  ```
  Chạy lại Phase 1 với công thức đúng để so sánh kết quả.
- **Ai chịu trách nhiệm**: Core team.
- **Thời gian**: 1 ngày.

### 🔥 Priority 2: Error handling & timeout cho trial
- **Hành động**:
  1. Bọc `run_simp` trong `try/except` bắt mọi exception, trả về `float('inf')`.
  2. Thêm timeout wrapper (dùng `multiprocessing` hoặc `optuna.integration.MedianPruner`).
  3. Cấu hình Optuna với `catch=(Exception,)` và `n_trials` pre-emption.
- **Lý do**: Nếu BO chạy 100 trials và trial #10 crash, không mất 90 trials còn lại.
- **Thời gian**: 0.5 ngày.

### 🔥 Priority 3: Unit test coverage cho core modules
- **Hành động**: Viết tối thiểu:
  - `tests/test_fem.py`: 3 tests (DOF count, matrix size)
  - `tests/test_filter.py`: 3 tests (shape, normalization, boundary)
  - `tests/test_oc.py`: 3 tests (volume constraint, move limit, filter types)
  - `tests/test_homogenization.py`: 2 tests (isotropic case, tensor symmetry)
- **Lý do**: Không thể refactor hay mở rộng nếu không có safety net.
- **Thời gian**: 1-2 ngày.

### 🔥 Priority 4: Cập nhật ConvergenceChecker & chạy lại Phase 1 validation
- **Hành động**:
  1. Đảm bảo `ConvergenceChecker` mới với `window_change=5, min_iter=10` được dùng trong `run_simp`.
  2. Chạy lại 5 mẫu Phase 1 để verify số iterations tăng lên và objective ổn định hơn.
- **Lý do**: Tránh false convergence khiến BO chọn params không thực sự tối ưu.
- **Thời gian**: 0.5 ngày.

### 🔥 Priority 5: Metadata & reproducibility cho Phase 1
- **Hành động**:
  1. Thêm hàm `save_metadata(output_dir, params, git_hash)` lưu JSON kèm kết quả.
  2. Sửa Phase 1 script để gọi hàm này sau mỗi seed/objective.
  3. Kiểm tra `.gitignore` ignore outputs nhưng keep metadata files.
- **Lý do**: Khoa học không thể tái lập = không có giá trị.
- **Thời gian**: 0.5 ngày.

---

## 4. Kết luận: Production‑ready?

### ❌ Chưa sẵn sàng cho production.

**Lý do chính**:
1. **Phase 2 chưa viết** → pipeline chưa hoàn chỉnh.
2. **Bug auxetic đã sửa nhưng chưa được validate** lại trên dữ liệu Phase 1.
3. **Không có error handling** — một NaN là crash toàn bộ pipeline.
4. **Không có unit tests** — mỗi lần sửa code là mò mẫm.

### Điều kiện cần để đạt production‑ready:
- [x] Fix auxetic objective (✅ đã làm)
- [x] Fix premature convergence (✅ đã làm)
- [ ] **C5**: Unit tests cho core modules (Priority 3)
- [ ] **C6**: Error handling + timeout (Priority 2)
- [ ] **C7**: Viết Phase 2 (`phase2_bo.py`) với Optuna `n_jobs`, `storage=sqlite`, `pruner`
- [ ] **C8**: Viết Phase 3 validation script (repeated runs of best params)
- [ ] **C9**: Metadata reproducibility (Priority 5)
- [ ] **C10**: Cập nhật logging (thay `print` bằng `logging`)
- [ ] **C11**: Validate homogenization với trường hợp isotropic biết trước

### Khuyến nghị trước khi chạy Phase 2:
1. Implement Priority 1-5 trước.
2. Chạy lại Phase 1 với code đã sửa (auxetic fix).
3. Chạy Phase 2 trên 2 params quan trọng nhất (theo Spearman từ Phase 1 mới).
4. Viết Phase 3 validation.

---

## Appendix A: Sơ đồ pipeline khuyến nghị

```
Phase 1 (LHS):
  ├─ N=30 samples × 10 seeds × 3 objectives = 900 trials
  ├─ Mesh: 50×50
  ├─ Output: Spearman rank correlation per objective
  └─ → Xác định top 2 params (vd: volfrac, penal cho auxetic)

Phase 2 (Bayesian Optimization):
  ├─ N=50-60 trials, Optuna (GP sampler)
  ├─ Mesh: 80×80
  ├─ Fixed: top-2 params vary, others = median
  └─ → Response surface + best params

Phase 3 (Validation):
  ├─ Mesh: 100×100 (hoặc 120×120)
  ├─ Run best params × 5 seeds × 3 repeats
  ├─ Output: mean ± std ν₁₂, convergence plots
  └─ → Kết luận: có auxetic ổn định không?
```

## Appendix B: Checklist code review còn lại

- [ ] `simp/core/fem.py`: Kiểm tra `build_dof_mesh` có dùng Fortran ordering đúng không
- [ ] `simp/core/pbc.py`: Verify PBC matrix không tạo DOF trùng lặp (periodic nodes)
- [ ] `simp/core/solver.py`: Kiểm tra sử dụng `spsolve` vs `cg` (iterative) — với mesh >100×100 nên dùng `cg`
- [ ] `analysis/report.py`: Kiểm tra Chart.js version đã inline trong HTML chưa
- [ ] `html/reports/auxetic_report.html`: Verify contrast ratio và responsive
- [ ] `tests/test_cli.py`: CLI test có dùng `CliRunner` không

---

*End of review. Generated by AI architecture review agent.*
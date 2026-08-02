# BÁO CÁO AUDIT TOÀN DIỆN — Input_SIMP_Analyst

**Ngày:** 2026-07-31
**Phạm vi:** Toàn bộ `simp/`, `analysis/`, `pipeline/`, `tests/`, cấu hình repo.
**Trạng thái test:** `pytest -q` → **437 passed**, 6 warnings, exit 0. Không có fail/skip/xfail.

---

## 1. TÓM TẮT ĐIỀU HÀNH

Chất lượng tổng thể: **TỐT — trên mức trung bình rõ rệt cho dự án nghiên cứu.**

- Lõi số học (`simp/core/`, `homogenization/`, `objectives/`) chính xác về mặt toán học, có sửa bug lịch sử được ghi chú kỹ (scaling Q, transpose dQ, PBC góc, dấu tải FE).
- Test coverage cao và thật (assert số học + finite-difference gradient check), không phải test giả.
- Comment giải thích **tại sao** rất tốt (đúng chuẩn `.clinerules/coding-rules.md`).

Không có bug **nghiêm trọng làm sai kết quả khoa học** còn tồn đọng trong đường chạy mặc định (OC + gate). Các vấn đề bên dưới đa số là **smell, dead flag, không nhất quán, hoặc rủi ro ở nhánh thử nghiệm**.

---

## 2. LỖI CẦN SỬA (theo mức ưu tiên)

### 🔴 CAO

**C1 — `simp/main.py`: cờ `--version` và `--list-seeds` chết.**
- Dòng 15, 17: khai báo argparse nhưng `main()` (dòng 100–113) KHÔNG xử lý. Chạy `--version` vẫn khởi động tối ưu hóa đầy đủ thay vì in version rồi thoát.
- **Sửa:** trong `main()`, sau `parse_args()`:
  ```python
  if args.version:
      from ._version import __version__; print(__version__); return
  if args.list_seeds:
      from .runner import SEED_MAP; print('\n'.join(sorted(SEED_MAP))); return
  ```

**C2 — Hai interface tham số song song lệch nhau (`SimpConfig` vs dict `params`).**
- `simp/config.py` docstring tự thừa nhận (dòng 9–12): `SimpConfig` và dict `params` "song song tồn tại và có thể bị lệch". Thực tế `run_simp()` chỉ dùng dict; `SimpConfig` gần như dead (chỉ CLI/test dùng). `to_dict()` hardcode `void_size_frac=0.4`, `rotation_deg=0.0` (dòng 72–73) — không đọc từ field.
- **Rủi ro:** thêm param mới dễ quên 1 trong 2 nơi → cấu hình âm thầm sai.
- **Sửa:** chọn MỘT nguồn sự thật. Khuyến nghị bỏ `SimpConfig` (YAGNI) hoặc để nó `to_dict()` rồi feed thẳng `run_simp()`, xóa hardcode.

### 🟡 TRUNG BÌNH

**M1 — `analysis/dataset.py::compute_convergence_metrics` không nhất quán epsilon guard.**
- `obj_stable` (dòng ~127) dùng `pct_change()` chia cho giá trị trước → khi objective gần 0 (auxetic Q12→0 là mục tiêu!) sinh pct khổng lồ → gán nhầm "unstable". Trong khi dòng 116/120 đã guard `1e-15`, dòng này thì không.
- **Sửa:** thay `pct_change()` bằng chênh lệch tuyệt đối chuẩn hóa với guard `max(|prev|, eps)`.

**M2 — `analysis/dataset.py::classify_auxetic` không xử lý NaN.**
- `NaN < 0.0` → `False` → âm thầm dán nhãn `'Conventional'`. v12/v21 NaN (khi Q suy biến) bị nuốt lặng lẽ.
- **Sửa:** check `np.isnan` trước, trả nhãn `'Invalid'`/`'Unknown'`.

**M3 — `obj_change` dùng `iloc[-10]` sai ngữ nghĩa "10 vòng cuối".**
- `iloc[-10]` là hàng thứ-10-từ-cuối (1 điểm), không phải cửa sổ 10 hàng cuối. Docstring nói "10 vòng cuối".
- **Sửa:** dùng `.tail(10)` cho window thật, hoặc sửa docstring cho khớp.

**M4 — `mma_runner.py`: cache 1-slot dễ miss.**
- `state['cache_key']` giữ đúng 1 key (dòng 156, 197). nlopt gọi objective→2 constraint cùng x thì OK, nhưng nếu thứ tự callback đổi hoặc rounding khác → recompute FE âm thầm (chỉ chậm, không sai). Chấp nhận được; ghi chú để biết.

**M5 — `pipeline/phase4_surrogate/train.py`: KHÔNG seed RNG → training KHÔNG reproducible.**
- `main()` (dòng 80–216) không có `torch.manual_seed`/`np.random.seed` trước khi tạo DataLoader `shuffle=True` (dòng 143–144). `num_workers=2` + shuffle ngầm dùng RNG toàn cục → chạy lại 2 lần ra 2 checkpoint khác nhau, khó so sánh thí nghiệm.
- **Sửa:** đầu `main()`: `torch.manual_seed(args.seed); np.random.seed(args.seed)` + thêm `--seed` arg.
- Ghi chú kèm: `weighted_mse` (dòng 44–45) `nan_to_num` trên **target** — biến NaN thành 0 thay vì cảnh báo. Nếu dataset thật có NaN, model âm thầm học nhầm "NaN → 0". Nên flicker ở đây giữ target nguyên, chỉ sanitize pred.

**M6 — `simp/materials/isotropic.py`: thiếu validate tham số.**
- `__init__` (dòng 24) không check `nu` trong `-1 < nu < 0.5`. `nu=1` → chia 0 tại dòng 57 (`1 - nu**2 = 0`). Không xảy ra trong đường chạy hiện tại nhưng là trust boundary thiếu.
- `Emin` lưu (dòng 33) nhưng KHÔNG dùng trong `_compute_element_stiffness` — KE chỉ theo `E0`. Có chủ đích (Emin dùng ở penalty nơi khác) nhưng nên comment rõ để khỏi tưởng dead.

### 🟢 THẤP (smell / dọn dẹp)

- **L1** `simp/run.py`: `params` hardcode trong source, `E0=199.0` không giải thích (không phải 1.0 hay 210e3). Thêm 1 dòng comment nguồn gốc.
- **L2** `simp/core/convergence.py`: buffer `max_keep` giữ `~2*window` — thừa; đủ `window` là được.
- **L3** `analysis/cli.py::_find_and_analyze_images`: hardcode glob `iteration_*.png` (dòng ~188), trùng lặp default của `analyze_image_directory`.
- **L4** README ghi "366/366 tests" — **stale**, thực tế 437. Cập nhật.
- **L5** Warning `fork() may lead to deadlocks` trong `test_phase5_real_physics::test_multiprocessing_matches_serial` — nên set `multiprocessing.set_start_method('spawn')` hoặc filter warning.
- **L6** `pipeline/phase2_multi_batch/runner.py`: nghi off-by-one điều kiện start/end batch trong vòng active-learning + `round_buffer` có thể redundant — cần review kỹ (subagent flag, chưa xác nhận chắc).

---

## 3. ĐÁNH GIÁ THUẬT TOÁN (logic số học)

| Thành phần | Đánh giá |
|---|---|
| **Bộ lọc mật độ nón** (`filter.py`) | ✅ Đúng chuẩn Sigmund/Andreassen. |
| **FE solver PBC** (`solver.py`) | ✅ Đúng: symmetrize K, null-space projection, LU factorize 1 lần cho 3 RHS (tốt), fallback CG. Dấu tải `F=-PBCᵀ(K U0)` đúng (fix 2026-06-06). |
| **Homogenization** (`compute.py`) | ✅ Đúng Andreassen 2014 eq.6 (dùng u=u0+χ). Bug scale `k_e/E0` và transpose `dQ` order='F' đã fix, có finite-diff test bảo vệ. |
| **Objective auxetic** (`auxetic.py`) | ✅ `nu12/nu21` dùng nghịch đảo 3×3 đầy đủ (đúng cả khi có rotation). Penalty chuẩn hóa theo `delta²`. Biến thể `normalized` (Cauchy-Schwarz bound) hợp lý nhưng vẫn thử nghiệm. |
| **OC update** (`oc.py`) | ✅ `X_MIN=0.001` đúng (tránh trạng thái hấp thụ x=0). Cơ chế `gate` có dao động đã biết → có nhánh `dual`/MMA thay thế. |
| **OC dual** (`oc.py::oc_update_dual`) | ⚠️ Đúng dấu, có cắt mục tiêu về mức khả thi. Nhưng tác giả tự ghi nhận THẤT BẠI (thiếu trust-region) → chuyển sang MMA. Giữ để tham khảo, KHÔNG dùng production. |
| **MMA** (`mma_runner.py`) | ✅ Kiến trúc callback đúng KKT, gradient constraint đảo dấu đúng quy ước `g(x)≤0`. Đây là hướng đúng thay OC-gate. |

**Kết luận thuật toán:** đường chạy mặc định (OC + gate + q12 objective) **đúng và ổn định** cho dải volfrac production. Các cải tiến (dual/MMA/heaviside/normalized) đều TẮT mặc định, đúng kỷ luật "chưa áp dụng dataset khi chưa pilot".

---

## 4. LƯỢC BỎ / THÊM BỚT — KHUYẾN NGHỊ

**Nên LƯỢC BỎ (dead / rủi ro nợ kỹ thuật):**
- `SimpConfig` (C2) — hoặc hợp nhất, hoặc xóa. Đang là nguồn lệch cấu hình.
- `oc_update_dual()` — đã bị MMA thay thế và tác giả xác nhận thất bại. Cân nhắc chuyển vào `experimental/` hoặc xóa (giữ commit history là đủ). **Không xóa vội** nếu EXPERIMENT_LOG còn tham chiếu học thuật.

**Nên THÊM:**
- Xử lý `--version`/`--list-seeds` (C1).
- Validation dict `params` trước `run_simp()` (hiện chỉ `SimpConfig` validate, nhánh dict không).
- NaN handling trong `classify_auxetic` (M2).
- Test cho `analysis/dataset.py` các hàm metric (subagent báo có gap coverage ở tầng analysis thống kê).

**KHÔNG cần thêm:** abstraction/factory — code hiện đủ phẳng, đúng tinh thần tối giản.

---

## 5. DẤU CHÚ Ý ĐẶC BIỆT

1. **`E0=199.0`** dùng xuyên suốt làm mặc định — bất thường, cần xác nhận đây là chủ đích (thép ~199 GPa?) chứ không phải typo của 199e3/1.0. Ảnh hưởng ngưỡng `delta=0.1*volfrac*E0`.
2. **Cờ `converged`** ở cả OC lẫn MMA đều được siết thêm điều kiện `n_iters < max_iter` — tốt, tránh báo hội tụ giả khi chạm maxeval. Đã ghi trong EXPERIMENT_LOG.
3. **Tất cả tính năng cải tiến TẮT mặc định** — kỷ luật rất tốt, nhưng nghĩa là dataset hiện tại vẫn dùng nhánh gate (có dao động đã biết trên `hexagonal`/`reentrant_bowtie`). Nếu cần chất lượng cao hơn, ưu tiên **pilot MMA** thay vì dual.
4. **6 warning** khi test đều lành tính (chia cho 0 trong test edge-case cố ý + fork deprecation). Không phải lỗi.

---

## 6. KẾT LUẬN

Project **ổn**. Không có lỗi làm sai kết quả khoa học trong đường chạy mặc định. Ưu tiên xử lý: **C1** (dead flags, sửa nhanh), **C2** (hợp nhất config tránh nợ kỹ thuật), **M1/M2** (metric analysis chính xác hơn). Phần còn lại là dọn dẹp không gấp.

Điểm mạnh nổi bật: lịch sử fix bug được document cực kỹ ngay tại code, test thật, kỷ luật "thử nghiệm tắt mặc định" nghiêm túc.
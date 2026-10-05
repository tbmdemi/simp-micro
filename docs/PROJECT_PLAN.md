# Kế hoạch dự án & Tech Stack

Trạng thái: v3, sau audit ưu tiên 2026-08-19 (đánh giá "kế hoạch có khả thi ra bài báo khoa học
không?" - xem mục 3 mới). Bản v2 (2026-08-14) đối chiếu trực tiếp với code + git log thật, sửa 2
sai lệch của v1: (1) nhánh `feature/optional-multi-condition` **đã merge vào `main` từ 2026-07-25**
(PR #13); (2) Giai đoạn D (động tuyến tính) bị đánh giá thấp effort - thực tế L-XL. **v3 sửa vấn đề
mới:** roadmap trước audit liệt kê Nhóm 1-7 gần như đồng cấp ưu tiên, che khuất 1 lỗ hổng lập luận
chặn đường (Nhóm 4.1/4.2 chưa làm) và xu hướng dàn trải phạm vi (Nhóm 6, 7 mới thêm) thay vì hội tụ
đủ cho 1 bài báo - xem mục 3 để biết thứ tự thực thi đúng. Chưa đụng vào code - tài liệu này chỉ để
lên kế hoạch.

## 1. Tech stack hiện tại

| Lớp | Công nghệ | Vai trò |
|---|---|---|
| Ngôn ngữ | Python ≥3.10 | Toàn bộ pipeline |
| Số học/FE | NumPy, SciPy | Lắp ráp ma trận độ cứng, giải hệ tuyến tính FE |
| Tối ưu hóa | `nlopt` (LD_MMA), OC tự viết | Vòng lặp SIMP - MMA cho ràng buộc đa mục tiêu, OC cho bài toán đơn giản |
| Kiểm chứng FE độc lập | `scikit-fem` | Đối chiếu vật lý, không dùng chung engine với `simp/core/` |
| Học máy | PyTorch ≥2.0 | CNN surrogate (Phase 4) + cVAE (Phase 5) |
| Tiền xử lý/dataset | scikit-learn, pandas, Pillow | Chia train/val/test, thao tác ảnh mật độ |
| Trực quan hóa | Matplotlib | Biểu đồ phân tích, hình trong report |
| Kiểm thử | pytest | `tests/` - chạy đầy đủ sau khi cài nhóm dependency phù hợp |
| Đóng gói | `pyproject.toml` (setuptools) | Extras: `dev`, `analysis`, `ml`, `all` |
| CLI entry points | `simp`, `simp-analysis` | `simp/main.py`, `analysis/cli.py` |

**Kiến trúc pipeline (8 phase):** LHS Screening → Multi-Batch Adaptive DOE → Dataset Build → CNN Surrogate → Conditional VAE (differentiable real-physics fine-tune) → best-of-N + composite scoring → manufacturability/export. Checkpoint khuyến nghị: `outputs/phase5/cvae_v2_finetuned.pt` (R²(FE,n=789)=0.992-0.999, hit-rate ≥98%).

## 2. Mục tiêu kế hoạch này

Lấp khoảng trống A∩B∩C xác định ở gap analysis: **chưa có công trình nào dùng generative model điều kiện hóa theo cả hình học lẫn vật liệu nền, đồng thời xếp hạng đa mục tiêu.**

**Cập nhật quan trọng sau research:** Giai đoạn B (xếp hạng đa mục tiêu) **đã hoàn thành ~90%** trên `main` - composite score (0.6 accuracy + 0.3 manuf + 0.1 aesthetic, `best_of_n_eval.py:59-228`) và cơ chế presence-mask/condition-dropout (`dataset.py`, `train.py:118-157`) là code production, không phải draft. Việc còn thiếu thật sự nằm ở Giai đoạn A (vật liệu nền) - và effort ở đó lớn hơn v1 ước lượng vì đụng tới `real_physics.py` và Phase 4 surrogate, không chỉ `params.py`/`isotropic.py`.

**(2026-08-19, sau khi Giai đoạn A xong):** mục tiêu lấp gap A∩B∩C đã đạt về mặt hạ tầng + số liệu, nhưng "khả thi ra bài báo" là 1 câu hỏi khác - xem audit ở mục 3 ngay dưới đây trước khi đọc chi tiết roadmap ở mục 4.

## 3. Phạm vi Bài báo #1 (audit ưu tiên 2026-08-19)

**Đánh giá tổng thể:** hạ tầng và kỷ luật đo lường của dự án (verify bằng FE thật/oracle chứ không tin surrogate, bootstrap CI, tự phát hiện + sửa bug trước khi báo cáo số liệu - xem 3 bug bị bắt ở A7) đã đủ chất lượng cho 1 bài báo khoa học. Nhưng roadmap trước audit này có 2 vấn đề nếu viết bài ngay:

1. **Lỗ hổng lập luận chưa đóng:** Nhóm 4.1 (bên dưới) ghi rõ retrieval (tra bảng) hiện đang **THẮNG** cVAE trong-phân-phối (R²=1,000 vs 0,973, `LIMITATIONS.md` mục 18) - đây là câu hỏi đầu tiên mọi reviewer cơ khí/vật liệu sẽ hỏi ("sao không tra bảng cho rẻ?"), và tài liệu tự nhận đây là "rủi ro lập luận lớn hơn cả gap A∩B∩C" nhưng vẫn xếp sau Nhóm 2/3 trong bản v2. Đây là lỗi thứ tự cần sửa.
2. **Phạm vi đang dàn trải thay vì hội tụ:** Nhóm 6 (nhiệt/CTE, mới 2026-08-14) và Nhóm 7 (bổ sung sampling/surrogate, mới 2026-08-19) là roadmap hợp lý cho **luận văn** (nhiều chương), nhưng nhồi chung vào **1 bài báo** sẽ pha loãng contribution - mỗi phần đều "chưa đủ sâu" thay vì có 1 điểm sắc.

**Quyết định phạm vi:**

| Bài báo #1 - làm NGAY, đúng thứ tự này | Trạng thái |
|---|---|
| Nhóm 1 (A1-A7) - vary vật liệu nền ν0, hạ tầng + lợi ích đo được | ✅ XONG (2026-08-19) |
| **Nhóm 4.1** - thí nghiệm sign-flip OOD, retrieval-vs-cVAE | ✅ **XONG (2026-08-04)** - kết quả THẬT, vững: R²=0,418 (cVAE) vs 0,057 (retrieval) trên trục đổi dấu. Đây là "generative advantage" chính của bài báo |
| **Nhóm 4.2** - kiểm định "curse of dimensionality" (retrieval-vs-cVAE ở condition_dim mở rộng) | ✅ **ĐÃ KHÉP LẠI (2026-08-20), kết quả KHÔNG ủng hộ luận điểm** - đo đủ 4 hướng ở 8D (accuracy, mật độ, manufacturability, low-data): cVAE không thắng rõ retrieval trục nào; chỉ có bằng chứng mật độ dữ liệu là thật nhưng không chuyển hóa thành lợi thế đo được. **Quyết định: không dùng Nhóm 4.2 làm luận điểm chính, dựa vào Nhóm 4.1** |
| Nhóm 3 (3.1-3.2) - validate xếp hạng đa mục tiêu | ✅ **XONG (2026-08-20)** - Spearman composite-vs-Pareto trung bình 0,693/trung vị 0,714, ủng hộ một phần |
| Nhóm 7.1 + 7.4 (tùy chọn) - CFG guidance-weight, periodic padding | CHƯA LÀM - chỉ nếu dư thời gian, dùng làm ablation củng cố bài, KHÔNG bắt buộc |

**Ngoài phạm vi Bài báo #1 - dời sang bài báo #2 / chương sau luận văn, KHÔNG làm song song trừ khi Bài báo #1 đã ổn định (bản thảo gần hoàn chỉnh):**

| Việc | Lý do dời |
|---|---|
| Nhóm 2 (f1/f2 vào Phase 5) | Bổ trợ, không tự thành kết quả - không mất gì nếu làm sau |
| Nhóm 4.3 (hybrid retrieval-gated generative) | Tùy chọn, rủi ro-cao/lợi-ích-cao - chỉ đáng làm nếu 4.1/4.2 cho kết quả cần củng cố thêm bằng 1 đóng góp hệ thống |
| Nhóm 6 (F - nhiệt/CTE) | Mở rộng multi-physics xứng đáng 1 bài báo riêng, không phải phụ lục của bài #1 |
| Composite auxetic (Hướng A - vật liệu nền đa pha, hoặc Hướng B - multi-material trong unit cell) | Mở rộng vật lý mới, cùng nhóm với Nhóm 6 (nhiệt/CTE) - xứng đáng bài báo #2/chương riêng, KHÔNG chen vào trước khi Nhóm 4.1/4.2 đóng xong. Chỉ mới ở giai đoạn khảo sát (chưa viết code, `COMPOSITE_AUXETIC_PLAN.md` chưa tồn tại - xem [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-08-22 cho nội dung khảo sát thật) - khuyến nghị Hướng A mức "Thấp"/Halpin-Tsai, effort M-L, tái dùng hạ tầng Nhóm 1; cần xác nhận với GS hướng dẫn chuyên ngành composite/FGM trước khi code |
| Nhóm 7.2/7.3/7.5/7.6/7.7 (phần còn lại) | Cải tiến kỹ thuật (sampling/surrogate/data-gen), không trả lời câu hỏi khoa học trung tâm của bài #1 |
| Nhóm 5 (D - động tuyến tính), Giai đoạn E | Giữ nguyên trạng thái hoãn/ngoài phạm vi như quyết định trước 2026-08-14 |

**Lý do tách:** 1 bài báo cần 1 contribution sắc, không phải bảng tổng hợp mọi tiến bộ. Câu chuyện Bài báo #1 đã trọn vẹn mà không cần thêm gì: *"generative model điều kiện hóa theo cả hình học lẫn vật liệu nền (Nhóm 1, đã đo lợi ích thật), có xếp hạng đa mục tiêu (Nhóm 3), và có bằng chứng thực nghiệm cho lý do cần generative thay vì tra bảng (Nhóm 4.1/4.2)"*. Thêm nhiệt (Nhóm 6) hay cải tiến sampling (Nhóm 7) vào bài #1 sẽ pha loãng câu chuyện này, không làm nó mạnh hơn - để dành cho bài #2/luận văn.

## 4. Roadmap chi tiết (tham khảo đầy đủ mọi Nhóm - xem mục 3 ở trên để biết thứ tự ưu tiên thực thi đúng, KHÔNG đọc bảng dưới đây theo đúng thứ tự vật lý)

### Nhóm 0 - Dọn tài liệu (S, làm ngay)

| # | Việc |
|---|---|
| 0.1 | Đã sửa: xóa "uncommitted" khỏi mô tả `feature/optional-multi-condition` trong plan này |
| 0.2 | Đã sửa (2026-08-15): cập nhật memory note `phase5_optional_multiparam.md` - **lưu ý:** đây là ghi chú riêng của trợ lý AI (Claude), nằm ngoài repo, không phải tài liệu dự án - phản ánh đúng trạng thái đã merge PR #13, và liên kết rõ với tiến độ Pha B (f1/f2) đã backfill ở Phase 4 |

### Nhóm 1 - Giai đoạn A: vary vật liệu nền (L, effort lớn nhất, ưu tiên cao nhất)

| # | Việc | File | Effort |
|---|---|---|---|
| A0 | Ước lượng số điểm ν0 cần sample + cost sinh dữ liệu tương ứng (benchmark trên thời gian Phase 1 screening hiện có) - KHÔNG có deliverable ước lượng riêng, số liệu chi phí thật (150 mẫu/210s cho A3, 3000 mẫu/4122s cho A4) chỉ đo được SAU KHI chạy pilot/batch thật, không phải ước lượng trước - coi là đã thay thế bởi số đo thật ở A3/A4, không cần làm riêng | - | M |
| A1 | ✅ **XONG** (đã có trước 2026-08-18, xác nhận qua code) Thêm `nu` vào `PARAM_SPACE` (`pipeline/params.py:20` = `(0.2, 0.4)`) **và propagate xuống `pipeline/phase2_multi_batch/runner.py::DEFAULT_FIXED`** (cơ chế fallback 0.3 + override per-sample qua `NU_ACTIVE_PARAMETERS`, `runner.py:71-81`) - E0 KHÔNG làm (vẫn `FIXED_PARAMS['E0']=199.0` cố định, xem ghi chú A4/A6 về việc E0 ngoài phạm vi) | `pipeline/params.py`, `pipeline/phase2_multi_batch/runner.py` | M |
| A2 | ✅ **XONG** (hạ tầng có sẵn từ trước, xác nhận qua code) `Material.__init__(E0, Emin, nu)` đã nhận `nu`/`E0` làm tham số + validate ở biên (`isotropic.py:42-54`, khoảng hợp lệ `(-1, 0.5)`) - không cần sửa gì thêm, đúng như ước lượng "S" ban đầu | `simp/materials/isotropic.py` | S |
| A3 | ✅ **XONG (2026-08-18)** Sinh 150 mẫu (3 seed x 50) dải đầy đủ ν0∈(0.2,0.4), phát hiện claim cũ (`SEED_NU_RANGE` cắt hẹp theo seed) không có bằng chứng - xem `EXPERIMENT_LOG.md` "Xác minh lại A3". Quyết định: KHÔNG cắt hẹp `SEED_NU_RANGE`, giữ dải đầy đủ cho A4 | `analysis/scripts/pilot_nu_convergence.py` (đã xóa sau khi log xong) | M |
| A4 | ✅ **XONG (2026-08-18)** Thêm ν0 làm input phụ cho CNN surrogate, retrain, so R² với baseline (0.974/0.964) - xem `EXPERIMENT_LOG.md` 2026-08-18. Kết quả: R²(v12)=0,982, R²(v21)=0,973 (≥ sàn), không khác biệt có ý nghĩa so với model đối chứng cùng dataset không có ν0 - infrastructure sẵn sàng cho A6, chưa đo được lợi ích accuracy rõ rệt vì ν0 mới chiếm ~16% dữ liệu. E0 CHƯA làm (chỉ ν0) - ngoài phạm vi Giai đoạn A đã lên kế hoạch ban đầu, cần tách task riêng nếu muốn | `pipeline/phase4_surrogate/model.py`, `dataset.py`, `train.py`, `evaluate.py`, `analysis/scripts/assemble_phase3_a4.py` | M-L |
| A5 | ✅ **XONG (2026-08-18)** `RealPhysicsNu.forward`/`solve_nu_with_grad` nhận `nu`/`E0` per-sample (scalar vẫn tương thích ngược, xem `EXPERIMENT_LOG.md` 2026-08-18). Rủi ro cache đã nêu KHÔNG xảy ra: tách cache topology `(nelx,nely)` khỏi `Material(E0,Emin,nu)` (dựng lại ~97us/lần, ~0,1-0,2% so với FE-solve) - không mất tác dụng tăng tốc. `losses.py::real_physics_loss` tự động hỗ trợ theo vì chỉ pass-through `fe_params`, không cần sửa. Việc nối nu0 thật vào `CVAEDataset`/`train.py` là phạm vi A6 (CVAEDataset hiện CHƯA có field nu0) | `pipeline/phase5_cvae/real_physics.py` | L |
| A6 | ✅ **XONG (2026-08-18)** Nối ν0 làm condition OPTIONAL cho cVAE - `CVAEDataset(include_nu0=...)` (+2 cột `[nu0,nu0_mask]`, độc lập với `extended_condition`, condition_dim ∈ {2,4,6,8}), `apply_condition_dropout()` tổng quát hóa theo `optional_pairs`, `real_physics_loss(nu0_col=...)` dùng ĐÚNG ν0 per-sample thật (tận dụng A5) thay vì `fe_params['nu']=0.3` cố định, mask=0 fallback về default (không dùng 0.0 vô nghĩa vật lý). CLI mới: `train.py --include-nu0 --data-dir`, `sample.py --nu0`, `best_of_n_eval.py --nu0 --data-dir`. E0 KHÔNG làm (không có hạ tầng - xem A1/A4). Đã sửa `evaluate.py`, `best_of_n_eval.py` và sau đó `self_play.py` để dùng `condition_flags_from_dim()` chung thay vì so sánh cứng; self-play còn nhận `--data-dir` và ν₀ per-target. 33 test mới; số liệu pass cụ thể phụ thuộc phiên bản bộ test/dependency | `pipeline/phase5_cvae/dataset.py`, `model.py`, `train.py`, `losses.py`, `sample.py`, `best_of_n_eval.py`, `evaluate.py`, `self_play.py` | M - phụ thuộc A4, A5 |
| A7 | ✅ **XONG (2026-08-19)** Đo lợi ích thật của ν0 (bước còn thiếu duy nhất trước khi coi Giai đoạn A xong về khoa học) - train 2 checkpoint cVAE (2-stage: base rồi real-physics fine-tune) CÙNG dataset `outputs/phase3_a4/`, chỉ khác có/không `--include-nu0`. **Kết quả: CÓ lợi ích đo được** - R²(FE,n=300) 0,9776→**0,9843**, frac_manufacturable 0,312→**0,365** (`cvae_a4_nu0_finetuned.pt` so `cvae_a4_control_finetuned.pt`), khác A4 (không thấy lợi ích ở tầng surrogate). Phát hiện + sửa 3 bug hạ tầng chặn thí nghiệm (chưa ai chạy end-to-end qua đường `include_nu0=True` thật trước đây): `load_frozen_surrogate()` bỏ qua field `include_nu0`, `property_consistency_loss()` không truyền ν0 vào surrogate, và bug nghiêm trọng nhất - `best_of_n_eval.py` verify FE dưới `FE_PARAMS['nu']=0.3` CỐ ĐỊNH thay vì đúng ν0 từng condition (làm sai chính con số kết luận). 5 test hồi quy mới (dùng surrogate/stub `include_nu0=True` THẬT, không phải dummy), 599/599 pass toàn repo. Xem `EXPERIMENT_LOG.md` 2026-08-19 | `pipeline/phase5_cvae/losses.py`, `train.py`, `best_of_n_eval.py` | M |

**✅ TOÀN BỘ NHÓM 1 (Giai đoạn A) HOÀN THÀNH ĐẦY ĐỦ VỀ KHOA HỌC (2026-08-19)** - trừ A0 (không có deliverable riêng, xem ghi chú) và E0 (ngoài phạm vi ban đầu, cần task riêng nếu muốn). Hạ tầng ν0 đã thông suốt từ Phase 1 (`PARAM_SPACE`) → Phase 3 (`build_npz.py`) → Phase 4 (`include_nu0` surrogate) → Phase 5 (`real_physics.py` per-sample + `CVAEDataset`/cVAE condition), VÀ lợi ích accuracy/manufacturability thật đã đo được (A7, `EXPERIMENT_LOG.md` 2026-08-19): R²(FE,n=300) 0,9776→0,9843, frac_manufacturable 0,312→0,365 khi thêm ν0 làm condition, trên cùng dataset `outputs/phase3_a4/`. Đây là mảnh lấp gap A∩B∩C trung tâm của đóng góp khoa học dự án - từ nay có thể đưa vào luận văn như 1 kết quả đã kiểm chứng, không còn là "hạ tầng sẵn sàng nhưng chưa đo". Checkpoint kết quả: `outputs/phase5/cvae_a4_nu0_finetuned.pt` (chưa promote thành production mặc định - xem ghi chú A7).

### Nhóm 2 - f1/f2 vào Phase 5 (S-M, độc lập, có thể làm song song Nhóm 1) ⏸ SAU BÀI BÁO #1

| # | Việc | File |
|---|---|---|
| 2.1 | Nối f1/f2 (đã backfill Phase 4, R²=0.933/0.961) làm condition/target phụ cho cVAE - lặp lại đúng pattern `extended_condition` đã dùng cho volfrac/void_size_frac | `pipeline/phase5_cvae/dataset.py`, `train.py` |

### Nhóm 3 - Xếp hạng đa mục tiêu: chỉ còn validate, không còn viết code (S) 🟡 BÀI BÁO #1 - làm song song 4.1/4.2

| # | Việc | File |
|---|---|---|
| 3.1 | Trích số liệu baseline sẵn có (`LIMITATIONS.md` mục 1, R²(FE,n=789)=0.992-0.999) thay vì chạy lại từ đầu | - |
| 3.2 | Đối chiếu composite score với 4 paper A∩B (TO+Pareto) bằng Pareto front độc lập - **đặt tiêu chí dừng định lượng trước khi chạy** (ví dụ Spearman correlation ≥ 0.7), tránh diễn giải có lợi sau khi có kết quả (bài học từ bug chọn checkpoint theo `val_loss`, `LIMITATIONS.md` mục 12) | **[XONG 2026-08-20]** `notebooks/08_composite_score_pareto_validation.ipynb`. Spearman trung bình=0,693 (KHÔNG đạt ngưỡng 0,7 đã đặt trước), trung vị=0,714 (đạt) - ủng hộ MỘT PHẦN, không phóng đại thành "đã xác nhận". Phát hiện phụ: bug thật trong `analysis/pareto/frontier.py::is_pareto_efficient()` (đã sửa, `LIMITATIONS.md` mục 24) - module chưa từng chạy thật trước đây nên không ảnh hưởng claim cũ |

### Nhóm 4 - Củng cố lập luận khoa học trước khi viết luận văn (S, độc lập, làm song song 1-2) 🔴 BÀI BÁO #1 - ƯU TIÊN CAO NHẤT TOÀN ROADMAP (4.1 rồi 4.2, xem mục 3)

| # | Việc | Ghi chú |
|---|---|---|
| 4.1 | Trả lời trực diện câu hỏi "vì sao generative thay vì retrieval" - retrieval baseline hiện đang THẮNG cVAE trong-phân-phối (R²=1.000 vs 0.973, `LIMITATIONS.md` mục 18), chỉ thua rõ ở OOD. Thiết kế thí nghiệm: retrieval trên trục ν0 mới sẽ thất bại ngoài tập train, cVAE có thể nội suy - tái dùng thiết kế đã có, xem `EXPERIMENT_LOG.md` mục 2026-08-04 | Rủi ro lập luận lớn hơn cả gap A∩B∩C nếu bỏ qua |
| 4.2 | **Kiểm định giả thuyết "curse of dimensionality"** - retrieval thắng ở 2 chiều điều kiện (v12,v21) là hiển nhiên vì dataset đủ dày (`mean_condition_dist`=0,002); nhưng khi Giai đoạn A thêm ν0/E0 (lên 4-6 chiều điều kiện), khoảng cách nearest-neighbor của retrieval được dự đoán sẽ tăng nhanh hơn sai số cVAE (dữ liệu cần tăng cấp số nhân theo số chiều để giữ cùng độ dày, cVAE thì không). Đo lại đúng phép so sánh retrieval-vs-cVAE (`analysis/scripts/baseline_comparison.py`) ở không gian điều kiện mở rộng - đây là luận điểm trung tâm để trả lời "vì sao cần inverse design" thay vì tra bảng, quan trọng hơn cả thí nghiệm sign-flip OOD đã có | **[KẾT LUẬN 2026-08-20, sau 4 thí nghiệm liên tiếp cùng ngày, `EXPERIMENT_LOG.md`] Ở 8D in-distribution: cVAE KHÔNG thắng rõ retrieval trên bất kỳ trục nào đã đo.** Mật độ: khoảng cách retrieval nhảy +992% (bằng chứng thật). Accuracy: sau khi loại confound checkpoint chưa tune (`--select-by fe_r2` đưa cVAE R² 0,943→0,997), 2 bên gần hòa (0,997 vs 0,998). Manufacturability: retrieval THẮNG rõ (0,889 vs 0,667 manuf_score, 83,3% vs 26,3% passes-all) - ngược hoàn toàn giả thuyết "cVAE bảo toàn cấu trúc tốt hơn". Low-data: retrieval KHÔNG bế tắc khi thu nhỏ tập tra cứu (R²=0,984 dù chỉ còn 500/68.286 mẫu) - loại bỏ luôn hướng "low-data regime" mà không cần train lại cVAE (tiết kiệm ~1,5-3 giờ GPU dự kiến). **Quyết định: dừng đầu tư thêm vào Nhóm 4.2 ở dạng in-distribution 8D - luận điểm trung tâm thật sự vững của dự án là Nhóm 4.1 (sign-flip OOD, đã xong từ 2026-08-04), không phải Nhóm 4.2.** Checkpoint cuối: `cvae_a4_full8d_finetuned_v2.pt` |
| 4.3 | (Tùy chọn) Thiết kế **hybrid retrieval-gated generative**: router theo khoảng cách (retrieval khi điều kiện nằm trong vùng dày, cVAE khi ngoài vùng dày/OOD) + latent warm-start (encode sample retrieval gần nhất thành z0 thay vì sample ngẫu nhiên từ prior) + gradient refinement khả vi qua `real_physics.py`. Biến câu hỏi "ai thắng ai" thành đóng góp hệ thống - kết hợp đúng chỗ tốt hơn dùng riêng lẻ, tránh phải tuyên bố cVAE "thắng" retrieval một cách gượng ép | Tái dùng `baseline_comparison.py` + `best_of_n_eval.py` + `real_physics.py`; rủi ro: warm-start "lười" có thể suy biến thành retrieval trá hình - theo dõi bằng `novelty_diversity_eval.py` |

### Nhóm 5 - Giai đoạn D (động tuyến tính): hoãn, effort thực tế L-XL không phải "thấp" ⏸ NGOÀI PHẠM VI, giữ nguyên trạng thái hoãn

Đã xác nhận qua code: **không tồn tại** bất kỳ hạ tầng mass-matrix/eigenvalue/Bloch-Floquet nào (`grep` toàn repo cho `eigsh|frequency|bandgap|omega` → 0 kết quả vật lý thật). `simp/core/pbc.py` chỉ là projection tĩnh thực (real-valued) cho 3 load case biến dạng đơn vị - band-gap cần PBC **phức** (Bloch phase `e^{ik·L}`) và sweep qua Brillouin zone, gần như viết lại từ đầu chứ không phải "mở rộng".

| # | Việc | File | Effort |
|---|---|---|---|
| D0 | Xác nhận với GS liệu D có cần cho luận văn hiện tại hay để đề tài con - áp dụng logic tương tự Giai đoạn E | - | - |
| D1 | Lắp ma trận khối lượng M từ trường mật độ ρ | `simp/core/` | M |
| D2 | Giải trị riêng suy rộng `Kφ=ω²Mφ`; xử lý spurious low-frequency mode do `Emin=1e-9` ở vùng gần-rỗng (vấn đề kinh điển trong eigenfrequency TO - cần mode-tracking/MAC criterion) | `simp/core/` | M-L |
| D3 | Band-gap qua Bloch-Floquet - viết PBC phức mới, sweep Brillouin zone | `simp/core/pbc.py` (viết mới phần lớn) | L |
| D4 | Nếu đưa tần số/band-gap làm condition cVAE: kéo theo retrain Phase 4 + Phase 5 đầy đủ (effort ≈ Nhóm 1) | `pipeline/phase4_surrogate/`, `pipeline/phase5_cvae/` | L |

### Giai đoạn E - Động phi tuyến (nổ/va đập): vẫn ngoài phạm vi

Cần solver FE động tường minh hoàn toàn mới, không tái dùng `simp/core/`. Không đưa vào roadmap này trừ khi GS yêu cầu chính thức và đồng ý tách thành đề tài con.

### Nhóm 6 - Giai đoạn F: Nhiệt (CTE) ⏸ SAU BÀI BÁO #1 - đề xuất tách thành bài báo #2 riêng (audit 2026-08-19, sửa quyết định "ưu tiên cùng cấp Nhóm 1" của 2026-08-14 - xem mục 3)

**Quyết định phạm vi (2026-08-14):** trong 3 domain vật lý mới (động tuyến tính, áp điện, nhiệt),
chọn **vật liệu nền + nhiệt** làm ưu tiên - cùng họ toán homogenization đã có, effort M-L, có sẵn
công thức tham khảo trực tiếp trong literature review (Nhóm 2: Sigmund & Torquato 1997 - 3-phase TO
cho CTE cực trị; Guo et al. 2024 - NTE bằng energy-based homogenization, cùng kỹ thuật PBC dự án
đang dùng). **Áp điện bị loại khỏi roadmap này** - cần constitutive model điện-cơ ghép đôi hoàn
toàn mới, effort nặng hơn cả Giai đoạn D, không tái dùng được `simp/core/`. **Động tuyến tính (D)
giữ nguyên trạng thái hoãn** như Nhóm 5 đã phân tích.

| # | Việc | File | Effort |
|---|---|---|---|
| F0 | Đọc kỹ công thức energy-based homogenization cho CTE trong Guo et al. (2024) - xác nhận mức độ tái dùng được với `simp/homogenization/compute.py` (cùng PBC, khác phương trình cấu thành: cần thêm hệ số giãn nở nhiệt α vào vật liệu, giải thêm 1 load case "nhiệt đơn vị" bên cạnh 3 load case biến dạng đơn vị hiện có) | - | S |
| F1 | Thêm `alpha` (hệ số giãn nở nhiệt vật liệu nền) vào `Material` - cùng cơ chế với `nu`/`E0` ở A2 | `simp/materials/isotropic.py` | S |
| F2 | Mở rộng homogenization: giải thêm load case nhiệt, tính tensor CTE hiệu dụng α* (không chỉ Q) | `simp/homogenization/compute.py`, `simp/core/pbc.py` | M-L |
| F3 | Đối tượng tối ưu mới hoặc mở rộng `simp/objectives/auxetic.py`: CTE cực trị/âm, có thể đồng thời với auxetic (2 tính chất) - tham khảo trực tiếp cách Sigmund & Torquato ràng buộc 3-pha | `simp/objectives/` | M |
| F4 | Sinh dataset nhỏ kiểm tra hội tụ trước khi tích hợp vào Phase 1-3 đầy đủ | `pipeline/phase1_screening/` | M |
| F5 | Thêm α (CTE) làm output/condition cho surrogate + cVAE - lặp lại đúng pattern đã dùng cho f1/f2 (Nhóm 2) | `pipeline/phase4_surrogate/`, `pipeline/phase5_cvae/` | M |

**Phụ thuộc với Nhóm 1:** F1 dùng chung cơ chế mở rộng `Material` với A2 - nên làm A2+F1 cùng lúc
(cả hai đều là "thêm 1 thuộc tính vật liệu nền"), tránh sửa `isotropic.py` 2 lần riêng biệt.

### Nhóm 7 - Bổ sung tầng sampling/inference cho cVAE (KHÔNG đổi kiến trúc VAE) + nâng cấp Phase 1/4 (quyết định 2026-08-19, sau vòng research 2 đợt) ⏸ SAU BÀI BÁO #1, trừ 7.1/7.4 (tùy chọn, xem mục 3)

**Ràng buộc phạm vi (quyết định của GS/người dùng):** không thay thế mạng cVAE bằng kiến trúc khác (diffusion/flow đã cân nhắc và loại - xem ghi chú research bên dưới) - chỉ (a) bổ sung cơ chế xung quanh cVAE đã train (`cvae_realphysics.pt`/`cvae_a4_nu0_finetuned.pt`) ở tầng sampling/inference, hoặc (b) thay thế các thành phần KHÁC trong pipeline (surrogate Phase 4, chiến lược sinh dữ liệu Phase 1-2) mà không đụng kiến trúc VAE.

**Đã cân nhắc và loại (research 2026-08-19, đợt 1):** physics-guided diffusion model thay cVAE (`arxiv 2603.16209`, `2401.13570`), TopoDiff-style conditional diffusion (giảm 8x sai số compliance, 11x mẫu không chế tạo được so với VAE/GAN theo `arxiv 2208.09591`) - kỹ thuật mạnh, tận dụng đúng `real_physics.py` (A5) làm guidance signal, nhưng bị loại vì phạm vi đợt này giữ nguyên VAE. Cân nhắc lại sau nếu Nhóm 7 không đủ thu hẹp gap retrieval-vs-generative (Nhóm 4.1).

| # | Việc | File | Effort | Vấn đề giải quyết |
|---|---|---|---|---|
| 7.1 | CFG guidance-weight extrapolation lúc sample: `sample.py` hiện CHỈ dùng condition-dropout lúc train (A6, `train.py::apply_condition_dropout`) để decoder học xử lý condition bị mask, nhưng `sample.py` (`--v12/--v21/--volfrac/--nu0`, dòng 87-102) chưa từng khai thác cơ chế này lúc inference - decode 2 lần (condition thật + condition null) rồi extrapolate `out = decode(z,cond_null) + w·(decode(z,cond) − decode(z,cond_null))`, w>1 khuếch đại độ bám điều kiện (đánh đổi với đa dạng mẫu - xem `arxiv 2607.19725` về guidance-weight schedule thay vì hằng số) | `pipeline/phase5_cvae/sample.py` | S | tận dụng hạ tầng A6 (condition-dropout) hiện chưa dùng hết ở tầng sample |
| 7.2 | Gradient-refine z bằng `real_physics_loss` trước khi chấm composite score: `best_of_n_eval.py` hiện sample N vector z NGẪU NHIÊN rồi chọn tốt nhất - bỏ phí gradient có sẵn từ `real_physics.py` (A5). Thêm vài bước gradient descent trên z (decoder ĐÓNG BĂNG, không train lại) tối thiểu hóa `real_physics_loss(decode(z), target)` trước khi đưa vào composite scoring - biến "random best-of-N" thành "gradient-refined best-of-N", tái dùng 100% checkpoint + hàm loss đã có (pattern giống GB-FESO, `arxiv 2607.06421`; và Liu et al. tối ưu latent VAE bằng evolutionary cho metamaterial) | `pipeline/phase5_cvae/best_of_n_eval.py` | M | random search lãng phí gradient real-physics đã có sẵn từ A5 |
| 7.3 | CMA-ES polish cho phần KHÔNG khả vi của composite score (manuf check rời rạc, aesthetic score) - chạy sau 7.2 (gradient lo phần accuracy khả vi, CMA-ES lo 2 phần còn lại). Rủi ro: CMA-ES suy giảm hiệu quả ở latent_dim cao (`model.py:54` mặc định `latent_dim=32` - biên giới chấp nhận được theo literature, cần đo thời gian hội tụ thật trước khi coi là khả thi) | `pipeline/phase5_cvae/best_of_n_eval.py` | M | 2/3 thành phần composite score không tối ưu được bằng gradient thuần |
| 7.4 | Periodic padding (CircularPad) thay zero-padding mặc định cho CNN surrogate (`phase4_surrogate/model.py`) VÀ encoder cVAE (`phase5_cvae/model.py`) - đúng vật lý PBC đã dùng ở tầng FE (`simp/core/pbc.py`) nhưng CHƯA áp ở tầng conv, rẻ, không đổi kiến trúc tổng thể (chỉ đổi padding_mode) | `pipeline/phase4_surrogate/model.py`, `pipeline/phase5_cvae/model.py` | S-M | conv hiện không tôn trọng đúng biên tuần hoàn của ô đơn vị |
| 7.5 | Equivariant/group-equivariant CNN (G-CNN, tôn trọng đối xứng D4 xoay/phản chiếu ô vuông) thay CNN surrogate Phase 4 - literature: "equivariance is a powerful remedy for data scarcity", trực tiếp giải quyết phát hiện A4 (`EXPERIMENT_LOG.md` 2026-08-18): thêm ν0 KHÔNG cải thiện R² đo được vì mẫu ν0 biến thiên chỉ ~16% dataset - equivariance nhân hiệu quả dữ liệu mà không cần sinh thêm mẫu FE. Cũng chuẩn bị cho Nhóm 6 (CTE) - domain mới nào cũng sẽ khan hiếm dữ liệu ban đầu | `pipeline/phase4_surrogate/model.py` | L | data-scarcity gốc rễ của A4, tái diễn ở mọi domain vật lý mới |
| 7.6 | Ensemble surrogate (nhiều CNN Phase 4 train độc lập) → epistemic uncertainty → dùng làm acquisition function cho active learning ở Phase 1-2, thay LHS/adaptive-DOE thuần túy hiện tại - giảm chi phí sinh dữ liệu (hiện 3000 mẫu/4122s cho A4, sẽ tăng khi mở thêm chiều điều kiện ở Nhóm 6) | `pipeline/phase1_screening/`, `pipeline/phase2_multi_batch/` | M | chi phí FE-solve tăng theo số chiều điều kiện khi mở rộng Nhóm 6 trở đi |
| 7.7 | **Tự động chọn giá trị tối ưu cho điều kiện bị thiếu** (đặc tả tính năng, quyết định 2026-08-19): hiện tại bỏ trống 1 trường ở `sample.py` (`--nu0`/`--volfrac`/`--void-size-frac` mặc định `None`) → mask=0 → decoder sinh KHÔNG ràng buộc trường đó (ngầm định, không có gì đảm bảo "tốt"); ở tầng loss lúc train, mask=0 lại fallback về hằng số cố định (`losses.py:123`, nu0→0.3, KHÔNG liên quan tới sample-time). Hành vi mới: khi user không truyền 1 trường, coi giá trị đó là 1 biến tự do BỔ SUNG (cùng z) để tối ưu bằng đúng cơ chế 7.2/7.3 - gradient descent (qua `real_physics_loss`, trong khoảng hợp lệ của `PARAM_SPACE`) và/hoặc CMA-ES cho phần không khả vi - sao cho composite score tốt nhất, với ĐIỀU KIỆN các trường ĐÃ truyền (vd `--v12`/`--v21`) vẫn được giữ cố định làm target. Nói cách khác: "thiếu 1 điều kiện" không còn nghĩa là "bỏ qua nó" mà là "tìm giá trị của nó sao cho tối ưu, có ràng buộc bởi các điều kiện còn lại đã cho" | `pipeline/phase5_cvae/sample.py`, `best_of_n_eval.py` | M - phụ thuộc 7.2/7.3 xong trước (tái dùng đúng cơ chế tối ưu, chỉ mở rộng biến tối ưu từ `z` sang `(z, condition_thiếu)`) | thay behavior "không ràng buộc = bất kỳ giá trị nào" bằng "không ràng buộc = giá trị TỐT NHẤT" |

**Thứ tự làm:** 7.1 và 7.4 trước (rẻ, không cần retrain lớn, đo ngay bằng `best_of_n_eval.py` sẵn có) → 7.2/7.3 (xây cơ chế tối ưu z + condition tự do, nền tảng cho 7.7) → 7.7 (tính năng cuối, tái dùng 7.2/7.3) → 7.5/7.6 độc lập, có thể làm song song bất cứ lúc nào vì không phụ thuộc 7.1-7.3/7.7.

**Rủi ro cần theo dõi riêng cho Nhóm 7:** 7.7 tối ưu điều kiện thiếu có thể trôi ra ngoài vùng dữ liệu đã train (vd tối ưu ν0 chọn giá trị hiếm gặp trong 16% mẫu ν0 biến thiên của A4) - cần clip về `PARAM_SPACE` VÀ log rõ khi giá trị tối ưu tìm được nằm ở biên (dấu hiệu ngoại suy, độ tin cậy composite score thấp hơn) thay vì báo cáo như kết quả trong-phân-phối bình thường.

## 5. Rủi ro cần theo dõi

- **A3** có thể phá hội tụ FE nếu ν0 âm hoặc gần biên `(-1, 0.5)` - bắt đầu dải hẹp, mở rộng dần.
- **A5** là điểm rủi ro kỹ thuật lớn nhất của Nhóm 1 - mất tác dụng tăng tốc của mesh/PBC-cache có thể làm chậm đáng kể vòng lặp differentiable-physics training.
- **A4 trước A6** là ràng buộc thứ tự bắt buộc - bỏ qua sẽ khiến mọi số liệu R² dựa trên surrogate cũ sai lệch có hệ thống.
- **Nhóm 4 là rủi ro số 1 của TOÀN BỘ kế hoạch, không chỉ 1 mục phụ** (nâng cấp mức độ nghiêm trọng ở audit 2026-08-19 - xem mục 3) - nếu bài báo #1 tập trung vào A∩B∩C mà chưa xử lý câu hỏi "vì sao không dùng retrieval", đây là lỗ hổng lập luận reviewer sẽ chỉ ra ngay ở vòng đầu, nghiêm trọng hơn gap A∩B∩C chính. KHÔNG viết bài báo #1 trước khi 4.1/4.2 có kết quả.
- **Nhóm 6/7 làm song song Nhóm 4 (thay vì sau)** là rủi ro tiến độ mới (audit 2026-08-19) - dễ khiến bài báo #1 bị trì hoãn vô thời hạn vì cứ có việc "thú vị" mới chen vào trước khi đóng lỗ hổng 4.1/4.2 - xem ràng buộc thứ tự ở mục 3.
- **Nhóm 5 (D)** effort thực tế gần bằng cả Nhóm 1 nếu D4 được chọn - không nên coi là "mở rộng nhẹ" khi trình bày với GS.
- **Nhóm 6 (F)** F2 là bước rủi ro nhất - cần xác nhận công thức CTE homogenization của Guo et al. (2024) áp dụng đúng với cách project tính Q trước khi code, tránh vừa viết vừa suy diễn công thức sai (bài học từ lỗi hoán vị `dQ` từng gặp, LIMITATIONS.md mục 10).

## 6. Ước tính độ phức tạp, khả thi, ETA, khả năng ra báo

ETA tính theo person-week giả định làm bán thời gian (song song với việc học/nghiên cứu khác),
**không** tính thời gian chờ GS duyệt hoặc review. Đây là ước tính, không phải cam kết - dùng để
sắp xếp thứ tự ưu tiên, không dùng để báo cáo tiến độ cứng.

| Nhóm | Độ phức tạp | Khả thi (trong khung luận văn hiện tại) | ETA | Khả năng ra báo/đóng góp khoa học |
|---|---|---|---|---|
| 0 - Dọn tài liệu | Rất thấp (S) | Cao | ~1 buổi | Không - chỉ vệ sinh tài liệu |
| 1 (A) - Vary vật liệu nền | ✅ **HOÀN THÀNH ĐẦY ĐỦ (2026-08-19)** - tất cả 8 sub-task (trừ A0 không cần deliverable riêng), hạ tầng + lợi ích thật đã đo | Đã đạt, hạ tầng thông suốt Phase 1→5 + kết quả đã kiểm chứng | Đã xong hoàn toàn - R²(FE,n=300) 0,9776→0,9843, frac_manufacturable 0,312→0,365 khi thêm ν0 (`EXPERIMENT_LOG.md` 2026-08-19) | **Cao, đã hiện thực hóa** - mảnh lấp gap A∩B∩C, thành phần trung tâm của đóng góp, sẵn sàng đưa vào luận văn như 1 kết quả đã kiểm chứng |
| 2 - f1/f2 vào Phase 5 | Thấp (S-M) | Cao - chỉ lặp lại pattern `extended_condition` đã có | 3–5 ngày | Bổ trợ - làm đầy đủ hơn output, không tự thành đóng góp riêng |
| 3 - Validate xếp hạng đa mục tiêu | Thấp (S) | Cao - số liệu FE baseline đã có sẵn | ~1 tuần (chờ chạy Pareto đối chiếu) | Bổ trợ - củng cố phần B đã có, không phải kết quả mới |
| 4.1–4.2 - Củng cố lập luận retrieval-vs-generative | Thấp-Trung bình (S, nhưng 4.2 phụ thuộc Nhóm 1 xong) | Cao cho 4.1 (dùng lại thiết kế OOD sẵn có); 4.2 khả thi trung bình vì cần chờ Giai đoạn A | 4.1: ~1 tuần độc lập; 4.2: ~1 tuần sau khi Nhóm 1 xong | **Bắt buộc phải có** - không tự thành 1 kết quả để "khoe", nhưng thiếu nó thì toàn bộ lập luận vì sao dùng generative dễ bị phản biện bác |
| 4.3 - Hybrid retrieval-gated generative (tùy chọn) | Trung bình-Cao (M-L) | Trung bình - phụ thuộc 4.1/4.2 xong và Nhóm 1 ổn định | 2–3 tuần nếu làm | Cao nếu thành công - biến câu hỏi thắng/thua thành 1 đóng góp phương pháp riêng biệt, nhưng là hạng mục rủi ro-cao/lợi-ích-cao, không phải việc bắt buộc |
| 5 (D) - Động tuyến tính | Rất cao (L-XL) | Thấp trong khung thời gian hiện tại - chưa có hạ tầng mass-matrix/eigenvalue nào, D0 cần GS quyết định trước | 6–10+ tuần nếu làm đủ D0-D4 | Cao nếu làm trọn (auxetic + band-gap ít người kết hợp), nhưng effort/rủi ro vượt xa lợi ích trong khung luận văn - **đang hoãn**, không đưa vào ETA tổng |
| E - Động phi tuyến (nổ/va đập) | Rất cao, cần solver mới hoàn toàn | Rất thấp - không tái dùng được `simp/core/` | Không ước tính (ngoài phạm vi) | Ngoài phạm vi trừ khi GS yêu cầu tách đề tài con |
| 6 (F) - Nhiệt (CTE) | Trung bình-Cao (M-L) | Trung bình-Cao - F2 (verify công thức homogenization) là rủi ro chính, còn lại tái dùng hạ tầng sẵn có | 3–5 tuần | **Cao** - mở rộng multi-physics (auxetic + CTE) tự nhiên trên cùng nền A∩B∩C, cùng họ toán đã kiểm chứng |
| 7 - Bổ sung sampling/inference cVAE + surrogate/data-gen (KHÔNG đổi kiến trúc VAE) | Thấp-Trung bình (S-L tùy sub-task) | Cao cho 7.1/7.4 (rẻ, độc lập); trung bình cho 7.2/7.3/7.7 (chuỗi phụ thuộc); trung bình cho 7.5/7.6 (đổi kiến trúc surrogate/data pipeline, không đổi VAE) | 7.1/7.4: vài ngày; 7.2/7.3/7.7: 1–2 tuần chuỗi; 7.5/7.6: 2–3 tuần mỗi việc | Bổ trợ trực tiếp Nhóm 4.1 (nếu thu hẹp đủ gap retrieval-vs-generative bằng gradient-refine thay vì đổi kiến trúc) và trực tiếp giải quyết data-scarcity gốc rễ của A4/Nhóm 6 (7.5) |

**Thứ tự ưu tiên CHỐT (audit 2026-08-19 - thay thế đoạn ưu tiên cũ bên dưới đã lỗi thời vì xếp Nhóm
6/7 ngang hàng Nhóm 4, xem phân tích đầy đủ ở mục 3):**

1. **Nhóm 4.1 → 4.2** (blocking, không có lựa chọn khác - đây là điều kiện CẦN để bài báo #1 đứng
   vững trước phản biện, không phải 1 hạng mục ưu tiên cao trong nhiều hạng mục).
2. **Nhóm 3** (3.1-3.2) chen song song 4.1/4.2 - rẻ, không cạnh tranh nguồn lực.
3. Viết bài báo #1 (Nhóm 1 + 3 + 4.1/4.2) - có thể gộp thêm Nhóm 7.1/7.4 làm ablation NẾU rảnh, không
   bắt buộc và không được phép làm trễ bước 1-2.
4. **Chỉ sau khi bài báo #1 ổn định** (bản thảo gần hoàn chỉnh hoặc đã nộp): mở lại Nhóm 2, Nhóm 6
   (đề xuất tách bài báo #2 riêng - không nhồi chung), Nhóm 7.2/7.3/7.5/7.6/7.7, và cân nhắc Nhóm 4.3
   nếu 4.1/4.2 cần củng cố thêm bằng 1 đóng góp hệ thống.
5. Nhóm 5 (D) và Giai đoạn E: giữ nguyên trạng thái hoãn/ngoài phạm vi, không xét lại trừ khi GS yêu
   cầu tách đề tài con.

**Sai lầm cần tránh (lý do audit này tồn tại):** trước 2026-08-19, roadmap đã đúng về mặt liệt kê việc
cần làm nhưng SAI về thứ tự - xếp Nhóm 6 (nhiệt) "ưu tiên cùng cấp Nhóm 1" (quyết định 2026-08-14) và
thêm Nhóm 7 (2026-08-19) mà chưa đóng Nhóm 4, trong khi chính tài liệu đã tự nhận Nhóm 4 "rủi ro lập
luận lớn hơn cả gap A∩B∩C nếu bỏ qua" (mục 5). Bài học: 1 quyết định phạm vi đúng lúc ghi ra vẫn có
thể bị lấn át bởi các hạng mục "thú vị" thêm sau nếu không có 1 lần audit thứ tự thực thi riêng biệt.

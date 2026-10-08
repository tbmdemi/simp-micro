# Phạm vi Claim Khoa học & Giới hạn Đã biết

> Trạng thái tại **2026-10-07** (sau P1.9, pipeline cuối KHÔNG `force_periodic`). Số mục (#N) giữ
> nguyên làm ID để các tài liệu khác trỏ tới; mục đã giải quyết nằm ở bảng cuối. Chi tiết từng thí
> nghiệm: [`plan.md`](plan.md) mục 3-4, [`EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md). Mọi số liệu
> dưới đây truy được về `outputs/phase5/plan_v3/` (bảng nguồn: [`paper1/README.md`](paper1/README.md)).

## Phạm vi Claim Khoa học (đọc trước khi trích dẫn)

**ĐƯỢC ủng hộ bởi bằng chứng:**
1. Engine FE/homogenization đúng vật lý: khớp một implementation `scikit-fem` độc lập tới ~1e-9 trên
   24 thiết kế (#15).
2. **Refine không gian ẩn nhận thức nhị phân hóa** (objective tính trên đúng thiết kế được verify):
   R²(ν₁₂) mẫu đơn 0,874 → 0,998 (IN100), 0,831 → 0,990 (IN100-B); sau best-of-30 0,983-0,990 →
   0,995-0,998 trên 3 tập mục tiêu rời nhau (#29).
3. Khoảng lệch objective ↔ verify bị khai thác bởi CẢ refine liên tục của cVAE lẫn SIMP có Heaviside
   β=64 (#29, #32) - luận điểm trung tâm của Bài #1.
4. cVAE + refine **ngang** SIMP chạy từ đầu (hội tụ, 4 khởi tạo) về độ chính xác trong phân phối
   (R²(ν₁₂) 0,995-0,998 cả 2 phương pháp, 3 tập) với ~8× ít FE (107 vs ~893/mục tiêu); với mục tiêu
   dị hướng vừa phải (r < 10) sai số cặp thấp hơn - kiểm định đăng ký trước trên IN100-C: thấp
   hơn 27% [8; 42] (#32, #37).
5. Trên trục đổi dấu ngoài phân phối (train chỉ có ν₁₂ < 0), cVAE sinh được ν₁₂ > 0, điều retrieval
   từ thư viện không thể làm (#19).
6. Thiết kế sinh ra không phải bản sao gần của dữ liệu train (0/480, #18).

**CHƯA được ủng hộ (không claim):**
- ν₁₂ chính xác hơn SIMP (tiêu chí đăng ký trước không đạt; chênh ~0,001 nhỏ hơn sai số lưới) (#32, #33, #37).
- Vượt SIMP về khả năng chế tạo, OOD đổi dấu, dị hướng cực đoan, hoặc chi phí cho 1 mục tiêu (#2, #19, #32, #37).
- Ngoại suy biên độ auxetic vượt biên train (bão hòa từ ν₁₂* ≈ −2,25) (#29).
- Độ chính xác tốt hơn sai số rời rạc hóa của lưới 50² (#33).
- KAN head / WIRE decoder tốt hơn baseline tích chập + đầu tuyến tính (#30).
- Lợi thế của cVAE so với retrieval trong phân phối khi số chiều điều kiện tăng (8D) (#23).
- `hit_rate` như bằng chứng độc lập (#17); proxy Q₁₂ tương đương mục tiêu auxetic dưới mọi phép xoay.

**Không bao giờ claim:** "đã giải quyết inverse design", "cVAE tốt hơn phương pháp kinh điển trong
mọi trường hợp", "surrogate thay được FE ở vùng chưa kiểm chứng".

---

## Giới hạn còn hiệu lực

### A. Phạm vi vật lý và mô hình

14. **FEM tuyến tính, biến dạng nhỏ, ô 2D, ν₀ = 0,3.** Auxetic thiết kế ở biến dạng nhỏ có thể mất ν âm
    khi biến dạng lớn; chưa có 3D, chưa in 3D/thử nghiệm.
15. **Chưa đối chiếu với FE thương mại** (Abaqus/ANSYS/COMSOL). Đã đối chiếu `scikit-fem` độc lập
    (mean |Δν₁₂| 3,7e-9, 24 mẫu) nên engine đúng vật lý, nhưng skfem ít được dùng làm chuẩn tham chiếu.
33. **Độ chính xác là so với mô hình FE 50×50**, không phải ν hội tụ lưới. Cùng hình học, 50² lệch
    200² trung vị 0,015 (3,8%), tối đa 0,042, lưới mịn âm hơn ở 27/30 thiết kế - lớn hơn MAE ~0,005
    đang báo. Xếp hạng cVAE/SIMP giữ nguyên trên 200² (R²(ν₁₂) 0,992 vs 0,985).
31. **Tính chất suy từ Q chỉ là đàn hồi tĩnh.** E/B/M/tốc độ sóng dọc dự đoán được 92-97% từ 5 điều
    kiện hiện có (f1/f2 vì vậy ít giá trị làm condition); chỉ G_xy mang thông tin mới đáng kể (R² 0,84).
    Tốc độ sóng chỉ đúng ở bước sóng dài; ấn lõm chỉ là proxy Hertz 2D; nhiệt/thấm/band gap cần solver mới.
4. **Phạt `mu` trong mục tiêu auxetic tắt** (`mu=0`, `q12`) cho toàn bộ dataset; biến thể
    `normalized` đã thử và bị loại (yield giảm 60-76 điểm % do mất ổn định hội tụ).

### B. Dữ liệu

11. **Chỉ 23,7% lần chạy SIMP hội tụ thật theo tolerance** (76,3% chạm `max_iter`). Mẫu rời rạc đã
    được lọc (0/300 mẫu kiểm tra bị rời), nhưng nhãn của mẫu chưa hội tụ có thể chưa ổn định.
20. **Dataset 57k sinh trước một số sửa lỗi code sinh dữ liệu, chưa rebuild:** seed `reentrant_bowtie`
    từng nhận `void_size_frac` thay vì `volfrac`; tâm seed lệch nửa phần tử (mất đối xứng gương ~2%
    pixel); chưa có MMA dispatch theo seed; CSV Phase 1 từng ghi ν = 0,0 thành rỗng. Ảnh hưởng tới
    kết quả cuối chưa đánh giá lại (chỉ ảnh hưởng mật độ khởi tạo/compute, hoặc phân tích Phase 1).
37. **Dị hướng cực đoan thưa trong dữ liệu.** Mục tiêu r = max(ν₂₁/ν₁₂, ν₁₂/ν₂₁) ≥ 10 (≈ E_y/E_x) chỉ
    có ở 2,8% mẫu train; 3/5 mục tiêu như vậy qua 3 tập cho ô đứt (E ≈ 2e-11·E₀), cả 30/30 ứng viên
    đều đứt → lọc lúc chọn không cứu được; SIMP giải được các mục tiêu này.
18. **Nhãn dataset lệch ν verify của chính ảnh đó**: 0,019 (khứ hồi 64→50) đến 0,032 (sau nhị phân hóa).
    Retrieval verify trung thực trên IN100: R²(ν₁₂) 0,941 (1 mẫu), 0,977 (best-of-30) - thấp hơn
    cVAE best-of-30 (0,988) và cVAE + refine (0,998). Con số "retrieval R² = 1,000" cũ đo trên nhãn.

### C. Phương pháp - kết quả âm và giới hạn

29. **Refine nhận thức nhị phân hóa:** (a) ngoài dải train: tại ν₁₂* = −2,0 sai số trung vị 1,2% nhưng
    phân phối 2 cụm (6/10 lần ≤ 1,7%, 4/10 lần 17-31%); từ −2,25 bão hòa (14-29%) → không claim ngoại
    suy; (b) R²(ν₂₁) thấp hơn nhiều R²(ν₁₂) ở IN100/IN100-B (0,75-0,88); (c) ~77 FE/mục tiêu; (d) refine
    liên tục (không nhận thức nhị phân hóa) làm TỆ best-of-30 (MAE +62 đến +74%).
32. **So với SIMP chạy từ đầu** (MMA, Heaviside β→64, 4 khởi tạo, cùng đường verify): ngang về độ chính
    xác, thua về chế tạo (SIMP 0,74-0,83 vs cVAE 0,32-0,35), OOD đổi dấu, dị hướng cực đoan. Tiêu chí
    đăng ký trước (IN100-C, pipeline có fp, r < 10): sai số cặp đạt, ν₁₂ không đạt (+19% [−10; 40]).
    Pipeline không fp cho ν₁₂ +37-39% ở cả 3 tập - chỉ là phân tích độ nhạy. Chi phí sinh dataset
    ≈ 2,7-3,0M FE → hòa vốn với SIMP sau ~3 500 mục tiêu.
36. **Lai cVAE → SIMP không đạt** tiêu chí đăng ký trước (2/4): khởi tạo SIMP từ thiết kế cVAE (β 8→64,
    ≤ 61 FE, tổng 166 FE) không thừa hưởng khả năng chế tạo (0,33 < 0,70) và không cứu được ô đứt.
19. **OOD đổi dấu: chỉ thắng retrieval, thua SIMP.** cVAE + refine đạt ν₁₂ > 0 nhưng MAE ν₁₂ 0,06-0,09
    vs SIMP 0,005-0,03 (28 mục tiêu). Mục tiêu OOD phải thỏa ν₁₂·ν₂₁ < 1: bộ "extreme auxetic" cũ
    (vd (−2,1; −2,1)) không tồn tại về vật lý và đã bị loại khỏi mọi kết luận.
30. **KAN và WIRE không cải thiện** khi đo bằng FE thật: ablation có kiểm soát (2 seed, IN100) - KAN
    không thắng chế độ nào, Linear thắng ở best-of-30, KAN tốn 3,2× tham số; WIRE: hit-rate 12,5%,
    R²(FE) −7,19 → đã đóng. "KAN vượt Linear ~2×" cũ là artifact của thước đo surrogate.
23. **8D trong phân phối (+ν₀, volfrac, void_size_frac): cVAE không thắng retrieval ở trục nào**
    (accuracy 0,997 vs 0,998; chế tạo 0,67 vs 0,89; retrieval vẫn R² 0,984 khi chỉ còn 500 mẫu). Chỉ có
    bằng chứng mật độ dữ liệu loãng (+992% khoảng cách láng giềng).

### D. Đánh giá và báo cáo

2. **Khả năng chế tạo thấp** (liên thông + nét tối thiểu 2 pixel): cVAE 0,22-0,35 qua các lần chạy,
    SIMP 0,74-0,83. Refine không thay đổi có hệ thống (−6 đến +9 điểm %, gộp 61 thêm / 44 mất, p = 0,12).
34. **`force_periodic` đã bỏ (quyết định C5, 2026-10-07):** lưới FE dựa trên phần tử + PBC nên mọi ảnh
    pixel đều lát được; kiểm tra "cạnh khớp" chỉ đạt vì fp ép nó. Bài báo cả 2 phiên bản số liệu;
    kết quả đăng ký trước (P1.8) giữ nguyên phiên bản có fp.
38. **Quy tắc chọn ứng viên đổi R²:** best-of-30 n=300 chọn thuần độ chính xác R² 0,995-0,997, chọn
    composite 0,6/0,3/0,1 R² 0,983-0,988. Mọi bảng phải ghi quy tắc chọn.
25. **Composite score khớp Pareto ở mức trung bình:** Spearman TB 0,693 (có fp, trượt ngưỡng 0,7 đặt
    trước), 0,718 không fp; trọng số là lựa chọn thiết kế; điểm thẩm mỹ chưa kiểm chứng với người.
17. **`hit_rate` (đúng dấu ν₁₂) là metric yếu:** ~92% dataset vốn auxetic, baseline ngẫu nhiên cũng đạt
    1,000 ở n=24 - luôn đọc kèm R²/MAE và CI.

### E. Hạ tầng

6. **Test chưa phủ I/O nặng:** vòng lặp `screening_parallel.py`, `pipeline/seeds/*.py`, `visualize.py`,
    lời gọi FE thật trong `multi_batch/runner.py::evaluate_single` (đang mock).
22. **Self-play round-trip đầy đủ chưa chạy được với checkpoint điều kiện mở rộng** (bước train lại qua
    subprocess chưa truyền `--data-dir`/`--extended-condition`/`--include-nu0`; verify 1 checkpoint thì đã sửa).
8. **Tài liệu chủ yếu tiếng Việt**; chỉ file này có bản tiếng Anh (rút gọn, bên dưới).

---

## Đã giải quyết (bài học giữ lại; chi tiết ở `EXPERIMENT_LOG.md` theo ngày)

| # | Ngày | Vấn đề → cách sửa / bài học |
|---|---|---|
| 1 | 07-24 | Số Phase 5 trên n=24 có CI rộng → luôn báo n ≥ 100 + bootstrap CI ghép cặp |
| 3, 9 | 07-24 | Single-shot không đáng tin do surrogate exploitation → fine-tune bằng FE khả vi (differentiable physics) |
| 10 | 07-24 | Hoán vị `dQ` trong homogenization làm 3 seed bất đối xứng hội tụ sai → sửa, rebuild 2 160 mẫu |
| 12 | 07-24 | Chọn checkpoint theo `val_loss` bị KL-warmup đánh lừa → luôn `--select-by fe_r2` |
| 13 | 07-24 | 33,6% nhãn dao động limit-cycle → lọc `osc_score` |
| 16 | 07-30 | Yield seed `hexagonal`/`hourglass` thấp với OC → dispatch MMA theo seed (`reentrant_bowtie` giữ OC) |
| 20-21 | 08-15 | 2 vòng audit code: 8 bug (crash/reproducibility/validate ở biên, tham số seed, tâm seed) → sửa + 55 test; phần chưa lan sang dataset xem #20 ở trên |
| 24 | 08-20 | `is_pareto_efficient()` luôn loại phần tử cuối → viết lại O(n²) + test |
| 28 | 09-25 | `load_cvae()` bỏ qua `use_kan`/`enforce_symmetry` → mọi đánh giá qua loader này 09-11 → 09-25 sai; dùng `cvae_kwargs_from_checkpoint()`; train mới luôn `--disable-symmetry` |
| 35 | 10-07 | "Refine làm giảm chế tạo 30% → 24%" chỉ là 1 lần chạy → gộp vào #2 |

Mục đã gộp hoặc bỏ: #5 (f1/f2) → #31; #26, #27 (8D) → #23; #7 (dashboard HTML cũ, đã xóa khỏi repo) bỏ.

---

## Scientific claim scope & known limitations (English summary)

**Supported:** (1) FE engine matches an independent `scikit-fem` implementation to ~1e-9 (#15).
(2) Binarization-aware latent refinement raises verified R²(ν₁₂) from 0.874 to 0.998 (single sample)
and to 0.995-0.998 after best-of-30 on three disjoint target sets (#29). (3) Both continuous latent
refinement and SIMP with Heaviside β=64 exploit the objective-verification gap (#29, #32). (4) The cVAE
pipeline matches converged multi-start SIMP accuracy with ~8× fewer FE solves, and has a lower joint
error for moderate anisotropy (r < 10; pre-registered IN100-C: 27% lower [8; 42]) (#32, #37). (5) It reaches
the unseen positive sign of ν₁₂, which retrieval cannot (#19). (6) Generated cells are not near-copies
of training data (#18).

**Not supported:** ν₁₂ more accurate than SIMP; better than SIMP in manufacturability, sign-flip OOD,
extreme anisotropy or single-target cost; magnitude extrapolation beyond ν₁₂* ≈ −2.25; accuracy beyond
the 50² mesh error; KAN/WIRE gains; an 8D in-distribution advantage over retrieval; `hit_rate` alone.

**Active limitations:** linear-elastic small-strain 2D cells with ν₀ = 0.3, no experiments, no commercial
FE check (#14, #15); accuracy is relative to the 50×50 FE model whose discretization error (median 0.015)
exceeds the reported MAE (#33); only 23.7% of SIMP data runs converged by tolerance and the dataset
predates several generator fixes (#11, #20); extremely anisotropic targets (r ≥ 10, 2.8% of data) yield
disconnected cells (#37); OOD magnitude saturates and is bimodal at −2.0 (#29); SIMP wins
manufacturability (0.74-0.83 vs 0.22-0.35) and sign-flip OOD (#2, #19, #32); the hybrid cVAE→SIMP failed
its pre-registered criteria (#36); KAN/WIRE gave no gain (#30); best-of-N R² depends on the selection
rule (#38); composite score agrees only moderately with Pareto ranking (#25); `force_periodic` was
dropped and both versions of the numbers are reported (#34).

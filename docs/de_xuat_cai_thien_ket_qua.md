# Đề xuất cải thiện kết quả (N2) - tra cứu + kế hoạch, CHƯA CHẠY

> 2026-10-10. Mục tiêu do tác giả đặt: **kết quả tốt hơn** (trùng phương pháp đã có vẫn được).
> Tài liệu này tổng hợp tra cứu, chẩn đoán điểm yếu, và đề xuất thí nghiệm kèm tiêu chí ghi trước.
> Chưa chạy thí nghiệm nào trong tài liệu này; chờ tác giả duyệt.
> Ràng buộc máy: RAM 14 GB, dùng ≤ 4 tiến trình CPU (tác giả yêu cầu 2026-10-10).

---

## 1. Điểm yếu hiện tại cần cải thiện (số đã đo)

| # | Điểm yếu | Số hiện tại | Nguồn |
|---|---|---|---|
| W1 | Sai số thật lớn hơn sai số báo | MAE ν12 = 0,0046 trên lưới verify 50², nhưng 0,0149 trên lưới mịn (128²), 0,0164 trên hiện thực hóa làm mượt 200² | N1 bước 1 |
| W2 | Refine phần lớn "khớp lưới" | refine giảm MAE 68% trên 50², chỉ 26% trên lưới mịn | N1 bước 1 |
| W3 | Chế tạo được thấp | cVAE 0,31 vs SIMP 0,72 (4 hướng); K1 lên 0,72 nhưng ν12 tệ hơn 37% (trên 50²) | K1 |
| W4 | Dị hướng cực đoan r ≥ 10 | 30/30 ứng viên đứt rời; dữ liệu train chỉ 2,8% r > 10 | P1.6, C6 |
| W5 | Ngoài dải / đổi dấu | bão hòa ở ν12 ≈ −1,95; SIMP thắng đổi dấu | P1.1d, P1.6b |
| W6 | Thiết kế kém bền với sai lệch hình học | σ = 1,0: MAE 0,090 (A) | N1 pilot |

**Phát hiện mới từ dữ liệu sẵn có (chưa chạy gì thêm):** chi phí độ chính xác của K1 **co lại một nửa**
khi chấm bằng hiện thực hóa làm mượt thay vì lưới 50²: +37% [CI +0,0003; +0,0033] → +19% [+0,0010;
+0,0054] (IN100, n = 100). Tức là một phần "đánh đổi" K1 là do thước đo 50² thưởng cho thiết kế mảnh.
Còn lại vẫn có ý nghĩa → K1 chưa miễn phí, nhưng rẻ hơn ta tưởng.

---

## 2. Tra cứu: các phương pháp liên quan và bài học rút ra

> Danh sách đầy đủ 47 tài liệu đã tham khảo (link, mức đọc, đã có trong `refs.bib` chưa, dùng cho phần
> nào): `docs/paper1/ghi_chu_viet_bao.md` mục D. Dự định tiếp theo: `docs/plan.md` mục N3.

### 2.1 Bền hình học + kích thước nét tối thiểu: robust formulation (co / gốc / giãn)
- Sigmund 2009; Wang, Lazarov & Sigmund 2011: tối ưu đồng thời 3 hiện thực hóa co (eroded), gốc,
  giãn (dilated) bằng các ngưỡng chiếu khác nhau trên trường đã lọc. Hệ quả: **tự sinh kích thước
  nét tối thiểu** cho cả pha rắn lẫn rỗng, và hội tụ ổn định hơn tới thiết kế gần nhị phân
  ([phân tích kích thước nét](https://arxiv.org/pdf/2101.08605),
  [mở rộng](https://arxiv.org/pdf/2003.00263)).
- **Đúng bài toán của ta:** Andreassen, Lazarov & Sigmund 2014 dùng robust formulation cho vi cấu
  trúc auxetic (inverse homogenization), in được không cần hậu xử lý, ν = −0,5 kiểm thực nghiệm
  ([Mech. Mater. 69](https://backend.orbit.dtu.dk/ws/files/119929621/Design_of_manufacturable_3D_extremal_elasticc_microstructure.pdf)).
  Sigmund 2000 đã chỉ ra auxetic tối ưu topo hay ra bản lề mảnh không in được - đúng W3.
- Trong tối ưu bằng mạng nơ-ron (photonics, GLOnet): loss = 0,5·gốc + 0,25·co + 0,25·giãn
  ([Chen et al.](https://arxiv.org/pdf/2007.12991)) - dạng trung bình có trọng số, dễ cho L-BFGS hơn
  dạng max.
- Nhược điểm: ~3 lần giải FE mỗi bước; thiết kế có thể kém hơn bản không ràng buộc một chút.
- **Bài học cho ta:** robust formulation có thể giải **cả W3 lẫn W6 cùng lúc**, với lý do vật lý
  rõ ràng (thay cho phạt góc/nét mảnh của K1 vốn là heuristic). Đây là đối thủ/ứng viên số 1.

### 2.2 Sai số "bậc thang" của lưới pixel: composite voxel, subpixel smoothing
- Trong đồng nhất hóa trên lưới pixel/voxel, biên bậc thang gây sai số hệ thống; tính chất hiệu dụng
  hội tụ bậc O(h) (Schneider; [X-FFT](https://arxiv.org/pdf/2601.02172)). Khớp số đo của ta: biên bậc
  thang hội tụ đều 50² → 400², lệch ~0,022.
- **Composite voxel** (Kabel, Merkert & Schneider 2015, CMAME 294): phần tử chứa biên được gán độ
  cứng của **lớp ghép (laminate)** theo tỉ lệ thể tích + pháp tuyến biên → giảm mạnh sai số tính chất
  hiệu dụng ở cùng độ phân giải ([ComBo](https://arxiv.org/pdf/2204.13624)).
- **SSP - subpixel-smoothed projection** (Hammond, Oskooi, Johnson et al., Opt. Express 2025,
  [arXiv 2503.20189](https://arxiv.org/abs/2503.20189), [code](https://github.com/NanoComp/SSP)):
  phép chiếu phụ thuộc cả mật độ đã lọc ρ̃ lẫn ‖∇ρ̃‖; khoảng cách tới biên d = (η − ρ̃)/‖∇ρ̃‖, trong dải
  |d| < R̂ (R̂ = 0,55Δx) mật độ = tỉ lệ lấp đầy F(d) (đa thức bậc 5). Khả vi **kể cả β = ∞**, hội tụ
  nhanh hơn tanh-Heaviside ở β lớn (tanh ở β = ∞ gradient bằng 0). Mới demo cho photonics.
  Tiếp nối: SSP bậc hai ([2601.10737](https://arxiv.org/html/2601.10737)); ràng buộc nét tối thiểu
  không cần chỉnh tham số dựa trên SSP + ràng buộc hình học Zhou 2015
  ([Arrieta, Romano, Johnson 2025](https://arxiv.org/pdf/2507.16108)) - chi phí hiệu năng thường
  ≤ 3-20%.
- **Bài học cho ta:** (a) SSP là tài liệu gần nhất với luận điểm P1.1e "chiếu nhị phân khả vi" -
  **bài #1 cần trích dẫn**; (b) sai số W1 có cách sửa chuẩn: tối ưu/verify ở lưới mịn hơn hoặc
  biên subpixel. Lưu ý: SSP để lại pixel xám ở biên; với nội suy SIMP p = 3 pixel xám bị đánh giá
  thấp độ cứng, cần nội suy kiểu laminate mới đúng vật lý → không cắm thẳng được.

### 2.3 Liên thông (W4): ràng buộc vật lý
- Virtual temperature method (Liu et al. 2015) và họ "virtual scalar field": giải bài toán dẫn nhiệt
  phụ, đảo rắn rời/lỗ kín làm nhiệt độ max tăng vọt → ràng buộc nhiệt độ max = ràng buộc liên thông,
  khả vi. Tổng quan so sánh 5 cách: Cool, Aage & Sigmund
  ([arXiv 2501.09402](https://ar5iv.labs.arxiv.org/html/2501.09402)) - lựa chọn ràng buộc ảnh hưởng
  lớn tới kết quả.
- **Bài học cho ta:** ở r ≥ 10, cả 30 ứng viên đều đứt, nên refine từ điểm đứt khó cứu bằng phạt.
  Ràng buộc liên thông chỉ hữu ích khi đi kèm điểm xuất phát tốt hơn (mục 2.5).

### 2.4 Vật lý của dị hướng cực đoan (W4)
- Tương hỗ trực hướng: ν12/E1 = ν21/E2 ⇒ **r = ν21/ν12 = E2/E1**; ổn định 2D cần ν12·ν21 < 1
  ([tổng hợp](https://link.springer.com/10.1038/s41467-023-39792-9)). Mục tiêu r ≥ 10 = yêu cầu một
  phương **mềm hơn ≥ 10 lần** → về mặt hình học là gần cơ cấu; cách "rẻ" nhất để mềm là đứt rời -
  giải thích vì sao cVAE ra ô đứt.
- **Bài học cho ta:** với r ≥ 10, nên ràng buộc độ cứng tối thiểu (min(E_x, E_y)/E0 ≥ ε, E đã có sẵn
  từ Q) thay vì chỉ khớp ν - biến "đứt rời" thành vi phạm rõ ràng. Gốc rễ vẫn là thiếu dữ liệu.

### 2.5 Phía học máy (W4, W5)
- Guided diffusion cho vi cấu trúc voxel (2024, [arXiv 2401.13570](https://arxiv.org/pdf/2401.13570)):
  dùng **active learning** để mở rộng vùng tính chất thiếu dữ liệu; dùng thiết kế diffusion làm
  **điểm khởi tạo cho tối ưu topo** (ν −0,54 → −0,63).
- DiffuMeta ([2507.15753](https://arxiv.org/pdf/2507.15753)): ngoại suy trong không gian tính chất
  nhờ nội suy trong không gian thiết kế (claim của tác giả).
- Tối ưu latent trên diffusion + VAE (Nature 2025,
  [s44455-025-00005-6](https://www.nature.com/articles/s44455-025-00005-6)): claim ngoại suy, nhưng
  R² một số thành phần chỉ ~0,6.
- DiffOPT (AISTATS 2025): tối ưu theo surrogate ra ngoài phân phối sẽ khai thác lỗi surrogate - cùng
  bài học P1.1.
- Khởi tạo tối ưu topo từ mô hình sinh giảm 36-58% chi phí (báo cáo trong các nghiên cứu warm-start).
- **Bài học cho ta:** W4/W5 chủ yếu là bài toán **phủ dữ liệu**; cách đáng tin nhất là sinh dữ liệu
  có mục tiêu (SIMP cho vùng r ≥ 10 / ν ngoài dải, verify bằng FE thật) rồi fine-tune, hoặc định tuyến
  SIMP khởi tạo từ cVAE cho mục tiêu ngoài dải. Active learning tổng quát đã thất bại ở Phase 7 vì
  checkpoint gần trần trong phân phối - khác với sinh dữ liệu nhắm đúng vùng thiếu.

---

## 3. Tổng hợp: vì sao các hướng trước chưa "tốt hơn toàn diện"

Mọi pilot đến nay cho cùng một quy luật: **mỗi objective chỉ cải thiện đúng loại hiện thực hóa nó
tối ưu** (lưới 100² cải thiện lưới mịn nhưng kém bền; dịch-lệch-lưới cải thiện độ bền nhưng không cải
thiện lưới; K1 cải thiện chế tạo). Muốn "tốt hơn toàn diện" phải:
1. **Định nghĩa thước đo đáng tin trước** (nếu không, "tốt hơn" trên lưới 50² có thể là ảo - W2).
2. Tối ưu một **họ hiện thực hóa đại diện cho thứ sẽ được chế tạo và đo**: lưới đủ mịn + sai lệch
   hình học kiểu co/giãn. Robust formulation (2.1) chính là họ co/giãn có cơ sở; lưới mịn (2.2) xử lý
   sai số bậc thang.

### Bảng điểm đề xuất (dùng cho mọi thí nghiệm dưới đây, ghi trước)
- **Độ chính xác chính:** MAE ν12 và sai số cặp trên **lưới 200² (kron) của thiết kế 50²** (e_mesh).
- **Độ bền:** MAE ν12 trên hiện thực hóa làm mượt 200² ở σ ∈ {0,5; 1,0} (e_real).
- **Chế tạo:** liên thông 4 hướng + nét tối thiểu trên ảnh 64².
- **Chi phí:** số lần giải FE quy đổi ra lưới 50² (1 lần 100² ≈ 2 lần 50²; đo thời gian thật).
- e_verify (50²) vẫn báo nhưng **không dùng để kết luận**.
- **"Tốt hơn A"** = không tệ hơn có ý nghĩa ở mục nào (CI hiệu số chứa 0 hoặc tốt hơn) VÀ tốt hơn có
  ý nghĩa ở ít nhất 1 mục.

---

## 4. Thí nghiệm đề xuất (theo thứ tự lợi ích / chi phí)

Tất cả trên `cvae_v2_finetuned.pt`, best-of-30, β {1, 4, 16, 64}, không force_periodic, guarded như
hiện tại. Pilot = IN100-B 20 condition đầu (giống K1/N1); xác nhận = IN100 (chính) + IN100-C.

### E1 - Robust formulation trong refine (ứng viên số 1 cho W3 + W6)
- **Cách làm:** 3 hiện thực hóa từ ảnh decoder: làm mượt σ = 0,5 phần tử → Heaviside với ngưỡng
  η ∈ {0,3 (giãn); 0,5 (gốc); 0,7 (co)} → FE 50². Loss = 0,25·MSE_giãn + 0,5·MSE_gốc + 0,25·MSE_co
  (dạng GLOnet; dạng max để làm biến thể phụ nếu bản trung bình đạt). Thêm vào `realization.py` (tham
  số danh sách η) + test. Chi phí: 3 FE/bước (song song 3 tiến trình).
- **Giả thuyết + tiêu chí pilot (ghi trước):** so với A: chế tạo ≥ 0,60; e_real(σ = 1,0) tốt hơn có ý
  nghĩa; e_mesh không tệ hơn có ý nghĩa.
- **ETA:** code + test ~30 phút; pilot ~10-15 phút; chấm điểm ~10 phút.

### E2 - Refine đa độ phân giải (W1, W2)
- **Cách làm:** 3 mức β đầu ở lưới 50², mức β = 64 cuối ở lưới 100² (cờ `--fe-upsample` đã có, chỉ
  cần áp riêng cho mức cuối). Lý do: pilot C (100² mọi mức) giảm e_mesh 44% nhưng tốn 2× FE; mức β
  cuối mới là lúc quyết định hình học nhị phân.
- **Tiêu chí pilot:** e_mesh tốt hơn A có ý nghĩa; giữ ≥ 70% mức cải thiện của C; chi phí ≤ 1,4× A.
- **ETA:** code ~20 phút; pilot ~15 phút (1 tiến trình FE).

### E3 - Kết hợp tốt nhất + xác nhận (mục tiêu chính: tốt hơn A toàn diện)
- Ghép E1 + E2 (robust formulation, mức cuối ở 100²); nếu chế tạo vẫn < 0,6 thì thêm phạt góc K1 với
  λ nhỏ. Cấu hình chốt **trên IN100-B trước khi** chạy IN100 / IN100-C (quy trình chống rò rỉ như K1).
- **Tiêu chí xác nhận (n = 100, IN100):** "tốt hơn A" theo định nghĩa mục 3 trên cả 3 mục độ chính
  xác / độ bền / chế tạo; tái lập độ chính xác + chế tạo trên IN100-C.
- **ETA:** ~2-3 giờ mỗi tập 100 condition (3 tiến trình), chấm điểm ~30 phút/tập.

### E4 - Dị hướng cực đoan r ≥ 10 (W4) - lớn hơn, làm sau E1-E3
- (a) Chẩn đoán rẻ: liệt kê mục tiêu r ≥ 10, xác nhận chúng đòi E2/E1 ≥ 10 qua tương hỗ.
- (b) Sinh dữ liệu có mục tiêu: SIMP (đúng pipeline P1.6a) cho ~200-300 mục tiêu r ∈ [5; 30], lọc ô
  liên thông, fine-tune cVAE (công thức 2 giai đoạn hiện hành, chọn theo `fe_r2`).
- (c) Thêm điều kiện độ cứng tối thiểu min(E_x, E_y)/E0 ≥ 1e-3 vào refine (chặn ô đứt).
- **ETA (b):** SIMP ~2-3 giờ CPU (4 tiến trình) + fine-tune GPU ~1-2 giờ. Tiêu chí ghi trước khi làm.

### E5 - Ngoài dải / đổi dấu (W5) - tùy chọn
- Định tuyến: mục tiêu xa phân phối train → SIMP khởi tạo từ thiết kế cVAE tốt nhất (thay vì 4 khởi
  tạo hình học). Đo: giữ độ chính xác SIMP với ít FE hơn. Khác P1.7 (lai cho mọi mục tiêu, trượt vì
  chế tạo): ở đây chỉ dùng cho OOD, nơi SIMP vốn thắng.

### E6 - SSP thay tanh-Heaviside (chi phí) - ưu tiên thấp
- Lợi ích kỳ vọng: ít bước L-BFGS hơn, thiết kế nhị phân hơn. Rủi ro: pixel xám ở biên lệch vật lý
  với nội suy SIMP p = 3 (mục 2.2) → phải giải quyết nội suy trước.

---

## 5. Thứ tự đề xuất và tổng thời gian
1. **E1 + E2 pilot** (~1,5 giờ tổng, ≤ 4 tiến trình) → báo kết quả.
2. Nếu ít nhất 1 cái đạt: **E3** chốt cấu hình trên IN100-B, xác nhận IN100 + IN100-C (~5-7 giờ).
3. E4 / E5 sau khi E3 xong (mỗi cái ~nửa ngày).

## 6. Ảnh hưởng tới bài báo #1 (cần tác giả quyết)
- Nên trích dẫn SSP (Hammond 2025) và robust formulation (Wang 2011; Andreassen 2014) ở phần liên quan.
- Nên báo sai số trên lưới mịn bên cạnh 50² (W1, W2), vì reviewer SMO (cộng đồng Sigmund) rất có thể
  hỏi đúng điểm này.
- Nếu E3 đạt, đó là kết quả mạnh hơn hẳn: "cVAE + refine bền vững ngang/hơn SIMP trên thước đo đáng
  tin, chế tạo được ngang SIMP".

---

## 7. Sau pilot E1/E2 (2026-10-10): đề xuất E1′ - CHƯA CHẠY

Kết quả pilot: `docs/plan.md` mục N2. E1 trượt vì (1) đốm vật liệu nhỏ không bị số hạng nào của
loss phạt, (2) guard chấm bằng lưới 50² loại các refine bền vững. E2 cải thiện lưới mịn nhưng làm
kém độ bền.

**E1′ - tham số hóa thiết kế qua bộ lọc (đúng robust formulation chuẩn):**
- Thiết kế vật lý x_phys = Heaviside_η(làm mượt_σ(ảnh decoder)) lấy mẫu trên lưới 50². Bộ lọc là một
  phần của tham số hóa (như density filter trong tối ưu topo), nên đốm nhỏ hơn ~σ tự biến mất.
- Bản gốc / co / giãn đều từ CÙNG trường đã lọc (η = 0,5 / 0,75 / 0,25), đúng Wang et al. 2011.
- FE verify, kiểm chế tạo và hình vẽ đều trên CÙNG ảnh x_phys (bỏ khe 64² ↔ 50²).
- Guard chọn theo chính loss robust (tối ưu gì thì chọn bằng nấy), không theo lưới 50².
- σ = 1,0 (co/giãn ≈ 0,67 phần tử); nếu chế tạo chưa đạt, σ = 1,5 làm biến thể phụ.
- Tiêu chí ghi trước khi chạy (đề xuất): chế tạo ≥ 0,60 VÀ e_real(σ=1,0) tốt hơn A VÀ e_mesh không
  tệ hơn A. Lưu ý: đây là vòng lặp thứ 2 sau khi thấy kết quả E1 → bắt buộc xác nhận trên tập mới
  (IN100 / IN100-C, n = 100) trước khi claim.

---

## 8. Đề xuất E4 - dị hướng cực đoan r ≥ 10 (CHƯA CHẠY, chờ tác giả duyệt)

**Chẩn đoán (2026-10-10, dữ liệu sẵn có):**
- Train v2 (57 216 mẫu sau tăng cường ×6): r ∈ [1; 2) 89,0%; [2; 5) 7,8%; **[5; 10) 0,46%**; [10; 20)
  1,55%; ≥ 20 1,21% → tổng r ≥ 10 là 2,8% (~260 thiết kế gốc), có **khoảng trống ở r ∈ [5; 10)**.
- Mẫu r ≥ 10 trong train: |ν| nhỏ hơn trung vị 0,066, |ν| lớn hơn trung vị 1,26 → toàn cực trị.
- Mục tiêu thất bại IN100 #46 (−0,036; −1,25), r ≈ 35, nằm TRONG vùng có dữ liệu (≥ 20: 1,2%) nhưng 30/30
  ứng viên vẫn đứt rời → không chỉ là thiếu dữ liệu; theo tương hỗ r = E2/E1 = 35 đòi một phương mềm hơn
  35 lần, cách "rẻ" là đứt rời. F (thiết kế qua bộ lọc) cũng không cứu được (#46 vẫn suy biến).
- Tập test hiện chỉ có 1 mục tiêu r ≥ 10 mỗi tập → không đủ lực thống kê để đo cải thiện.

**Đề xuất:**
1. Tập đánh giá riêng **ANISO30**: 30 mục tiêu r ∈ [5; 40] lấy từ test set (hoặc tổng hợp khả thi vật lý
   ν12·ν21 < 1), đo trước baseline A, F, SIMP (SIMP ~2 phút/mục tiêu ×4 khởi tạo ≈ 1 giờ với 8 tiến trình).
2. Ràng buộc độ cứng tối thiểu trong refine: phạt khi min(E_x, E_y)/E0 < 1e-3 (E đã có từ Q) - chặn đường
   "đứt rời". Rẻ, thử trước trên ANISO30.
3. Nếu (2) chưa đủ: sinh ~200-300 mẫu SIMP nhắm r ∈ [5; 30] (lấp khoảng trống [5; 10)), lọc liên thông,
   fine-tune cVAE theo công thức 2 giai đoạn, chọn theo `fe_r2`; kiểm không làm tệ IN100 (sàn cứng
   CLAUDE.md: R² không thấp hơn checkpoint hiện hành). Ước tính SIMP ~2-3 giờ CPU + GPU ~1-2 giờ.
- Tiêu chí ghi trước khi chạy từng bước (tỉ lệ ô suy biến trên ANISO30, sai số cặp vs SIMP, IN100 không
  tệ hơn).

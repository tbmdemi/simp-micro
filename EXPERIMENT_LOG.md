# Nhật ký Thử nghiệm & Lỗi đã Sửa

Tài liệu này chứa **lịch sử điều tra** (điều gì đã được thử, tại sao thất bại/thành công, dữ liệu đo được ở từng bước) cho các phase của SIMP Analyst. [`README.md`](README.md) chỉ mô tả **cách hệ thống hoạt động ở trạng thái hiện tại**; nếu muốn biết *tại sao* nó hoạt động như vậy — bao gồm các ngõ cụt đã thử — thì tài liệu này là nơi tra cứu.

> Quy ước: mỗi mục ghi ngày xác nhận (nếu có) và trỏ ngược tới mục tương ứng trong README bằng liên kết `README.md#...`.

---

## Phase 1 — LHS Screening

Liên quan tới [README § Pipeline / 1. LHS Screening](README.md#1-lhs-screening-phase-1) và [README § Hàm Mục tiêu](README.md#hàm-mục-tiêu-auxetic).

Lần chạy LHS screening đầu tiên trả về **0 mẫu auxetic** trên toàn bộ không gian tham số quét (`volfrac`, `penal`, `rmin`, `move`, `void_size_frac`, `rotation_deg`). Truy nguyên đến ba nguyên nhân đồng thời:

1. **Công thức tắt trực hướng (orthotropic shortcut) cho ν₁₂ khi xoay** — `ν₁₂ = Q₁₂/Q₂₂` chỉ đúng khi liên kết cắt-pháp `Q₁₃ = Q₂₃ = 0`. Với `rotation_deg = 32,4°` và các góc tương tự, các thành phần này khác 0, khiến công thức tắt cho kết quả sai dấu. Sửa bằng cách viết lại `compute_nu12`/`compute_nu21` để dùng **nghịch đảo đầy đủ ma trận độ mềm 3×3** (`S = Q⁻¹`) thay vì công thức tắt.
2. **Các hệ số phạt được chuẩn hóa kém** trong hàm mục tiêu auxetic, khiến việc tối ưu bị lệch hướng.
3. **Lỗi chuyển vị FE trong đồng nhất hóa** — trường dao động χ (chi) được dùng trực tiếp làm chuyển vị tổng thay vì `U_total = U0 + χ`, âm thầm làm sai lệch tensor đồng nhất hóa `Q` ở *mọi* lần chạy, không chỉ khi xoay. Sửa trong `compute_homogenized_tensor()`: giờ dùng `U_total = U0 + U`.

Sau khi sửa cả ba, Phase 2 (multi-batch DOE, xem README) đạt 82,1% mẫu auxetic trên 7.920 mẫu.

Phân tích độ nhạy (tương quan Spearman) trên kết quả đã sửa xác nhận **`volfrac` là tham số chi phối** (r ≈ 0,87–0,96); `move`, `rmin`, `void_size_frac` không có ý nghĩa thống kê — đây là cơ sở cho việc thu hẹp khoảng `volfrac` qua các lô refine của Phase 2.

---

## Phase 5 — cVAE: Gamma-sweep & Kiểm chứng FE

Liên quan tới [README § Pipeline / 5. Conditional VAE](README.md#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực).

**Quét `gamma`** (trọng số property-loss; không nhầm với `--beta`, trọng số KL), đánh giá tại thời điểm test với `z ~ N(0, 1)` (không rò rỉ từ encoder), **R² tính qua surrogate đóng băng** (chưa kiểm chứng FE):

| gamma | R² (ν₁₂) qua surrogate | MAE (ν₁₂) | pixel_std (đa dạng @ v12=v21=-0,6) |
|---|---|---|---|
| 1 | −0,418 | 0,174 | 0,326 |
| 5 | 0,450 | 0,106 | 0,274 |
| 20 | 0,633 | 0,086 | 0,314 |
| 30 | 0,60 | — | — |
| 50 | 0,63 | — | — |
| 80 | 0,71 | — | — |
| 100 | 0,76 | — | — |
| 150 | 0,79 | — | — |
| 200 | 0,79 | — | — |
| 250 | 0,81 | — | — |
| 300 | 0,86 | — | — |

R² qua surrogate tăng đơn điệu theo gamma, không bao giờ chững lại (plateau) — dữ liệu đầy đủ ở `outputs/phase5/gamma_sweep_results/`.

**⚠️ Giá trị R² qua surrogate không đáng tin cậy.** `pipeline/phase5_cvae/verify_fe.py` (độc lập với surrogate: nhị phân hóa ảnh sinh ra, resize về lưới FE thực, chạy `simp/core/solver.py` + `simp/homogenization/compute.py` thực để lấy ν₁₂/ν₂₁ chuẩn thực) cho thấy:

| gamma | R²(v12) qua surrogate | R²(v12) qua FE thực | tỷ lệ trúng auxetic thực (trên 24 mẫu mục tiêu auxetic) |
|---|---|---|---|
| 1 | −0,42 | **−1,97** | 7/24 |
| 20 | 0,63 | **−1,16** | 12/24 |
| 100 | 0,76 | **−2,23** | 6/24 |
| 300 | 0,86 | **−2,41** | 4/24 |

R² thực âm sâu ở mọi mức gamma, và khoảng cách giữa surrogate với thực tế *nới rộng* khi gamma tăng — bằng chứng của **khai thác surrogate** (decoder ngày càng giỏi đánh lừa CNN đóng băng, chứ không phải giỏi sinh hình học auxetic thực). Tỷ lệ trúng auxetic thực thực ra *tệ hơn* ở gamma cao so với gamma=20. Dữ liệu đầy đủ: `outputs/phase5/fe_verification_report.json`.

Trước khi tin tưởng/mở rộng bất kỳ kết quả gamma-sweep nào trong tương lai, hãy chạy `python3 pipeline/phase5_cvae/verify_fe.py --sanity-check` trước (phải cho sai số trung bình < 0,05 khi dùng `penal` **thực** của từng mẫu, không phải giá trị mặc định cố định) rồi mới chạy kiểm chứng đầy đủ.

Kết luận (2026-07-23): checkpoint `gamma=20` được giữ làm baseline mặc định không phải vì R² qua surrogate tốt nhất, mà vì nó là điểm dữ liệu duy nhất có kiểm chứng FE đầy đủ trước khi chuyển hướng sang biện pháp inference-time (best-of-N, xem README).

---

## Phase 5 — cVAE: Self-play (không khắc phục được vấn đề)

Liên quan tới [README § Pipeline / 5. Conditional VAE](README.md#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực). Thử ngày 2026-07-23.

`pipeline/phase5_cvae/self_play.py`: định kỳ fine-tune surrogate trên các ảnh do cVAE sinh ra và được chấm điểm bằng FE thực, đồng thời chọn checkpoint cVAE theo R² tính bằng FE thực thay vì `val_loss` dễ bị "chơi gian".

Một proof-of-concept đầu tiên (2 vòng, 8 điều kiện) có vẻ cải thiện R² đơn điệu (−9,50→−4,90→−4,17) — nhưng hóa ra đây là **nhiễu đo lường**: `verify_round()` (a) dùng seed ngẫu nhiên khác nhau mỗi vòng (`123+k`, nên mỗi vòng được chấm trên một tập con test khác nhau — không so sánh tương đương) và (b) `model.generate()` lấy mẫu vector tiềm ẩn từ RNG toàn cục không có seed, nên chấm lại *cùng một* checkpoint hai lần vẫn ra số khác nhau.

Cả hai lỗi đã được sửa (seed cố định 123 không đổi giữa các vòng, `torch.manual_seed(seed)` trước khi sinh trong `verify_round()`). Kiểm chứng lại các vòng 0–8 trên một tập giữ riêng 96 mẫu cố định cho R² gần như phẳng, nhiễu (−2,27 đến −3,30, không có xu hướng) — mức "cải thiện" ban đầu (−9,50→−1,49 qua các vòng) không phản ánh tiến bộ thực.

Một lần chạy v2 đã sửa (dự kiến 10 vòng, huấn luyện lại từ đầu, với một sửa lỗi thêm — các mẫu adversarial chỉ chiếm ~0,1% mỗi batch fine-tune, không đủ để gradient descent nhận ra, được oversample ×40 để khắc phục qua `--adversarial-oversample 40`) vẫn không cho thấy cải thiện thực sự sau 5 vòng (baseline −2,41 so với −2,65/−2,78/−2,93/−2,67) trước khi bị dừng để chuyển sang biện pháp inference-time (best-of-N).

**Kết luận: self-play, kể cả sau khi sửa lỗi đo lường, không khắc phục được việc sinh single-shot trong ngân sách thời gian đã thử.**

---

## Phase 5 — cVAE: Ensemble Surrogate (chưa được chứng minh)

Liên quan tới [README § Pipeline / 5. Conditional VAE](README.md#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực).

`pipeline/phase5_cvae/losses.py` (`load_frozen_surrogate_ensemble` / `property_consistency_loss_ensemble`) và `train.py --surrogate-paths --lambda-disagreement`: dùng 3 surrogate được huấn luyện độc lập, huấn luyện dựa trên giá trị dự đoán trung bình của chúng, và phạt các đầu ra của decoder ở nơi 3 mô hình bất đồng — một biện pháp chống khai thác mang tính cấu trúc (khó đánh lừa đồng thời 3 mô hình độc lập, và sự bất đồng là một proxy cho "vùng ngoại suy").

Hạ tầng đã sẵn sàng và có unit test (đã xác minh forward+backward pass); một lần chạy huấn luyện đầy đủ **không hội tụ trong ngân sách thời gian của phiên làm việc**. Được để lại như một tùy chọn đã có tài liệu, sẵn sàng dùng (`--surrogate-paths a.pt b.pt c.pt --lambda-disagreement 0.1`) cho công việc sau này — **chưa phải là một biện pháp khắc phục đã được chứng minh.**

---

## Khả năng chế tạo — Biện pháp huấn luyện lại đã thử

Liên quan tới [README § Pipeline / 5. Conditional VAE — Khả năng chế tạo](README.md#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực). Thử ngày 2026-07-23.

Sau khi xác nhận hệ số Poisson đúng ≠ hình học khả thi để chế tạo (xem README), hai biện pháp khắc phục ở giai đoạn huấn luyện đã được thử trước khi chuyển sang biện pháp inference-time (`--require-manufacturable`, xem README):

1. **Chính quy hóa (regularize) phần tái tạo posterior** (`train.py --lambda-tv --lambda-bin --lambda-periodic`, tái sử dụng `tv_loss`/`binarization_loss` hiện có cộng thêm `periodicity_loss` mới, fine-tune 25 epoch từ `cvae_gamma20.pt`) — **không có tác dụng** (tỷ lệ khả năng chế tạo không đổi trong biên độ nhiễu). Nguyên nhân gốc: loss này được tính trên `recon`, giải mã từ `z` lấy mẫu từ *posterior* (encoder thực, với ảnh huấn luyện thực) — mà ảnh huấn luyện thực (xây từ các hình seed đối xứng: circle/square/reentrant_bowtie/hexagonal) vốn đã gần tuần hoàn/liên thông sẵn, nên gần như không có gì để sửa trong chế độ đó. `model.generate()` ở giai đoạn inference thì lại lấy mẫu `z` từ *prior* (`N(0,1)`, không qua encoder) — một phân phối mà loss phía posterior không hề chạm tới.
2. **Chính quy hóa trực tiếp các bản giải mã lấy mẫu từ prior** (`train.py --regularize-prior-samples`, hàm mới `losses.prior_sample_regularization()`: giải mã một batch `z ~ N(0,1)` và áp dụng cùng các loss tv/bin/periodicity lên *ảnh đó* — đúng chế độ mà `generate()` thực sự dùng), fine-tune 40 epoch, trọng số mạnh hơn (`--lambda-periodic 0.3`) — cho **cải thiện nhỏ nhưng nhất quán** (tỷ lệ đạt cả hai cao hơn khoảng 1,5–2×: 0/2, 1/4, 7/10 trên 200 mẫu ở 3 điều kiện test, so sánh baseline với bản fine-tune), mà không làm hại độ chính xác thuộc tính (R² của checkpoint được chọn qua FE thực = -1,11, trong khoảng bình thường). Không ấn tượng — checkpoint được lưu tại `outputs/phase5/cvae_manuf_prior.pt`, được cung cấp như một lựa chọn thay thế tùy chọn, **không** thay thế mặc định hiện tại.

**Kết luận: biện pháp huấn luyện lại cho cải thiện tối đa (prior-sample regularization), biện pháp inference-time (`--require-manufacturable` + N lớn, xem README) là biện pháp giảm nhẹ thực tế được dùng mặc định.**

---

## Các Lỗi Chính đã Sửa

| Lỗi | Ảnh hưởng | Cách sửa |
|-----|--------|-----|
| **Lỗi chuyển vị FE trong đồng nhất hóa** | Trường dao động χ được dùng trực tiếp làm chuyển vị tổng thay vì `U_total = U0 + χ` — âm thầm làm sai lệch tensor đồng nhất hóa ở mọi lần chạy | Đã sửa trong `compute_homogenized_tensor()`: giờ dùng `U_total = U0 + U` |
| **Công thức tắt trực hướng ν₁₂ khi xoay** | `ν₁₂ = Q₁₂/Q₂₂` chỉ đúng khi `Q₁₃ = Q₂₃ = 0`; sai ở `rotation_deg = 32,4°` và các góc tương tự, gây ra **0 mẫu auxetic** trong lần screening Phase 1 đầu tiên | Viết lại `compute_nu12`/`compute_nu21` để dùng nghịch đảo đầy đủ ma trận độ mềm 3×3 (`S = Q⁻¹`) |
| **Thành phần phạt `mu` trong mục tiêu auxetic** | Dự định đẩy `Q₁₂` âm hơn nữa nhưng có sai sót về khái niệm — có thể gây sụp đổ vùng rỗng mà không cải thiện auxeticity một cách đáng tin cậy | Tắt theo mặc định (`mu=0.0`); đang chờ thiết kế lại |
| **Dấu của mục tiêu (loại bỏ các mục tiêu không phải auxetic)** | Hướng cập nhật OC bị sai với các mục tiêu `first`/`second` được hỗ trợ trước đây | Loại bỏ hoàn toàn các loại mục tiêu đó; chỉ còn `auxetic` |
| **Dùng `max` thay vì `min` trong `aggregate_correlations.py`** | Việc chọn mẫu tốt nhất lại chọn ra giá trị mục tiêu *tệ nhất* | Đổi `max(...)` → `min(...)` |
| **Nhiễu đo lường xuyên vòng trong self-play** | `verify_round()` của `self_play.py` dùng seed `123+k` (khác nhau mỗi vòng) và `model.generate()` lấy mẫu từ RNG toàn cục không seed — mức cải thiện R² có vẻ như thấy được qua các vòng 0-8 (−9,50→−1,49) chủ yếu là nhiễu do chấm mỗi vòng trên một tập con điều kiện *khác nhau*, không phải tiến bộ thực (kiểm chứng lại cho kết quả phẳng ở mức −2,27 đến −3,30 trên một tập cố định) | Cố định seed (123, không đổi giữa các vòng) + `torch.manual_seed(seed)` trước khi sinh trong `verify_round()` |
| **Dữ liệu adversarial vô hình trong fine-tune self-play** | `phase4_surrogate/train.py --adversarial-npz` thêm ~16-32 mẫu adversarial vào một batch 33.120 mẫu (~0,1%) — tín hiệu gradient từ dữ liệu đặc thù cho việc khai thác quá nhỏ để làm thay đổi surrogate | Thêm `--adversarial-oversample N` để lặp lại dataset adversarial N lần trước khi ghép nối (self-play v2 dùng N=40) |

---

*Xem [`CHANGELOG.md`](CHANGELOG.md) cho lịch sử thay đổi theo phiên bản (tính năng thêm/sửa/đổi theo từng release), và [`README.md`](README.md) cho trạng thái/cách hoạt động hiện tại của dự án.*

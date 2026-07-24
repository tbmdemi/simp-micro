# Nhật ký Thử nghiệm & Lỗi đã Sửa

Tài liệu này chứa **lịch sử điều tra** (điều gì đã được thử, tại sao thất bại/thành công, dữ liệu đo được ở từng bước) cho các phase của AuxForge. [`README.md`](README.md) chỉ mô tả **cách hệ thống hoạt động ở trạng thái hiện tại**; nếu muốn biết *tại sao* nó hoạt động như vậy — bao gồm các ngõ cụt đã thử — thì tài liệu này là nơi tra cứu.

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

## Phase 4 — Bootstrap CI cho R2 của surrogate

Liên quan tới [README § Pipeline / 4. CNN Surrogate Model](README.md#4-cnn-surrogate-model-phase-4---hoàn-thành). Thêm 2026-07-23.

Đối chiếu với Phase 5 (CI rất rộng do n=19-24 điều kiện — xem mục dưới): `pipeline/phase4_surrogate/bootstrap_ci.py` đo CI cho R² của surrogate trên **toàn bộ 1.184 mẫu test** (không cần huấn luyện lại — chỉ 1 lần forward pass trên checkpoint `surrogate_best.pt` đã có, ~2,4s). Vì cỡ mẫu lớn hơn Phase 5 gần 50-60 lần, CI hẹp hơn nhiều:

| Target | R² điểm ước lượng | CI 95% (bootstrap, n=1.184) |
|---|---|---|
| ν₁₂ | 0,9099 | [0,896, 0,923] |
| ν₂₁ | 0,9107 | [0,892, 0,926] |
| volfrac_achieved | 0,9817 | [0,979, 0,984] |

**Kết luận:** khác với các con số R² của Phase 5 (cần đọc kèm CI vì cỡ mẫu quá nhỏ), R²=0,91 của Phase 4 là một ước lượng đáng tin cậy — CI hẹp (~0,03 mỗi bên) phản ánh đúng cỡ mẫu test lớn, không phải may rủi thống kê. Dữ liệu đầy đủ: `outputs/phase4/bootstrap_ci_report.json`. Chạy lại: `python3 pipeline/phase4_surrogate/bootstrap_ci.py`.

---

## Phase 5 — Bootstrap CI cho R2 và tỷ lệ trúng của best-of-N

Liên quan tới [README § Pipeline / 5. Conditional VAE](README.md#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực). Thêm 2026-07-23.

Các con số R²/hit_rate của `best_of_n_eval.py` (README §5) là điểm ước lượng đơn trên tập giữ riêng rất nhỏ (24 điều kiện, 19 auxetic; chỉ 3 điều kiện cho biến thể `--require-manufacturable N=1500`). `pipeline/phase5_cvae/bootstrap_ci.py` bổ sung khoảng tin cậy 95% mà **không cần chạy lại FE** — chỉ resample lại `per_condition` đã lưu sẵn trong các file JSON kết quả:

- **R²(v12, FE thực):** percentile bootstrap (10.000 lần resample có hoàn lại trên các điều kiện).
- **tỷ lệ trúng auxetic:** Wilson score interval (không dùng bootstrap thường — với tỷ lệ trúng 100% trên n nhỏ, bootstrap percentile thường suy biến về `[1.0, 1.0]` vì mọi lần resample lại đều toàn giá trị 1, che giấu mất sự bất định thực sự ở n nhỏ; Wilson không có vấn đề này).

Kết quả (`outputs/phase5/self_play/bootstrap_ci_report.json`):

| file | n điều kiện | R² điểm ước lượng | R² CI 95% | hit_rate điểm ước lượng | hit_rate CI 95% (Wilson) |
|---|---|---|---|---|---|
| `best_of_n_result.json` (oracle) | 24 (19 auxetic) | +0,5955 | [0,003, 0,845] | 1,000 | [0,832, 1,000] |
| `best_of_n_k10_result.json` (thực dụng) | 24 (19 auxetic) | +0,4384 | [−0,372, 0,748] | 1,000 | [0,832, 1,000] |
| `best_of_n_manuf_n300.json` | 6 (5 auxetic) | −1,9557 | [−13,741, 0,827] | 0,800 | [0,376, 0,964] |
| `best_of_n_manuf_n1500.json` | 3 (3 auxetic) | +0,1871 | [−2,191, 0,903] | 1,000 | [0,439, 1,000] |

**Nhận xét:** CI rộng ở mọi trường hợp — kể cả kết quả headline (+0,5955) có CI dưới gần bằng 0, nghĩa là "khả năng cải thiện thực = 0" vẫn nằm trong khoảng hợp lý thống kê ở n=24. Kết quả `N=1500 + require-manufacturable` (R²=+0,19) đo trên chỉ 3 điều kiện — CI [−2,19, 0,90] gần như vô nghĩa về mặt thống kê, chỉ nên đọc như tín hiệu sơ bộ, không phải kết luận. Trước khi trích dẫn các con số R² của Phase 5 trong báo cáo/bài báo, nên: (1) luôn kèm CI thay vì chỉ điểm ước lượng, và (2) mở rộng tập giữ riêng vượt quá 24 điều kiện (và vượt quá 3 điều kiện cho biến thể manufacturability) nếu ngân sách FE cho phép.

Chạy lại: `python3 pipeline/phase5_cvae/bootstrap_ci.py <file1.json> <file2.json> ... --out <report.json>`.

---

## Khả năng chế tạo — Biện pháp huấn luyện lại đã thử

Liên quan tới [README § Pipeline / 5. Conditional VAE — Khả năng chế tạo](README.md#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực). Thử ngày 2026-07-23.

Sau khi xác nhận hệ số Poisson đúng ≠ hình học khả thi để chế tạo (xem README), hai biện pháp khắc phục ở giai đoạn huấn luyện đã được thử trước khi chuyển sang biện pháp inference-time (`--require-manufacturable`, xem README):

1. **Chính quy hóa (regularize) phần tái tạo posterior** (`train.py --lambda-tv --lambda-bin --lambda-periodic`, tái sử dụng `tv_loss`/`binarization_loss` hiện có cộng thêm `periodicity_loss` mới, fine-tune 25 epoch từ `cvae_gamma20.pt`) — **không có tác dụng** (tỷ lệ khả năng chế tạo không đổi trong biên độ nhiễu). Nguyên nhân gốc: loss này được tính trên `recon`, giải mã từ `z` lấy mẫu từ *posterior* (encoder thực, với ảnh huấn luyện thực) — mà ảnh huấn luyện thực (xây từ các hình seed đối xứng: circle/square/reentrant_bowtie/hexagonal) vốn đã gần tuần hoàn/liên thông sẵn, nên gần như không có gì để sửa trong chế độ đó. `model.generate()` ở giai đoạn inference thì lại lấy mẫu `z` từ *prior* (`N(0,1)`, không qua encoder) — một phân phối mà loss phía posterior không hề chạm tới.
2. **Chính quy hóa trực tiếp các bản giải mã lấy mẫu từ prior** (`train.py --regularize-prior-samples`, hàm mới `losses.prior_sample_regularization()`: giải mã một batch `z ~ N(0,1)` và áp dụng cùng các loss tv/bin/periodicity lên *ảnh đó* — đúng chế độ mà `generate()` thực sự dùng), fine-tune 40 epoch, trọng số mạnh hơn (`--lambda-periodic 0.3`) — cho **cải thiện nhỏ nhưng nhất quán** (tỷ lệ đạt cả hai cao hơn khoảng 1,5–2×: 0/2, 1/4, 7/10 trên 200 mẫu ở 3 điều kiện test, so sánh baseline với bản fine-tune), mà không làm hại độ chính xác thuộc tính (R² của checkpoint được chọn qua FE thực = -1,11, trong khoảng bình thường). Không ấn tượng — checkpoint được lưu tại `outputs/phase5/cvae_manuf_prior.pt`, được cung cấp như một lựa chọn thay thế tùy chọn, **không** thay thế mặc định hiện tại.

**Kết luận: biện pháp huấn luyện lại cho cải thiện tối đa (prior-sample regularization), biện pháp inference-time (`--require-manufacturable` + N lớn, xem README) là biện pháp giảm nhẹ thực tế được dùng mặc định.**

---

## Khả năng chế tạo — `force_periodic()`: ép cứng periodicity bằng 1 phép gán, không cần train

Liên quan tới mục "Khả năng chế tạo — Biện pháp huấn luyện lại đã thử" ngay phía trên. Phát hiện từ 1 nhánh nghiên cứu riêng (`research/auxetic-breakthrough`, thử nghiệm kiến trúc diffusion thay cVAE - không liên quan trực tiếp tới Phase 5, xem nhánh đó để biết toàn bộ bối cảnh), tích hợp ngược lại vào `FixLoss` ngày 2026-07-24 vì áp dụng được thẳng, không phụ thuộc kiến trúc sinh.

Quan sát mấu chốt: periodicity (`manufacturability.py::check_periodicity`) không phải 1 thuộc tính cVAE cần HỌC - nó là hệ quả toán học trực tiếp của phép lát tịnh tiến: ô kề bên khi lát là **bản sao y hệt** ô hiện tại, nên cạnh phải của 1 ô PHẢI bằng cạnh trái của CHÍNH NÓ (không phải bằng cạnh trái của 1 ô khác). Ràng buộc tự-nhất-quán này ép được trực tiếp bằng 1 phép gán: `img[:,-1] = img[:,0] = trung_bình(img[:,0], img[:,-1])` (tương tự cho hàng trên/dưới) - không cần gradient hay epoch train nào. Đã thêm `manufacturability.py::force_periodic()`.

**Đo trên chính checkpoint `cvae_gamma20.pt`** (không train lại, chỉ hậu xử lý output của `model.generate()`, 600 mẫu = 6 điều kiện × 100 mẫu, seed=123):
- `passes_all`: **1,7% → 19,5%** (>10×) - khớp với con số 0-3,5% đã ghi trong README §5 cho baseline không lọc.
- `periodic_ok`: → 100% tuyệt đối (đúng theo thiết kế).
- Chi phí property: đo qua FE thật trên 10 mẫu, `mean |Δv12| = 0,0193` - không có outlier lớn trong lần đo này (tối đa 0,0385).

Đã tích hợp làm hậu xử lý **mặc định BẬT** (cờ `--no-force-periodic` để tắt, tái hiện hành vi gốc) ở:
- `pipeline/phase5_cvae/best_of_n_eval.py` - áp lên mọi ứng viên trước khi chấm manufacturability/FE.
- `pipeline/phase5_cvae/sample.py::save_png()` - áp trước khi ghi PNG.

Test mới: `tests/test_phase5_manufacturability.py::TestForcePeriodic` (5 test), `tests/test_phase5_sample.py::TestSavePngForcePeriodic` (2 test) - cộng 1 test cũ (`test_save_png_roundtrip`) phải sửa để truyền `apply_force_periodic=False` (test đó cố ý tạo ảnh có cạnh trên/dưới lệch nhau để kiểm tra riêng phép chuyển tensor->PNG, nên sẽ vỡ nếu bị `force_periodic` ghi đè cạnh theo mặc định mới).

**Lưu ý:** đây là 1 cải tiến hậu xử lý độc lập, không phải bằng chứng cho việc kiến trúc diffusion tốt hơn cVAE nói chung (xem nhánh `research/auxetic-breakthrough` cho so sánh đầy đủ 2 kiến trúc - kết quả ở đó cho thấy single-shot của diffusion thực ra tệ hơn cVAE, best-of-N vẫn bắt buộc ở cả 2). `force_periodic()` chỉ tình cờ được phát hiện trong lúc thử nghiệm kiến trúc đó, và hoá ra tổng quát cho cả 2.

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

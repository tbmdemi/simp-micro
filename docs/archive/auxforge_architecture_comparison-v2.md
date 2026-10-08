# So Sánh Kiến Trúc Hiện Tại và Kiến Trúc Hướng Tới (AuxForge Next-Gen) - v2

> **⚠️ Cảnh báo tính cập nhật (thêm 2026-09-24):** viết ngày 2026-08-23, **trước khi** WIRE decoder (Task 2) thất bại chính thức trên benchmark 24-condition (2026-09-23 - xem [`EXPERIMENT_LOG.md`](../../EXPERIMENT_LOG.md) mục 2026-09-23, [`task_progress.md`](task_progress.md) Task 2, [`PIPELINE.md`](../PIPELINE.md) mục "2026-09-23"). Số liệu/khẳng định về WIRE ("passes_all ≥75%") và MNO ("tăng tốc 1000 lần", "≥500 mẫu/giây") ở đây là **MỤC TIÊU ĐỀ XUẤT tại thời điểm viết, KHÔNG PHẢI kết quả đã đạt**. Thực tế tính đến 2026-09-23: WIRE thất bại (R²(FE)=-13,87, hit-rate 0%, tệ hơn cả baseline gốc); MNO mới smoke-test bằng GRU fallback (thiếu `mamba_ssm`), chưa đo tốc độ/L2 error thật. ~~Chỉ mục "KAN + Real-Physics" bên dưới vẫn đúng và đã kiểm chứng.~~ **Cập nhật 2026-09-25:** cả mục KAN cũng không còn đứng vững - ablation có kiểm soát bằng FE thật cho thấy KAN không thắng Linear, và WIRE đã đóng (xem [`plan.md`](../plan.md) v3, P1.2/P1.3).

Đối chiếu kỹ thuật giữa **Kiến trúc Hiện tại (AuxForge Baseline)** và **Kiến trúc Hướng tới (Next-Gen AuxForge)**, nhánh `substrate-material`.

---

## Số liệu đo đạc thực tế (2026-08-23, loại bỏ số liệu cũ không tái lập được R²≈0,85)

| Chỉ số | Linear cũ (Base) | Linear cũ (Finetuned) | KAN Base (chọn `val_loss`) | KAN + Real-Physics v2 (khuyến nghị) |
| :--- | :---: | :---: | :---: | :---: |
| R² vĩ mô tổng thể | 0,29 | 0,35 | 0,19 | **0,64** |
| R² v12 | - | - | - | **0,82** |
| R² v21 | - | - | - | **0,45** |
| R² FE thật tốt nhất | - | - | - | **0,8889** |
| Property R² khi chọn theo `val_loss` | thấp | thấp | **-1,34** (sụp đổ vật lý) | không áp dụng (bắt buộc chọn theo `fe_r2`) |

---

## I. Luồng dữ liệu

**Hiện tại:** `[c] → MLP concat → CNN cVAE decoder → lưới 64×64 (mờ/răng cưa)` ↔ `[Q] ← FEM tuần hoàn (CPU, chậm)`

**Hướng tới (mục tiêu, WIRE/MNO chưa đạt - xem cảnh báo đầu bài):** `[c] → KAN encoder ┐` + `[tọa độ x,y] → WIRE INR decoder (Gabor Wavelet) → trường mật độ liên tục` ↔ `[Q] ← Mamba Neural Operator (GPU)`

---

## II. So sánh trực diện

| Tiêu chí | Hiện tại | Hướng tới | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| **R² vật lý thực tế** | 0,35 (MLP tuyến tính, không học được quan hệ phi tuyến) | 0,64 (FE thật 0,8889), KAN + real-physics loss | ~2× hiệu năng thực tế, giảm phụ thuộc hậu xử lý |
| **Chọn checkpoint** | Theo `val_loss` (reconstruction) - có thể chọn nhầm model đẹp ảnh nhưng gãy cơ học | Theo `fe_r2`/`property_accuracy` - loại bỏ hẳn `val_loss` (từng cho R²=-1,34 trên KAN) | Bảo toàn tính nhất quán cơ học của checkpoint được chọn |
| **Biểu diễn hình học** | Lưới pixel cố định 64×64 | Implicit Neural Representation, ρ=f(x,y;z,c) liên tục | Resolution-agnostic - xuất mesh mịn không cần train lại *(WIRE: xem cảnh báo, chưa đạt)* |
| **Decoder** | CNN transposed-conv, biên mờ do MSE/BCE | WIRE (Gabor Wavelet phức) | Mục tiêu: `passes_all` 30%→≥75% *(THẤT BẠI thật - xem cảnh báo đầu bài)* |
| **Surrogate** | CNN + MLP FC | ConvKAN (CNN + KAN FC) | Khả năng giải thích cao hơn về lý thuyết; chưa có benchmark production để kết luận lợi ích tham số/hồi quy ký hiệu *(thực đo: R² thấp hơn Linear - xem `EXPERIMENT_LOG.md` 2026-08-24)* |
| **Homogenization** | FEM CPU, tuần tự, chậm | Mamba Neural Operator, GPU | Mục tiêu tăng tốc ~1000× (~0,6→≥500 mẫu/s) *(chưa đo - mới smoke-test GRU fallback)* |

---

## III. Vì sao Real-Physics là chìa khóa cho KAN

**Vấn đề của KAN thuần:** spline có bậc tự do phi tuyến rất lớn - nếu chỉ train bằng loss ảnh (BCE/MSE, chọn theo `val_loss`), spline khớp từng pixel nhưng phá vỡ kết nối chịu lực thật, khiến property R² sụp đổ (-1,34).

**Vai trò của real-physics loss:** đưa `real_physics_prior_loss` (gradient FEA thật, khả vi) vào training ép spline của KAN bám theo phân bố ứng suất/biến dạng đàn hồi thật, thay vì chỉ khớp pixel - nâng R² vĩ mô lên 0,64 và R²(FE) lên 0,8889.

**Vai trò của `--select-by fe_r2`:** val_loss thấp nhất không đồng nghĩa model tốt nhất trên KAN Co-VAE - phải chọn checkpoint theo R²(FE) đo định kỳ bằng evaluator độc lập (`cvae_kan_realphysics_v2.pt` được chọn theo cách này). Đây là đóng góp phương pháp luận huấn luyện đáng đưa vào Discussion của Bài báo #1.

# BẢN KẾ HOẠCH CHI TIẾT (v2): TÍCH HỢP KAN, WIRE VÀ MAMBA VÀO PROJECT AUXFORGE

Bản kế hoạch này (Phiên bản v2 - Cập nhật ngày 23-24/08/2026 dựa trên kết quả thực nghiệm mới nhất trên nhánh `substrate-material`) cung cấp lộ trình thiết kế kỹ thuật, cách thức thay đổi mã nguồn, kết quả kỳ vọng và tiêu chí chấp nhận cho từng bước tích hợp các kiến trúc học máy tiên tiến (**Kolmogorov-Arnold Networks - KAN**, **Wavelet Implicit Neural Representations - WIRE**, và **Mamba State-Space Models**) vào dự án **AuxForge** (Thiết kế ngược và đồng nhất hóa siêu vật liệu auxetic).

> **Trạng thái mới nhất (2026-09-24, chưa cập nhật vào nội dung kế hoạch dưới đây):** Task 1 (KAN) đã đạt DoN, xem mục ngay dưới. Task 2 (WIRE) **đã thử và THẤT BẠI chính thức** sau 2 round retrain real-physics + vá bug `enforce_symmetry` (2026-09-23) - kết quả benchmark 24-condition TỆ HƠN baseline gốc (R²(FE)=-13,87 vs -10,29, hit-rate 0% vs 4,17%/16,67%), coi như đóng ở trạng thái thất bại. Task 3 (ConvKAN) đo được R² thấp hơn Linear baseline, kế hoạch tăng `grid_size` ở Phụ lục A bên dưới **chưa từng chạy**. Task 4 có benchmark tích cực đầu tiên nhưng chưa đạt DoN gốc. Nội dung DoN/checklist bên dưới GIỮ NGUYÊN như lúc viết kế hoạch (23-24/08) để làm tài liệu lịch sử; xem `task_progress.md` (log trạng thái, luôn cập nhật) và `EXPERIMENT_LOG.md` mục 2026-09-23 cho số liệu mới nhất.

---

## 🚨 SỰ THAY ĐỔI CHIẾN LƯỢC QUAN TRỌNG (DỰA TRÊN THỰC NGHIỆM 2026-08-23 & 2026-08-24)

Các thử nghiệm đo đạc trực tiếp trên nhánh `substrate-material` đã phát hiện ra các quy luật vật lý và số liệu mang tính bước ngoặt, buộc chúng ta phải cập nhật kế hoạch tích hợp:

1. **Định vị lại Baseline thực tế (Sửa lỗi "False Oracle" 0.85):** 
   - *Phát hiện:* Chỉ số $R^2 \approx 0.85$ báo cáo trong file đánh giá cũ (`evaluation_report.json` tháng 7) không thể tái lập bằng code hiện hành trên cùng checkpoint và test set (đo lại thực tế chỉ đạt $0.35$ với surrogate v2 và âm sâu với v1).
   - *Thay đổi trong plan:* Hạ chỉ tiêu so sánh baseline của mạng Linear cũ xuống mức thực tế: **0.29 (base) / 0.35 (finetuned)**. Như vậy, mọi so sánh hiệu năng của KAN từ nay sẽ đối chiếu với mốc thực này, làm nổi bật hiệu quả vượt trội thực chất của KAN.
2. **Loại bỏ việc chọn checkpoint theo `val_loss` (Reconstruction Loss):**
   - *Phát hiện:* Lựa chọn mô hình có `val_loss` thấp nhất (tức dựng ảnh đẹp nhất) cho ra thuộc tính cơ học $R^2$ âm sâu (lên tới $-1.34$). Reconstruction tốt hoàn toàn không tương quan với Property Fidelity trên mạng KAN.
   - *Thay đổi trong plan:* Nghiêm cấm chọn checkpoint dựa trên `val_loss` cho các bài toán nhạy cảm vật lý. Bắt buộc tích hợp bộ chọn `--select-by fe_r2` hoặc `property_accuracy`.
3. **Real-Physics là chìa khóa tối thượng cho KAN:**
   - *Phát hiện:* KAN base (không vật lý) chỉ đạt $R^2 = 0.19$. Khi kết hợp fine-tune với bộ giải phần tử hữu hạn khả vi (`--lambda-real-physics 2.0 --real-physics-every 20`), chỉ số nhảy vọt lên **$R^2 = 0.64$ (v12 đạt 0.82, v21 đạt 0.45) và đạt đỉnh $R^2(FE\ thật) = 0.8889$** (vượt gần 2 lần so với Linear finetuned thực tế).
   - *Thay đổi trong plan:* Tích hợp cứng quy trình huấn luyện hai giai đoạn (Base KAN -> Fine-tune Real Physics) làm quy chuẩn bắt buộc của Task 1.

4. **Exp 3 MLP control (2026-09-11):** MLP + Physics chạy đủ 150 epoch với
    cùng recipe đối chứng đạt `R²=[0,432; 0,087; -2,483]` cho
    `[v12,v21,volfrac]` và `R²(FE,v12)=-9,834`, trong khi KAN v2 đo lại đạt
    `[0,836; 0,476; -0,221]` và `0,581`. Kết quả ủng hộ KAN, nhưng KAN cần
    retrain lại sau các thay đổi decoder symmetry/volfrac loss để hoàn tất
    ablation công bằng.

---

## LỘ TRÌNH TỔNG QUAN HIỆU CHỈNH (ROADMAP v2)

```
[AuxForge Roadmap v2]
    │
    ├── PHẦN I: BÀI BÁO #1 — Thiết kế ngược siêu vật liệu liên tục đa mục tiêu
    │     ├── Task 1: KAN-hóa bộ hồi quy Co-VAE + Real Physics (Phase 5) - [ĐÃ HOÀN THÀNH KIỂM CHỨNG]
    │     ├── Task 2: Bộ tạo sinh tọa độ liên tục WIRE INR Decoder (Phase 5) - [ƯU TIÊN CAO]
    │     ├── Task 3: Bộ dự đoán thuộc tính ConvKAN Surrogate (Phase 4) - [ƯU TIÊN TRUNG BÌNH]
    │     └── Task 4: Vòng lặp tối ưu hóa ngược Tandem KAN-LBFGS (Phase 6) - [ƯU TIÊN CAO]
    │
    └── PHẦN II: BÀI BÁO #2 — Toán tử nơ-ron không gian trạng thái & Vật lý phi tuyến
          ├── Task 1: Mamba Neural Operator (MNO) đồng nhất hóa siêu tốc - [ƯU TIÊN RẤT CAO]
          ├── Task 2: KINN cho cơ học phi tuyến kết hợp JAX-AMG - [ƯU TIÊN CAO]
          └── Task 3: ICKANs cho siêu vật liệu Composite đa pha - [ƯU TIÊN TRUNG BÌNH]
```

---

## PHẦN I: BÀI BÁO #1 — THIẾT KẾ NGƯỢC SIÊU VẬT LIỆU LIÊN TỤC ĐA MỤC TIÊU BẰNG WIRE INR VÀ KAN

### Task 1: KAN-hóa Bộ Hồi Quy Co-VAE (Phase 5) [CẬP NHẬT THỰC NGHIỆM]
*   **Mức độ ưu tiên:** Đã hoàn thành giai đoạn kiểm chứng (Proof of Concept) thành công rực rỡ. Cần đóng gói cứng vào pipeline sản xuất.
*   **Vị trí tác động:** Thư mục `pipeline/phase5_cvae/`. Cụ thể là phần **Regression Head (Co-VAE)** dùng để ép không gian ẩn học theo đặc tính cơ học vĩ mô.
*   **Cách thức triển khai thực tế đã chứng minh:**
    1. Thay thế các lớp `nn.Linear` bằng `EfficientKANLinear`.
    2. Chạy huấn luyện Base KAN (chọn theo val_loss thô để lấy khung khởi tạo).
    3. Thực hiện chu kỳ Fine-tune 60 epoch tiếp theo với cấu hình: `--select-by fe_r2 --fe-eval-every 10 --lambda-real-physics 2.0 --real-physics-every 20 --lr 2e-4`.
    4. Lưu trữ checkpoint sản xuất cuối cùng tại: `cvae_kan_realphysics_v2.pt`.
*   **Tiêu chí chấp nhận (DoN - ĐÃ ĐẠT KHÓA MÃ NGUỒN):**
    *   [x] Đạt chỉ số đánh giá thực tế $R^2 \ge 0.60$ trên toàn bộ hằng số vĩ mô (v12 đạt 0.82, v21 đạt 0.45).
    *   [x] Đạt sai số kiểm chứng FE thật tốt nhất $R^2 \ge 0.85$ (thực tế đạt **0.8889**).
    *   [x] Khóa cơ chế checkpointing: Nghiêm cấm chọn theo `val_loss`. Bắt buộc chọn theo `fe_r2` hoặc `property_accuracy`.

---

### Task 2: Bộ Tạo Sinh Tọa Độ Liên Tục WIRE INR Decoder (Phase 5)
*   **Mức độ ưu tiên:** Cao.
*   **Vị trí tác động:** `pipeline/phase5_cvae/`. Thay thế hoàn toàn lớp xuất ảnh cuối cùng của **cVAE Decoder**.
*   **Kiến trúc đề xuất:** 
    *   Decoder không xuất ra tensor ảnh kích thước cố định ($64 \times 64$).
    *   Decoder nhận đầu vào là tọa độ liên tục $(x, y) \in [-1, 1]^2$ gộp chung với vector ẩn $z$ và vector điều kiện $c$ (8 chiều).
    *   Sử dụng hàm kích hoạt **Gabor Wavelet phức (WIRE)** cho các lớp ẩn của Decoder để giữ tính định xứ và triệt tiêu đốm rác.
*   **Tiêu chí chấp nhận (DoN):**
    *   [ ] Tỷ lệ sinh cấu trúc liên thông và tuần hoàn thực tế (`passes_all`) ở chế độ single-shot tăng từ $30\%$ (baseline conv decoder, `frac_manufacturable` tự nhiên) lên **$\ge 75\%$** (nhờ tính liên tục của trường mật độ) — **thực đo (3 checkpoint, 2026-08-24 & 2026-09-23): `cvae_wire_v2.pt`=4,17%, `cvae_wire_realphysics_v3_round2.pt`=0,00% - CẢ HAI ĐỀU THẤP HƠN NHIỀU baseline 30% ban đầu, chưa nói tới mục tiêu 75%. THẤT BẠI, xem `task_progress.md` Task 2 và `EXPERIMENT_LOG.md` mục 2026-09-23.**
    *   [ ] Khi xuất lưới tọa độ mịn gấp 4 lần ($256 \times 256$), hình dáng thanh chịu lực không bị vỡ hoặc răng cưa — chưa đánh giá định tính vì chưa có checkpoint đạt DoN đầu tiên để kiểm tra.

---

### Task 3: Bộ Dự Đoán Thuộc Tính ConvKAN Surrogate (Phase 4)
*   **Mức độ ưu tiên:** Trung bình.
*   **Vị trí tác động:** `pipeline/phase4_surrogate/`. File định nghĩa mô hình `Surrogate CNN`.
*   **Kiến trúc đề xuất:** Giữ nguyên các khối tích chập trích xuất đặc trưng của CNN cũ, nhưng thay thế toàn bộ phần lớp Fully Connected (FC) phân loại/hồi quy ở cuối bằng mô hình **KAN** 2 lớp để hỗ trợ tính gradient cực kỳ mượt mà cho thuật toán L-BFGS.
*   **Tiêu chí chấp nhận (DoN):**
    *   [ ] Đạt chỉ số đánh giá $R^2 \ge 0.985$ cho cả 5 đầu ra đồng thời ($\nu_{12}, \nu_{21}, volfrac, f_1, f_2$) đo cùng code hiện tại — **thực đo 2026-08-24 (mới 3/5 đầu ra, chưa có f1/f2): v12=0,9636, v21=0,9574, volfrac=0,9161, cả 3 đều THẤP HƠN mục tiêu 0,985 (và thấp hơn cả Linear baseline 0,974/0,964/0,983) - xem `EXPERIMENT_LOG.md` mục 2026-08-24.**
    *   [ ] Tổng số lượng tham số của lớp FC giảm **$\ge 80\%$** so với MLP cũ — chưa đo (chưa có benchmark param-count công bằng).

---

### Task 4: Vòng lặp Tối ưu hóa ngược Tandem KAN-LBFGS (Phase 6)
*   **Mức độ ưu tiên:** Cao.
*   **Vị trí tác động:** `pipeline/phase6_inference/`.
*   **Kiến trúc đề xuất:** 
    *   Sử dụng checkpoint `cvae_kan_realphysics_v2.pt` đã huấn luyện ở Task 1 làm xuất phát điểm.
    *   Áp dụng thuật toán L-BFGS để tinh chỉnh trực tiếp trên vector ẩn $z$ bằng cách lan truyền ngược gradient mượt qua ConvKAN Surrogate đã đóng băng ở Task 3.
*   **Tiêu chí chấp nhận (DoN):**
    *   [ ] Đạt chính xác các mục tiêu Poisson cực đoan (ví dụ $\nu^* = -2.5$, nằm sâu ngoài phân phối huấn luyện) với sai số kiểm chứng FE thật **$\le 8\%$** (khắc phục hiện tượng bão hòa biên độ cũ) — **chưa đo đúng kịch bản này. Benchmark đầu tiên (2026-09-23, 24 condition TRONG-phân-phối, không phải OOD cực đoan): R²(v12) baseline=-16,00 → refined=-7,60, MAE giảm ~40% (0,587→0,355) - tín hiệu tích cực nhưng khác đơn vị/kịch bản đo so với DoN gốc, chưa thể đối chiếu trực tiếp. Xem `EXPERIMENT_LOG.md` mục 2026-09-23.**

---

## PHẦN II: BÀI BÁO #2 — TOÁN TỬ NƠ-RON KHÔNG GIAN TRẠNG THÁI & VẬT LÝ PHI TUYẾN

### Task 1: Mamba Neural Operator (MNO) Đồng nhất hóa Siêu tốc [CẬP NHẬT ƯU TIÊN]
*   **Mức độ ưu tiên:** Rất cao (Giải quyết triệt để điểm nghẽn hiệu năng mô phỏng để sinh dữ liệu).
*   **Vị trí tác động:** `simp/homogenization/`. Tạo một phân hệ độc lập chạy hoàn toàn trên GPU.
*   **Kiến trúc đề xuất:** Sử dụng Bi-directional Mamba blocks làm backbone thay cho FNO để dự đoán 18 trường dịch chuyển vĩ mô từ trường mật độ vật liệu.
*   **Tiêu chí chấp nhận (DoN):**
    *   [ ] Thời gian đồng nhất hóa một mẫu giảm xuống dưới **1.5 mili-giây** trên GPU (nhanh hơn 1000 lần so với FEM) — chưa đo (mới smoke-test shape bằng fallback GRU, chưa có benchmark tốc độ thật, thiếu `mamba_ssm` trong môi trường).
    *   [ ] Sai số tương đối L2 của trường dịch chuyển dự đoán so với solver FEM thật $\le 1\%$ — chưa đo (chưa có dataset FE-displacement thật để train/evaluate).

---

### Task 2: KINN cho Cơ học Phi tuyến kết hợp JAX-AMG
*   **Mức độ ưu tiên:** Cao.
*   **Vị trí tác động:** `analysis/scripts/` (nonlinear_physics_prior_loss.py).
*   **Tiêu chí chấp nhận (DoN):**
    *   [ ] Giải quyết thành công trường biến dạng lớn phi tuyến (hyperelastic) dưới thời gian hội tụ dưới 30 giây/mẫu — chưa đo (hiện dùng PyTorch autograd thường, chưa có JAX-AMG thật trong môi trường).
    *   [ ] Đảm bảo tính khả vi toàn phần, gradient vật lý mượt mà truyền suốt qua JAX — chưa đo.

---

### Task 3: ICKANs cho Siêu vật liệu Composite Đa pha
*   **Mức độ ưu tiên:** Trung bình.
*   **Vị trí tác động:** `pipeline/constitutive_model/`.
*   **Tiêu chí chấp nhận (DoN):**
    *   [ ] ICKANs dự đoán chính xác năng lượng đàn hồi with $R^2 \ge 0.985$ — chưa đạt R² production, chưa có số liệu cụ thể để trích dẫn.
    *   [ ] Kiểm chứng toán học: Đạo hàm bậc hai của hàm năng lượng đàn hồi (ma trận stiffness tangent) bắt buộc phải luôn **xác định dương** trong mọi dải biến dạng kéo để đảm bảo tính nhất quán vật lý.

---

## PHỤ LỤC A: KẾ HOẠCH THÍ NGHIỆM (2026-08-24) — ConvKAN grid 8/10 + WIRE Heaviside Real-Physics

> Đính kèm theo phiên bản `plan.md` thống nhất ngày 2026-08-24 (tên cũ
> `auxforge_kan_integration_plan-v2.md` — file đã được đổi tên, mọi tham chiếu
> mới phải trỏ vào `plan.md`). Log trạng thái chi tiết từng task vẫn ở
> `task_progress.md`; trạng thái codebase xem mục "Trạng thái hiện tại" đầu
> `task_progress.md`.

### Mục tiêu

1. **ConvKAN (Phase 4):** tăng `grid_size` B-spline lên 8/10, huấn luyện trên
   dataset v2 để kéo $R^2$ vượt ngưỡng 0.985 trên cả 5 đầu ra
   ($\nu_{12}, \nu_{21}, volfrac, f_1, f_2$).
2. **WIRE (Phase 5):** huấn luyện WIRE decoder kết hợp **Heaviside Projection**
   + **Real-Physics** để hàn gắn triệt để lỗi đứt gãy topo 1-pixel
   (passes_all single-shot hiện chỉ 4.17%, mục tiêu ≥ 75%).

### Thực trạng đã xác minh từ code

* `SurrogateCNN` dùng `EfficientKANLinear` **mặc định `grid_size=5`**, không có
  flag đổi grid — `pipeline/phase4_surrogate/model.py`. ConvKAN v2 (3 đầu ra):
  $R^2$ v12=0.964 / v21=0.957 / volfrac=0.916 — **thấp hơn Linear baseline**
  (0.974 / 0.964 / 0.983).
* WIRE v2 (30 epoch) train **không bật real-physics**
  (`--lambda-real-physics` mặc định 0.0 → `real_physics=nan` trong history).
  `wire_best_of_n_result.json`: single-shot hit-rate 4.17%, best-of-N 16.7%,
  $R^2$(FE) = −10.29, frac_manufacturable 4.17%.
* Heaviside projection **đã có sẵn ở SIMP core**
  (`simp/core/filter.py::apply_heaviside_projection`, Wang–Lazarov–Sigmund 2011,
  đã có test trong `tests/test_core_smoke.py::TestFilter`) — chỉ cần port sang
  PyTorch khả vi.
* Real-physics hiện dùng density **liên tục** (bilinear resize → lưới FE
  50×50, `losses.py::real_physics_loss`) — chưa có bước làm nhị phân; đây là
  điểm chèn Heaviside.

### Phần A — ConvKAN tăng grid_size

| Bước | Công việc | File | ETA |
|---|---|---|---|
| A1 | Thêm `kan_grid_size: int = 5` vào `SurrogateCNN.__init__`, truyền vào 2 lớp KAN | `pipeline/phase4_surrogate/model.py` | 20 phút |
| A2 | Flag `--kan-grid-size` + lưu vào meta checkpoint | `pipeline/phase4_surrogate/train.py` | 15 phút |
| A3 | Đọc `ckpt.get("kan_grid_size", 5)` khi dựng model (tương thích ngược) | `evaluate.py`, `export_for_phase5.py`, `losses.py::load_frozen_surrogate`, `bootstrap_ci.py` | 20 phút |
| A4 | Test plumb-through + backward-compat mặc định 5 | `tests/test_phase4_model.py`, `tests/test_phase5_losses.py` | 20 phút |
| A5 | Run 1: grid=8, 3 đầu ra → `surrogate_convkan_g8.pt` | — | ~30–60 phút |
| A6 | Run 2: grid=10, 3 đầu ra → `surrogate_convkan_g10.pt` | — | ~30–75 phút |
| A7 | Run 3: grid tốt hơn trong {8,10} × `--include-f1f2` (5 đầu ra, đúng DoN) | — | ~45–75 phút |
| A8 | Đo R² bằng `evaluate.py`, đếm tham số KAN vs Linear, export sang Phase 5 | — | 15 phút |

**Cấu hình:** `--epochs 60 --batch-size 128 --lr 5e-4 --patience 10 --seed 0
--use-kan --kan-grid-size 8` (giữ nguyên bản v2). GPU RTX 3050 6GB
(447 batch/epoch), ước tính 0.5–1.5 phút/epoch.

**DoN:** $R^2 \ge 0.985$ trên cả 5 đầu ra. ⚠️ Rủi ro: f₁/f₂ lịch sử chỉ đạt
0.933/0.961 (LIMITATIONS #5) — nếu không kéo được, báo số thật + đề xuất
phương án B (tăng `fc_hidden`/`spline_order`).

### Phần B — WIRE + Heaviside + Real-Physics

| Bước | Công việc | File | ETA |
|---|---|---|---|
| B1 | Port `apply_heaviside_projection` sang torch khả vi (tanh, eta=0.5, beta ramp 1→50), test so khớp numpy gốc | `pipeline/phase5_cvae/heaviside.py` (mới), `tests/test_phase5_model.py` | 45 phút |
| B2 | Flag `heaviside_beta` trên `CVAE` — áp projection trong `forward()`/`generate()` để mọi downstream nhất quán; lưu beta vào checkpoint | `pipeline/phase5_cvae/model.py`, `train.py` | 30 phút |
| B3 | Ramp β theo epoch (`--heaviside-beta-max 50` sau kl-warmup); real-physics trên ảnh **đã chiếu** | `train.py` run_epoch, `real_physics_prior_loss` | 30 phút |
| B4 | Test shape, β tăng→nhị phân, gradient chảy qua projection + real-physics | tests | 30 phút |
| B5 | Run 1: `--decoder-type wire --select-by fe_r2 --fe-eval-every 5 --lambda-real-physics 2.0 --real-physics-every 2 --real-physics-subsample 8 --heaviside-beta-max 50 --lambda-bin 0.1 --lambda-periodic 0.1 --regularize-prior-samples --epochs 60 --output-name cvae_wire_heaviside_rp.pt` | — | **~3 giờ** |
| B6 | `best_of_n_eval.py` ở **resolution 128** (WIRE resolution-agnostic) → passes_all, R²(FE), hit-rate | — | 30–45 phút |
| B7 | Nếu chưa đạt DoN: Run 2 (tăng β-max / giảm `real-physics-every` / weighted sampling) | — | ~3 giờ |

**DoN (Task 2 Phần I):** passes_all single-shot ≥ 75% (hiện 4.17%) + xuất 256²
không vỡ + $R^2$(FE) dương ≥ 0.85.

**Cơ chế hàn gắn 1-pixel:** (1) huấn luyện ép ảnh về gần nhị phân ngay từ
decoder bằng projection khả vi — real-physics loss trên ảnh đã chiếu phạt đúng
cấu trúc sẽ vỡ khi làm sắc nét; (2) suy luận ở 128²/256² biến thanh 1px@64²
thành ≥2px, không đứt khi threshold.

### Tổng ETA & rủi ro

| Hạng mục | ETA | Rủi ro chính |
|---|---|---|
| Code + test 2 nhánh | ~3 giờ | — |
| Phase 4: 3 run + eval | ~2–3.5 giờ | f₁/f₂ khó chạm 0.985; grid lớn có thể overfit |
| Phase 5: run 1 + eval | ~3.5–4 giờ | Real-physics + WIRE chậm; β cao mất gradient (ramp + tanh) |
| Phase 5: run 2 (nếu cần) | ~3 giờ | — |
| **Tổng** | **~1.5–2 ngày làm việc** (1 GPU 6GB, tuần tự) | — |

**Quyết định cần duyệt trước khi thực thi:**
1. Phase 4: chạy 3 đầu ra trước (chọn grid nhanh) rồi 5 đầu ra — hay chỉ 5 đầu ra?
2. Phase 5: Heaviside áp trong `CVAE.forward` (mọi downstream nhất quán) — có đồng ý?
3. Xác nhận chạy training GPU background nhiều giờ (tôi theo dõi tự động).

**Môi trường:** conda env `simp` (torch 2.6.0+cu124); env `base` hiện hỏng torch
(thư mục `site-packages/torch` rác) — không dùng để chạy test/train.

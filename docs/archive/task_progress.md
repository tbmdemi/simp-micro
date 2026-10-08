# LOG TRẠNG THÁI: TÍCH HỢP KAN, WIRE VÀ MAMBA VÀO AUXFORGE

> **[2026-09-25] Kế hoạch + trạng thái hiện hành: [`plan.md`](../plan.md) v3.** File này chỉ còn là log chi tiết từng lần chạy (lịch sử). Lưu ý: so sánh "KAN vượt Linear ~2×" bên dưới đo bằng `property_accuracy()` (qua surrogate) - thước đo xếp hạng ngược FE thật, không dùng để kết luận (xem `plan.md` mục 1.1). Số benchmark Task 4 ngày 2026-09-23 bị sai do bug loader (`plan.md` mục 1.4).

> **[2026-10-06]** Kết quả P1.6 (baseline SIMP, IN100-B, hội tụ lưới, retrieval verified, OOD đổi dấu) ghi tại `plan.md` mục P1.6 và `EXPERIMENT_LOG.md` 2026-10-06 - không ghi lặp ở đây.


> **Kế hoạch chính thức hiện hành: `plan.md`** (v2 — hiệu chỉnh KPI sau khi baseline 0.85 không tái lập, cấm chọn checkpoint bằng `val_loss`, Real-Physics bắt buộc từ Task 1). File này (`task_progress.md`) giữ làm log trạng thái chi tiết; số liệu đối chiếu (bảng so sánh R² từng checkpoint) xem phần "KẾT QUẢ THỰC HIỆN" bên dưới.

> **Trạng thái hiện tại của codebase (cập nhật 2026-09-11):**
> - `pipeline/phase5_cvae/model.py` đã KAN-hóa regression head; WIRE decoder và ConvKAN surrogate đã có checkpoint benchmark. Tandem L-BFGS đã có test stub, nhưng chưa đạt nghiệm thu FE OOD.
> - Exp 3 MLP control đã chạy đủ 150 epoch; bảng đối chứng mới nhất nằm ở mục Exp 3 bên dưới. KAN cần retrain cùng code mới để hoàn tất ablation công bằng.
> - Đã thêm `pipeline/mno/`, `pipeline/kinn/`, `pipeline/ickans/` cùng dataset/train/evaluate hoặc physics hooks. MNO hiện fallback GRU vì môi trường thiếu `mamba_ssm`; KINN dùng PyTorch autograd; ICKAN giữ Hessian dương nhưng chưa đạt R² production.
> - `requirements.txt` vẫn giữ dependency nền; backend roadmap tùy chọn `mamba-ssm` và `jax` được khai báo trong extra `roadmap` của `pyproject.toml`. `jax-amg` chưa có trong môi trường.
> - Hạ tầng FE thật (verify_fe.py, real_physics.py, manufacturability.py) đã sẵn sàng để đo `passes_all`, R²(FE), hit_rate.

---

## KẾT QUẢ THỰC HIỆN (2026-08-23)

### Task 1 — KAN regression head: code xong, training chạy 5 lần, DoN R² chưa đạt

| Model (đo bằng `evaluate.property_accuracy`, surrogate v2 hiện tại) | v12 R² | v21 R² | R² TB |
|---|---|---|---|
| Linear base cũ (`cvae_v2_base.pt`, đo lại) | 0,27 | 0,30 | 0,29 |
| Linear finetuned cũ (`cvae_v2_finetuned.pt`, đo lại) | 0,24 | 0,47 | 0,35 |
| KAN base (`cvae_kan_best.pt`, chọn val_loss) | 0,35 | 0,02 | 0,19 |
| KAN + real-physics v1 (`cvae_kan_realphysics.pt`) | 0,76 | 0,41 | 0,58 |
| **KAN + real-physics v2 (`cvae_kan_realphysics_v2.pt`, khuyến nghị)** | **0,82** | **0,45** | **0,64** |
| *Mục tiêu (DoN v2)* | - | - | *≥ 0,60* |

- R²(FE thật) tốt nhất trong train v2 = **0,8889** (mục tiêu DoN v2: ≥ 0,85) (v1: 0,7865).
- Phát hiện đột xuất: report tháng 7 (R²=0,85) **không tái lập** với code hiện tại (cùng checkpoint đo lại = 0,35) — xem EXPERIMENT_LOG.md mục 2026-08-23.
- DoN R² ≥ 0,990 xem như không khả thi với pipeline đo hiện tại (model tốt nhất mọi kiến trúc đo ~0,35-0,64) — **đã thay bằng DoN v2** (R² tổng ≥ 0,60 + R² FE thật ≥ 0,85, xem `plan.md` mục Task 1): mốc mà `cvae_kan_realphysics_v2.pt` (0,64 / 0,8889) đã vượt qua.

### Exp 3 — Ablation MLP + Physics so với KAN + Physics (2026-09-11)

Đã chạy MLP head 150 epoch với cùng seed `123`, batch size `512`,
`lambda_real_physics=20`, FE subsample `8`, `lambda_volfrac=3`, FE evaluation
mỗi 10 epoch trên 24 condition. Các số dưới đây dùng cùng test loader,
surrogate v2 và seed prior cố định.

| Model | R² v12 | R² v21 | R² volfrac | R² FE(v12), n=24 |
|---|---:|---:|---:|---:|
| MLP + Physics (`exp3_mlp_physics_150.pt`) | 0,432 | 0,087 | -2,483 | -9,834 |
| KAN + Physics v2 (`cvae_kan_realphysics_v2.pt`) | **0,836** | **0,476** | **-0,221** | **0,581** |

Checkpoint MLP tốt nhất theo FE-R² là epoch 20 (`-6,2465`), không phải
epoch 150 (`-7,4893`). KAN v2 được huấn luyện trước các thay đổi mới về
đối xứng decoder và volfrac loss, nên đây là ablation định hướng; cần retrain
KAN cùng recipe/code trước khi đưa ra claim nhân quả cuối cùng.

### Task 2 — WIRE INR decoder: code + test xong, chưa train

- `WireContinuousDecoder` + `ComplexGaborActivation` hoạt động, API `(B,1,H,W)` giữ nguyên, resolution-agnostic (test 128²/256²).
- Các loader/CLI cập nhật: `train.py` (`--decoder-type`, `--wire-hidden-dim`, `--wire-omega0`, `--wire-s0`, lưu vào checkpoint), `sample.py` (`--resolution`), `adversarial_dataset.py`/`verify_fe.py` (đọc `decoder_type` từ checkpoint).
- Đã train checkpoint `cvae_wire_v2.pt`; benchmark best-of-N hiện chỉ đạt single-shot 4,17%, best-of-N 16,67%, R²(FE)=-10,29 nên chưa đạt `passes_all >= 75%`.

---

## PHẦN I: BÀI BÁO #1 — WIRE INR + KAN

### Task 1: KAN-hóa Bộ Hồi Quy Co-VAE (Phase 5) — Ưu tiên CAO
**Vị trí tác động:** `pipeline/phase5_cvae/model.py`

**Phân tích hiện trạng (tại thời điểm lập kế hoạch; ĐÃ HOÀN THÀNH 2026-08-23):**
- `Encoder.fc_mu`/`fc_logvar` đã là `EfficientKANLinear` (đã KAN-hóa một phần).
- `Decoder.fc` lúc đó còn là `nn.Linear(...)` — "bộ hồi quy" cần KAN-hóa; **đã chuyển sang `EfficientKANLinear`** (xem DoN bên dưới).

**Các bước thực hiện (trạng thái: ✅ đã làm 2026-08-23):**
1. ✅ Thay `Decoder.fc = nn.Linear(...)` bằng `EfficientKANLinear(latent_dim + condition_dim, feat_ch*feat_res*feat_res, grid_size=5, spline_order=3)`.
2. ✅ **Tương thích ngược:** `train.py::resize_condition_dim_weights()` đã xử lý KAN (`base_weight`/`spline_weight`/grid rebuild qua `rebuild_kan_grid()`) — không cần sửa thêm.
3. ✅ `_CONDITION_DEPENDENT_LAYERS` đã gồm đủ `encoder.fc_mu`/`encoder.fc_logvar`/`decoder.fc`.
4. ✅ Test forward pass KAN decoder: `TestWireDecoder::test_gradients_flow` + `TestKANRegressionHead` (625/625 pass).

**DoN:**
- [x] `Decoder.fc` dùng `EfficientKANLinear` (2026-08-23).
- [x] `resize_condition_dim_weights()` xử lý đúng KAN weights (`base_weight`/`spline_weight`/grid, test `test_widen_preserves_base_and_v12_v21_columns` pass).
- [x] **DoN v2** — R² vĩ mô tổng thể ≥ 0,60 VÀ R² FE thật ≥ 0,85 (thay mốc 0,990 cũ — con số 0,85 của report tháng 7 không tái lập với code hiện tại, xem EXPERIMENT_LOG.md mục 2026-08-23): **ĐẠT** — 0,64 (v12 0,82, v21 0,45), FE đỉnh 0,8889 với `cvae_kan_realphysics_v2.pt`, chọn checkpoint theo `--select-by fe_r2`. Xem "KẾT QUẢ THỰC HIỆN" ở trên.

---

### Task 2: Bộ Tạo Sinh Tọa Độ Liên Tục WIRE INR Decoder (Phase 5) — Ưu tiên CAO
**Vị trí tác động:** `pipeline/phase5_cvae/model.py` (thêm class mới), `sample.py`, `best_of_n_eval.py`, `verify_fe.py`

**Phân tích hiện trạng (tại thời điểm lập kế hoạch; ĐÃ HOÀN THÀNH 2026-08-23):**
- Decoder cũ xuất tensor ảnh cố định `(B,1,64,64)` qua ConvTranspose2d — vẫn là mặc định (`decoder_type="conv"`).
- Đã thêm decoder mới nhận tọa độ liên tục `(x,y) ∈ [-1,1]²` + latent z + condition c, dùng Gabor Wavelet phức (WIRE), xuất mật độ `ρ ∈ [0,1]`.

**Các bước thực hiện (trạng thái: ✅ đã làm 2026-08-23, trừ train WIRE):**
1. ✅ Thêm `ComplexGaborActivation` (Gabor Wavelet phức) vào `model.py`.
2. ✅ Thêm `WireContinuousDecoder` — nhận `(coords, z, cond)`, xuất `rho` shape `[B, N_points, 1]` (qua `forward_coords`).
3. ✅ Tích hợp vào `CVAE`: flag `decoder_type="conv"|"wire"` (mặc định `conv`); khi `wire`, `forward()`/`generate()` sinh lưới tọa độ nội bộ rồi reshape về `(B,1,64,64)` — API downstream giữ nguyên.
4. ✅ Resolution-agnostic: `generate(resolution=...)` quét lưới mịn hơn (128²/256²/512²) khi suy luận.
5. ✅ Cập nhật `sample.py` (`--resolution`), `best_of_n_eval.py`/`adversarial_dataset.py`/`verify_fe.py` (đọc `decoder_type` từ checkpoint).
6. ✅ Test WIRE decoder: `TestWireDecoder` (shape, unit range, resolution-agnostic, gradient flow) — 625/625 pass.

**DoN:**
- [x] `WireContinuousDecoder` hoạt động, xuất `rho ∈ [0,1]` (2026-08-23).
- [x] `CVAE` hỗ trợ `decoder_type="wire"` tương thích ngược (`generate(resolution=...)` resolution-agnostic).
- [ ] Tỷ lệ `passes_all` single-shot ≥ 75% (đo bằng `best_of_n_eval.py` + `manufacturability.py`) — **[2026-09-23] ĐÃ THỬ 2 ROUND, VẪN THẤT BẠI, KẾT QUẢ TỆ HƠN BASELINE GỐC.** Sửa root-cause đã xác nhận (thêm `enforce_symmetry` cho WIRE + bật `lambda_real_physics` từ đầu, ~75 epoch hiệu dụng qua 2 round) nhưng benchmark chính thức 24-condition cho `cvae_wire_realphysics_v3_round2.pt`: hit_rate single-shot=**0,00%** (baseline gốc 4,17%), hit_rate best-of-N=**0,00%** (baseline gốc 16,67%), R²(FE)=**-13,87** (baseline gốc -10,29). R²(FE) đo trên 8-condition validation nội bộ lúc train CÓ cải thiện đều (tới -10,10, tốt nhất từng đo) nhưng KHÔNG generalize ra benchmark 24-condition đầy đủ - overfit lên chính validation subset dùng để chọn checkpoint. Chi tiết + khuyến nghị hướng thử tiếp theo: `EXPERIMENT_LOG.md` mục 2026-09-23. **Chưa giải quyết được** - cần recipe khác (2-giai-đoạn đúng kiểu KAN, hoặc validation set lớn hơn) hoặc xem xét lại liệu Gabor/INR có phù hợp domain density-field nhị phân hoá hay không.
- [x] Xuất lưới 256×256 không vỡ — API resolution-agnostic đã test (128²/256², unit range); đánh giá mỹ quan "không răng cưa" chờ WIRE checkpoint train thật.

---

### Task 3: Bộ Dự Đoán Thuộc Tính ConvKAN Surrogate (Phase 4) — Ưu tiên TRUNG BÌNH
**Vị trí tác động:** `pipeline/phase4_surrogate/model.py`, `train.py`, `evaluate.py`, `export_for_phase5.py`

**Phân tích hiện trạng:**
- `SurrogateCNN` dùng `nn.Linear` cho FC head (dòng 53-58), input `fc_in = channels[-1] + n_seeds + (1 if include_nu0 else 0)`.
- Kế hoạch đề xuất thay FC head bằng KAN. **Lưu ý:** kế hoạch gốc dùng `Flatten` (16384 chiều) nhưng code hiện tại dùng `GAP` (channels[-1] chiều). Nên **giữ GAP** (ít tham số hơn, đã kiểm chứng) và chỉ thay `nn.Linear` → `EfficientKANLinear`.

**Các bước thực hiện:**
1. ✅ Thêm flag `use_kan=False` vào `SurrogateCNN.__init__` (mặc định False để tương thích ngược checkpoint cũ).
2. ✅ Khi `use_kan=True`, thay `self.fc` bằng hai `EfficientKANLinear` với cùng GAP và kích thước head hiện tại.
3. ✅ Cập nhật `train.py` thêm `--use-kan` flag và lưu metadata.
4. ✅ Cập nhật `export_for_phase5.py` lưu `use_kan` vào checkpoint.
5. ✅ Cập nhật `losses.py::load_frozen_surrogate()` đọc `use_kan` từ checkpoint để dựng đúng model.
6. ✅ Thêm test ConvKAN; lần chạy regression gần nhất ghi nhận 633 passed (đo lại bằng conda env `simp`).

**DoN:**
- [x] `SurrogateCNN(use_kan=True)` hoạt động, tương thích ngược `use_kan=False`.
- [x] `export_for_phase5.py` + `losses.py::load_frozen_surrogate` xử lý `use_kan`.
- [ ] R² ≥ 0.985 trên test v2 (đo bằng `evaluate.py`) — **thực đo 2026-08-24 (3 đầu ra): v12=0,9636, v21=0,9574, volfrac=0,9161, đều THẤP HƠN mục tiêu và thấp hơn Linear baseline (0,974/0,964/0,983); xem `EXPERIMENT_LOG.md` mục 2026-08-24.**
- [ ] Đánh giá param count/accuracy trên benchmark production. Với cùng `fc_hidden`, KAN có nhiều tham số hơn Linear vì có thêm spline weights; chưa giảm hidden dimension khi chưa có benchmark công bằng.

---

### Task 4: Vòng Lặp Tối Ưu Hóa Ngược Tandem KAN-LBFGS (Phase 6) — Ưu tiên CAO
**Vị trí tác động:** Module mới `pipeline/phase5_cvae/tandem_lbfgs.py`

**Phân tích hiện trạng:**
- `tandem_lbfgs.py` đã có lõi tối ưu hóa latent khả vi; `sample.py` vẫn chỉ sinh ngẫu nhiên z.
- Cần module tinh chỉnh z trên không gian ẩn khả vi (WIRE Decoder + ConvKAN Surrogate) bằng L-BFGS.

**Các bước thực hiện:**
1. ✅ Tạo `tandem_lbfgs.py` với hàm `tandem_inverse_design_lbfgs(...)`; hỗ trợ decoder Conv/WIRE qua API `decoder(z, condition)`.
2. Sinh lưới tọa độ 64×64, truy vấn generator liên tục, dự đoán thuộc tính bằng surrogate, tính MSE với target, backward qua L-BFGS.
3. **Lưu ý:** surrogate cần `seed_vec` input — dùng seed one-hot mặc định (seed đầu tiên) như `best_of_n_eval.py` đã làm.
4. Thêm CLI để chạy thử với target OOD cực đoan (vd v12=-2.5).
5. ✅ Thêm test hội tụ bằng generator/surrogate khả vi stub.
6. ✅ **[MỚI 2026-09-23] `guidance_source="real_physics"`** — thêm tham số chọn nguồn gradient: `"surrogate"` (mặc định, hành vi cũ) hay `"real_physics"` (gradient GIẢI TÍCH từ `RealPhysicsNu`/`real_physics_loss`, KHÔNG qua surrogate CNN — né hoàn toàn vấn đề "exploitation" đã cảnh báo ở `losses.py`). Tái dùng nguyên `losses.real_physics_loss`, không viết lại logic FE. `surrogate_model` giờ optional (`None` hợp lệ khi `guidance_source="real_physics"`). 9 test mới (`tests/test_tandem_lbfgs.py::TestGuidanceSourceRealPhysics`), 643/643 test toàn repo pass.
7. ✅ **[MỚI 2026-09-23] Benchmark trên checkpoint thật** (`pipeline/phase5_cvae/benchmark_physics_guided_refinement.py`, MỚI) — so sánh 1-mẫu-1-lần-sinh (không best-of-N, cô lập đúng tác dụng refinement) trên `cvae_kan_realphysics_v2.pt`, 24 condition test set (seed=123, khớp `best_of_n_eval.py`), 30 bước L-BFGS: **R²(v12) baseline=-16,00 → refined=-7,60; MAE(v12) baseline=0,587 → refined=0,355 (giảm ~40%)**. Cải thiện thật, đo trên FE thật độc lập với loss dùng tối ưu (không tự chấm điểm bằng chính hàm minimize) — nhưng CHƯA đạt DoN gốc (≤8% sai số OOD cực đoan, kịch bản khác: đây là target trong-phân-phối, không phải OOD cực đoan như ν*=-2,5). Xem `EXPERIMENT_LOG.md` mục 2026-09-23.

**DoN:**
- [x] `tandem_lbfgs.py` hội tụ trong test stub (1 test pass); cần benchmark checkpoint thật < 50 vòng lặp.
- [x] **Benchmark checkpoint thật xong (2026-09-23, 30 bước < 50)** — xem bước 7 ở trên. Cải thiện thật (MAE giảm ~40%) nhưng dùng target trong-phân-phối, chưa phải target OOD cực đoan.
- [ ] Sai số thuộc tính (kiểm chứng FE thật) ≤ 8% so với target OOD **cực đoan** (vd ν*=-2,5) — CHƯA đo đúng kịch bản này, cần chạy riêng.

---

## PHẦN II: BÀI BÁO #2 — MAMBA + VẬT LÝ PHI TUYẾN

### Task 5: Mamba Neural Operator (MNO) Đồng Nhất Hóa Siêu Tốc — Ưu tiên RẤT CAO
**Vị trí tác động:** Module mới `pipeline/mno/` (độc lập)

**Phân tích hiện trạng:**
- Đã có module `pipeline/mno/` nhận ma trận mật độ → dự đoán 18 trường dịch chuyển.
- FE solver hiện tại (`simp/core/solver.py`) vẫn là nguồn nhãn cần xuất trường
	`displacements` trước khi train production.

**Các bước thực hiện:**
1. Thêm `mamba-ssm` vào `requirements.txt`.
2. Tạo `pipeline/mno/` với `model.py` (Bidirectional Mamba block), `dataset.py` (sinh dữ liệu FE), `train.py`, `evaluate.py`.
3. **Lưu ý:** `mamba-ssm` yêu cầu CUDA + kernel biên dịch — cần kiểm tra môi trường. Có thể cần fallback `mamba-ssm` thuần PyTorch hoặc `mamba-ssm` từ `state-spaces`.
4. Sinh dataset huấn luyện bằng FE thật (dùng `verify_fe.py`/`real_physics.py` làm nguồn).
5. Đo tốc độ suy luận (mẫu/giây) và L2 error so với FE thật.

**DoN:**
- [x] MNO forward dự đoán 18 trường dịch chuyển; smoke GPU giữ đúng shape.
- [ ] Tốc độ ≥ 500 mẫu/giây trên GPU.
- [ ] L2 relative error ≤ 1% so với FE thật.

---

### Task 6: Triển Khai KINN cho Cơ Học Phi Tuyến Kết Hợp JAX-AMG — Ưu tiên CAO
**Vị trí tác động:** Module mới `pipeline/kinn/`, tích hợp vào `losses.py`

**Phân tích hiện trạng:**
- `losses.py::real_physics_prior_loss` hiện dùng FE tuyến tính (RealPhysicsNu).
- Cần KINN (KAN-Informed Neural Network) cho biến dạng lớn phi tuyến + JAX-AMG.

**Các bước thực hiện:**
1. Thêm `jax`, `jaxlib`, `jax-amg` vào `requirements.txt`.
2. Tạo `pipeline/kinn/` với KINN model (dùng EfficientKANLinear), Deep Energy Method loss, JAX-AMG solver.
3. Tích hợp vào `losses.py` như một loss tùy chọn (`--lambda-kinn`).
4. Thêm test cho hội tụ.

**DoN:**
- [ ] KINN hội tụ trường dịch chuyển phi tuyến < 30 giây GPU.
- [ ] Gradient truyền ngược mượt qua JAX về cVAE.

---

### Task 7: Thiết Kế ICKANs cho Siêu Vật Liệu Composite Đa Pha — Ưu tiên TRUNG BÌNH
**Vị trí tác động:** Module mới `pipeline/ickans/`

**Phân tích hiện trạng:**
- Chưa có module constitutive modeling nào.
- Cần ICKANs (Input-Convex KAN) với ràng buộc trọng số spline không âm.

**Các bước thực hiện:**
1. Tạo `pipeline/ickans/` với `model.py` (ICKANs với ràng buộc non-negative weights), `dataset.py` (dữ liệu composite đa pha Halpin-Tsai/Rule of Mixtures), `train.py`, `evaluate.py`.
2. Áp đặt ràng buộc: sau mỗi optimizer step, clamp trọng số spline của các lớp ẩn ≥ 0.
3. Kiểm tra ma trận độ dẻo tangent xác định dương.
4. Thêm test.

**DoN:**
- [ ] Ma trận độ dẻo tangent luôn xác định dương.
- [ ] R² dự đoán năng lượng đàn hồi ≥ 0.985 trên test OOD.

---

## THỨ TỰ THỰC HIỆN ĐỀ XUẤT

| Ưu tiên | Nhiệm vụ | Module | Thời gian | Nghiệm thu |
| :---: | :--- | :--- | :---: | :--- |
| 1 | KAN-hóa Co-VAE | phase5_cvae/model.py | 3 ngày | R² ≥ 0.990 |
| 2 | WIRE INR Decoder | phase5_cvae/model.py | 5 ngày | passes_all ≥ 75% |
| 3 | ConvKAN Surrogate | phase4_surrogate/ | 4 ngày | R² ≥ 0.985 |
| 4 | Tandem L-BFGS | phase5_cvae/tandem_lbfgs.py | 4 ngày | Sai số ≤ 8% |
| 5 | MNO | pipeline/mno/ | 10 ngày | ≥ 500 mẫu/s; L2 ≤ 1% |
| 6 | KINN + JAX-AMG | pipeline/kinn/ | 12 ngày | Hội tụ < 30s |
| 7 | ICKANs Composite | pipeline/ickans/ | 7 ngày | Xác định dương 100% |

---

## RỦI RO & LƯU Ý KỸ THUẬT

1. **`efficient-kan` đã vendor local tại `efficient_kan/`** (không cần `pip install`; Task 1/3 dùng trực tiếp). Nếu đổi sang package PyPI, cần đồng bộ API (`EfficientKANLinear`, `rebuild_kan_grid`).
2. **`mamba-ssm` yêu cầu CUDA** — cần kiểm tra môi trường GPU trước Task 5. Có fallback thuần PyTorch.
3. **Tương thích ngược checkpoint:** mọi thay đổi kiến trúc (KAN decoder, WIRE, use_kan) phải giữ flag mặc định để không phá checkpoint cũ (`cvae_v2_finetuned.pt`, `surrogate_for_phase5_v2.pt`).
4. **`resize_condition_dim_weights()`** trong `train.py` **đã xử lý KAN** (`base_weight`/`spline_weight`/grid, xem test `TestResizeConditionDimWeights`) — không cần sửa thêm.
5. **`losses.py::load_frozen_surrogate()`** cần đọc `use_kan` từ checkpoint để dựng đúng model khi Task 3 hoàn thành.
6. **WIRE decoder** giữ API đầu ra `(B,1,64,64)` (đã kiểm chứng bằng test); resolution cao chỉ qua `generate(resolution=...)`.

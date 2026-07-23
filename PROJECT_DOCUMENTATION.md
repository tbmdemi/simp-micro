# SIMP-Micro: Inverse Design Auxetic Metamaterials — Tài liệu toàn diện

**Repo:** `github.com/tbmdemi/simp-micro`, branch `OnlyAuxetic`
**Mục tiêu cuối cùng:** cho trước một cặp hệ số Poisson mong muốn (ν₁₂, ν₂₁ < 0 — "auxetic"), sinh ra một hình học vi cấu trúc (microstructure) đạt được đúng tính chất đó — **inverse design**.

```
Phase 0  Core SIMP Engine (topology optimization)
Phase 1  LHS Screening (khảo sát không gian tham số)
Phase 2  Multi-batch Adaptive DOE (sinh dataset lớn)
Phase 3  Dataset Build (ảnh density field + targets, augment)
Phase 4  CNN Surrogate (density field -> ν₁₂, ν₂₁, volfrac)
Phase 5  Conditional VAE (ν₁₂, ν₂₁ -> density field)   <-- inverse design thật sự
Phase 6-8  FE verify thật, deployment, mở rộng (chưa triển khai)
```

Trạng thái hiện tại: **Phase 0–5 đã có code chạy được và có kết quả thực đo**; Phase 6–8 chưa triển khai.

---

## 0. Core SIMP Engine (`simp/`)

### 0.1 Bài toán & phương pháp

SIMP (Solid Isotropic Material with Penalization) — tối ưu hóa topology một ô cơ sở tuần hoàn (unit cell) để mật độ vật liệu `x ∈ [0,1]` tại mỗi phần tử hội tụ về 0 (rỗng) hoặc 1 (đặc), sao cho hệ số Poisson đồng nhất hóa của ô cơ sở càng âm càng tốt (auxetic).

Vòng lặp chính (`simp/runner.py`):

```
Seed generation → FE Analysis (PBC) → Homogenization (Q, dQ) →
Objective & Sensitivity → Density/Sensitivity Filter → OC Update →
Convergence Check → lặp lại
```

### 0.2 Từng module

| Module | Vai trò | Thuật toán |
|---|---|---|
| `core/fem.py` | Lưới FE, DOF mapping | Phần tử tứ giác 4 nút (Q4), đánh số kiểu Fortran (1-based), giống mã MATLAB gốc Sigmund (2001)/Andreassen (2011) |
| `core/pbc.py` | Periodic Boundary Conditions | Master–slave null-space projection: biên phải ràng buộc = biên trái (u_right=u_left), biên trên = biên dưới. Đúng vì chỉ áp lên trường dao động χ = u − u⁰ (biến dạng đồng nhất đã tách riêng trong U0) |
| `core/solver.py` | Giải hệ FE | Lắp K_global từ KE (SIMP-penalized), giải 3 trường hợp tải đơn vị (ε_xx, ε_yy, γ_xy) cho homogenization. `spsolve` (LU) trước, fallback `CG` + Jacobi preconditioner nếu suy biến |
| `homogenization/compute.py` | Đồng nhất hóa | Energy-based (Xia & Breitkopf, 2015): `Q_ij = (1/|Ω|) Σ_e E_e · χ_e^(i)ᵀ KE χ_e^(j)`, vector hóa bằng `np.einsum`. **Dùng `U_total = U0 + U`** (U0 = chuyển vị biến dạng đơn vị, U = trường dao động χ) |
| `objectives/auxetic.py` | Hàm mục tiêu | Xem 0.3 |
| `core/filter.py` | Lọc mật độ/độ nhạy | Cone filter cổ điển, trọng số giảm tuyến tính theo bán kính `rmin` |
| `core/oc.py` | Cập nhật biến thiết kế | Optimality Criteria, bisection tìm nhân tử Lagrange λ sao cho `mean(xPhys) = volfrac` |
| `core/convergence.py` | Điều kiện dừng | Đa tiêu chí (xem 0.4) |
| `seeds/*.py` | 11 mẫu khởi tạo hình học | circle, square, hourglass, four_circle, hexagonal, nine_circle, cross_rectangular, grid_circular_voids, small_square_cross, circle_half_quarter, **reentrant_bowtie** (mới nhất, khó auxetic nhất) |

### 0.3 Hàm mục tiêu auxetic (chi tiết toán học)

Từ compliance tensor `S = Q⁻¹` (Voigt [11,22,12]):

```
ν₁₂ = -S₁₂ / S₁₁         (công thức tổng quát, dùng nghịch đảo ma trận 3x3 đầy đủ)
ν₂₁ = -S₁₂ / S₂₂
```

Khi ô cơ sở orthotropic đúng trục (không xoay, Q₁₃ = Q₂₃ = 0), rút gọn được `ν₁₂ = Q₁₂/Q₂₂` — nhưng công thức rút gọn này **sai khi có rotation** (xem mục Bugfix). Code hiện tại (`compute_nu12`/`compute_nu21`) luôn dùng nghịch đảo 3×3 đầy đủ nên đúng trong mọi trường hợp, kể cả khi `rotation_deg ≠ 0`.

Objective tối thiểu hóa (dùng Q₁₂ làm proxy cho ν₁₂ vì cùng dấu):

```
c = Q₁₂ − μ·(Q₁₁ + Q₂₂) + penalty_terms
```

- `μ`: hệ số kéo Q₁₂ xuống âm mạnh hơn. **Hiện đang bị vô hiệu hóa (`μ = 0`)** vì công thức bị đánh giá là *conceptually flawed* — có thể gây void collapse mà không chắc cải thiện auxeticity — đang chờ redesign.
- `penalty_terms`: kích hoạt khi `Q₁₁` hoặc `Q₂₂` < `δ = 0.1·volfrac·E₀`, chuẩn hóa theo `δ²`, để tránh kết cấu sụp đổ (mất độ cứng).

### 0.4 Điều kiện hội tụ

Dừng khi **bất kỳ** điều kiện nào sau đúng:
1. Độ thay đổi thiết kế < `tol_change` (mặc định 0.01)
2. Objective ổn định — thay đổi tương đối < `tol_obj` (0.05) trong `window_size` (20) vòng liên tiếp
3. Đạt `max_iter` (mặc định 200)

Có `min_iter` để tránh dừng quá sớm. Trên 7.920 mẫu của Phase 2: **FE convergence rate = 100%**.

### 0.5 Các bug nghiêm trọng đã fix

| Bug | Ảnh hưởng | Fix |
|---|---|---|
| **FE displacement bug** trong `compute_homogenized_tensor()` | Dùng trực tiếp trường dao động χ làm tổng chuyển vị thay vì `U_total = U0 + U` → làm sai lệch toàn bộ tensor Q của **mọi** lần chạy | Sửa thành `U_total = U0 + U` |
| **Công thức ν₁₂ rút gọn dưới rotation** | `ν₁₂ = Q₁₂/Q₂₂` chỉ đúng khi `Q₁₃=Q₂₃=0`; sai ở `rotation_deg=32.4°` và tương tự → **Phase 1 screening ban đầu ra 0 mẫu auxetic** | Viết lại dùng nghịch đảo ma trận 3×3 đầy đủ `S=Q⁻¹` |
| **Tham số phạt `μ`** | Có thể gây collapse kết cấu mà không chắc cải thiện auxeticity | Vô hiệu hóa mặc định (`μ=0.0`), chờ redesign |
| **Dấu objective cho `first`/`second`** | OC update sai hướng cho các objective maximize | Đã loại bỏ, chỉ còn `auxetic` |
| **`max` thay `min`** trong `aggregate_correlations.py` | Chọn nhầm giá trị objective *tệ nhất* làm "best" | Sửa `max(...)` → `min(...)` |

Chi tiết đánh giá từng module xem [`REVIEW_ALGORITHMS_VI.md`](REVIEW_ALGORITHMS_VI.md) (báo cáo review độc lập, 2026-06-06 — dùng tên dự án cũ "Input_SIMP_Analyst", nay là AuxForge).

### 0.6 Chạy Phase 0 (1 lần chạy SIMP đơn lẻ)

```bash
pip install numpy scipy matplotlib pandas scikit-learn pillow
python -m simp.run                                    # cấu hình mặc định
python -m simp.main --nelx 80 --nely 60 --volfrac 0.35 --seed hexagonal --rotation_deg 30
```

**Điều kiện tiên quyết:** Python ≥3.10, các thư viện ở `requirements.txt`.

**Output:** `outputs/simp_results_{seed}/`
- `iteration_XXXXX.png` — density field (đen=rỗng, trắng=đặc)
- `iteration_data.csv` — lịch sử hội tụ (Iteration, Poisson_v12, Poisson_v21, Objective, Volume_Fraction)
- `metadata.json` — git_hash, timestamp, version, params

---

## 1. Phase 1 — LHS Screening

### Mục đích
Khảo sát không gian tham số (`volfrac`, `penal`, `rmin`, `move`, `void_size_frac`, `rotation_deg`) bằng Latin Hypercube Sampling để xác định tham số nào ảnh hưởng mạnh nhất đến kết quả auxetic, trước khi đầu tư chạy dataset lớn.

### Thuật toán
- `LatinHypercube(d=n_dims, seed=42)` — tái lập được.
- Chạy song song bằng multiprocessing (`pipeline/phase1_screening/screening_parallel.py`).
- Phân tích độ nhạy bằng **Spearman correlation** giữa từng tham số và kết quả (ν₁₂), lọc NaN, yêu cầu `n_valid ≥ 5`.

### Cách chạy
```bash
python -m pipeline.phase1_screening.screening_parallel --objective auxetic --seed hexagonal
python -m pipeline.phase1_screening.screening_parallel --all     # toàn bộ 11 seed
```

**Điều kiện tiên quyết:** Core engine (Phase 0) hoạt động đúng — đặc biệt công thức ν₁₂ và FE displacement, vì lần chạy Phase 1 đầu tiên đã bị 2 bug này che khuất hoàn toàn kết quả (0/N mẫu auxetic).

### Kết quả thực tế
- **Lần chạy đầu: 0 mẫu auxetic** — truy vết qua 3 tầng lỗi: (1) công thức ν₁₂ sai dưới rotation, (2) penalty term chưa scale đúng, (3) bug FE displacement.
- Sau khi fix: **`volfrac` là tham số chi phối mạnh nhất** (Spearman r ≈ 0,87–0,96); `move`, `rmin`, `void_size_frac` không có ý nghĩa thống kê.

### Output
`outputs/pipeline/phase1/` — JSON kết quả từng mẫu (params, obj_value, converged, v12, v21...) + báo cáo tổng hợp Spearman.

---

## 2. Phase 2 — Multi-batch Adaptive DOE

### Mục đích
Sinh dataset lớn, đa dạng, có kiểm soát chất lượng, bằng vòng lặp DOE (Design of Experiments) thích ứng — mỗi batch học từ batch trước để quyết định thu hẹp/mở rộng không gian tham số.

### Thuật toán (`pipeline/phase2_multi_batch/`)
1. **Sampling** (`sampling.py`): chiến lược Sobol (explore, batch đầu) hoặc Optimized LHS (refine, các batch sau), sinh ra bộ tham số mới cho mỗi batch.
2. **Coverage analysis** (`coverage.py`): ước lượng mật độ KDE trên không gian đặc trưng đã thu thập, phát hiện vùng thưa (sparse region) cần lấy mẫu thêm.
3. **Adaptive decision** (`adaptive.py`): sau mỗi batch, quyết định:
   - **refine** — thu hẹp range tham số quanh vùng tốt + nhắm vùng thưa
   - **expand** — thêm seed/objective mới
   - **stop** — khi N batch liên tiếp không cải thiện objective tốt nhất
4. **Runner** (`runner.py`): thực thi batch bằng `multiprocessing.Pool`.
5. **Visualize** (`visualize.py`): HTML report tiến trình batch (`outputs/multi_batch/reports/batch_progression.html`).

### Cách chạy
```bash
python -m pipeline.phase2_multi_batch.main --phase1-summary outputs/pipeline/phase1
```

**Điều kiện tiên quyết:** Kết quả Phase 1 (để khởi tạo range tham số ban đầu, dựa trên sensitivity `volfrac` chi phối).

### Kết quả thực tế (8 batch, dừng tự động)

| Batch | Chiến lược | n mẫu | % Auxetic | ν₁₂ tốt nhất |
|---|---|---|---|---|
| 1 | Sobol (explore) | 1.320 | 74,9% | −0,612 |
| 2 | Sobol (explore) | 600 | 79,7% | −0,519 |
| 3 | Sobol (explore) | 720 | 71,8% | −0,565 |
| 4 | Optimized LHS (refine) | 1.056 | 83,6% | −0,605 |
| 5 | Optimized LHS (refine) | 1.067 | 85,7% | −0,752 |
| 6 | Optimized LHS (refine) | 1.045 | 85,4% | −0,649 |
| 7 | Optimized LHS (refine) | 1.056 | 85,3% | −0,621 |
| 8 | Optimized LHS (refine) | 1.056 | 87,8% | **−0,807** |

- **Tổng: 7.920 mẫu, 100% FE hội tụ.**
- Range `volfrac` co hẹp từ `[0.45, 0.70]` xuống `[0.50, 0.58]` — nhất quán với kết luận sensitivity Phase 1.
- Pipeline tự dừng sau batch 8 khi `n_batches_no_improvement = 2` — range tham số đã hội tụ, sparsity ổn định ~18,5%.
- **Auxetic success rate theo seed** (đo trên toàn bộ 7.920 mẫu, tiêu chí ν₁₂<0 AND ν₂₁<0):

| Seed | % Auxetic | Seed | % Auxetic |
|---|---|---|---|
| grid_circular_voids | 99,4% | small_square_cross | 93,1% |
| nine_circle | 98,9% | four_circle | 87,9% |
| square | 94,0% | hourglass | 67,8% |
| circle | 93,8% | hexagonal | 64,4% |
| cross_rectangular | 93,3% | circle_half_quarter | 61,7% |
| | | **reentrant_bowtie** | **48,6% (khó nhất)** |

### Output
`outputs/multi_batch/batch_{1..8}/` — kết quả từng mẫu; `decision_batch{N}.json` — quyết định adaptive; `outputs/multi_batch/reports/batch_progression.html` — báo cáo trực quan.

---

## 3. Phase 3 — Dataset Build (density field + targets)

### Mục đích
Chuyển 7.920 kết quả SIMP thô (PNG + CSV) thành dataset ML-ready: ảnh 64×64 chuẩn hóa + nhãn (targets) + train/val/test split + augmentation.

### Pipeline (`pipeline/phase3_dataset/`)
1. **`scan_dataset.py`** → quét toàn bộ batch, tạo `manifest.csv` (đường dẫn ảnh + targets thô).
2. **`build_npz.py --resolution 64`** → resize PNG gốc (616×616, render matplotlib `imshow` từ lưới `xPhys` 50×50 thô) xuống 64×64 bằng box-filter downsampling, đóng gói `dataset_64.npz`.
3. **`augment_symmetry.py`** → augmentation dựa trên vật lý đối xứng (physics-aware), chỉ áp cho tập train.
4. **`finalize_dataset.py --resolution 64`** → lọc outlier + stratified split theo seed → `train.npz`, `val.npz`, `test.npz`.

### Thuật toán augmentation (physics-aware)
- Xoay **90°/270°**: hoán đổi vai trò 2 trục ô cơ sở → **ν₁₂ ↔ ν₂₁** (đổi vị trí target theo phép biến đổi hình học tương ứng).
- Xoay **180°**, lật ngang/dọc: giữ nguyên `ν₁₂, ν₂₁` (đối xứng bảo toàn tính chất).
- Tổng hệ số nhân: **×6** cho tập train.

### Lọc outlier
33/7.920 mẫu (0,4%) bị loại: topology suy biến — `volfrac_achieved` rơi ngoài `[0.05, 0.95]` dù `converged=True` (kết cấu gần như đặc hoàn toàn hoặc rỗng hoàn toàn, không có ý nghĩa thiết kế).

### Cách chạy
```bash
python3 pipeline/phase3_dataset/scan_dataset.py
python3 pipeline/phase3_dataset/build_npz.py --resolution 64
python3 pipeline/phase3_dataset/finalize_dataset.py --resolution 64
```

**Điều kiện tiên quyết:** Toàn bộ 8 batch Phase 2 đã chạy xong (`outputs/multi_batch/batch_*`).

### Kết quả thực tế (`outputs/phase3/split_report.json`)

| | Số mẫu | Ghi chú |
|---|---|---|
| Tổng thô | 7.920 | |
| Bị loại (degenerate) | 33 (0,4%) | |
| Train trước augment | 5.520 | |
| **Train sau augment** | **33.120** | ×6 |
| Val | 1.183 | |
| Test | 1.184 | |

- Split 70/15/15, **stratified theo seed** — cả 11 seed đều có mặt tỉ lệ đều trong mỗi split (~108 mẫu/seed ở val/test).
- Khoảng giá trị ν₁₂: train `[-0.807, 0.276]`, val `[-0.621, 0.236]`, test `[-0.627, 0.370]`.
- **Targets hiện có:** `v12`, `v21`, `volfrac_achieved`. (Roadmap gốc còn dự kiến `f1, f2` — độ cứng chuẩn hóa `E₁₁/E₀`, `E₂₂/E₀` — nhưng `compute_homogenized_tensor()` **chưa** xuất giá trị này; cần mở rộng trước khi thêm vào surrogate/cVAE.)

### Output
`outputs/phase3/{train,val,test}.npz` — mỗi file chứa mảng `images (N,64,64)`, `v12`, `v21`, `volfrac_achieved`, `seed_onehot (N,11)`, `seed_classes`.
`outputs/phase3/manifest.csv`, `split_report.json`.

---

## 4. Phase 4 — CNN Surrogate Model

### Mục đích
Học một mạng thay thế (surrogate) dự đoán nhanh `(ν₁₂, ν₂₁, volfrac)` trực tiếp từ ảnh density field, **không cần chạy FE** — dùng để đánh giá "property-consistency loss" khi train cVAE ở Phase 5 (vì FE thật quá chậm để đưa vào trong vòng lặp gradient descent).

### Kiến trúc (`pipeline/phase4_surrogate/model.py` — `SurrogateCNN`)
```
Input: density field (1,64,64) + seed one-hot (11,)
  4× ConvBlock: Conv2d(3x3) → BatchNorm → ReLU → MaxPool2d(2)
     kênh: 32 → 64 → 128 → 256   (64x64 → 32 → 16 → 8 → 4)
  → Global Average Pool → flatten (256,)
  → concat seed one-hot (11,) → (267,)
  → FC(267→128) → ReLU → Dropout(0.2) → FC(128→3)
Output: [v12, v21, volfrac_achieved]
```

### Loss & training (`train.py`)
- **Weighted MSE**: `weights=[1.0, 1.0, 0.3]` — ưu tiên ν₁₂/ν₂₁ hơn volfrac (roadmap 4.1).
- Optimizer: AdamW, `lr=5e-4` (đã giảm từ `1e-3` sau khi phát hiện val_loss dao động mạnh giữa epoch ở lần train đầu).
- `ReduceLROnPlateau` (factor 0.5, patience 4) + **gradient clipping** `max_norm=1.0` (để tránh update quá lớn gây val_loss "nhảy" 0,0055→0,030→0,0068 dù train_loss giảm đều).
- Early stopping: `patience=10` epoch không cải thiện val_loss.

### Cách chạy
```bash
python3 pipeline/phase4_surrogate/train.py
python3 pipeline/phase4_surrogate/train.py --epochs 100 --batch_size 256   # tùy chỉnh
```

**Điều kiện tiên quyết:** `outputs/phase3/{train,val}.npz` tồn tại (Phase 3 hoàn tất).

### Kết quả thực tế (`outputs/phase4/evaluation_report.json`, test set)

| Target | R² | MAE |
|---|---|---|
| ν₁₂ | **0,910** | 0,0373 |
| ν₂₁ | **0,911** | 0,0363 |
| volfrac_achieved | **0,982** | 0,0074 |

- Đạt mục tiêu roadmap R² ≥ 0,90 cho cả 3 target ngay ở lần train đầu (best checkpoint: epoch 38/48, val_loss=0,00551).
- MAE theo seed dao động 0,021–0,051 — `reentrant_bowtie` có MAE thấp nhất (0,021), `four_circle`/`square` cao nhất (~0,048–0,050).
- Vùng `ν₁₂ ∈ [-0.8,-0.6)` chỉ có 1 mẫu test → MAE ở vùng này (0,244) không đáng tin cậy (undersampled).

### Output
`outputs/phase4/surrogate_best.pt` (checkpoint), `train_history.json`, `evaluation_report.json`.
`export_for_phase5.py` đóng gói checkpoint kèm `usage_note` (surrogate chỉ đáng tin trong phân phối train: 11 seed đã thấy, ν₁₂ ∈ khoảng ~[-0,81, 0,37]) → `outputs/phase4/surrogate_for_phase5.pt`, dùng làm surrogate **đóng băng** cho Phase 5.

---

## 5. Phase 5 — Conditional VAE (Inverse Design)

### Mục đích
Đây là bước **inverse design thật sự**: học một mô hình sinh (generative model) nhận **target (ν₁₂, ν₂₁)** làm điều kiện, sinh ra ảnh density field (64×64) đạt tính chất đó — không cần biết trước seed hay bất kỳ tham số SIMP nào.

### Kiến trúc (`pipeline/phase5_cvae/model.py` — `CVAE`)

**Encoder:** giống hệt backbone SurrogateCNN (4× ConvBlock 32→64→128→256) nhưng **giữ nguyên feature map không gian** (không GAP) để decoder có đủ thông tin tái tạo hình học:
```
64×64×1 → 32 → 16 → 8 → 4×4×256 → flatten (4096,)
→ concat condition [v12,v21] (2,) → FC → (mu, logvar) kích thước latent_dim=32
```

**Decoder** (đối xứng ngược, ConvTranspose2d):
```
[z (32,), condition (2,)] → FC → reshape (256,4,4)
→ 4× ConvTranspose2d (upsample ×2): 256→128→64→32→1
→ Sigmoid → density field (1,64,64) ∈ [0,1]
```

**Quyết định thiết kế quan trọng:** condition vector **chỉ gồm [ν₁₂, ν₂₁]**, KHÔNG gồm seed one-hot — đúng tinh thần inverse design ("chỉ cần Poisson ratio mong muốn, không cần biết trước seed nào"). `seed_onehot` vẫn được Dataset trả về nhưng chỉ dùng phụ trong `evaluate.py` để phân tích latent space (VD: z có tự phân cụm theo seed family không).

### Loss (`losses.py`) — 3 thành phần

1. **Reconstruction (BCE)** — vì density field gần nhị phân (0=rỗng,1=đặc), BCE phù hợp bản chất phân loại pixel hơn MSE. `reduction="sum"` rồi chia theo batch.
2. **KL divergence** — có **KL annealing**: `beta` tăng tuyến tính 0→1 qua `kl-warmup` epoch (mặc định 30), **không cố định beta=1** như VAE gốc. Lý do: condition chỉ 2 chiều — nếu bật KL full ngay từ đầu, decoder rất dễ "bỏ qua" z (posterior collapse) vì chỉ cần dựa vào condition cũng giảm được recon loss khá nhiều.
3. **Property-consistency loss** — đưa **ảnh liên tục** (không binarize) do decoder sinh ra thẳng vào **surrogate đã đóng băng** (Phase 4) để dự đoán lại (ν₁₂, ν₂₁), so với target condition bằng MSE. Đây là loss bắt buộc decoder phải sinh hình học thật sự đạt đúng Poisson ratio, không chỉ "trông giống" density field.
   - Dùng `seed_vec` **thật** của ảnh gốc (vì surrogate cần input này) — là một xấp xỉ hợp lý cho baseline, nhưng không phản ánh đúng kịch bản generate thật (lúc đó không có seed thật). Đây là giới hạn đã biết, để TODO cho bản nâng cấp.

**Tổng hợp:**
```
total = recon + beta·kl + gamma · PROP_LOSS_SCALE · prop
```
`PROP_LOSS_SCALE = 1000` — hệ số cố định để đưa property loss (~0,01–0,05) về cùng thang với recon/kl (~1000) **trước khi** nhân gamma, để `gamma` giữ ý nghĩa "trọng số cơ sở hợp lý" thay vì phải tự dò số hàng trăm/nghìn. (Phát hiện quan trọng: nếu không có scale này, `gamma=1` mặc định khiến property loss gần như không có tiếng nói trong gradient — model chỉ tối ưu recon, và property accuracy thật sự (đánh giá bằng z~N(0,1) độc lập) rất tệ dù `prop` log ra vẻ tốt trong lúc train.)

### Training (`train.py`)

- `--latent-dim 32`, `--epochs 100`, `--batch-size 64`
- `--kl-warmup 30` — số epoch beta 0→1
- `--gamma` — trọng số property loss (đã scale) — **tham số quan trọng nhất cần tune**, xem sweep bên dưới
- Optimizer Adam `lr=1e-3`, `CosineAnnealingLR` (eta_min=1e-5)
- Early stopping `patience=15` theo val total loss
- `deterministic=True` (dùng mu, không sample) khi validate — loại nhiễu ngẫu nhiên khỏi so sánh giữa epoch.

### Cách chạy
```bash
python3 pipeline/phase4_surrogate/export_for_phase5.py   # (bắt buộc trước) tạo surrogate_for_phase5.pt
python3 pipeline/phase5_cvae/train.py --gamma 20 --kl-warmup 30 --epochs 100
python3 pipeline/phase5_cvae/evaluate.py                 # đánh giá sau train
python3 pipeline/phase5_cvae/sample.py --v12 -0.6 --v21 -0.6 --n 8   # sinh geometry mới
```

**Điều kiện tiên quyết:**
- `outputs/phase3/{train,val,test}.npz` (Phase 3)
- `outputs/phase4/surrogate_for_phase5.pt` (Phase 4, đã export và đóng băng)

### Đánh giá (`evaluate.py`) — 3 phép kiểm tra

1. **`property_accuracy()`** — quan trọng nhất: với mỗi mẫu test, sample `z ~ N(0,1)` **thật sự độc lập** (không "ăn gian" bằng z suy ra từ ảnh gốc như lúc train), sinh ảnh theo condition, đưa qua surrogate dự đoán lại (ν₁₂,ν₂₁), so với target → R²/MAE. Đây là con số phản ánh đúng khả năng "nghe lời" condition khi generate thật.
2. **`diversity_check()`** — giữ condition cố định, sample nhiều z khác nhau, đo pixel-wise std giữa các mẫu. std≈0 → nghi ngờ posterior collapse (decoder bỏ qua z hoàn toàn).
3. **`interpolation()`** — nội suy tuyến tính z giữa 2 mẫu test (giữ chung 1 condition) → chuỗi ảnh kiểm tra latent space có mượt không (không nhảy hình học đột ngột).

### Kết quả thực tế — **gamma sweep** (test set, 1.184 mẫu)

| gamma | R² (ν₁₂) | R² (ν₂₁) | MAE (ν₁₂) | Nhận xét |
|---|---|---|---|---|
| 1 | **−0,418** | −0,439 | 0,174 | Property loss gần như bị bỏ qua — tệ hơn cả dự đoán trung bình |
| 5 | 0,450 | 0,440 | 0,106 | Cải thiện rõ |
| **20** | **0,633** | **0,617** | **0,086** | **Cấu hình tốt nhất hiện tại** |

→ Xác nhận thực nghiệm: `gamma` càng lớn (đến một ngưỡng), property-consistency càng được tôn trọng khi generate thật; `gamma=1` mặc định gốc là không đủ.

**Diagnostics (gamma=20):**
- `diversity_check`: pixel_std = 0,314 tại condition (−0,6,−0,6) → **không có posterior collapse** (đủ đa dạng khi giữ condition cố định, đổi z).
- Best checkpoint: epoch 39/40, val_total=1201,1 (recon=1108,3, kl=67,8, prop_weighted=25,0).

**Đánh giá tổng thể:** cVAE đã "nghe lời" condition ở mức trung bình-khá (R²≈0,62–0,63) — đủ tốt cho baseline, nhưng **chưa đủ tin cậy để dùng trực tiếp** mà không verify: đúng như cảnh báo trong `sample.py`, mọi geometry sinh ra cần **verify lại bằng FE thật ở Phase 6** trước khi kết luận đạt đúng Poisson ratio mong muốn (vì property loss lúc train/eval chỉ dựa trên surrogate, không phải FE thật).

### Output
`outputs/phase5/cvae_best.pt`, `train_history.json`, `evaluation_report.json`, `eval_gamma{1,5,20}.json` (kết quả sweep), `diagnostics/diversity_*.png`, `diagnostics/interpolation_*.png`, `samples/v12_X_v21_Y/sample_XX.png` (từ `sample.py`).

---

## 6. Tổng kết trạng thái & việc còn lại

| Phase | Trạng thái | Kết quả chính |
|---|---|---|
| 0 — Core SIMP | ✅ Ổn định | 11 seed, PBC, energy-based homogenization, 2 bug nghiêm trọng đã fix |
| 1 — LHS Screening | ✅ Hoàn thành | volfrac là tham số chi phối (r≈0,87–0,96) |
| 2 — Multi-batch DOE | ✅ Hoàn thành | 7.920 mẫu, 87,8% auxetic ở batch cuối, ν₁₂ tốt nhất = −0,807 |
| 3 — Dataset Build | ✅ Hoàn thành | 33.120 train (sau augment) / 1.183 val / 1.184 test |
| 4 — CNN Surrogate | ✅ Hoàn thành | R²: ν₁₂=0,910, ν₂₁=0,911, volfrac=0,982 |
| 5 — cVAE | ✅ Baseline hoàn thành | R² (gamma=20): ν₁₂=0,633, ν₂₁=0,617 — chưa verify FE thật |
| 6 — FE verify thật | ⬜ Chưa triển khai | Cần thiết trước khi tin tưởng output Phase 5 |
| 7-8 — Deployment/mở rộng | ⬜ Chưa triển khai | |

### Việc còn lại quan trọng (theo mức ưu tiên)

1. **Phase 6 (bắt buộc trước khi dùng thật):** verify các mẫu sinh bởi cVAE bằng FE thật (chạy lại `simp/homogenization` trên ảnh binarize từ output cVAE) — vì mọi con số R² ở Phase 4/5 hiện tại đều dựa trên surrogate, chưa đối chiếu FE.
2. **Redesign tham số `μ`** trong objective auxetic (đang tắt) — có thể giúp Phase 2 đạt ν₁₂ âm hơn nữa nếu làm đúng.
3. **Property-consistency loss dùng seed thật** (Phase 5) — cần giải pháp tổng quát hơn không phụ thuộc seed ground-truth khi generate (trung bình dự đoán qua nhiều seed_vec, hoặc học seed phổ biến nhất).
4. **Mở rộng `compute_homogenized_tensor()`** để xuất `f1, f2` (độ cứng chuẩn hóa `E₁₁/E₀, E₂₂/E₀`) — cần cho roadmap gốc nhưng chưa có.
5. **Early-stopping Phase 5** nên loại trừ giai đoạn KL-warmup khỏi so sánh `best_val` (beta thay đổi làm `total` không so sánh được xuyên epoch trong lúc warmup).

---

## 7. Tham khảo

- Sigmund, O. (2001). *A 99 line topology optimization code written in Matlab.* Structural and Multidisciplinary Optimization, 21(2), 120–127.
- Andreassen, E., et al. (2011). *Efficient topology optimization in MATLAB using 88 lines of code.* Structural and Multidisciplinary Optimization, 43(1), 1–16.
- Xia, L., & Breitkopf, P. (2015). *Design of materials using topology optimization and energy‑based homogenization.* Archives of Computational Methods in Engineering, 22(2), 229–260.
- Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods, and Applications.* Springer.

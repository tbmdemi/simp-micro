# Kiến trúc hệ thống

> Tài liệu kiến trúc: bản đồ module hiện tại, luồng dữ liệu qua 8 phase, và các điểm mở rộng mà
> roadmap cũ [`PROJECT_PLAN.md`](archive/PROJECT_PLAN.md) (nay thay bằng [`plan.md`](plan.md) v3) sẽ chạm vào, cùng chi tiết layer-by-layer các mạng
> neural đang dùng (§2). Phần "kiến trúc đích" ở §5 viết **trước khi Nhóm 1 (ν0 vật liệu nền) hoàn
> thành** - nhánh A1-A7 trong sơ đồ/bảng §4-5 giờ đã là code thật trên `main` (xem
> [`PROJECT_PLAN.md` Nhóm 1](archive/PROJECT_PLAN.md#nhóm-1---giai-đoạn-a-vary-vật-liệu-nền-l-effort-lớn-nhất-ưu-tiên-cao-nhất),
> [`EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md) mục 2026-08-18/19), không còn "giả định"; chỉ nhánh F
> (nhiệt/CTE, Nhóm 6) vẫn thật sự giả định/chưa code. File `END_TO_END_SCENARIO.md` được nhắc tới ở
> §4-5 (bản nháp trước khi Nhóm 1 chạy thật) **không tồn tại** trong repo - dùng `PROJECT_PLAN.md` +
> `EXPERIMENT_LOG.md` làm nguồn thật thay thế. Không có sơ đồ nào ở đây thay thế
> [`docs/PIPELINE.md`](PIPELINE.md) (lệnh chạy + số liệu thật từng phase) hay
> [`docs/PHYSICS_AND_ML.md`](PHYSICS_AND_ML.md) (bản chất toán học) - tài liệu này chỉ tập trung vào
> **cấu trúc module, phụ thuộc giữa chúng, và kiến trúc mạng neural cụ thể**.

## 1. Bản đồ module hiện tại

```
simp/                           # Lõi vật lý - độc lập với pipeline/, không phụ thuộc PyTorch
├── core/                       # FE assembly, solver, PBC (real-valued, 3 load case biến dạng)
│   ├── fem.py                  #   build_dof_mesh - lưới bậc tự do
│   ├── solver.py               #   solve_fe - giải hệ tuyến tính KU=F
│   └── pbc.py                  #   build_pbc - điều kiện biên tuần hoàn (chỉ real-valued, không Bloch)
├── materials/
│   └── isotropic.py            #   Material - E0, Emin, nu (validate ở __init__), D matrix
├── homogenization/
│   └── compute.py              #   compute_homogenized_tensor - Q (3x3), dQ (3x3xnelyxnelx)
├── objectives/
│   └── auxetic.py              #   hàm mục tiêu SIMP (auxetic, v12/v21) + compute_elastic_constants (Ex,Ey,Gxy,B_eff) + compute_wave_speeds
├── seeds/                      #   hình dạng khởi tạo (hexagonal, hourglass, reentrant_bowtie)
└── io/                         #   đọc/ghi kết quả + reconstruct_3d_bent_surface (minh họa định tính, KHÔNG phải FE)

pipeline/                       # Orchestration 8-phase, PHỤ THUỘC simp/ (không ngược lại)
├── params.py                   #   PARAM_SPACE, FIXED_PARAMS - dùng cho Phase 1 (nelx=nely=80)
├── phase1_screening/           #   LHS sweep, phân tích độ nhạy Spearman
├── phase2_multi_batch/         #   Adaptive DOE (nelx=nely=50, ĐỘC LẬP DEFAULT_FIXED riêng)
│   ├── params.py                #   ACTIVE_PARAMETERS (production), NU_ACTIVE_PARAMETERS (Nhóm 1)
│   └── runner.py                #   DEFAULT_FIXED, build_params_dict - override theo sample_values
├── phase3_dataset/             #   scan_dataset, build_npz, finalize_dataset -> {train,val,test}.npz
├── phase4_surrogate/            #   CNN surrogate (PyTorch), đóng băng khi dùng làm loss cho Phase 5
│   ├── dataset.py, model.py, train.py, evaluate.py, export_for_phase5.py
├── phase5_cvae/                 #   Conditional VAE + differentiable-physics fine-tune
│   ├── dataset.py, model.py, train.py       #   KAN regression head (fc = EfficientKANLinear), decoder_type="conv"|"wire" (WIRE INR, Task 2), condition_dim, extended_condition, condition-dropout
│   ├── real_physics.py                       #   RealPhysicsNu (torch.autograd.Function, FE thật) + solve_elastic_with_grad (E/G/B + gradient)
│   ├── losses.py                             #   recon + beta*kl + gamma*prop_loss (+volfrac_consistency)
│   ├── best_of_n_eval.py                     #   composite_score (0.6/0.3/0.1), oracle FE selection
│   └── sample.py                             #   inference, force_periodic() mặc định bật, --resolution (chỉ wire)

analysis/scripts/                # Script sinh dữ liệu ĐỘC LẬP với pipeline/phase2_multi_batch/
└── generate_production_batch.py #   nguồn thật của outputs/phase3/*.npz hiện tại (xem PIPELINE.md)
```

**Ranh giới phụ thuộc quan trọng (không được vi phạm khi mở rộng):**
`simp/` không import gì từ `pipeline/` hay PyTorch - đây là lớp vật lý thuần NumPy/SciPy, dùng lại
được ở cả Phase 1-3 (NumPy) lẫn Phase 5 (`real_physics.py` gọi `simp/core/solver.py` trực tiếp bên
trong `torch.autograd.Function`, không qua surrogate).

## 2. Kiến trúc chi tiết các mạng neural (layer-by-layer)

Tham số đo thật bằng `sum(p.numel() for p in model.parameters())` trên chính code hiện hành (không phải ước lượng lý thuyết).

### 2.1 `SurrogateCNN` (Phase 4) — [`pipeline/phase4_surrogate/model.py`](../pipeline/phase4_surrogate/model.py)

Dự đoán (ν₁₂, ν₂₁, volfrac[, f1, f2]) trực tiếp từ ảnh mật độ 64×64 - oracle nhanh, đóng băng khi dùng làm loss cho Phase 5.

```
(B,1,64,64)
 → 4× ConvBlock: Conv2d(3×3,pad=1) → BatchNorm2d → ReLU → MaxPool2d(2)
     kênh 1→32→64→128→256, không gian 64→32→16→8→4
 → AdaptiveAvgPool2d(1) (GAP, KHÔNG Flatten toàn bộ feature map)
 → flatten → (B,256) → concat [seed_onehot(11)[, nu0(1)]]
 → FC head: Linear(256+11[+1], 128) → ReLU → Dropout(0.2) → Linear(128, n_outputs)
```

`use_kan=True` thay 2 lớp FC cuối bằng `EfficientKANLinear` (§2.4), giữ nguyên phần conv+GAP. Tham số (`n_seeds=11, n_outputs=3`): **423.491** (head Linear) vs **699.840** (head KAN, +65%). GAP thay vì Flatten 4096 chiều (kế hoạch gốc) là quyết định đã chốt - ít tham số hơn, không giảm R² đo được.

### 2.2 `CVAE` — Encoder + Decoder conv (Phase 5, mặc định) — [`pipeline/phase5_cvae/model.py`](../pipeline/phase5_cvae/model.py)

**Encoder** (cấu trúc conv giống `SurrogateCNN` nhưng **giữ nguyên feature map không gian**, không GAP - decoder cần đủ thông tin không gian để tái tạo):
```
(B,1,64,64) → 4× EncoderBlock (1→32→64→128→256, mỗi block /2 không gian: 64→4)
 → flatten → (B,4096) → concat condition (2 chiều: v12,v21)
 → 2× EfficientKANLinear(4098, 32) song song → (mu, logvar)   [latent_dim=32]
```

**Decoder (`decoder_type="conv"`):**
```
[z(32), condition(2)] → concat(34) → EfficientKANLinear(34, 4096) → reshape (B,256,4,4)
 → 4× DecoderBlock: ConvTranspose2d(4×4,stride2,pad1) [→BatchNorm→ReLU, trừ lớp cuối]
     kênh đảo ngược 256→128→64→32→1, không gian 4→8→16→32→64
 → Sigmoid → (B,1,64,64)∈[0,1]
 → enforce_symmetry: image = 0.5·(image + image.transpose(-1,-2))
```

Reparameterization chuẩn VAE: `z = mu + eps·exp(0.5·logvar)` lúc train; `z=mu` (deterministic) lúc validation. Tham số (condition_dim=2, latent_dim=32): **1.483.809** (Linear head) → **4.691.937** (KAN head, +3,16×) - phần lớn phình ra ở `fc_mu`/`fc_logvar`/`decoder.fc` vì `in_features` lớn (4096+), mỗi kết nối KAN có thêm 8 hệ số spline thay vì 1 trọng số Linear.

### 2.3 `WireContinuousDecoder` (decoder thay thế, `decoder_type="wire"`, Task 2 - đã thất bại benchmark chính thức, xem [`task_progress.md`](archive/task_progress.md) Task 2)

Thay ConvTranspose2d bằng implicit neural representation (INR) truy vấn tọa độ liên tục:
```
input: coords(x,y)∈[-1,1]²(2) ⊕ z(32) ⊕ condition(2) = 36 chiều
 → ComplexGaborActivation(36→128)  [layer1]
 → ComplexGaborActivation(128→128) [layer2]
 → Linear(128→1) → Sigmoid → ρ∈[0,1] tại đúng tọa độ truy vấn
```
`ComplexGaborActivation` = dạng thực của Gabor wavelet phức (Saragadam et al., WIRE, ICCV 2023):
```
act(x) = cos(ω₀·(Wx+b)) · exp(-s₀²·(Wx+b)²)
```
`cos` cho dao động tần số cao, `exp(-s₀²p²)` là envelope Gaussian định xứ (mỗi neuron chỉ "sáng" ở 1 vùng nhỏ input → biên sắc, ít cấu trúc rác cô lập). Khởi tạo weight `U(±√(6/in)/s₀)` để `s₀·p` giữ thang O(1) (đã verify `act.std≈0,37`, không suy biến gradient).

`forward(z, condition, resolution=...)` tự sinh lưới raster `resolution×resolution` rồi gọi `forward_coords()` - cơ chế **resolution-agnostic**: 1 checkpoint suy luận được ở 64²/128²/256²/512² không cần train lại, vì mạng học hàm liên tục `ρ=f(x,y;z,c)` thay vì 1 lưới pixel cố định. `wire_hidden_dim=128`: CVAE tổng (encoder KAN + WIRE decoder) **2.770.625** tham số.

### 2.4 `EfficientKANLinear` - lớp KAN dùng chung ở §2.1/2.2 — [`efficient_kan/__init__.py`](../efficient_kan/__init__.py) (vendor local, không có trên PyPI)

Mỗi kết nối có 2 nhánh cộng lại thay vì 1 trọng số vô hướng như `nn.Linear`:
```
y = base_activation(x)·base_weight^T  +  B(x)·spline_weight^T
```
- `base_weight`: (out,in) - nhánh tuyến tính cơ sở, `base_activation=SiLU`.
- `B(x)`: giá trị B-spline bậc 3 (đệ quy Cox-de Boor) trên lưới cố định phủ `[-1,1]` với `grid_size=5` khoảng (mở rộng thêm `spline_order=3` điểm mỗi bên để đệ quy không cần xử lý biên riêng).
- `spline_weight`: (out,in,grid_size+spline_order=8) - hệ số spline **learnable** - đây là phần "Kolmogorov-Arnold" thật sự (hàm kích hoạt học được trên từng cạnh, thay vì trọng số cố định + activation cố định kiểu MLP).
- `grid` là buffer không train; `rebuild_kan_grid()` cho phép đổi `in_features` khi resume checkpoint với `condition_dim` khác mà không mất `base_weight`/`spline_weight` đã học ở phần input chung.

### 2.5 `RealPhysicsNu` - không phải mạng neural, nhưng là nguồn gradient huấn luyện KAN/WIRE — [`pipeline/phase5_cvae/real_physics.py`](../pipeline/phase5_cvae/real_physics.py)

`torch.autograd.Function` bọc trực tiếp FE solver thật (`simp/core/solver.py` + `simp/homogenization/compute.py`), không phải mạng học xấp xỉ. Forward giải FE thật (~50-100ms/mẫu, lưới 50×50); backward tận dụng `dQ_ij/dx_e` (đạo hàm giải tích có sẵn nhờ tính tự-adjoint của bài toán năng lượng) nên gần như miễn phí - chỉ vài phép nhân ma trận 3×3, không giải FE lại. Đã kiểm chứng bằng finite-difference, sai số <1e-4. Đây là "chìa khóa" nâng R² của KAN từ 0,19 (chỉ BCE/MSE ảnh) lên 0,64/0,8889 khi dùng làm loss phụ - xem `EXPERIMENT_LOG.md` mục 2026-08-23.

## 3. Luồng dữ liệu qua 8 phase

```mermaid
flowchart LR
    P1[Phase 1\nLHS Screening\nnelx=nely=80] -->|độ nhạy Spearman\nvolfrac chi phối| P2
    P2[Phase 2\nAdaptive DOE\nnelx=nely=50] -->|8 lô, 7.920 mẫu\nHOẶC generate_production_batch.py\n57.216 mẫu v2| P3
    P3[Phase 3\nDataset Build\nfilter + augment] -->|train/val/test.npz\nv12,v21,volfrac,f1,f2| P4
    P4[Phase 4\nCNN Surrogate\nR²=0.974/0.964] -->|đóng băng, làm loss| P5
    simp_core[simp/core + homogenization\nFE thật] -.->|solve_fe trực tiếp\ntrong autograd.Function| P5
    P5[Phase 5\ncVAE + differentiable-physics\ncondition_dim=2/6] -->|best-of-N + composite score| P6
    P6[Phase 6-8\nmanufacturability/export\nchưa hoàn thiện đầy đủ] 
```

**Điểm khác biệt kiến trúc quan trọng ở Phase 5:** không giống Phase 1-4 (chỉ dùng surrogate/FE
NumPy tuần tự), Phase 5 có **2 đường dẫn vật lý song song**: (a) surrogate CNN đóng băng (nhanh,
nhưng từng gây "surrogate exploitation" - decoder học đánh lừa CNN); (b) `real_physics.py` (chậm
hơn nhưng chính xác - FE thật + đạo hàm giải tích ngay trong backward). Sửa tận gốc surrogate
exploitation chính là chuyển từ (a) sang (b) làm nguồn loss chính, giữ (a) chỉ để lọc sơ bộ nhanh
(`--k-fe-verify`).

## 4. Điểm mở rộng cho roadmap hiện tại

| File | Vai trò hiện tại | Điểm chạm khi mở rộng (Nhóm nào) |
|---|---|---|
| `pipeline/params.py` | `PARAM_SPACE` cho Phase 1 | A1 - đã thêm `nu` (`PARAM_SPACE['nu']=(0.2,0.4)`) |
| `pipeline/phase2_multi_batch/params.py` | `ACTIVE_PARAMETERS` cho Phase 2 | A1 - đã thêm `NU_ACTIVE_PARAMETERS` tách biệt, tránh đổi hành vi default |
| `pipeline/phase2_multi_batch/runner.py::DEFAULT_FIXED` | fallback khi sample không có cột | A1 - `nu` giữ 0.3 làm fallback, override qua `sample_values` |
| `analysis/scripts/generate_production_batch.py` | nguồn sinh dataset production thật | A3 - đã thêm `--vary-nu`, `SEED_NU_RANGE` |
| `simp/materials/isotropic.py::Material` | validate `nu` ở biên | A2 (đã có sẵn) - điểm chạm cho F1 (`alpha`) theo cùng pattern |
| `pipeline/phase4_surrogate/{dataset,model}.py` | input: ảnh + one-hot seed | A4 (thêm `nu` scalar), F5 (thêm `alpha`/CTE) |
| `pipeline/phase5_cvae/real_physics.py::RealPhysicsNu.forward` | `nu` là 1 scalar/batch | A5 - đổi thành `nu: (B,)`, **vỡ cache mesh key theo `(nelx,nely,E0,Emin,nu)`** |
| `pipeline/phase5_cvae/{dataset,model,train}.py` | `condition_dim` 2→6 (`extended_condition`) | A6 (→8), Nhóm 2 (f1/f2), F5 (CTE) - đều dùng chung pattern presence-mask + condition-dropout |
| `pipeline/phase5_cvae/model.py` | fc Linear → `EfficientKANLinear` (Task 1, đã xong); decoder conv → `decoder_type="wire"` (Task 2, code xong, chưa train) | Task 3 (ConvKAN surrogate - cùng pattern `use_kan`), Task 4 (Tandem L-BFGS - dùng `forward_coords` liên tục của WIRE decoder) |
| `simp/homogenization/compute.py::compute_homogenized_tensor` | `U` shape `(ndof,3)`, trả `Q` 3×3 | F2 - cần thêm 1 load case nhiệt, **công thức chờ F0 xác nhận trước khi đổi shape** |
| `simp/core/pbc.py` | PBC real-valued cho 3 load case biến dạng | F2 (PBC cho bài toán nhiệt); D3 nếu mở lại Nhóm 5 (PBC phức Bloch-Floquet - viết mới phần lớn) |
| `simp/objectives/auxetic.py` | mục tiêu auxetic đơn | F3 - mở rộng hoặc tách `thermal.py` cho CTE cực trị |

## 5. Kiến trúc đích (giả định, sau khi chạy hết Nhóm 1+2+6 theo kịch bản)

```mermaid
flowchart TB
    subgraph simp_core["simp/ (không đổi ranh giới phụ thuộc)"]
        Material["Material\n+nu (đã có) +alpha (F1, mới)"]
        Homog["compute_homogenized_tensor\n+load case nhiệt (F2, mới, chờ F0)"]
        PBC["pbc.py\n+PBC nhiệt (F2, mới)"]
        Obj["objectives/\n+thermal.py hoặc auxetic.py mở rộng (F3, mới)"]
    end

    subgraph p4["Phase 4 surrogate"]
        SurMat["surrogate_material.pt\n+input nu (A4, mới)"]
        SurTh["surrogate_thermal.pt\n+input alpha (F5, mới)"]
    end

    subgraph p5["Phase 5 cVAE - condition_dim tăng dần"]
        C2["condition_dim=2\nv12,v21 (hiện có)"]
        C6["condition_dim=6\n+volfrac,void_size_frac (hiện có, PR#13)"]
        C8["condition_dim=8\n+nu,nu_mask (A6, mới)"]
        C10["condition_dim=10+\n+f1/f2 (Nhóm 2) và/hoặc CTE (F5)"]
        C2 --> C6 --> C8 --> C10
    end

    RealPhys["real_physics.py\nnu per-sample (A5, sửa)\ncache mesh mất hiệu lực"]

    Material --> SurMat
    Material --> RealPhys
    Homog --> SurTh
    RealPhys --> C8
    SurMat --> C8
    SurTh --> C10
    Obj --> SurTh
```

**Rủi ro kiến trúc lớn nhất khi đi tới kiến trúc đích:** `condition_dim` tăng tuyến tính theo số
tính chất vật lý mới (2→6→8→10+), nhưng **độ dày dữ liệu cần thiết để giữ retrieval baseline cạnh
tranh tăng theo cấp số nhân** (giả thuyết Nhóm 4.2, chưa kiểm chứng) - đây là lý do kiến trúc
"generative + differentiable-physics" được chọn thay vì tiếp tục mở rộng retrieval/tra bảng đơn
thuần. Rủi ro thứ hai là **mesh/PBC-cache trong `real_physics.py`** vốn được thiết kế cho 1 vật
liệu nền cố định (key theo `(nelx,nely,E0,Emin,nu)` - 3 tham số đầu cố định trong toàn bộ project
tới nay) - khi `nu` (và có thể `alpha`) trở thành per-sample, giả định "cache hit gần 100%" của
thiết kế gốc không còn đúng, cần đánh giá lại chi phí trước khi coi differentiable-physics fine-tune
vẫn khả thi ở quy mô hiện tại - **đã đo thật ở A5** (`EXPERIMENT_LOG.md` 2026-08-18): dựng lại
`Material` mỗi lần ~97µs/lần, ~0,1-0,2% chi phí FE-solve, rủi ro không xảy ra trong thực tế.

## 6. Quy ước đặt tên & versioning

- **Không ghi đè checkpoint** - mỗi thay đổi ý nghĩa (dataset mới, condition mới, loss mới) tạo tên
  file mới (`_v2`, `_clean`, `_extended`, `_material`, `_thermal`...), giữ bản cũ làm tham chiếu
  lịch sử có thể tái lập.
- **Cờ mặc định TẮT cho hành vi mới** - `--extended-condition`, `--vary-nu`, `--require-manufacturable`
  đều mặc định tắt để không đổi hành vi pipeline production hiện có; bật tường minh qua CLI flag.
- **Dải tham số hẹp trước, mở rộng sau khi xác nhận hội tụ** - đúng pattern đã áp dụng cho
  `volfrac`/`void_size_frac` (Phase 2 adaptive) và lặp lại cho `nu` (A3: 0,45-0,70 gốc → hẹp dần;
  `nu`: 0,2-0,4 → thu hẹp theo seed sau khi đo hội tụ thật).
- **`ACTIVE_PARAMETERS`/`PARAM_SPACE` mặc định KHÔNG đổi khi thêm tham số mới** - tham số mới luôn
  đi vào 1 dict riêng (`NU_ACTIVE_PARAMETERS`) thay vì gộp thẳng vào dict mặc định, để không âm
  thầm đổi hành vi tái tạo dataset production cũ.

## 7. Tài liệu liên quan

- [`docs/plan.md`](plan.md) - kế hoạch + trạng thái hiện hành (nguồn duy nhất, v3)
- [`docs/archive/`](archive/) - roadmap cũ `PROJECT_PLAN.md`, log `task_progress.md`, bản nháp paper1 `.md` (lịch sử)
- [`docs/PIPELINE.md`](PIPELINE.md) - lệnh chạy + số liệu THẬT từng phase hiện có
- [`docs/PHYSICS_AND_ML.md`](PHYSICS_AND_ML.md) - bản chất toán học/vật lý + vai trò ML
- [`docs/LIMITATIONS.md`](LIMITATIONS.md) - giới hạn đã biết, phạm vi claim khoa học

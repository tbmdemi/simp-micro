# AuxForge

**Tối ưu hóa Topology cho Thiết kế Vi cấu trúc Vật liệu Auxetic (Auxetic Metamaterial)**

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)
[![Version](https://img.shields.io/badge/version-1.4.0-blueviolet)](simp/__init__.py)
[![Branch](https://img.shields.io/badge/branch-main-orange)](#)

---

## Tổng quan

**AuxForge** triển khai phương pháp **Solid Isotropic Material with Penalization (SIMP)** cho bài toán tối ưu hóa topology của các vi cấu trúc ô đơn vị tuần hoàn (periodic unit-cell), nhắm tới **hành vi auxetic** (hệ số Poisson âm). Mục tiêu cuối cùng của dự án là **thiết kế ngược (inverse design)**: cho trước một hệ số Poisson mục tiêu, sinh ra một hình học vi cấu trúc đạt được giá trị đó, sử dụng mô hình sinh có điều kiện (cVAE) được huấn luyện trên bộ dữ liệu tăng cường bằng surrogate, do chính engine SIMP này tạo ra.

```
Seed Generation → FE Analysis → Homogenization → Objective & Sensitivity →
Density/Sensitivity Filtering → OC Update → Convergence Check → Repeat
```

Codebase này là bản triển khai lại bằng Python của các đoạn mã SIMP MATLAB kinh điển (88 dòng / 99 dòng), được mở rộng thêm điều kiện biên tuần hoàn (periodic boundary conditions), phép đồng nhất hóa dựa trên năng lượng (energy-based homogenization), và một pipeline DOE (Design of Experiments) đa lô thích ứng (adaptive multi-batch) để sinh dữ liệu quy mô lớn.

---

## Mục lục

- [Trạng thái Dự án](#trạng-thái-dự-án)
- [Phạm vi Claim Khoa học](#phạm-vi-claim-khoa-học-đọc-trước-khi-trích-dẫn)
- [Bắt đầu](#bắt-đầu)
- [Cấu trúc Package](#cấu-trúc-package)
- [Các Seed Có sẵn](#các-seed-có-sẵn)
- [Hàm Mục tiêu (Auxetic)](#hàm-mục-tiêu-auxetic)
- [Pipeline chi tiết (Phase 1-5.1)](#pipeline-chi-tiết) - xem [docs/PIPELINE.md](docs/PIPELINE.md)
- [Tham chiếu CLI](#tham-chiếu-cli)
- [Sử dụng Lập trình](#sử-dụng-lập-trình)
- [File Đầu ra](#file-đầu-ra)
- [Tiêu chí Hội tụ](#tiêu-chí-hội-tụ)
- [Kiểm thử](#kiểm-thử)
- [Giới hạn Đã biết / Known Limitations](#giới-hạn-đã-biết--known-limitations) - xem [docs/LIMITATIONS.md](docs/LIMITATIONS.md)
- [Tài liệu](#tài-liệu)
- [Tài liệu Tham khảo](#tài-liệu-tham-khảo)
- [Giấy phép](#giấy-phép)

---

## Trạng thái Dự án

Lộ trình thiết kế ngược gồm 8 giai đoạn (phase). Phase 1-4 đã hoàn thành và được xác thực trên dữ liệu thực; Phase 5 (cVAE) đã sửa tận gốc bằng differentiable real-physics, đạt R²(FE,n=789)≈0,995-0,999/hit-rate ≥98%; Phase 6-8 (hậu xử lý & kiểm định FEA / active-learning loop / xác thực cuối & đóng gói) - xem bảng bên dưới, chi tiết đầy đủ tại [`outputs/phase5/reports/final_status_report_2026-07-31.md`](outputs/phase5/reports/final_status_report_2026-07-31.md). **Đi sâu vào từng phase:** [docs/PIPELINE.md](docs/PIPELINE.md).

| Phase | Thành phần | Trạng thái | Ghi chú |
|-------|-----------|--------|-------|
| 0 | Core SIMP Engine | ✅ Ổn định | 11 loại seed, mục tiêu auxetic, PBC, đồng nhất hóa dựa trên năng lượng |
| 1 | LHS Screening | ✅ Hoàn thành | Phân tích độ nhạy: `volfrac` là tham số chi phối (r ≈ 0,87–0,96). Lịch sử debug lần chạy đầu (0 mẫu auxetic): xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| 2 | Multi-Batch Adaptive DOE | ✅ Hoàn thành + cải tiến manufacturability + rebuild dQ | **8/8 lô gốc + batch 11 rebuild**, 7.920 mẫu, **91,9% auxetic** (đã rebuild 3 seed bị lỗi dQ, tăng từ 82,1%). Pipeline thích ứng tự dừng sau 2 lô liên tiếp không cải thiện mục tiêu. **2026-07-24**: (a) phân tích ngược xác nhận SEED chi phối manufacturability - thêm phân bổ mẫu theo seed; (b) phát hiện + sửa lỗi `dQ` + rebuild đầy đủ 2.160 mẫu (3 seed bất đối xứng). Xem [Phase 2](#2-multi-batch-adaptive-doe-phase-2----hoàn-thành) bên dưới và [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#bảng-tổng-hợp-lỗi-đã-sửa) |
| 3 | Dataset Build (trường mật độ + target) | ✅ Hoàn thành, **đã rebuild lần 2 (v2, 2026-07-25)** | **57.216 mẫu train** (sau aug) / 2.044 val / 2.044 test - dataset production hiện tại. Xem mục "Rebuild v2" ngay dưới bảng này |
| 4 | CNN Surrogate Model | ✅ Hoàn thành, đã retrain trên dataset v2 | Dự đoán (ν₁₂, ν₂₁, volfrac) từ trường mật độ. R² trên test set v2 (`surrogate_v2.pt`, 2026-07-25): ν₁₂ = **0,974**, ν₂₁ = **0,964**, volfrac = 0,983. Xem [Phase 4](#4-cnn-surrogate-model-phase-4----hoàn-thành) bên dưới |
| 5 | Conditional VAE | ✅✅ Surrogate-exploitation đã sửa TẬN GỐC bằng differentiable-physics; đã retrain trên dataset v2 | `cvae_v2_finetuned.pt` (checkpoint production khuyến nghị, train 2-stage trên dataset v2): R²(FE,n=300)=**0,9953** (oracle), hit rate single-shot=**99,7%**, frac manufacturable=0,247. `cvae_realphysics.pt` (dataset v1, giữ nguyên tham chiếu) ngang ngửa: R²=0,9984/hit-rate 98,3%/manuf=0,280. Xem [docs/PIPELINE.md § 5](docs/PIPELINE.md#5-conditional-vae-phase-5---đã-sửa-tận-gốc-bằng-differentiable-physics-2026-07-24) chi tiết; toàn bộ quá trình thử-sai xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Nâng cấp tuỳ chọn cGAN/conditional diffusion (mục 5.6/5.7 trong roadmap chi tiết): ⬜ chưa bắt đầu. **Đã merge vào `main` (2026-07-25):** thêm `volfrac`/`void_size_frac` làm condition **optional** (presence-mask + condition-dropout, cờ `--extended-condition`) và chấm điểm tổng hợp accuracy/manufacturability/aesthetic (`0,6/0,3/0,1`) thay `argmin(Δv12)` thuần túy trong `best_of_n_eval.py` - xem [docs/PIPELINE.md § 5.1](docs/PIPELINE.md#51-tham-số-input-tùy-chọn-volfrac-và-void-size-frac-chấm-điểm-toàn-diện) |
| 6 | Hậu xử lý & kiểm định FEA | ✅ Hoàn thành | Nhị phân hoá + connectivity/min-feature/periodicity (`manufacturability.py`) + lọc/verify bằng FE thật (`best_of_n_eval.py`) - đã có sẵn trong pipeline Phase 5 |
| 7 | Active-learning loop | ✅ Kết luận (2026-07-31) | Implement + chạy production (`pipeline/phase5_cvae/active_learning.py`) - **không cải thiện** checkpoint đã tối ưu (`cvae_realphysics.pt` gần mức trần, mean_abs_error tệ đi 0,0246→0,0686 sau 1 vòng) - giữ nguyên checkpoint production. Chi tiết: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) mục 2026-07-31 |
| 8 | Xác thực cuối & đóng gói | 🟨 Một phần | **8.1 đối chiếu FE độc lập**: ✅ xong, `scikit-fem` khớp engine nội bộ tới 1e-9 (R²=1,000000, n=24) - xác nhận đúng vật lý. **8.3 thư viện thiết kế**: ✅ xong, 24 thiết kế, hit-rate 100%, 87,5% xác nhận manufacturable. **8.2 biến dạng lớn**, **8.5 chuẩn bị STL**: ⬜ chưa làm (tuỳ chọn) |

> Chi tiết từng phase con (2.1-2.9, 3.1-3.6, v.v.): xem dashboard `html/dashboards/workflow.html`.
> **Khoảng trống đã biết** (tóm tắt - xem đầy đủ tại [docs/LIMITATIONS.md](docs/LIMITATIONS.md)): phạt `mu` trong mục tiêu auxetic đang tắt (`mu=0.0`, đang chờ thiết kế lại); đồng nhất hóa chưa xuất độ cứng `E₁₁/E₀, E₂₂/E₀` nên `f1, f2` (roadmap gốc, Pha B) chưa khả dụng làm condition (dù `volfrac`/`void_size_frac` đã khả dụng dạng optional, xem [docs/PIPELINE.md § 5.1](docs/PIPELINE.md#51-tham-số-input-tùy-chọn-volfrac-và-void-size-frac-chấm-điểm-toàn-diện)); khả năng chế tạo - xem [docs/PIPELINE.md § 5](docs/PIPELINE.md#5-conditional-vae-phase-5---đã-sửa-tận-gốc-bằng-differentiable-physics-2026-07-24); các R²/hit-rate của Phase 5 đo trên cỡ mẫu rất nhỏ (n=3-24 điều kiện), CI rộng - đọc kỹ trước khi trích dẫn.

### Phạm vi Claim Khoa học (đọc trước khi trích dẫn)

Mục này trả lời trực tiếp câu hỏi lặp lại nhiều nhất trong 2 báo cáo phản biện bên ngoài ngày 2026-08-02: "bạn đang claim cái gì, thật sự, và bằng chứng nào?" Bản đầy đủ (kèm 18 mục Giới hạn Đã biết): [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

- **Claim ĐƯỢC ủng hộ bởi bằng chứng:** engine FE/homogenization đúng vật lý (đối chiếu độc lập bằng `scikit-fem`); checkpoint cVAE hiện tại sinh single-shot đáng tin cậy để dự đoán (v₁₂,v₂₁) đúng DẤU ở tỉ lệ cao (n=789); cVAE sinh thiết kế KHÔNG phải sao chép gần training set (novelty).
- **Claim CHƯA được ủng hộ bởi bằng chứng (dù có thể vẫn đúng):** proxy Q₁₂ tương đương mục tiêu auxetic "thật" dưới mọi điều kiện xoay; cVAE vượt trội baseline nearest-neighbor retrieval về độ chính xác trong-phân-phối (baseline mới đo 2026-08-02 cho thấy CHƯA có bằng chứng); tổng quát hoá ngoài phân phối train; `hit_rate` như chỉ số độc lập (base rate ~92% auxetic khiến metric này yếu, R² đáng tin hơn).
- **Không claim:** "đã giải quyết inverse design", "cVAE tốt hơn phương pháp classical trong mọi trường hợp", "surrogate có thể thay FE ở vùng chưa kiểm chứng".

---

## Bắt đầu

### Yêu cầu & cài đặt

**Python** ≥ 3.10, **numpy/scipy/matplotlib** (core), **pandas/scikit-learn/Pillow** (`pipeline/phase3_dataset/`), **torch** (`pipeline/phase4_surrogate/`, `pipeline/phase5_cvae/` - không có trong `requirements.txt` gốc, cần cài thủ công):

```bash
pip install numpy scipy matplotlib pandas scikit-learn pillow torch
```

### Chạy nhanh

```bash
python -m simp.run                                                    # 1 lần chạy mặc định (hourglass, auxetic)
python -m simp.main --nelx 80 --nely 60 --volfrac 0.35 --seed hexagonal  # ghi đè tham số qua CLI (`simp ...` nếu cài editable)
```

Kết quả ghi vào `outputs/simp_results_{seed}/`: `iteration_XXXXX.png` (trường mật độ) + `iteration_data.csv` (lịch sử hội tụ ν₁₂/ν₂₁/mục tiêu/thể tích). Ví dụ log:

```
Loop: 134  obj:-2.8750e-01  vol:0.400  chg:0.003  v12:-0.8510  v21:-0.8510
[DONE] Hội tụ tại lần lặp 134 (45.2s)
```

### Pipeline dataset đầy đủ (Phase 2 → Phase 3)

```bash
# Phase 2: chạy/mở rộng adaptive multi-batch DOE (xem pipeline/phase2_multi_batch/)
python -m pipeline.phase2_multi_batch.main --phase1-summary outputs/pipeline/phase1

# Phase 3: xây dựng dataset sẵn sàng cho ML từ các lô đã hoàn thành
python3 pipeline/phase3_dataset/scan_dataset.py
python3 pipeline/phase3_dataset/build_npz.py --resolution 64
python3 pipeline/phase3_dataset/finalize_dataset.py --resolution 64
```

Đầu ra: `outputs/phase3/{train,val,test}.npz` - xem [Dataset Build (Phase 3)](#3-dataset-build-phase-3----hoàn-thành) bên dưới.

---

## Cấu trúc Package

```
├── simp/                     # Package SIMP lõi (v1.4.0): run.py/main.py (entry point), runner.py, config.py
│   ├── core/                 # fem, filter, pbc (null-space projection), solver (LU+CG), oc, convergence
│   ├── materials/, objectives/, homogenization/   # vật liệu, hàm mục tiêu auxetic, đồng nhất hóa (U_total = U0+χ)
│   ├── seeds/                # 11 bộ sinh mẫu rỗng ban đầu (xem bảng Các Seed Có sẵn)
│   └── io/                   # logger CSV, visualizer PNG
│
├── pipeline/
│   ├── phase1_screening/       # Phase 1: screening_parallel, refine_params, analyst (LHS screening)
│   ├── phase2_multi_batch/    # Phase 2: adaptive DOE - sampling (Sobol/LHS), runner, adaptive (quyết định), coverage
│   ├── phase3_dataset/        # Phase 3: scan_dataset, build_npz, augment_symmetry, finalize_dataset
│   ├── phase4_surrogate/      # Phase 4: dataset, model (SurrogateCNN), train, evaluate, export_for_phase5
│   └── phase5_cvae/           # Phase 5: dataset, model, losses, train, evaluate, sample, verify_fe,
│                               #   adversarial_dataset, self_play, best_of_n_eval (inference chính thức),
│                               #   manufacturability, coverage_eval, bootstrap_ci (CI cho R²/hit_rate)
│
├── analysis/                 # Phân tích độ nhạy (ANOVA, Sobol, regression), Pareto front, dataset QC
├── notebooks/, html/         # Jupyter notebook + dashboard/báo cáo (xem html/index.html)
├── html/dashboards/workflow.html        # Dashboard workflow toàn dự án (chi tiết từng phase con)
├── tests/                     # Bộ kiểm thử PyTest (483 test)
├── outputs/                   # Dữ liệu sinh ra - phần lớn (metadata/CSV/figures nhỏ, outputs/multi_batch/, outputs/pipeline/) ĐÃ commit; chỉ *.npz/*.npy/*.pt và outputs/phase3/*.npz bị gitignore (quá lớn)
├── docs/                      # PIPELINE.md, LIMITATIONS.md, PHYSICS_AND_ML.md - xem mục Tài liệu
├── EXPERIMENT_LOG.md, CHANGELOG.md   # xem mục Tài liệu
└── pyproject.toml, requirements.txt, README.md
```

---

## Các Seed Có sẵn

| Seed | Mô tả | Tỷ lệ thành công auxetic* |
|------|-------------|------------------------|
| `circle` | Một lỗ rỗng hình tròn ở tâm | 93,8% |
| `square` | Một lỗ rỗng hình vuông ở tâm | 94,0% |
| `hourglass` | Hai lỗ rỗng hình tam giác | 67,8% |
| `four_circle` | Bốn lỗ rỗng hình tròn, đối xứng | 87,9% |
| `hexagonal` | Một lỗ rỗng hình lục giác | 64,4% |
| `nine_circle` | Lưới 3×3 các lỗ rỗng hình tròn | 98,9% |
| `cross_rectangular` | Lỗ rỗng hình chữ thập | 93,3% |
| `grid_circular_voids` | Lưới N×N đều các lỗ rỗng hình tròn | 99,4% |
| `small_square_cross` | Chữ thập vuông nhỏ ở tâm | 93,1% |
| `circle_half_quarter` | Hình tròn ở tâm + bốn phần tư hình tròn ở góc | 61,7% |
| `reentrant_bowtie` | Lỗ rỗng hình nơ (hình học re-entrant) - seed mới nhất | 48,6% |

\* Tỷ lệ mẫu có cả ν₁₂ < 0 và ν₂₁ < 0, đo trên toàn bộ 7.920 mẫu của multi-batch DOE đã hoàn thành (Phase 2). `reentrant_bowtie` và `hexagonal` là các hình học khó đẩy về auxetic nhất và là ứng viên tốt để tiếp tục tinh chỉnh tham số.

Phép xoay (`--rotation_deg`) có thể áp dụng cho bất kỳ seed nào.

---

## Hàm Mục tiêu (Auxetic)

```
c = Q₁₂ − μ·(Q₁₁ + Q₂₂) + penalty_terms
penalty: kích hoạt khi Q₁₁ hoặc Q₂₂ < δ = 0.1·volfrac·E₀, chuẩn hóa theo δ²
```

- `compute_nu12` / `compute_nu21` dùng **nghịch đảo đầy đủ của ma trận độ mềm 3×3** (`S = Q⁻¹`), không dùng công thức tắt trực hướng (orthotropic) `ν₁₂ = Q₁₂/Q₂₂` - công thức tắt này sai bất cứ khi nào ô đơn vị bị xoay (liên kết cắt-pháp `Q₁₃, Q₂₃ ≠ 0`). Đây là nguyên nhân gốc của lỗi "0 mẫu auxetic" ở Phase 1 (xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)).
- Thành phần `μ` được thiết kế để đẩy `Q₁₂` âm hơn nữa thay vì dừng lại gần 0, nhưng công thức hiện tại **có sai sót về mặt khái niệm** (đang chờ thiết kế lại) - hiện bị tắt theo mặc định (`mu=0.0`), là cấu hình dùng cho toàn bộ 8 lần chạy multi-batch DOE đã hoàn thành.
- Một hệ số phạt độ cứng được kích hoạt khi `Q₁₁` hoặc `Q₂₂` giảm dưới `δ`, ngăn sụp đổ cấu trúc (topology suy biến dạng rỗng vẫn xảy ra ở ~0,4% lần chạy - bị lọc bỏ ở Phase 3).

---

## Pipeline chi tiết

Toàn bộ quy trình 5 bước **Screening → Multi-Batch DOE → Dataset → Surrogate → cVAE** (lệnh chạy từng bước, số liệu R²/hit-rate đầy đủ, lịch sử phát hiện+sửa bug, và mục 5.1 "condition tuỳ chọn volfrac/void_size_frac + chấm điểm toàn diện") đã được tách sang **[docs/PIPELINE.md](docs/PIPELINE.md)** để giữ README ngắn gọn.

Tóm tắt nhanh:

1. **LHS Screening (Phase 1)** - xác định `volfrac` là tham số chi phối (r≈0,87-0,96).
2. **Multi-Batch Adaptive DOE (Phase 2)** - 8 lô, 7.920 mẫu, tỷ lệ hội tụ FE 100%, auxetic rate 91,9% sau rebuild.
3. **Dataset Build (Phase 3)** - dataset v2 hiện hành: train=57.216 / val=2.044 / test=2.044.
4. **CNN Surrogate (Phase 4)** - `surrogate_v2.pt`: R²(v12)=0,974, R²(v21)=0,964, R²(volfrac)=0,983.
5. **Conditional VAE (Phase 5)** - `cvae_v2_finetuned.pt` (khuyến nghị production): R²(FE,n=300)=0,9953 (oracle), hit rate single-shot=99,7%. Bao gồm mục 5.1: condition `volfrac`/`void_size_frac` optional (`--extended-condition`) + chấm điểm tổng hợp accuracy/manufacturability/aesthetic trong `best_of_n_eval.py`.

---

## Tham chiếu CLI

Các tham số hay chỉnh nhất:

| Tùy chọn | Mặc định | Mô tả |
|--------|---------|-------------|
| `--nelx`, `--nely` | 100, 100 | Số phần tử lưới FE |
| `--volfrac` | 0.4 | Tỷ lệ thể tích mục tiêu |
| `--penal` | 3.0 | Hệ số phạt SIMP |
| `--rmin` | 3.0 | Bán kính bộ lọc |
| `--seed` | hourglass | Mẫu seed ban đầu (11 loại, xem [Các Seed Có sẵn](#các-seed-có-sẵn)) |
| `--rotation_deg` | 0.0 | Góc xoay seed |
| `--void_size_frac` | 0.4 | Tỷ lệ kích thước lỗ rỗng khi sinh seed |
| `--max_iter` | 200 | Số vòng lặp tối đa |

Tham số còn lại (`--ft`, `--E0`, `--Emin`, `--nu`, `--move`, `--tol_change`, `--tol_obj`, `--window_size`, `--objective`, `--beta`, `--save_every`, `--scale_factor`, `--output_dir`, `--quiet`) - xem `python -m simp.main --help`.

---

## Sử dụng Lập trình

```python
from simp.runner import run_simp

params = {
    'nelx': 120, 'nely': 120, 'volfrac': 0.35, 'penal': 3.0, 'rmin': 2.5,
    'seed': 'hexagonal', 'objective': 'auxetic', 'void_size_frac': 0.45,
    'max_iter': 300, 'save_every': 5,
}
result = run_simp(params)

print(f'ν₁₂ = {result["v12"]:.4f}, ν₂₁ = {result["v21"]:.4f}, converged: {result["converged"]}')

xPhys = result['xPhys']      # trường mật độ (nely, nelx)
Q = result['Q']              # tensor độ cứng đồng nhất hóa 3×3
history = result['history']  # dict: iteration, v12, v21, objective, volume
```

---

## File Đầu ra

### Ảnh PNG (`iteration_XXXXX.png`)
Trường mật độ thang xám - đen (0) = rỗng, trắng (1) = rắn.

### Dữ liệu CSV (`iteration_data.csv`)

| Cột | Mô tả |
|--------|-------------|
| `Iteration` | Số thứ tự vòng lặp |
| `Poisson_v12` | ν₁₂, tính bằng nghịch đảo đầy đủ ma trận độ mềm 3×3 |
| `Poisson_v21` | ν₂₁, tính bằng nghịch đảo đầy đủ ma trận độ mềm 3×3 |
| `Objective` | Giá trị hàm mục tiêu |
| `Volume_Fraction` | Giá trị trung bình của `xPhys` |

### Metadata (`metadata.json`)
`git_hash`, `timestamp`, `version`, toàn bộ `params` dùng cho lần chạy.

---

## Tiêu chí Hội tụ

Dừng khi **bất kỳ** điều kiện nào sau được thỏa mãn:
1. Thay đổi thiết kế < `tol_change`
2. Độ ổn định mục tiêu - thay đổi tương đối < `tol_obj` trong `window_size` vòng lặp liên tiếp
3. Đạt `max_iter`

Được xử lý bởi `ConvergenceChecker` (`simp/core/convergence.py`), với `min_iter` để tránh dừng quá sớm. Trên toàn bộ 7.920 mẫu của multi-batch DOE, **tỷ lệ hội tụ FE đạt 100%**.

---

## Kiểm thử

```bash
pytest tests/ -v
```

Trạng thái hiện tại: **483/483 test pass** (`pytest tests/ -q`, ~7s) - bao gồm test cho tính năng optional multi-condition (xem [docs/PIPELINE.md § 5.1](docs/PIPELINE.md#51-tham-số-input-tùy-chọn-volfrac-và-void-size-frac-chấm-điểm-toàn-diện)).

| Module | Trạng thái |
|--------|--------|
| Phân tích tham số dòng lệnh CLI | ✅ |
| Xác thực SimpConfig | ✅ |
| Bộ kiểm tra hội tụ | ✅ |
| Smoke test lõi (FEM, vật liệu, filter, OC, solver, PBC) | ✅ |
| Nạp dataset & phân loại auxetic | ✅ |
| Định dạng CSV của logger | ✅ |
| `pipeline/phase4_surrogate/` (model, dataset, evaluate, export, train, **bootstrap_ci**) | ✅ |
| `pipeline/phase5_cvae/` (model, dataset, losses, verify_fe, sample, adversarial_dataset, self_play, train, **best_of_n_eval**, **manufacturability**, **coverage_eval**, **bootstrap_ci**) | ✅ |
| `pipeline/phase1_screening/` (refine_params: quyết định ACTIVE/FIXED; analyst: Spearman, top3, đếm success/converged) | ✅ |
| `pipeline/phase2_multi_batch/` (params, sampling: Sobol/LHS/random, coverage: sparse-region + coverage_report, adaptive: stop/refine/expand) | ✅ |
| `pipeline/phase3_dataset/` (augment_symmetry: swap ν₁₂↔ν₂₁ khi xoay 90/270°, finalize_dataset: stratify chống rò rỉ dữ liệu, scan_dataset, build_npz) | ✅ |

> Test dùng fixture `.npz` tổng hợp nhỏ (không phụ thuộc `outputs/phase3/*.npz` thực, bị gitignore) nên chạy nhanh (~4s) ở mọi nơi. `pipeline/seeds/*.py` và các hàm CLI/orchestration nặng I/O (`screening_parallel.py`'s main loop, `multi_batch/main.py`, `multi_batch/runner.py`'s `evaluate_single`, `visualize.py`) vẫn chưa có test - coverage mới thêm cho phase1/2/3 tập trung vào logic thuần (quyết định ACTIVE/FIXED, sampling, coverage/adaptive, augment/stratify), không phải toàn bộ pipeline end-to-end.
>
> **Lưu ý khi thêm test:** `phase4_surrogate/` và `phase5_cvae/` định nghĩa module con trùng tên (`dataset.py`, `model.py`...) qua import trần (`sys.path.insert` + `from dataset import X`) - import 2 module cùng tên từ *phase khác nhau* trong 1 tiến trình sẽ đè cache `sys.modules`. Fixture `_isolate_pipeline_bare_imports` (`tests/conftest.py`) reset cache này, nhưng chỉ hoạt động nếu import nằm **bên trong** hàm test (không phải top-level file) - luôn import trễ (lazy).

---

## Giới hạn Đã biết / Known Limitations

Danh sách đầy đủ 18 mục (song ngữ Việt/English) đã được tách sang **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** cùng với mục "Phạm vi Claim Khoa học". Các điểm đáng chú ý nhất:

- R²/hit-rate của Phase 5 chỉ đáng tin ở cỡ mẫu lớn (n≈300-789); ở n=24 CI rất rộng.
- Manufacturability của đầu ra gốc (không lọc) rất thấp; cần `force_periodic()`/`--require-manufacturable`.
- Phạt `mu` trong mục tiêu auxetic đang tắt (`mu=0.0`).
- `f1, f2` (Pha B, độ cứng chuẩn hóa) chưa khả dụng làm condition - chỉ `volfrac`/`void_size_frac` (Pha A) đã xong.
- Test tự động (483/483 pass) chưa phủ hết đường I/O nặng (screening loop, seeds, FE call thật).
- Toàn bộ pipeline dùng FEM tuyến tính (giả định biến dạng nhỏ) - xem mục 14.
- `hit_rate` là metric yếu do base rate ~92% auxetic của dataset; chưa có bằng chứng cVAE vượt trội baseline nearest-neighbor trong-phân-phối (mục 17-18, phát hiện 2026-08-02).

---

## Tài liệu
- [`docs/PIPELINE.md`](docs/PIPELINE.md) - chi tiết từng bước pipeline (Phase 1-5.1): lệnh chạy, số liệu R²/hit-rate, lịch sử phát hiện+sửa bug
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) - phạm vi claim khoa học + 18 mục giới hạn đã biết (song ngữ)
- [`docs/PHYSICS_AND_ML.md`](docs/PHYSICS_AND_ML.md) - bản chất toán học/cơ học/vật lý của SIMP + đồng nhất hóa, và vai trò cụ thể của ML/DL (surrogate, cVAE, differentiable-physics) trong pipeline
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) - nhật ký các phát hiện/sửa lỗi và đột phá chính thay đổi kết quả dự án
- `html/dashboards/workflow.html` - dashboard workflow, chi tiết từng phase con (2.1-2.9, 3.1-3.6, v.v.)
- `html/index.html` - dashboard/báo cáo bổ sung (lưu ý: một số trang chỉ phản ánh screening Phase 1, chưa tái sinh theo Phase 2-5)
- `CHANGELOG.md` - lịch sử thay đổi theo phiên bản
- [`outputs/pilot_damping_decay/PILOT_REPORT.md`](outputs/pilot_damping_decay/PILOT_REPORT.md) - pilot có kiểm soát (N=400/config, Wilson CI) cho thấy `use_sqrt` (damping η=0.5) + `penal_init=2.0` (continuation nhẹ) gần gấp đôi yield `reentrant_bowtie` (46%→76-80%) - **kết quả tốt, đã kiểm chứng thống kê, nhưng CHƯA được áp dụng vào cấu hình production**
- `outputs/{phase3,phase4,phase5}/` - báo cáo/kết quả từng phase (`evaluation_report.json`, `fe_verification_report.json`, `self_play/`, v.v.)
- `notebooks/01-06_*.ipynb`, `gamma_sweep_analysis.ipynb` - notebook phân tích Phase 1-5 và tổng kết end-to-end

---

## Tài liệu Tham khảo

- Sigmund, O. (2001). *A 99 line topology optimization code written in Matlab.* Structural and Multidisciplinary Optimization, 21(2), 120–127.
- Andreassen, E., et al. (2011). *Efficient topology optimization in MATLAB using 88 lines of code.* Structural and Multidisciplinary Optimization, 43(1), 1–16.
- Xia, L., & Breitkopf, P. (2015). *Design of materials using topology optimization and energy‑based homogenization.* Archives of Computational Methods in Engineering, 22(2), 229–260.
- Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods, and Applications.* Springer.
- Pahlavani, H., et al. (2024). *Deep Learning for Size-Agnostic Inverse Design of Random-Network 3D Printed Mechanical Metamaterials.* Advanced Materials, 36(6). DOI: [10.1002/adma.202303481](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202303481) - pipeline "Deep-DRAM" cVAE + surrogate + chọn lọc bằng FE thực đã truyền cảm hứng cho biện pháp khắc phục best-of-N ở Phase 5 nêu trên.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017 - ý tưởng bất đồng-giữa-các-ensemble đằng sau `load_frozen_surrogate_ensemble`/`property_consistency_loss_ensemble`.

---

## Giấy phép

MIT - xem [`simp/__init__.py`](simp/__init__.py).
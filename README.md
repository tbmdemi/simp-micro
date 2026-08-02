# AuxForge

**Tối ưu hóa Topology cho Thiết kế Vi cấu trúc Vật liệu Auxetic (Auxetic Metamaterial)**

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)
[![Version](https://img.shields.io/badge/version-1.4.0-blueviolet)](simp/__init__.py)
[![Branch](https://img.shields.io/badge/branch-FixLoss-orange)](#)

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
- [Bắt đầu](#bắt-đầu)
- [Cấu trúc Package](#cấu-trúc-package)
- [Các Seed Có sẵn](#các-seed-có-sẵn)
- [Hàm Mục tiêu (Auxetic)](#hàm-mục-tiêu-auxetic)
- [Pipeline: Screening → Multi-Batch DOE → Dataset](#pipeline-screening--multi-batch-doe--dataset--surrogate--cvae)
- [Tham chiếu CLI](#tham-chiếu-cli)
- [Sử dụng Lập trình](#sử-dụng-lập-trình)
- [File Đầu ra](#file-đầu-ra)
- [Tiêu chí Hội tụ](#tiêu-chí-hội-tụ)
- [Kiểm thử](#kiểm-thử)
- [Giới hạn Đã biết / Known Limitations](#giới-hạn-đã-biết--known-limitations)
- [Tài liệu](#tài-liệu)
- [Tài liệu Tham khảo](#tài-liệu-tham-khảo)
- [Giấy phép](#giấy-phép)

---

## Trạng thái Dự án

Lộ trình thiết kế ngược gồm 8 giai đoạn (phase). Phase 1-4 đã hoàn thành và được xác thực trên dữ liệu thực; Phase 5 (cVAE) đã sửa tận gốc bằng differentiable real-physics, đạt R²(FE,n=789)=0,999/hit-rate 98,7%; Phase 6-8 (hậu xử lý & kiểm định FEA / active-learning loop / xác thực cuối & đóng gói) - xem bảng bên dưới, chi tiết đầy đủ tại [`outputs/phase5/reports/final_status_report_2026-07-31.md`](outputs/phase5/reports/final_status_report_2026-07-31.md).

| Phase | Thành phần | Trạng thái | Ghi chú |
|-------|-----------|--------|-------|
| 0 | Core SIMP Engine | ✅ Ổn định | 11 loại seed, mục tiêu auxetic, PBC, đồng nhất hóa dựa trên năng lượng |
| 1 | LHS Screening | ✅ Hoàn thành | Phân tích độ nhạy: `volfrac` là tham số chi phối (r ≈ 0,87–0,96). Lịch sử debug lần chạy đầu (0 mẫu auxetic): xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| 2 | Multi-Batch Adaptive DOE | ✅ Hoàn thành + cải tiến manufacturability + rebuild dQ | **8/8 lô gốc + batch 11 rebuild**, 7.920 mẫu, **91,9% auxetic** (đã rebuild 3 seed bị lỗi dQ, tăng từ 82,1%). Pipeline thích ứng tự dừng sau 2 lô liên tiếp không cải thiện mục tiêu. **2026-07-24**: (a) phân tích ngược xác nhận SEED chi phối manufacturability - thêm phân bổ mẫu theo seed; (b) phát hiện + sửa lỗi `dQ` + rebuild đầy đủ 2.160 mẫu (3 seed bất đối xứng). Xem [Phase 2](#2-multi-batch-adaptive-doe-phase-2---hoàn-thành) bên dưới và [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#bảng-tổng-hợp-lỗi-đã-sửa) |
| 3 | Dataset Build (trường mật độ + target) | ✅ Hoàn thành, **đã rebuild lần 2 (v2, 2026-07-25)** | **57.216 mẫu train** (sau aug) / 2.044 val / 2.044 test - dataset production hiện tại. Xem mục "Rebuild v2" ngay dưới bảng này |
| 4 | CNN Surrogate Model | ✅ Hoàn thành, đã retrain trên dataset v2 | Dự đoán (ν₁₂, ν₂₁, volfrac) từ trường mật độ. R² trên test set v2 (`surrogate_v2.pt`, 2026-07-25): ν₁₂ = **0,974**, ν₂₁ = **0,964**, volfrac = 0,983. Xem [Phase 4](#4-cnn-surrogate-model-phase-4---hoàn-thành) bên dưới |
| 5 | Conditional VAE | ✅✅ Surrogate-exploitation đã sửa TẬN GỐC bằng differentiable-physics; đã retrain trên dataset v2 | `cvae_realphysics.pt` (fine-tune real-physics, 2026-07-24, checkpoint production hiện hành): R²(FE,n=789)=**0,999** (oracle), hit rate single-shot=**98,7%**, frac manufacturable=0,35. Xem [Phase 5](#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực) bên dưới; toàn bộ quá trình thử-sai xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Nâng cấp tuỳ chọn cGAN/conditional diffusion (mục 5.6/5.7 trong roadmap chi tiết): ⬜ chưa bắt đầu |
| 6 | Hậu xử lý & kiểm định FEA | ✅ Hoàn thành | Nhị phân hoá + connectivity/min-feature/periodicity (`manufacturability.py`) + lọc/verify bằng FE thật (`best_of_n_eval.py`) - đã có sẵn trong pipeline Phase 5 |
| 7 | Active-learning loop | ✅ Kết luận (2026-07-31) | Implement + chạy production (`pipeline/phase5_cvae/active_learning.py`) - **không cải thiện** checkpoint đã tối ưu (`cvae_realphysics.pt` gần mức trần, mean_abs_error tệ đi 0,0246→0,0686 sau 1 vòng) - giữ nguyên checkpoint production. Chi tiết: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) mục 2026-07-31 |
| 8 | Xác thực cuối & đóng gói | 🟨 Một phần | **8.1 đối chiếu FE độc lập**: ✅ xong, `scikit-fem` khớp engine nội bộ tới 1e-9 (R²=1,000000, n=24) - xác nhận đúng vật lý. **8.3 thư viện thiết kế**: ✅ xong, 24 thiết kế, hit-rate 100%, 87,5% xác nhận manufacturable. **8.2 biến dạng lớn**, **8.5 chuẩn bị STL**: ⬜ chưa làm (tuỳ chọn) |

> Chi tiết từng phase con (2.1-2.9, 3.1-3.6, v.v.): xem dashboard `html/dashboards/workflow.html`.
> **Khoảng trống đã biết** (tóm tắt - xem đầy đủ tại [Giới hạn Đã biết](#giới-hạn-đã-biết--known-limitations)): phạt `mu` trong mục tiêu auxetic đang tắt (`mu=0.0`, đang chờ thiết kế lại); đồng nhất hóa chưa xuất độ cứng `E₁₁/E₀, E₂₂/E₀` nên `f1, f2` (roadmap gốc) chưa khả dụng; khả năng chế tạo - xem [Phase 5](#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực); các R²/hit-rate của Phase 5 đo trên cỡ mẫu rất nhỏ (n=3-24 điều kiện), CI rộng - đọc kỹ trước khi trích dẫn.

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

Đầu ra: `outputs/phase3/{train,val,test}.npz` - xem [Dataset Build (Phase 3)](#3-dataset-build-phase-3---hoàn-thành) bên dưới.

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
├── tests/                     # Bộ kiểm thử PyTest (307 test)
├── outputs/                   # Dữ liệu sinh ra - phần lớn (metadata/CSV/figures nhỏ, outputs/multi_batch/, outputs/pipeline/) ĐÃ commit; chỉ *.npz/*.npy/*.pt và outputs/phase3/*.npz bị gitignore (quá lớn)
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

## Pipeline: Screening → Multi-Batch DOE → Dataset → Surrogate → cVAE

### 1. LHS Screening (Phase 1)

Quét không gian tham số (`volfrac`, `penal`, `rmin`, `move`, `void_size_frac`, `rotation_deg`) bằng Latin Hypercube Sampling.

```bash
python -m pipeline.phase1_screening.screening_parallel --objective auxetic --seed hexagonal
python -m pipeline.phase1_screening.screening_parallel --all   # quét toàn bộ, tất cả seed
python -m pipeline.phase1_screening.analyst                    # tổng hợp kết quả -> _all_correlations.json, _all_summaries_parallel.json
```

Phân tích độ nhạy (tương quan Spearman) xác định **`volfrac` là tham số chi phối** (r ≈ 0,87–0,96); `move`, `rmin`, `void_size_frac` không có ý nghĩa thống kê. (Lịch sử debug lần chạy đầu - 0 mẫu auxetic - xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).)

### 2. Multi-Batch Adaptive DOE (Phase 2) - ✅ hoàn thành

Các lô tuần tự, mỗi lô được định hướng bởi phân tích độ phủ (KDE + phát hiện vùng thưa) trên kết quả tích lũy. `adaptive.py` quyết định **tinh chỉnh** (thu hẹp khoảng tham số + nhắm vào vùng thưa), **mở rộng** (thêm seed/mục tiêu), hoặc **dừng**.

```bash
python -m pipeline.phase2_multi_batch.main --phase1-summary outputs/pipeline/phase1
```

**Kết quả (8 lô, 7.920 mẫu, tỷ lệ hội tụ FE 100%):**

| Lô | Chiến lược | Số mẫu | % Auxetic | ν₁₂ tốt nhất |
|-------|----------|-----------|-----------|----------|
| 1 | Sobol (khám phá) | 1.320 | 74,9% | −0,612 |
| 2 | Sobol (khám phá) | 600 | 79,7% | −0,519 |
| 3 | Sobol (khám phá) | 720 | 71,8% | −0,565 |
| 4 | Optimized LHS (tinh chỉnh) | 1.056 | 83,6% | −0,605 |
| 5 | Optimized LHS (tinh chỉnh) | 1.067 | 85,7% | −0,752 |
| 6 | Optimized LHS (tinh chỉnh) | 1.045 | 85,4% | −0,649 |
| 7 | Optimized LHS (tinh chỉnh) | 1.056 | 85,3% | −0,621 |
| 8 | Optimized LHS (tinh chỉnh) | 1.056 | 87,8% | **−0,807** |

Khoảng `volfrac` hội tụ từ `[0,45, 0,70]` xuống `[0,50, 0,58]` ở lô 8; pipeline tự dừng sau 2 lô liên tiếp không cải thiện mục tiêu (độ thưa ổn định ~18,5%).

**2026-07-24 - 2 cải tiến tích hợp** (chi tiết đầy đủ + bảng số liệu: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)):
- Phân tích ngược 7.920 mẫu cho thấy manufacturability hầu như KHÔNG tương quan với tham số DOE liên tục (|Spearman r|<0,12) - biến chi phối là **SEED** (7,9% ở `hexagonal` tới 62,8% ở `reentrant_bowtie`). Thêm đo manufacturability tại thời điểm sinh (miễn phí) + `compute_seed_sample_allocation()` phân bổ mẫu lệch theo seed thay vì chia đều. Validated bằng 99 mẫu thật (lô 9): joint rate (auxetic∧manufacturable) 17,6%→**25,3%**.
- Lỗi hoán vị `dQ` (xem [Giới hạn #10](#giới-hạn-đã-biết--known-limitations)) khiến 3 seed bất đối xứng hội tụ dưới tiềm năng thật (2/3 từng sai cả dấu) → rebuild đầy đủ 2.160 mẫu (lô 11). Auxetic rate toàn dataset: 82,1%→**91,9%**.

### 3. Dataset Build (Phase 3) - ✅ hoàn thành

```bash
python3 pipeline/phase3_dataset/scan_dataset.py       # -> outputs/phase3/manifest.csv
python3 pipeline/phase3_dataset/build_npz.py --resolution 64   # -> dataset_64.npz
python3 pipeline/phase3_dataset/finalize_dataset.py --resolution 64  # -> train/val/test.npz
```

- Ảnh PNG mật độ (từ lưới `xPhys` 50×50) resize về 64×64 bằng box-filter downsampling; 33/7.920 mẫu (0,4%) bị loại (topology suy biến, `volfrac_achieved` ngoài `[0,05, 0,95]`).
- **Chia train/val/test 70/15/15, phân tầng theo seed**; **tăng cường đối xứng** (chỉ train): xoay 90°/270° hoán đổi `ν₁₂↔ν₂₁`, xoay 180°/lật giữ nguyên. Train: 5.520 → 33.120 mẫu (×6) *(số liệu lịch sử - xem cập nhật ngay dưới)*.
- Target xuất ra: `v12`, `v21`, `volfrac_achieved`. `f1, f2` (roadmap gốc) chưa khả dụng - cần mở rộng `compute_homogenized_tensor()` để xuất độ cứng chuẩn hóa trước.

**2026-07-24 - dọn dữ liệu tận gốc:** manifest hiện tại đã lọc bỏ 2.662/7.920 mẫu (33,6%, nhãn dao động/rời rạc/ngoài khoảng vật lý - xem [Giới hạn #13](#giới-hạn-đã-biết--known-limitations)). Pipeline cho ra `train.npz`=**22.080** mẫu, `val.npz`/`test.npz`=**789** mẫu mỗi tập (khác số liệu lịch sử 33.120/33.246 ở trên). `cvae_gamma20.pt` cũ vẫn train trên bản CŨ và được giữ nguyên làm baseline lịch sử; checkpoint mới train trên bản sạch - xem `surrogate_clean.pt`/`cvae_clean_v2.pt` ở mục 4-5 ngay dưới.

**2026-07-25 - Rebuild v2 (mở rộng quy mô):** sinh thêm raw sample bằng `analysis/scripts/generate_production_batch.py` (song song, độc lập với `pipeline/phase2_multi_batch/`), gộp với pool sạch cũ qua `analysis/scripts/assemble_phase3_v2.py`. Quá trình phát hiện + sửa 2 bug trong chính script sinh dữ liệu mới (range tham số quá rộng cho seed nhạy cảm; `beta=0,8` thay vì mặc định production `1,0`) - chi tiết đầy đủ (mục "2026-07-25"): [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). **Kết quả: train=57.216 (đạt mục tiêu 40-50k), val=2.044, test=2.044.** Đã thay thế `outputs/phase3/{train,val,test,dataset_64}.npz` (bộ cũ backup nguyên vẹn tại `outputs/phase3_backup/`).

### 4. CNN Surrogate Model (Phase 4) - ✅ hoàn thành

```bash
python3 pipeline/phase4_surrogate/train.py
python3 pipeline/phase4_surrogate/evaluate.py
python3 pipeline/phase4_surrogate/export_for_phase5.py
```

Kiến trúc (baseline "Phương án A"): 4× khối `Conv(3x3) + BatchNorm + ReLU + MaxPool` → global average pool → nối với one-hot của seed → 2 lớp FC → 3 đầu ra (ν₁₂, ν₂₁, volfrac_achieved). Huấn luyện trên `outputs/phase3/train.npz`, xác thực trên `val.npz`.

**Hiệu năng trên test set** (`outputs/phase4/evaluation_report.json`, `test.npz` giữ riêng - **1.184 mẫu**, không rò rỉ dữ liệu):

| Target | R² | 95% CI (bootstrap, n=1.184) | MAE |
|---|---|---|---|
| ν₁₂ | 0,910 | [0,896, 0,923] | 0,037 |
| ν₂₁ | 0,911 | [0,892, 0,926] | 0,036 |
| volfrac_achieved | 0,982 | [0,979, 0,984] | 0,007 |

CI tính bằng `pipeline/phase4_surrogate/bootstrap_ci.py` (percentile bootstrap trên 1.184 mẫu test - không cần train lại). Khác hẳn Phase 5 (CI rất rộng do chỉ 19-24 điều kiện, xem [Giới hạn Đã biết](#giới-hạn-đã-biết--known-limitations)), CI ở đây **hẹp** vì cỡ mẫu lớn - con số R²=0,91 đáng tin cậy, không phải một điểm ước lượng may rủi.

MAE theo seed dao động 0,021–0,048, không seed nào kém nghiêm trọng. Nếu R² < 0,90 trên bất kỳ target nào, thử mở rộng `channels` trong `SurrogateCNN` trước khi đổi kiến trúc.

> **Bảng trên đo trên `surrogate_best.pt` (data cũ, lẫn 33,6% nhãn lỗi).** Sau khi dọn dữ liệu (mục 3), `surrogate_clean.pt` đo trên cùng 1 test set sạch (789 mẫu): **v12 R²=0,922** [CI 0,907,0,935], **v21 R²=0,889** [CI 0,860,0,913] - so với `surrogate_best.pt` đo trên CHÍNH test set sạch này chỉ 0,484/0,384. Bằng chứng thực nghiệm rằng 33,6% mẫu nhãn lỗi là nguyên nhân chính khiến R² cũ thấp, không phải giới hạn kiến trúc (chi tiết + ablation xác nhận: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)). `surrogate_best.pt`/`surrogate_for_phase5.pt` **không bị ghi đè** - dùng `--surrogate-path outputs/phase4/surrogate_clean.pt` cho công việc mới.

> **2026-07-25 - retrain trên dataset v2** (57.216 mẫu train): `surrogate_v2.pt` đo trên test set v2 (2.044 mẫu) - **v12 R²=0,974**, **v21 R²=0,964**, **volfrac R²=0,983** - cải thiện so với `surrogate_clean.pt` (0,922/0,889), chủ yếu nhờ quy mô dữ liệu lớn hơn. Export cho Phase 5 tại `outputs/phase4/surrogate_for_phase5_v2.pt`. Không ghi đè `surrogate_best.pt`/`surrogate_for_phase5.pt` (quy ước cũ) - dùng `--ckpt`/`--surrogate-path` trỏ tới bản `_v2` cho công việc mới.

### 5. Conditional VAE (Phase 5) - ✅✅ đã sửa tận gốc bằng differentiable-physics (2026-07-24)

> **Checkpoint khuyến nghị hiện tại: `outputs/phase5/cvae_realphysics.pt`** - không phải `cvae_gamma20.pt` (baseline lịch sử) hay `cvae_clean_weighted.pt` (tốt nhất TRƯỚC differentiable-physics). Đo ở n=789 (toàn bộ test set): R²(FE thực) = **0,999** (oracle), **0,992** (K=10), **hit rate single-shot 98,7%** - 1 lần `generate()` duy nhất giờ đã đáng tin, không bắt buộc phải lọc qua best-of-N nữa (dù vẫn nên dùng best-of-N khi cần độ chính xác tối đa, gần như miễn phí về R² so với single-shot ở checkpoint này). Chi tiết đầy đủ + kiểm tra chống ngộ nhận: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) (mục "đột phá differentiable-physics").

```bash
# Fine-tune từ checkpoint có sẵn với differentiable-physics (khuyến nghị - cách đã tạo ra cvae_realphysics.pt):
python3 pipeline/phase5_cvae/train.py --gamma 20.0 --epochs 35 --kl-warmup 1 \
  --surrogate-path outputs/phase4/surrogate_clean.pt --resume-from outputs/phase5/cvae_clean_weighted.pt \
  --output-name cvae_realphysics.pt --select-by fe_r2 --fe-eval-every 2 \
  --lambda-real-physics 20.0 --real-physics-subsample 8 --real-physics-every 2 --real-physics-workers 8

# Train từ đầu, KHÔNG differentiable-physics (baseline lịch sử, phần dưới đây mô tả câu chuyện đã sửa):
python3 pipeline/phase5_cvae/train.py --gamma 20.0 --epochs 50
python3 pipeline/phase5_cvae/evaluate.py
python3 pipeline/phase5_cvae/sample.py --ckpt outputs/phase5/cvae_realphysics.pt   # single-shot giờ đáng tin ở checkpoint này
```

`sample.py` sinh 1 ứng viên/lần gọi. Với các checkpoint CŨ (không differentiable-physics), không lọc FE nên chỉ để xem qua - in cảnh báo mỗi lần chạy, dùng `best_of_n_eval.py` mới đáng tin. Với `cvae_realphysics.pt`, single-shot đã đủ tin cậy (98,7% hit rate) nhưng `best_of_n_eval.py` vẫn là cách đo lường chính thức/nghiêm ngặt nhất.

Huấn luyện trên `train.npz`, dùng surrogate Phase-4 **đóng băng** để tính loss property-consistency (`recon + beta·kl + gamma·PROP_LOSS_SCALE·prop_loss`).

**Quy trình đo lường chính thức (`best_of_n_eval.py`):** sinh N ứng viên/điều kiện, để **FE thực** (không phải surrogate) chọn người thắng cuộc - công thức lấy cảm hứng từ pipeline Deep-DRAM đã công bố (Pahlavani et al. 2024, xem [Tài liệu Tham khảo](#tài-liệu-tham-khảo)).

```bash
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 30                        # oracle (FE trên toàn bộ N)
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 30 --k-fe-verify 10        # thực dụng (lọc sơ bộ bằng surrogate)
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 1500 --k-fe-verify 8 --require-manufacturable   # bắt buộc khả năng chế tạo
```

**Tóm tắt hành trình phát hiện + sửa** (mọi bảng số liệu/CI đầy đủ: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)):

1. R² đo qua surrogate đóng băng không đáng tin (**surrogate exploitation** - decoder học đánh lừa CNN thay vì sinh hình học đúng): FE thực cho R² âm sâu ở mọi mức gamma đã thử. Hai biện pháp khắc phục lúc huấn luyện (self-play, ensemble surrogate) đều **không** khắc phục được vấn đề single-shot.
2. `force_periodic()` (ép cứng periodicity bằng 1 phép gán pixel, không cần train) tích hợp làm mặc định BẬT trong `best_of_n_eval.py`/`sample.py` (cờ `--no-force-periodic` để tắt) - nâng manufacturability >10× gần như miễn phí.
3. Dọn dữ liệu (loại 33,6% mẫu nhãn dao động, [Giới hạn #13](#giới-hạn-đã-biết--known-limitations)) + weighted-sampling theo phổ auxetic + chọn checkpoint bằng `--select-by fe_r2 --fe-eval-every N` (**không dùng `val_loss` mặc định** - bị KL-warmup đánh lừa, xác nhận độc lập 2 lần) → `cvae_clean_weighted.pt`, checkpoint tốt nhất trước differentiable-physics (oracle R²=0,83, K=10 R²=0,65 ở n=24).
4. Đánh giá lại ở quy mô lớn (n=789, TOÀN BỘ test set): oracle mode giữ được cải thiện (khiêm tốn hơn n=24: R²=0,44 chứ không phải 0,83). Chế độ "thực dụng" K=10 **sụp đổ về CI âm/cắt-0 cho mọi checkpoint TRƯỚC differentiable-physics** - khuyến nghị trước đây ("K=10 giữ gần hết R² gain") đã SAI và bị gỡ bỏ.
5. **Differentiable-physics** (`real_physics_prior_loss()` - FE-solve thật + gradient giải tích ngay trong training loop, không qua surrogate) sửa **tận gốc**: `cvae_realphysics.pt` đạt R²(FE, n=789)=**0,999** (oracle)/**0,992** (K=10), hit rate single-shot **98,7%**. `train.py` hỗ trợ `--lambda-real-physics/--real-physics-subsample/--real-physics-every/--real-physics-workers` để giữ chi phí FE-solve hợp lý (~50-100ms/mẫu tuần tự, song song qua `multiprocessing.Pool`).

**Khả năng chế tạo:** best-of-N mặc định tối ưu độ chính xác Poisson; thêm `--require-manufacturable` + N lớn khi cần đảm bảo khả thi chế tạo (đổi lấy R² thấp hơn ở checkpoint chưa dùng differentiable-physics - không cần đánh đổi này ở `cvae_realphysics.pt`, `frac_manufacturable`=0,350 đã cao sẵn). Biện pháp huấn luyện lại từng thử (regularize posterior/prior-samples) kém hiệu quả hơn `force_periodic()` - chi tiết: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

**2026-07-25 - retrain trên dataset v2:** train from-scratch với real-physics loss ngay từ epoch 1 **thất bại** (R²(FE) âm sâu) - xác nhận quy trình 2-stage (base rồi fine-tune, xem mục trên) là **bắt buộc**, không thể gộp 1 bước. `cvae_v2_finetuned.pt` (đúng recipe 2-stage) so với `cvae_realphysics.pt`, đo trên CÙNG test set v2 (n=300):

| Checkpoint | R²(FE) oracle | hit rate single-shot | frac manufacturable |
|---|---|---|---|
| `cvae_v2_finetuned.pt` (train trên dataset v2) | 0,9953 | **0,997** | 0,247 |
| `cvae_realphysics.pt` (train trên dataset cũ, giữ nguyên) | **0,9984** | 0,983 | **0,280** |

Hai checkpoint ngang ngửa nhau (cả hai đạt hit-rate best-of-N=1,0); `cvae_v2_finetuned.pt` nhỉnh hơn ở hit-rate single-shot (số quan trọng nhất cho dùng thực tế không lọc), `cvae_realphysics.pt` nhỉnh hơn nhẹ ở R²/manufacturability. **Khuyến nghị: dùng `cvae_v2_finetuned.pt` cho nhất quán với dataset production hiện tại**; `cvae_realphysics.pt` vẫn được giữ nguyên làm tham chiếu lịch sử, không bị ghi đè.

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

Trạng thái hiện tại: **437/437 test pass** (`pytest tests/ -q`, ~7s).

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

### Tiếng Việt

Mục này gộp lại toàn bộ khoảng trống/giới hạn đã biết của dự án ở một chỗ, cho mục đích đánh giá khoa học (thay vì rải rác trong README/EXPERIMENT_LOG). Không có mục nào dưới đây là mới - tất cả đã được ghi nhận ở nơi khác trong tài liệu; đây là bản tổng hợp.

1. **R²/hit-rate của Phase 5 lịch sử đo trên cỡ mẫu nhỏ (n=24), CI rộng.** Mở rộng lên n=300/789 (toàn bộ test set) cho thấy oracle mode giữ được cải thiện nhưng khiêm tốn hơn (0,44 chứ không phải 0,83); chế độ "thực dụng" K=10 **sụp đổ về CI âm/cắt-0** cho mọi checkpoint TRƯỚC differentiable-physics - khuyến nghị cũ ("K=10 giữ gần hết R² gain") đã bị gỡ bỏ. Với checkpoint hiện tại (`cvae_realphysics.pt`), CI ở n=789 cực hẹp và cả 2 chế độ đều gần hoàn hảo (R²=0,999/0,992) - bài học về cỡ mẫu nhỏ vẫn áp dụng cho MỌI số liệu tương lai. Chi tiết: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

2. **Manufacturability của đầu ra gốc (không lọc) rất thấp** (0-3,5%, đồng đều toàn không gian thuộc tính, không phải vùng chết cục bộ). `force_periodic()` + `--require-manufacturable` cải thiện đáng kể (R²=0,745, hit rate 100% ở checkpoint sạch+weighted, n=24) - không còn cần đánh đổi này ở `cvae_realphysics.pt` (`frac_manufacturable`=0,350 tự nhiên).

3. **[ĐÃ GIẢI QUYẾT 2026-07-24] Single-shot generation từng không đáng tin** (R² FE âm sâu mọi gamma, surrogate exploitation - self-play/ensemble không khắc phục được). Differentiable-physics sửa tận gốc, xem mục 9.

4. **Thành phần phạt `mu` trong hàm mục tiêu auxetic đang tắt** (`mu=0.0`, sai sót khái niệm chưa thiết kế lại) - toàn bộ dataset hiện có (Phase 2-5) dùng cấu hình này.

5. **`f1, f2` (mục tiêu độ cứng chuẩn hóa theo roadmap gốc) chưa khả dụng** - `compute_homogenized_tensor()` chưa xuất `E₁₁/E₀, E₂₂/E₀`; dataset/surrogate/cVAE chỉ dùng `ν₁₂, ν₂₁, volfrac_achieved`.

6. **Test tự động (437/437 pass) chưa phủ hết I/O nặng**: vòng lặp chính của `screening_parallel.py`, `pipeline/seeds/*.py`, `visualize.py`, và lệnh gọi SIMP FE thật bên trong `multi_batch/runner.py::evaluate_single` (được mock trong test hiện có).

7. **Một số HTML dashboard từng lỗi thời** so với kết quả đã xác thực (vd `html/inverse_auxetic_report.html`, sinh 2026-07-16) - đã gắn banner cảnh báo trỏ về README + báo cáo hiện hành. Rủi ro mang tính hệ thống (tài liệu trực quan không tự đồng bộ với code/dữ liệu) cần lưu ý khi thêm dashboard mới.

8. **Tài liệu chủ yếu bằng tiếng Việt** (README, EXPERIMENT_LOG) - mục Limitations có bản dịch English, các tài liệu chi tiết khác thì chưa.

9. **[ĐÃ GIẢI QUYẾT 2026-07-24] Differentiable-physics đã chạy huấn luyện thật, kết quả đột phá.** Fine-tune 35 epoch (`--lambda-real-physics 20.0`, ~15-18 phút) → `cvae_realphysics.pt`: R²(FE, n=789)=0,999 (oracle)/0,992 (K=10), hit rate single-shot 98,7%. Sửa tận gốc mục 3, không phải workaround. Chi tiết: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). **Còn lại:** train from-scratch (chỉ mới fine-tune); quét `--lambda-real-physics` khác 20,0.

10. **[ĐÃ GIẢI QUYẾT 2026-07-24] Lỗi hoán vị `dQ` trong `compute_homogenized_tensor()` - đã rebuild đầy đủ dataset bị ảnh hưởng.** 3 seed bất đối xứng (`hourglass`,`hexagonal`,`reentrant_bowtie`) từng hội tụ dưới tiềm năng thật (2/3 sai cả dấu). Rebuild 2.160 mẫu tích hợp vào Phase 3 (backup: `manifest_pre_dqfix_backup.csv`) - auxetic rate toàn dataset **82,1%→91,9%**. Phase 4/5 đã train lại trên dataset này, xem mục 13.

11. **`converged=True`/150 iteration KHÔNG đồng nghĩa hội tụ thật, và ~1/4 dataset có hình rời rạc** (audit 2026-07-24, yêu cầu advisor). Chỉ **23,7%** hội tụ thật vì tolerance (76,3% chỉ chạm `max_iter`); **23,9%** mẫu rời rạc (`check_connectivity()`), tập trung ở `grid_circular_voids`/`nine_circle`/`four_circle`. Dữ liệu: `outputs/phase3/manifest_quality.csv`. Chưa dùng để lọc lại dataset trước khi train Phase 4/5.

12. **[ĐÃ GIẢI QUYẾT 2026-07-24] Weighted-sampling theo phổ auxetic từng inconclusive do bug chọn checkpoint** (`val_loss` chọn nhầm epoch 2/50 vì bị KL-warmup đội lên cơ học). Train lại đúng cách với `--select-by fe_r2` KẾT HỢP dataset sạch → `cvae_clean_weighted.pt`, checkpoint tốt nhất trước differentiable-physics (oracle R²=0,833, thực dụng=0,653, n=24).

13. **[ĐÃ GIẢI QUYẾT 2026-07-24] Dataset đã dọn tận gốc (loại 33,6% mẫu nhãn dao động/limit-cycle, 2.662/7.920) - Phase 4/5 đã train lại trên bản sạch.** Metric `osc_score` phát hiện trực tiếp trên lịch sử tối ưu (không suy diễn qua pixel). Surrogate R²(v12) 0,484→0,922; cVAE best-of-N oracle R² 0,314→0,679 (không weighted) / 0,833 (+ weighted). **Ablation 3-seed độc lập** xác nhận: phần lớn mức tăng R² Phase 4 đến từ bản vá `dQ` (mục 10), KHÔNG phải từ bộ lọc `osc_score` này (0,952 không lọc vs 0,922 đã lọc, CI tách biệt) - nhưng ở Phase 5 best-of-N, điểm ước lượng lại nghiêng có lợi cho việc lọc (0,68 vs 0,52, CI chồng lấn ở n=24, chưa đủ ý nghĩa thống kê). Vẫn giữ bộ lọc mặc định (mẫu bị lọc có nhãn sai rõ ràng, giá trị vệ sinh dữ liệu độc lập với R²). Chi tiết đầy đủ: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

14. **Toàn bộ pipeline dùng FEM tuyến tính (giả định biến dạng nhỏ).** Tài liệu (vd CMES 2024, "Topology Optimization of Metamaterial Microstructures for NPR under Large Deformation") cho thấy metamaterial auxetic thiết kế dưới giả định biến dạng nhỏ có thể suy giảm nhanh hệ số Poisson âm khi bị kéo/nén thật đáng kể. Đây là lựa chọn kiến trúc hợp lý cho giai đoạn hiện tại (chưa có bước in 3D/thử nghiệm vật lý trong roadmap Phase 6-8), nhưng mọi số liệu `ν₁₂` trong dataset chỉ đảm bảo đúng ở chế độ biến dạng vô cùng nhỏ - cần lưu ý nếu roadmap tương lai hướng tới chế tạo/thử nghiệm vật lý thật.

15. **[GIẢI QUYẾT PHẦN LỚN 2026-07-31] Physics oracle (loss huấn luyện + oracle đánh giá R²=0,999) dùng CHUNG một engine FE/homogenization nội bộ** (`simp/core/solver.py` + `simp/homogenization/compute.py`) cho cả training lẫn evaluation. Đã đối chiếu bằng 1 implementation FE **độc lập hoàn toàn** viết mới trên thư viện `scikit-fem` (`analysis/scripts/skfem_homogenization.py` - mesh/assembly/solve dùng skfem, periodic BC + công thức homogenize Q tự viết mới, KHÔNG tái dùng `simp/core/pbc.py`/`compute.py`) - kết quả trên 24 mẫu thật trải khắp property space: `mean|Δv12|=3,7e-9, max|Δv12|=6,8e-9, R²=1,000000` (khớp tới sai số làm tròn dấu phẩy động, không phải trùng hợp). Xác nhận engine hiện tại đúng vật lý, không chỉ nhất quán nội bộ. Chi tiết: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) mục 2026-07-31. **Còn lại (nếu hướng tới công bố khoa học ở mức cao hơn):** chưa đối chiếu với phần mềm FE thương mại/đã được công nhận rộng rãi (ANSYS/Abaqus/FEniCS/COMSOL) - scikit-fem tuy độc lập thật nhưng ít được dùng làm "ground truth" tham chiếu trong literature hơn các công cụ trên.

16. **[CHẤP NHẬN 2026-07-29] Seed `hexagonal` có yield thấp hơn lịch sử (~80,5% so với 93,9%) sau khi sửa 2 bug nghiêm trọng (Q-scaling A1 + trạng thái hấp thụ x=0 trong OC), đã xác nhận không có fix đơn giản.** Bug A1 (đã sửa) từng vô tình đóng vai trò regularizer cho `hexagonal` (seed mỏng/khó nhất) suốt lịch sử dự án - sau khi sửa đúng, ngưỡng stiffness `delta=10%` (khớp thiết kế tham chiếu gốc) không đủ mạnh để giữ ổn định như trước cho riêng seed này. Đã thử 3 hướng khắc phục - tăng `delta` (10%→25%), tăng `beta` (1,0→4,0), và `use_sqrt`+`penal_init=2,0` (damping/continuation đã kiểm chứng giúp `reentrant_bowtie` trên code cũ, xem `PILOT_REPORT.md`) - **cả 3 đều phản tác dụng hoặc vô ích** (sạch giảm còn 0-75%, do bùng nổ dao động/rời rạc, bộ tối ưu bỏ hẳn mục tiêu auxetic, hoặc thiếu ngân sách vòng lặp để continuation hoàn tất) - xác nhận cấu hình mặc định hiện tại đã là tốt nhất trong các hướng đã thử. `hourglass` (98,5%→~85% ở dải volfrac khớp lịch sử) và `reentrant_bowtie` (77,2%→**87,0%, cải thiện thật**) không có vấn đề này. `degenerate=0` (hết sụp cấu trúc vĩnh viễn) ở cả 3 seed (`hourglass` n=100, `hexagonal` n=400, `reentrant_bowtie` n=100). Chưa rebuild dataset 57k mẫu sản xuất - dataset hiện có sinh bằng code cũ (trước các fix này), không bị ảnh hưởng. Chi tiết đầy đủ: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

### English

This section consolidates every known gap/limitation of the project in one place for scientific-review purposes (rather than scattered across README/EXPERIMENT_LOG). Nothing below is new information - all of it is documented elsewhere; this is a summary.

1. **Historical Phase 5 R²/hit-rate numbers were measured on a small sample (n=24), wide CIs.** Expanding to n=300/789 (the full test set) showed oracle mode keeps its improvement but more modestly (0.44, not 0.83); the "practical" K=10 mode **collapses to a negative/zero-crossing CI** for every checkpoint BEFORE differentiable-physics - the old recommendation ("K=10 keeps most of the R² gain") has been withdrawn. With the current checkpoint (`cvae_realphysics.pt`), the n=789 CI is extremely tight and both modes are near-perfect (R²=0.999/0.992) - the small-sample lesson still applies to any future numbers. Details: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

2. **Manufacturability of raw (unfiltered) output is very low** (0-3.5%, uniformly low across the property space, not a localized dead zone). `force_periodic()` + `--require-manufacturable` help substantially (R²=0.745, 100% hit rate on the clean+weighted checkpoint, n=24) - no longer a real tradeoff on `cvae_realphysics.pt` (`frac_manufacturable`=0.350 naturally).

3. **[RESOLVED 2026-07-24] Single-shot generation used to be unreliable** (deeply negative real-FE R² at every gamma, surrogate exploitation - self-play/ensemble did not fix it). Differentiable-physics fixed it at the root, see item 9.

4. **The `mu` penalty term in the auxetic objective is disabled** (`mu=0.0`, unresolved conceptual flaw) - the entire existing dataset (Phase 2-5) uses this configuration.

5. **`f1, f2` (normalized stiffness targets from the original roadmap) are not available** - `compute_homogenized_tensor()` doesn't export `E₁₁/E₀, E₂₂/E₀`; dataset/surrogate/cVAE only use `ν₁₂, ν₂₁, volfrac_achieved`.

6. **Automated tests (437/437 pass) don't cover the heaviest I/O paths**: `screening_parallel.py`'s main loop, `pipeline/seeds/*.py`, `visualize.py`, and the real SIMP FE call inside `multi_batch/runner.py::evaluate_single` (mocked in the existing tests).

7. **Some HTML dashboards had gone stale** relative to validated results (e.g. `html/inverse_auxetic_report.html`, generated 2026-07-16) - now carry a warning banner pointing to the README and current reports. A systemic risk (visual docs don't auto-sync with code/data) worth watching for new dashboards.

8. **Primary documentation is in Vietnamese** (README, EXPERIMENT_LOG) - the Limitations section is translated, the other detailed documents are not.

9. **[RESOLVED 2026-07-24] Differentiable-physics has now been used for real training, with breakthrough results.** 35-epoch fine-tune (`--lambda-real-physics 20.0`, ~15-18 min) → `cvae_realphysics.pt`: R²(FE, n=789)=0.999 (oracle)/0.992 (K=10), 98.7% single-shot hit rate. A root-cause fix for item 3, not a workaround. Details: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). **Still not done:** training from scratch (only fine-tuning tried); other `--lambda-real-physics` values besides 20.0.

10. **[RESOLVED 2026-07-24] The `dQ` pixel-permutation bug in `compute_homogenized_tensor()` - affected dataset fully rebuilt.** Three asymmetric seeds (`hourglass`, `hexagonal`, `reentrant_bowtie`) used to converge well below their true potential (2/3 with the wrong sign entirely). A 2,160-sample rebuild is integrated into Phase 3 (backup: `manifest_pre_dqfix_backup.csv`) - full-dataset auxetic rate **82.1%→91.9%**. Phase 4/5 have since been retrained on this dataset, see item 13.

11. **`converged=True`/reaching 150 iterations does NOT mean true convergence, and ~1/4 of the dataset has disconnected geometry** (audit 2026-07-24, advisor request). Only **23.7%** truly converged via tolerance (76.3% just hit `max_iter`); **23.9%** of samples are disconnected (`check_connectivity()`), concentrated in `grid_circular_voids`/`nine_circle`/`four_circle`. Data: `outputs/phase3/manifest_quality.csv`. Not yet used to re-filter the dataset before training Phase 4/5.

12. **[RESOLVED 2026-07-24] Weighted-sampling by the auxetic spectrum used to be inconclusive due to a checkpoint-selection bug** (`val_loss` mistakenly picked epoch 2/50, mechanically inflated during KL-warmup). Retrained correctly with `--select-by fe_r2` combined with the clean dataset → `cvae_clean_weighted.pt`, the best checkpoint before differentiable-physics (oracle R²=0.833, practical=0.653, n=24).

13. **[RESOLVED 2026-07-24] The training dataset has been cleaned at the root (33.6% of samples removed, 2,662/7,920, limit-cycle label oscillation) - Phase 4/5 have since been retrained on the clean version.** New metric `osc_score` detects oscillation directly on the optimization trajectory (not inferred from pixels). Surrogate R²(v12) 0.484→0.922; cVAE best-of-N oracle R² 0.314→0.679 (unweighted) / 0.833 (+weighted). A **3-seed independent ablation** confirmed most of the Phase 4 R² gain comes from the `dQ` fix (item 10), NOT from this `osc_score` filter (0.952 unfiltered vs 0.922 filtered, non-overlapping CIs) - but at the Phase 5 best-of-N level, the point estimate favors filtering (0.68 vs 0.52, overlapping CIs at n=24, not yet statistically significant). Filter kept as default anyway (removed samples are genuinely mislabeled, independent data-hygiene value). Full details: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

14. **The entire pipeline uses linear FEM (small-strain assumption).** Literature (e.g. CMES 2024, "Topology Optimization of Metamaterial Microstructures for NPR under Large Deformation") shows auxetic metamaterials designed under a small-strain assumption can lose their negative Poisson's ratio quickly under real, significant compression/tension. A reasonable architectural choice for the current stage (no 3D-printing/physical-testing step in the Phase 6-8 roadmap yet), but every `ν₁₂` number in the dataset is only guaranteed accurate in the infinitesimal-strain regime - worth flagging if a future roadmap targets physical fabrication/testing.

15. **[MOSTLY RESOLVED 2026-07-31] The physics oracle (training loss + evaluation oracle, R²=0.999) uses the SAME internal FE/homogenization engine** (`simp/core/solver.py` + `simp/homogenization/compute.py`) for both training and evaluation. Cross-checked against a **fully independent** FE implementation written fresh on `scikit-fem` (`analysis/scripts/skfem_homogenization.py` - mesh/assembly/solve via skfem, periodic BC + Q homogenization hand-written, NOT reusing `simp/core/pbc.py`/`compute.py`) - result on 24 real samples spanning the property space: `mean|Δv12|=3.7e-9, max|Δv12|=6.8e-9, R²=1.000000` (agreement at floating-point round-off level, not coincidence). Confirms the existing engine is physically correct, not merely internally self-consistent. Details: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), 2026-07-31 entry. **Remaining (only relevant for a higher publication bar):** not yet cross-checked against a widely-recognized commercial/established FE tool (ANSYS/Abaqus/FEniCS/COMSOL) - scikit-fem is genuinely independent but less commonly cited as a reference "ground truth" in the literature than those tools.

16. **[ACCEPTED 2026-07-29] The `hexagonal` seed has lower yield than historical (~80.5% vs 93.9%) after fixing 2 serious bugs (A1 Q-scaling + an x=0 absorbing-state bug in the OC loop), confirmed to have no simple fix.** The (now-fixed) A1 bug had accidentally acted as a regularizer for `hexagonal` (the thinnest/hardest seed) throughout the project's history - once corrected, the reference-matching `delta=10%` stiffness threshold is not strong enough to keep this specific seed as stable as before. Tried raising `delta` (10%→25%) and `beta` (1.0→4.0) - **both backfired sharply** (clean rate dropped to 0-10%, from oscillation/disconnection blowing up or the optimizer abandoning the auxetic objective entirely) - confirming the current default is already the best of the tested configurations. `hourglass` (98.5%→~85% at the historically-matched volfrac range) and `reentrant_bowtie` (77.2%→**87.0%, a genuine improvement**) don't show this issue. `degenerate=0` (no more permanent structural collapse) across all 3 seeds (`hourglass` n=100, `hexagonal` n=400, `reentrant_bowtie` n=100). The 57k-sample production dataset has not been rebuilt - it was generated with code predating these fixes and is unaffected. Full details: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

---

## Tài liệu

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
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

Lộ trình thiết kế ngược gồm 8 giai đoạn (phase). Phase 1-4 đã hoàn thành và được xác thực trên dữ liệu thực; baseline của Phase 5 huấn luyện được nhưng **không vượt qua kiểm chứng vật lý thực** (xem hàng bên dưới); Phase 6 trở đi đang được tiến hành.

| Phase | Thành phần | Trạng thái | Ghi chú |
|-------|-----------|--------|-------|
| 0 | Core SIMP Engine | ✅ Ổn định | 11 loại seed, mục tiêu auxetic, PBC, đồng nhất hóa dựa trên năng lượng |
| 1 | LHS Screening | ✅ Hoàn thành | Phân tích độ nhạy: `volfrac` là tham số chi phối (r ≈ 0,87–0,96). Lịch sử debug lần chạy đầu (0 mẫu auxetic): xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#phase-1--lhs-screening) |
| 2 | Multi-Batch Adaptive DOE | ✅ Hoàn thành | **8/8 lô (batch)**, 7.920 mẫu, **82,1% auxetic**, ν₁₂ tốt nhất = −0,807. Pipeline thích ứng tự dừng sau 2 lô liên tiếp không cải thiện mục tiêu |
| 3 | Dataset Build (trường mật độ + target) | ✅ Hoàn thành | 7.920 mẫu → trường mật độ 64×64, lọc outlier, chia tập 70/15/15 theo phân tầng seed, tăng cường đối xứng theo vật lý (train: 33.120 mẫu) |
| 4 | CNN Surrogate Model | ✅ Hoàn thành | Dự đoán (ν₁₂, ν₂₁, volfrac) từ trường mật độ. R² trên test set: ν₁₂ = 0,910, ν₂₁ = 0,911, volfrac = 0,982 (MAE 0,037 / 0,036 / 0,007). Xem [Phase 4](#4-cnn-surrogate-model-phase-4---hoàn-thành) bên dưới |
| 5 | Conditional VAE | ✅ Thiết kế ngược qua best-of-N + FE thực | Một mẫu cVAE đơn lẻ (single-shot) không đáng tin cậy về auxeticity (khai thác surrogate). Quy trình chính thức: sinh N ứng viên, chọn ứng viên tốt nhất bằng **FE thực** — R²(v12, FE thực) = **+0,44 đến +0,60**, tỷ lệ trúng auxetic thực = **100%**. Hình học sinh ra cũng hiếm khi *khả thi để chế tạo* kể cả khi chính xác — giảm nhẹ bằng lọc ở bước inference (`--require-manufacturable`, N lớn) với cái giá về R². Xem [Phase 5](#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực) bên dưới; toàn bộ quá trình thử-sai (gamma-sweep, self-play, ensemble, regularization) xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| 6 | cGAN / Conditional Diffusion (nâng cấp tùy chọn) | ⬜ Chưa bắt đầu | |
| 7-8 | Xác thực, triển khai | ⬜ Chưa bắt đầu | |

> Chi tiết từng phase con (2.1-2.9, 3.1-3.6, v.v.): xem dashboard `html/dashboards/workflow.html`.
> **Khoảng trống đã biết** (tóm tắt — xem đầy đủ tại [Giới hạn Đã biết](#giới-hạn-đã-biết--known-limitations)): phạt `mu` trong mục tiêu auxetic đang tắt (`mu=0.0`, đang chờ thiết kế lại); đồng nhất hóa chưa xuất độ cứng `E₁₁/E₀, E₂₂/E₀` nên `f1, f2` (roadmap gốc) chưa khả dụng; khả năng chế tạo — xem [Phase 5](#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực); các R²/hit-rate của Phase 5 đo trên cỡ mẫu rất nhỏ (n=3-24 điều kiện), CI rộng — đọc kỹ trước khi trích dẫn.

---

## Bắt đầu

### Yêu cầu & cài đặt

**Python** ≥ 3.10, **numpy/scipy/matplotlib** (core), **pandas/scikit-learn/Pillow** (`pipeline/phase3_dataset/`), **torch** (`pipeline/phase4_surrogate/`, `pipeline/phase5_cvae/` — không có trong `requirements.txt` gốc, cần cài thủ công):

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

Đầu ra: `outputs/phase3/{train,val,test}.npz` — xem [Dataset Build (Phase 3)](#3-dataset-build-phase-3---hoàn-thành) bên dưới.

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
│   ├── phase2_multi_batch/    # Phase 2: adaptive DOE — sampling (Sobol/LHS), runner, adaptive (quyết định), coverage
│   ├── phase3_dataset/        # Phase 3: scan_dataset, build_npz, augment_symmetry, finalize_dataset
│   ├── phase4_surrogate/      # Phase 4: dataset, model (SurrogateCNN), train, evaluate, export_for_phase5
│   └── phase5_cvae/           # Phase 5: dataset, model, losses, train, evaluate, sample, verify_fe,
│                               #   adversarial_dataset, self_play, best_of_n_eval (inference chính thức),
│                               #   manufacturability, coverage_eval, bootstrap_ci (CI cho R²/hit_rate)
│
├── analysis/                 # Phân tích độ nhạy (ANOVA, Sobol, regression), Pareto front, dataset QC
├── notebooks/, html/         # Jupyter notebook + dashboard/báo cáo (xem html/index.html)
├── html/dashboards/workflow.html        # Dashboard workflow toàn dự án (chi tiết từng phase con)
├── tests/                     # Bộ kiểm thử PyTest (208 test)
├── outputs/                   # Dữ liệu sinh ra — phần lớn (metadata/CSV/figures nhỏ, outputs/multi_batch/, outputs/pipeline/) ĐÃ commit; chỉ *.npz/*.npy/*.pt và outputs/phase3/*.npz bị gitignore (quá lớn)
├── PROJECT_DOCUMENTATION.md, EXPERIMENT_LOG.md, INSTRUCTIONS.md, CHANGELOG.md   # xem mục Tài liệu
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
| `reentrant_bowtie` | Lỗ rỗng hình nơ (hình học re-entrant) — seed mới nhất | 48,6% |

\* Tỷ lệ mẫu có cả ν₁₂ < 0 và ν₂₁ < 0, đo trên toàn bộ 7.920 mẫu của multi-batch DOE đã hoàn thành (Phase 2). `reentrant_bowtie` và `hexagonal` là các hình học khó đẩy về auxetic nhất và là ứng viên tốt để tiếp tục tinh chỉnh tham số.

Phép xoay (`--rotation_deg`) có thể áp dụng cho bất kỳ seed nào.

---

## Hàm Mục tiêu (Auxetic)

```
c = Q₁₂ − μ·(Q₁₁ + Q₂₂) + penalty_terms
penalty: kích hoạt khi Q₁₁ hoặc Q₂₂ < δ = 0.1·volfrac·E₀, chuẩn hóa theo δ²
```

- `compute_nu12` / `compute_nu21` dùng **nghịch đảo đầy đủ của ma trận độ mềm 3×3** (`S = Q⁻¹`), không dùng công thức tắt trực hướng (orthotropic) `ν₁₂ = Q₁₂/Q₂₂` — công thức tắt này sai bất cứ khi nào ô đơn vị bị xoay (liên kết cắt-pháp `Q₁₃, Q₂₃ ≠ 0`). Đây là nguyên nhân gốc của lỗi "0 mẫu auxetic" ở Phase 1 (xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#phase-1--lhs-screening)).
- Thành phần `μ` được thiết kế để đẩy `Q₁₂` âm hơn nữa thay vì dừng lại gần 0, nhưng công thức hiện tại **có sai sót về mặt khái niệm** (đang chờ thiết kế lại) — hiện bị tắt theo mặc định (`mu=0.0`), là cấu hình dùng cho toàn bộ 8 lần chạy multi-batch DOE đã hoàn thành.
- Một hệ số phạt độ cứng được kích hoạt khi `Q₁₁` hoặc `Q₂₂` giảm dưới `δ`, ngăn sụp đổ cấu trúc (topology suy biến dạng rỗng vẫn xảy ra ở ~0,4% lần chạy — bị lọc bỏ ở Phase 3).

---

## Pipeline: Screening → Multi-Batch DOE → Dataset → Surrogate → cVAE

### 1. LHS Screening (Phase 1)

Quét không gian tham số (`volfrac`, `penal`, `rmin`, `move`, `void_size_frac`, `rotation_deg`) bằng Latin Hypercube Sampling.

```bash
python -m pipeline.phase1_screening.screening_parallel --objective auxetic --seed hexagonal
python -m pipeline.phase1_screening.screening_parallel --all   # quét toàn bộ, tất cả seed
python -m pipeline.phase1_screening.analyst                    # tổng hợp kết quả -> _all_correlations.json, _all_summaries_parallel.json
```

Phân tích độ nhạy (tương quan Spearman) xác định **`volfrac` là tham số chi phối** (r ≈ 0,87–0,96); `move`, `rmin`, `void_size_frac` không có ý nghĩa thống kê. (Lịch sử debug lần chạy đầu — 0 mẫu auxetic — xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#phase-1--lhs-screening).)

### 2. Multi-Batch Adaptive DOE (Phase 2) — ✅ hoàn thành

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

### 3. Dataset Build (Phase 3) — ✅ hoàn thành

```bash
python3 pipeline/phase3_dataset/scan_dataset.py       # -> outputs/phase3/manifest.csv
python3 pipeline/phase3_dataset/build_npz.py --resolution 64   # -> dataset_64.npz
python3 pipeline/phase3_dataset/finalize_dataset.py --resolution 64  # -> train/val/test.npz
```

- Ảnh PNG mật độ (từ lưới `xPhys` 50×50) resize về 64×64 bằng box-filter downsampling; 33/7.920 mẫu (0,4%) bị loại (topology suy biến, `volfrac_achieved` ngoài `[0,05, 0,95]`).
- **Chia train/val/test 70/15/15, phân tầng theo seed**; **tăng cường đối xứng** (chỉ train): xoay 90°/270° hoán đổi `ν₁₂↔ν₂₁`, xoay 180°/lật giữ nguyên. Train: 5.520 → 33.120 mẫu (×6).
- Target xuất ra: `v12`, `v21`, `volfrac_achieved`. `f1, f2` (roadmap gốc) chưa khả dụng — cần mở rộng `compute_homogenized_tensor()` để xuất độ cứng chuẩn hóa trước.

### 4. CNN Surrogate Model (Phase 4) — ✅ hoàn thành

```bash
python3 pipeline/phase4_surrogate/train.py
python3 pipeline/phase4_surrogate/evaluate.py
python3 pipeline/phase4_surrogate/export_for_phase5.py
```

Kiến trúc (baseline "Phương án A"): 4× khối `Conv(3x3) + BatchNorm + ReLU + MaxPool` → global average pool → nối với one-hot của seed → 2 lớp FC → 3 đầu ra (ν₁₂, ν₂₁, volfrac_achieved). Huấn luyện trên `outputs/phase3/train.npz`, xác thực trên `val.npz`.

**Hiệu năng trên test set** (`outputs/phase4/evaluation_report.json`, `test.npz` giữ riêng, không rò rỉ dữ liệu):

| Target | R² | MAE |
|---|---|---|
| ν₁₂ | 0,910 | 0,037 |
| ν₂₁ | 0,911 | 0,036 |
| volfrac_achieved | 0,982 | 0,007 |

MAE theo seed dao động 0,021–0,048, không seed nào kém nghiêm trọng. Nếu R² < 0,90 trên bất kỳ target nào, thử mở rộng `channels` trong `SurrogateCNN` trước khi đổi kiến trúc.

### 5. Conditional VAE (Phase 5) — ✅ thiết kế ngược được giải quyết qua best-of-N + chọn lọc bằng FE thực

```bash
python3 pipeline/phase5_cvae/train.py --gamma 20.0 --epochs 50
python3 pipeline/phase5_cvae/evaluate.py
python3 pipeline/phase5_cvae/sample.py    # chỉ để xem nhanh single-shot — xem cảnh báo bên dưới
```

`sample.py` sinh 1 ứng viên/lần gọi, không lọc FE — chỉ để xem qua, in cảnh báo mỗi lần chạy. `best_of_n_eval.py` là quy trình **chính thức** để lấy hình học đáng tin cậy (xem bên dưới).

Huấn luyện trên `train.npz` (33.120 mẫu), dùng surrogate Phase-4 **đóng băng** để tính loss property-consistency. Tổng loss: `recon + beta·kl + gamma·PROP_LOSS_SCALE·prop_loss` (`PROP_LOSS_SCALE=1000`); `gamma=20` là checkpoint mặc định (`outputs/phase5/cvae_gamma20.pt`).

> **⚠️ R² đo qua surrogate đóng băng không đáng tin cậy — kiểm chứng bằng FE thực (`verify_fe.py`) cho R² âm sâu ở mọi mức gamma.** Đây là hiện tượng **khai thác surrogate**: decoder học đánh lừa CNN đóng băng thay vì sinh hình học auxetic thực, và khoảng cách surrogate-vs-thực *nới rộng* khi gamma tăng. Hai biện pháp khắc phục ở giai đoạn huấn luyện (self-play adversarial retraining, ensemble surrogate) đã được thử và **không** khắc phục được vấn đề single-shot. Toàn bộ bảng số liệu gamma-sweep, kiểm chứng FE, và hai nỗ lực khắc phục nói trên xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#phase-5--cvae-gamma-sweep--kiểm-chứng-fe).

**Quy trình chính thức (`best_of_n_eval.py`):** sinh N ứng viên/điều kiện, để **FE thực** (không phải surrogate) chọn người thắng cuộc — cùng công thức pipeline Deep-DRAM đã công bố (Pahlavani et al. 2024, xem [Tài liệu Tham khảo](#tài-liệu-tham-khảo)). Đo trên checkpoint `gamma=20` (không cần huấn luyện lại), tập giữ riêng 24 điều kiện (19 auxetic):

| chiến lược | # lần gọi FE thực / điều kiện | R²(v12, FE thực) [95% CI bootstrap] | tỷ lệ trúng auxetic thực [95% CI Wilson] |
|---|---|---|---|
| single-shot (1 mẫu, không lọc) | 1 | âm sâu | 0,526 (10/19) |
| best-of-N, **oracle** (FE trên toàn bộ N=30, giữ ứng viên gần mục tiêu nhất) | 30 | **+0,5955** [0,003, 0,845] | **1,000** (19/19) [0,832, 1,000] |
| best-of-N, **thực dụng** (surrogate xếp hạng N=30, FE chỉ kiểm chứng top K=10) | 10 | **+0,4384** [−0,372, 0,748] | **1,000** (19/19) [0,832, 1,000] |

Biến thể thực dụng (lọc sơ bộ bằng surrogate, rẻ hơn 3×) giữ được gần như toàn bộ mức tăng R². Dữ liệu đầy đủ: `outputs/phase5/self_play/best_of_n_result.json` (oracle), `best_of_n_k10_result.json` (thực dụng).

> **⚠️ Cỡ mẫu nhỏ (n=19-24 điều kiện) — đọc khoảng tin cậy, đừng chỉ đọc điểm ước lượng.** CI 95% trên được tính bằng `pipeline/phase5_cvae/bootstrap_ci.py` (percentile bootstrap cho R², Wilson score interval cho tỷ lệ trúng — không chạy lại FE, chỉ resample lại `per_condition` đã lưu sẵn trong JSON). R² dao động trong khoảng rộng [0,00, 0,85] cho biến thể oracle — nghĩa là con số điểm +0,60 có thể lạc quan hơn thực tế đáng kể nếu lặp lại trên một tập điều kiện khác. Tỷ lệ trúng 19/19 cũng không đồng nghĩa "luôn luôn trúng": CI dưới chỉ ~83%. Trước khi trích dẫn các con số này như một kết luận mạnh, nên mở rộng tập giữ riêng vượt quá 24 điều kiện. Chạy lại: `python3 pipeline/phase5_cvae/bootstrap_ci.py outputs/phase5/self_play/best_of_n_result.json outputs/phase5/self_play/best_of_n_k10_result.json`.

```bash
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 30                      # oracle (FE trên toàn bộ N)
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 30 --k-fe-verify 10      # thực dụng (lọc sơ bộ bằng surrogate)
```

**Khả năng chế tạo — hệ số Poisson đúng ≠ khả thi để chế tạo.** `manufacturability.py` kiểm tra `check_connectivity()` + `check_periodicity()` (ghép lát không bước nhảy). Trên đầu ra gốc: đạt cả hai đồng thời chỉ **0–3,5%**, đồng đều toàn không gian thuộc tính (không phải vùng chết — `coverage_eval.py`). Giảm nhẹ: `--require-manufacturable` với N nhỏ (30-300) **gây hại** (R² +0,44→-1,96, đo trên 6 điều kiện, CI 95% [−13,74, 0,83] — cực rộng vì n quá nhỏ); N=**1500** khôi phục **tỷ lệ trúng 1,0, R²=+0,19** — nhưng con số này đo trên **chỉ 3 điều kiện** (chi phí FE ở N=1500 quá lớn để test rộng hơn trong phiên này), CI 95% bootstrap cho R² là **[−2,19, 0,90]** và CI Wilson cho tỷ lệ trúng là **[0,44, 1,00]** — tức là **chưa đủ bằng chứng để khẳng định chắc chắn**, chỉ nên đọc như một tín hiệu sơ bộ đáng theo đuổi, không phải kết luận:

```bash
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 1500 --k-fe-verify 8 --require-manufacturable
```

**Kết luận:** dùng best-of-N mặc định khi cần độ chính xác Poisson; thêm `--require-manufacturable` + N lớn khi cần đảm bảo khả thi chế tạo (đổi lấy R² thấp hơn). Chi tiết & biện pháp huấn luyện lại đã thử: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#khả-năng-chế-tạo--biện-pháp-huấn-luyện-lại-đã-thử).

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

Tham số còn lại (`--ft`, `--E0`, `--Emin`, `--nu`, `--move`, `--tol_change`, `--tol_obj`, `--window_size`, `--objective`, `--beta`, `--save_every`, `--scale_factor`, `--output_dir`, `--quiet`) — xem `python -m simp.main --help`.

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
Trường mật độ thang xám — đen (0) = rỗng, trắng (1) = rắn.

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
2. Độ ổn định mục tiêu — thay đổi tương đối < `tol_obj` trong `window_size` vòng lặp liên tiếp
3. Đạt `max_iter`

Được xử lý bởi `ConvergenceChecker` (`simp/core/convergence.py`), với `min_iter` để tránh dừng quá sớm. Trên toàn bộ 7.920 mẫu của multi-batch DOE, **tỷ lệ hội tụ FE đạt 100%**.

---

## Kiểm thử

```bash
pytest tests/ -v
```

Trạng thái hiện tại: **208/208 test pass** (`pytest tests/ -q`, ~4s).

| Module | Trạng thái |
|--------|--------|
| Phân tích tham số dòng lệnh CLI | ✅ |
| Xác thực SimpConfig | ✅ |
| Bộ kiểm tra hội tụ | ✅ |
| Smoke test lõi (FEM, vật liệu, filter, OC, solver, PBC) | ✅ |
| Nạp dataset & phân loại auxetic | ✅ |
| Định dạng CSV của logger | ✅ |
| `pipeline/phase4_surrogate/` (model, dataset, evaluate, export, train) | ✅ |
| `pipeline/phase5_cvae/` (model, dataset, losses, verify_fe, sample, adversarial_dataset, self_play, train, **best_of_n_eval**, **manufacturability**, **coverage_eval**, **bootstrap_ci**) | ✅ |

> Test dùng fixture `.npz` tổng hợp nhỏ (không phụ thuộc `outputs/phase3/*.npz` thực, bị gitignore) nên chạy nhanh (~3s) ở mọi nơi. Chưa có test tự động cho `seeds/*.py`, `pipeline/phase2_multi_batch/*`, `pipeline/phase3_dataset/`.
>
> **Lưu ý khi thêm test:** `phase4_surrogate/` và `phase5_cvae/` định nghĩa module con trùng tên (`dataset.py`, `model.py`...) qua import trần (`sys.path.insert` + `from dataset import X`) — import 2 module cùng tên từ *phase khác nhau* trong 1 tiến trình sẽ đè cache `sys.modules`. Fixture `_isolate_pipeline_bare_imports` (`tests/conftest.py`) reset cache này, nhưng chỉ hoạt động nếu import nằm **bên trong** hàm test (không phải top-level file) — luôn import trễ (lazy).

---

## Giới hạn Đã biết / Known Limitations

### Tiếng Việt

Mục này gộp lại toàn bộ khoảng trống/giới hạn đã biết của dự án ở một chỗ, cho mục đích đánh giá khoa học (thay vì rải rác trong README/EXPERIMENT_LOG). Không có mục nào dưới đây là mới — tất cả đã được ghi nhận ở nơi khác trong tài liệu; đây là bản tổng hợp.

1. **Các con số R²/tỷ lệ trúng của Phase 5 (best-of-N) đo trên cỡ mẫu rất nhỏ, khoảng tin cậy rộng.** Tập giữ riêng chỉ có 24 điều kiện (19 auxetic); biến thể `--require-manufacturable N=1500` chỉ đo trên **3 điều kiện** (chi phí FE ở N lớn quá tốn để mở rộng trong ngân sách đã thử). Bootstrap CI 95% (`pipeline/phase5_cvae/bootstrap_ci.py`, xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#phase-5--bootstrap-ci-cho-r2-và-tỷ-lệ-trúng-của-best-of-n)):
   - R² oracle: điểm ước lượng +0,5955, **CI [0,003, 0,845]** — cận dưới gần 0.
   - R² manufacturability N=1500: điểm ước lượng +0,1871, **CI [−2,191, 0,903]** — gần như không có ý nghĩa thống kê ở n=3.
   - Hit rate 100% (19/19, 3/3) không đồng nghĩa "luôn luôn trúng": CI Wilson dưới chỉ 83% (n=19) hoặc 44% (n=3).
   - **Hệ quả:** các con số này nên được trình bày kèm CI, không phải chỉ điểm ước lượng, khi đưa vào báo cáo/bài báo. Trước khi công bố như một kết luận chính, nên mở rộng tập giữ riêng.

2. **Khả năng chế tạo (manufacturability) rất thấp ở đầu ra gốc.** Chỉ 0–3,5% hình học sinh ra (không lọc) vừa đúng Poisson ratio vừa liên thông + ghép ô tuần hoàn được — đồng đều trên toàn không gian thuộc tính, không phải vùng chết cục bộ. Biện pháp giảm nhẹ (`--require-manufacturable` + N lớn) có tác dụng nhưng dựa trên cỡ mẫu rất nhỏ (mục 1) và đánh đổi R² thấp hơn đáng kể.

3. **Mô hình sinh single-shot (1 lần gọi `cvae.generate()`, không lọc FE) hoàn toàn không đáng tin cậy** — R² qua FE thực âm sâu ở mọi mức gamma đã thử (1 đến 300), và khoảng cách surrogate-vs-FE-thực *nới rộng* khi gamma tăng (khai thác surrogate). Hai biện pháp khắc phục ở giai đoạn huấn luyện (self-play adversarial retraining, ensemble surrogate 3-mô hình) đã thử và **không** khắc phục được vấn đề trong ngân sách thời gian đã thử — đây không phải bằng chứng "không thể khắc phục được", chỉ là "chưa khắc phục được với nỗ lực đã bỏ ra". `best_of_n_eval.py` (sinh N, chọn bằng FE thật) là biện pháp **inference-time** né vấn đề bằng cách đổi tiêu chí thành công, không phải một bản sửa cho decoder.

4. **Thành phần phạt `mu` trong hàm mục tiêu auxetic đang tắt (`mu=0.0`)** do có sai sót về mặt khái niệm chưa được thiết kế lại — toàn bộ 8 lô Phase 2 (7.920 mẫu) và các phase sau đều dùng cấu hình `mu=0.0`. Nếu `mu` được thiết kế lại và bật lên, kết quả downstream (Phase 2-5) sẽ cần chạy lại để phản ánh cấu hình mới.

5. **`f1, f2` (mục tiêu độ cứng chuẩn hóa theo roadmap gốc) chưa khả dụng** — `compute_homogenized_tensor()` chưa xuất `E₁₁/E₀, E₂₂/E₀`. Dataset/surrogate/cVAE hiện tại chỉ dùng `ν₁₂, ν₂₁, volfrac_achieved` làm target, không phải bộ target đầy đủ trong roadmap ban đầu.

6. **Thiếu test tự động** cho `pipeline/seeds/*.py`, `pipeline/phase2_multi_batch/*`, `pipeline/phase3_dataset/` — các module này chưa có coverage như `phase4_surrogate/`/`phase5_cvae/` (208/208 test hiện tại tập trung ở core SIMP + Phase 4/5).

7. **Một số tài liệu trực quan (HTML dashboard) từng bị lỗi thời** so với kết quả đã xác thực — cụ thể `html/inverse_auxetic_report.html` (sinh ngày 2026-07-16) có bảng xếp hạng seed trái ngược hoàn toàn với dữ liệu Phase 2 đã xác thực; đã gắn banner cảnh báo rõ ràng ở đầu trang và tại mục Kết luận trỏ về `README.md` + báo cáo Phase 3-5 hiện hành (`html/reports/ml_pipeline_phase3to5.html`). Đây là rủi ro mang tính hệ thống (tài liệu trực quan không tự động đồng bộ với code/dữ liệu) cần lưu ý khi thêm dashboard mới.

8. **Ngôn ngữ tài liệu chủ yếu là tiếng Việt** (README, EXPERIMENT_LOG, PROJECT_DOCUMENTATION). Người đọc quốc tế cần bản tóm tắt tiếng Anh riêng — xem bản dịch English bên dưới cho mục Limitations, nhưng các tài liệu chi tiết khác chưa có bản dịch đầy đủ.

### English

This section consolidates every known gap/limitation of the project in one place for scientific-review purposes (rather than scattered across README/EXPERIMENT_LOG). Nothing below is new information — all of it is documented elsewhere; this is a summary.

1. **Phase 5 (best-of-N) R²/hit-rate numbers are measured on very small samples, with wide confidence intervals.** The held-out set has only 24 conditions (19 auxetic); the `--require-manufacturable N=1500` variant is measured on just **3 conditions** (FE cost at large N was too high to expand further within the time budget tried). 95% bootstrap CIs (`pipeline/phase5_cvae/bootstrap_ci.py`, see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#phase-5--bootstrap-ci-cho-r2-và-tỷ-lệ-trúng-của-best-of-n)):
   - Oracle R²: point estimate +0.5955, **CI [0.003, 0.845]** — lower bound is nearly zero.
   - Manufacturability N=1500 R²: point estimate +0.1871, **CI [−2.191, 0.903]** — essentially uninformative at n=3.
   - 100% hit rate (19/19, 3/3) does not mean "always succeeds": the Wilson CI lower bound is only 83% (n=19) or 44% (n=3).
   - **Implication:** these numbers should be reported with CIs, not point estimates alone, in any paper/report. The held-out set should be expanded before treating these as a headline conclusion.

2. **Manufacturability of raw (unfiltered) output is very low.** Only 0–3.5% of generated geometries are simultaneously Poisson-accurate, connected, and periodic-tileable — uniformly low across the property space, not a localized dead zone. The mitigation (`--require-manufacturable` + large N) helps but rests on the very small samples in item 1 and trades off substantially lower R².

3. **Single-shot generation (one call to `cvae.generate()`, no FE filtering) is fundamentally unreliable** — real-FE R² is deeply negative at every gamma tested (1 to 300), and the surrogate-vs-real-FE gap *widens* as gamma increases (surrogate exploitation). Two training-time remedies (self-play adversarial retraining, 3-model ensemble surrogate) were tried and **did not** fix this within the time budget attempted — this is not evidence the problem is unfixable, only that it wasn't fixed with the effort spent. `best_of_n_eval.py` (generate N, select with real FE) is an **inference-time** workaround that changes the success criterion, not a fix to the decoder itself.

4. **The `mu` penalty term in the auxetic objective is disabled (`mu=0.0`)** due to an unresolved conceptual flaw — all 8 Phase 2 batches (7,920 samples) and every downstream phase use `mu=0.0`. If `mu` is redesigned and re-enabled, downstream results (Phase 2-5) would need to be regenerated to reflect the new configuration.

5. **`f1, f2` (normalized stiffness targets from the original roadmap) are not available** — `compute_homogenized_tensor()` does not yet export `E₁₁/E₀, E₂₂/E₀`. The current dataset/surrogate/cVAE only use `ν₁₂, ν₂₁, volfrac_achieved` as targets, not the full target set originally planned.

6. **No automated tests** for `pipeline/seeds/*.py`, `pipeline/phase2_multi_batch/*`, `pipeline/phase3_dataset/` — these modules lack the coverage that `phase4_surrogate/`/`phase5_cvae/` have (the current 208/208 passing tests concentrate on the core SIMP engine and Phase 4/5).

7. **Some visual documentation (HTML dashboards) had gone stale** relative to validated results — specifically `html/inverse_auxetic_report.html` (generated 2026-07-16) contained a seed ranking table that directly contradicted the validated Phase 2 data; it now carries a clear warning banner at the top and in its Conclusion section pointing to `README.md` and the current Phase 3-5 report (`html/reports/ml_pipeline_phase3to5.html`). This is a systemic risk (visual docs don't auto-sync with code/data) worth watching when adding new dashboards.

8. **Primary documentation is in Vietnamese** (README, EXPERIMENT_LOG, PROJECT_DOCUMENTATION). International readers need a separate English summary — this Limitations section is translated, but the other detailed documents are not fully translated.

---

## Tài liệu

- `PROJECT_DOCUMENTATION.md` — tài liệu toàn diện của dự án (tiếng Việt)
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — nhật ký thử nghiệm (bug, gamma-sweep, self-play, ensemble, khả năng chế tạo — kể cả biện pháp thất bại)
- `html/dashboards/workflow.html` — dashboard workflow, chi tiết từng phase con (2.1-2.9, 3.1-3.6, v.v.)
- `html/index.html` — dashboard/báo cáo bổ sung (lưu ý: một số trang chỉ phản ánh screening Phase 1, chưa tái sinh theo Phase 2-5)
- `INSTRUCTIONS.md` — hướng dẫn chạy gamma sweep Phase 5; `CHANGELOG.md` — lịch sử thay đổi theo phiên bản
- [`REVIEW_ALGORITHMS_VI.md`](REVIEW_ALGORITHMS_VI.md) — báo cáo review thuật toán độc lập (2026-06-06, trước khi đổi tên sang AuxForge)
- `outputs/{phase3,phase4,phase5}/` — báo cáo/kết quả từng phase (`evaluation_report.json`, `fe_verification_report.json`, `gamma_sweep_results/`, `self_play/`, v.v.)
- `notebooks/01-06_*.ipynb`, `gamma_sweep_analysis.ipynb` — notebook phân tích Phase 1-5 và tổng kết end-to-end

---

## Tài liệu Tham khảo

- Sigmund, O. (2001). *A 99 line topology optimization code written in Matlab.* Structural and Multidisciplinary Optimization, 21(2), 120–127.
- Andreassen, E., et al. (2011). *Efficient topology optimization in MATLAB using 88 lines of code.* Structural and Multidisciplinary Optimization, 43(1), 1–16.
- Xia, L., & Breitkopf, P. (2015). *Design of materials using topology optimization and energy‑based homogenization.* Archives of Computational Methods in Engineering, 22(2), 229–260.
- Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods, and Applications.* Springer.
- Pahlavani, H., et al. (2024). *Deep Learning for Size-Agnostic Inverse Design of Random-Network 3D Printed Mechanical Metamaterials.* Advanced Materials, 36(6). DOI: [10.1002/adma.202303481](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202303481) — pipeline "Deep-DRAM" cVAE + surrogate + chọn lọc bằng FE thực đã truyền cảm hứng cho biện pháp khắc phục best-of-N ở Phase 5 nêu trên.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017 — ý tưởng bất đồng-giữa-các-ensemble đằng sau `load_frozen_surrogate_ensemble`/`property_consistency_loss_ensemble`.

---

## Giấy phép

MIT — xem [`simp/__init__.py`](simp/__init__.py).

---

*Được duy trì bởi AuxForge Team.*

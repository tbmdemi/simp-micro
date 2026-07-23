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

> Chi tiết từng phase con (2.1-2.9, 3.1-3.6, v.v.): xem dashboard `docs/workflow.html`.
> **Khoảng trống đã biết:** phạt `mu` trong mục tiêu auxetic đang tắt (`mu=0.0`, đang chờ thiết kế lại); đồng nhất hóa chưa xuất độ cứng `E₁₁/E₀, E₂₂/E₀` nên `f1, f2` (roadmap gốc) chưa khả dụng; khả năng chế tạo — xem [Phase 5](#5-conditional-vae-phase-5---thiết-kế-ngược-được-giải-quyết-qua-best-of-n--chọn-lọc-bằng-fe-thực).

---

## Bắt đầu

### Yêu cầu & cài đặt

**Python** ≥ 3.10, **numpy/scipy/matplotlib** (core), **pandas/scikit-learn/Pillow** (`pipeline/phase3/`), **torch** (`pipeline/phase4_surrogate/`, `pipeline/phase5_cvae/` — không có trong `requirements.txt` gốc, cần cài thủ công):

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
# Phase 2: chạy/mở rộng adaptive multi-batch DOE (xem pipeline/multi_batch/)
python -m pipeline.multi_batch.main --phase1-summary outputs/pipeline/phase1

# Phase 3: xây dựng dataset sẵn sàng cho ML từ các lô đã hoàn thành
python3 pipeline/phase3/scan_dataset.py
python3 pipeline/phase3/build_npz.py --resolution 64
python3 pipeline/phase3/finalize_dataset.py --resolution 64
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
│   ├── phase1_screening_parallel.py, phase1_refine_params.py   # Phase 1: LHS screening
│   ├── multi_batch/          # Phase 2: adaptive DOE — sampling (Sobol/LHS), runner, adaptive (quyết định), coverage
│   ├── phase3/                # Phase 3: scan_dataset, build_npz, augment_symmetry, finalize_dataset
│   ├── phase4_surrogate/      # Phase 4: dataset, model (SurrogateCNN), train, evaluate, export_for_phase5
│   └── phase5_cvae/           # Phase 5: dataset, model, losses, train, evaluate, sample, verify_fe,
│                               #   adversarial_dataset, self_play, best_of_n_eval (inference chính thức),
│                               #   manufacturability, coverage_eval
│
├── analysis/                 # Phân tích độ nhạy (ANOVA, Sobol, regression), Pareto front, dataset QC
├── notebooks/, html/         # Jupyter notebook + dashboard/báo cáo (xem html/index.html)
├── docs/workflow.html        # Dashboard workflow toàn dự án (chi tiết từng phase con)
├── tests/                     # Bộ kiểm thử PyTest (194 test)
├── outputs/                   # Dữ liệu sinh ra (gitignored — .npz/.png lớn không commit)
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
python -m pipeline.phase1_screening_parallel --objective auxetic --seed hexagonal
python -m pipeline.phase1_screening_parallel --all   # quét toàn bộ, tất cả seed
python -m pipeline.phase1_analyst                    # tổng hợp kết quả -> _all_correlations.json, _all_summaries_parallel.json
```

Phân tích độ nhạy (tương quan Spearman) xác định **`volfrac` là tham số chi phối** (r ≈ 0,87–0,96); `move`, `rmin`, `void_size_frac` không có ý nghĩa thống kê. (Lịch sử debug lần chạy đầu — 0 mẫu auxetic — xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#phase-1--lhs-screening).)

### 2. Multi-Batch Adaptive DOE (Phase 2) — ✅ hoàn thành

Các lô tuần tự, mỗi lô được định hướng bởi phân tích độ phủ (KDE + phát hiện vùng thưa) trên kết quả tích lũy. `adaptive.py` quyết định **tinh chỉnh** (thu hẹp khoảng tham số + nhắm vào vùng thưa), **mở rộng** (thêm seed/mục tiêu), hoặc **dừng**.

```bash
python -m pipeline.multi_batch.main --phase1-summary outputs/pipeline/phase1
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
python3 pipeline/phase3/scan_dataset.py       # -> outputs/phase3/manifest.csv
python3 pipeline/phase3/build_npz.py --resolution 64   # -> dataset_64.npz
python3 pipeline/phase3/finalize_dataset.py --resolution 64  # -> train/val/test.npz
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

| chiến lược | # lần gọi FE thực / điều kiện | R²(v12, FE thực) | tỷ lệ trúng auxetic thực |
|---|---|---|---|
| single-shot (1 mẫu, không lọc) | 1 | âm sâu | 0,526 (10/19) |
| best-of-N, **oracle** (FE trên toàn bộ N=30, giữ ứng viên gần mục tiêu nhất) | 30 | **+0,5955** | **1,000** (19/19) |
| best-of-N, **thực dụng** (surrogate xếp hạng N=30, FE chỉ kiểm chứng top K=10) | 10 | **+0,4384** | **1,000** (19/19) |

Biến thể thực dụng (lọc sơ bộ bằng surrogate, rẻ hơn 3×) giữ được gần như toàn bộ mức tăng R². Dữ liệu đầy đủ: `outputs/phase5/self_play/best_of_n_result.json` (oracle), `best_of_n_k10_result.json` (thực dụng).

```bash
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 30                      # oracle (FE trên toàn bộ N)
python3 pipeline/phase5_cvae/best_of_n_eval.py --n-samples 30 --k-fe-verify 10      # thực dụng (lọc sơ bộ bằng surrogate)
```

**Khả năng chế tạo — hệ số Poisson đúng ≠ khả thi để chế tạo.** `manufacturability.py` kiểm tra `check_connectivity()` + `check_periodicity()` (ghép lát không bước nhảy). Trên đầu ra gốc: đạt cả hai đồng thời chỉ **0–3,5%**, đồng đều toàn không gian thuộc tính (không phải vùng chết — `coverage_eval.py`). Giảm nhẹ: `--require-manufacturable` với N nhỏ (30-300) **gây hại** (R² +0,44→-1,96); N=**1500** khôi phục **tỷ lệ trúng 1,0, R²=+0,19**:

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

Trạng thái hiện tại: **194/194 test pass** (`pytest tests/ -q`, ~3s).

| Module | Trạng thái |
|--------|--------|
| Phân tích tham số dòng lệnh CLI | ✅ |
| Xác thực SimpConfig | ✅ |
| Bộ kiểm tra hội tụ | ✅ |
| Smoke test lõi (FEM, vật liệu, filter, OC, solver, PBC) | ✅ |
| Nạp dataset & phân loại auxetic | ✅ |
| Định dạng CSV của logger | ✅ |
| `pipeline/phase4_surrogate/` (model, dataset, evaluate, export, train) | ✅ |
| `pipeline/phase5_cvae/` (model, dataset, losses, verify_fe, sample, adversarial_dataset, self_play, train, **best_of_n_eval**, **manufacturability**, **coverage_eval**) | ✅ |

> Test dùng fixture `.npz` tổng hợp nhỏ (không phụ thuộc `outputs/phase3/*.npz` thực, bị gitignore) nên chạy nhanh (~3s) ở mọi nơi. Chưa có test tự động cho `seeds/*.py`, `pipeline/multi_batch/*`, `pipeline/phase3/`.
>
> **Lưu ý khi thêm test:** `phase4_surrogate/` và `phase5_cvae/` định nghĩa module con trùng tên (`dataset.py`, `model.py`...) qua import trần (`sys.path.insert` + `from dataset import X`) — import 2 module cùng tên từ *phase khác nhau* trong 1 tiến trình sẽ đè cache `sys.modules`. Fixture `_isolate_pipeline_bare_imports` (`tests/conftest.py`) reset cache này, nhưng chỉ hoạt động nếu import nằm **bên trong** hàm test (không phải top-level file) — luôn import trễ (lazy).

---

## Tài liệu

- `PROJECT_DOCUMENTATION.md` — tài liệu toàn diện của dự án (tiếng Việt)
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — nhật ký thử nghiệm (bug, gamma-sweep, self-play, ensemble, khả năng chế tạo — kể cả biện pháp thất bại)
- `docs/workflow.html` — dashboard workflow, chi tiết từng phase con (2.1-2.9, 3.1-3.6, v.v.)
- `html/index.html` — dashboard/báo cáo bổ sung (lưu ý: một số trang chỉ phản ánh screening Phase 1, chưa tái sinh theo Phase 2-5)
- `INSTRUCTIONS.md` — hướng dẫn chạy gamma sweep Phase 5; `CHANGELOG.md` — lịch sử thay đổi theo phiên bản
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

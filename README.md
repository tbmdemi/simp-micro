# AuxForge

**Tối ưu hóa Topology cho Thiết kế Vi cấu trúc Vật liệu Auxetic (Auxetic Metamaterial)**

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)
[![Version](https://img.shields.io/badge/version-1.4.0-blueviolet)](simp/__init__.py)

---

## Tổng quan

**AuxForge** triển khai phương pháp **Solid Isotropic Material with Penalization (SIMP)** cho bài toán tối ưu hóa topology của các vi cấu trúc ô đơn vị tuần hoàn (periodic unit-cell), nhắm tới **hành vi auxetic** (hệ số Poisson âm). Mục tiêu cuối cùng của dự án là **thiết kế ngược (inverse design)**: cho trước một hệ số Poisson mục tiêu, sinh ra một hình học vi cấu trúc đạt được giá trị đó, sử dụng mô hình sinh có điều kiện (cVAE) được huấn luyện trên bộ dữ liệu tăng cường bằng surrogate, do chính engine SIMP này tạo ra.

```
Seed Generation → FE Analysis → Homogenization → Objective & Sensitivity →
Density/Sensitivity Filtering → OC Update → Convergence Check → Repeat
```

Codebase này là bản triển khai lại bằng Python của các đoạn mã SIMP MATLAB kinh điển (88 dòng / 99 dòng), được mở rộng thêm điều kiện biên tuần hoàn (periodic boundary conditions), phép đồng nhất hóa dựa trên năng lượng (energy-based homogenization), và một pipeline DOE (Design of Experiments) đa lô thích ứng (adaptive multi-batch) để sinh dữ liệu quy mô lớn.

## Mục lục

- [AuxForge](#auxforge)
  - [Tổng quan](#tổng-quan)
  - [Mục lục](#mục-lục)
  - [Trạng thái Dự án](#trạng-thái-dự-án)
    - [Phạm vi Claim Khoa học (đọc trước khi trích dẫn)](#phạm-vi-claim-khoa-học-đọc-trước-khi-trích-dẫn)
  - [Bắt đầu](#bắt-đầu)
    - [Yêu cầu \& cài đặt](#yêu-cầu--cài-đặt)
    - [Chạy nhanh](#chạy-nhanh)
    - [Pipeline dataset đầy đủ (Phase 2 → Phase 3)](#pipeline-dataset-đầy-đủ-phase-2--phase-3)
  - [Cấu trúc Package](#cấu-trúc-package)
  - [Các Seed Có sẵn](#các-seed-có-sẵn)
  - [Hàm Mục tiêu (Auxetic)](#hàm-mục-tiêu-auxetic)
  - [Pipeline chi tiết](#pipeline-chi-tiết)
  - [Tham chiếu CLI](#tham-chiếu-cli)
  - [Sử dụng Lập trình](#sử-dụng-lập-trình)
  - [File Đầu ra](#file-đầu-ra)
    - [Ảnh PNG (`iteration_XXXXX.png`)](#ảnh-png-iteration_xxxxxpng)
    - [Dữ liệu CSV (`iteration_data.csv`)](#dữ-liệu-csv-iteration_datacsv)
    - [Metadata (`metadata.json`)](#metadata-metadatajson)
  - [Tiêu chí Hội tụ](#tiêu-chí-hội-tụ)
  - [Kiểm thử](#kiểm-thử)
  - [Giới hạn Đã biết / Known Limitations](#giới-hạn-đã-biết--known-limitations)
  - [Tài liệu](#tài-liệu)
  - [Tài liệu Tham khảo](#tài-liệu-tham-khảo)
  - [Giấy phép](#giấy-phép)

---

## Trạng thái Dự án

Lộ trình thiết kế ngược gồm 8 giai đoạn (phase). Phase 1-4 đã hoàn thành và được xác thực trên dữ liệu thực; Phase 5 (cVAE) đã sửa tận gốc bằng differentiable real-physics, đạt R²(FE,n=789)≈0,995-0,999/hit-rate ≥98%. Giai đoạn A đã nối ν₀ (vật liệu nền) xuyên suốt Phase 1-5 và đo được lợi ích thực; Phase 6-8 đã hoàn thành hậu xử lý, kiểm định FE độc lập, active-learning và thư viện thiết kế, còn biến dạng lớn/STL là tùy chọn chưa làm. Xem [báo cáo trạng thái tổng hợp](outputs/phase5/reports/final_status_report_2026-07-31.md) và [docs/PIPELINE.md](docs/PIPELINE.md).

| Phase | Thành phần | Trạng thái | Ghi chú |
|-------|-----------|--------|-------|
| 0 | Core SIMP Engine | ✅ Ổn định | 11 loại seed, mục tiêu auxetic, PBC, đồng nhất hóa dựa trên năng lượng |
| 1 | LHS Screening | ✅ Hoàn thành | Phân tích độ nhạy: `volfrac` là tham số chi phối (r ≈ 0,87–0,96). Lịch sử debug lần chạy đầu (0 mẫu auxetic): xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| 2 | Multi-Batch Adaptive DOE | ✅ Hoàn thành + cải tiến manufacturability + rebuild dQ | **8/8 lô gốc + batch 11 rebuild**, 7.920 mẫu, **91,9% auxetic** (đã rebuild 3 seed bị lỗi dQ, tăng từ 82,1%). Pipeline thích ứng tự dừng sau 2 lô liên tiếp không cải thiện mục tiêu. **2026-07-24**: (a) phân tích ngược xác nhận SEED chi phối manufacturability - thêm phân bổ mẫu theo seed; (b) phát hiện + sửa lỗi `dQ` + rebuild đầy đủ 2.160 mẫu (3 seed bất đối xứng). Xem [Phase 2](#2-multi-batch-adaptive-doe-phase-2----hoàn-thành) bên dưới và [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md#bảng-tổng-hợp-lỗi-đã-sửa) |
| 3 | Dataset Build (trường mật độ + target) | ✅ Hoàn thành, **đã rebuild lần 2 (v2, 2026-07-25)** | **57.216 mẫu train** (sau aug) / 2.044 val / 2.044 test - dataset production hiện tại. Xem mục "Rebuild v2" ngay dưới bảng này |
| 4 | CNN Surrogate Model | ✅ Hoàn thành, đã retrain trên dataset v2 | Dự đoán (ν₁₂, ν₂₁, volfrac) từ trường mật độ. R² trên test set v2 (`surrogate_v2.pt`, 2026-07-25): ν₁₂ = **0,974**, ν₂₁ = **0,964**, volfrac = 0,983. Xem [Phase 4](#4-cnn-surrogate-model-phase-4----hoàn-thành) bên dưới |
| 5 | Conditional VAE | ✅✅ Đã sửa surrogate-exploitation bằng differentiable-physics; đã retrain và mở rộng condition | `cvae_v2_finetuned.pt` là checkpoint production khuyến nghị: R²(FE,n=300)=**0,9953** (oracle), hit rate single-shot=**99,7%**, frac manufacturable=0,247. `cvae_realphysics.pt` trên dataset v1 đạt R²=0,9984/hit-rate 98,3%/manuf=0,280 ở phép đo tương ứng. Condition optional gồm `volfrac`/`void_size_frac` (`--extended-condition`) và ν₀ (`--include-nu0`); A/B trên cùng dataset cho ν₀ tăng R² 0,9776→0,9843 và manufacturability 0,312→0,365. Composite score dùng trọng số accuracy/manufacturability/aesthetic `0,6/0,3/0,1`. Xem [docs/PIPELINE.md § 5](docs/PIPELINE.md#5-conditional-vae-phase-5---đã-sửa-tận-gốc-bằng-differentiable-physics-2026-07-24) và [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). **2026-09-25 (nhánh `substrate-material`, [docs/plan.md](docs/plan.md) v3):** tinh chỉnh latent bằng gradient FE khả vi *nhận thức nhị phân hóa* (`--projection-betas`) đưa R²(FE,v12) single-shot 0,874→**0,998** và best-of-30 0,986→0,997 trên `cvae_v2_finetuned.pt`; ablation có kiểm soát KAN vs Linear (2 seed, FE thật) - **KAN không thắng**, giữ Linear; WIRE decoder đã đóng (không đạt tiêu chí dừng). Chi tiết: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) mục 2026-09-25. |
| 6 | Hậu xử lý & kiểm định FEA | ✅ Hoàn thành | Nhị phân hoá + connectivity/min-feature/periodicity (`manufacturability.py`) + lọc/verify bằng FE thật (`best_of_n_eval.py`) - đã có sẵn trong pipeline Phase 5 |
| 7 | Active-learning loop | ✅ Kết luận (2026-07-31) | Implement + chạy production (`pipeline/phase5_cvae/active_learning.py`) - **không cải thiện** checkpoint đã tối ưu (`cvae_realphysics.pt` gần mức trần, mean_abs_error tệ đi 0,0246→0,0686 sau 1 vòng) - giữ nguyên checkpoint production. Chi tiết: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) mục 2026-07-31 |
| 8 | Xác thực cuối & đóng gói | 🟨 Hầu hết hoàn thành | **8.1** đối chiếu FE độc lập: ✅ `scikit-fem` khớp tới 1e-9 (R²=1,000000, n=24). **8.3** thư viện thiết kế: ✅ 24 thiết kế, hit-rate 100%, 87,5% manufacturable. **8.4** báo cáo: ✅. **8.2** biến dạng lớn và **8.5** STL: ⬜ tùy chọn, chưa làm. |

**Cập nhật 2026-09-25 (plan v3, nhánh `substrate-material`):** so sánh "KAN vượt Linear ~2×"
và Exp 3 (2026-09-11) trước đây dựa trên `property_accuracy()` (chấm qua surrogate - xếp hạng ngược
FE thật) và một phần trên bug loader `load_cvae` (ép đối xứng lúc đánh giá, `docs/LIMITATIONS.md`
mục 28) - **không còn dùng làm kết luận**. Ablation lại có kiểm soát bằng FE thật: KAN không thắng
Linear. Xem [docs/plan.md](docs/plan.md) mục P1.2.

**Cập nhật 2026-10-06 (plan v3 P1.6):** so với **SIMP chạy từ đầu** (inverse homogenization, cùng
đường verify), cVAE + refine **ngang** SIMP hội tụ về độ chính xác với ít FE hơn ~8×; trên đủ
100 mục tiêu cặp (ν₁₂, ν₂₁) hòa (cVAE sinh ô đứt ở mục tiêu dị hướng cực đoan r ≥ 10); SIMP thắng về khả năng chế tạo
(77% vs ~25%) và OOD đổi dấu. Refine tái lập trên tập mới IN100-B. Độ chính xác báo cáo là so với
mô hình FE 50×50 (sai số rời rạc hóa so với 200² ~0,015). Retrieval verified chỉ đạt R² 0,94-0,98
(con số 1,000 cũ đo trên nhãn). Xem [docs/plan.md](docs/plan.md) mục P1.6 và `docs/LIMITATIONS.md` #32-35.

**Cập nhật 2026-10-07 (plan v3 P1.7-P1.9, tác giả chốt C1-C5):** lai cVAE → SIMP **không đạt** tiêu
chí đăng ký trước (chế tạo 0,33 < 0,70); kiểm định phân tầng đăng ký trước trên tập mới IN100-C:
với r < 10 sai số cặp cVAE thấp hơn SIMP 27% [8; 42] (đạt), ν₁₂ không đạt → **không claim ν₁₂ hơn
SIMP**. Bỏ `force_periodic` (C5) và chạy lại mọi số (P1.9): độ chính xác giữ nguyên/tốt hơn, chế tạo
0,24 → 0,32, OOD −2,0 sai số trung vị 12,5% → 1,2% (4/10 lần vẫn sai 17-31%); SIMP vẫn thắng chế tạo
và OOD đổi dấu. Bài #1: Khung A, tên "Optimizing What Is Verified…", nộp SMO. Xem `docs/plan.md`
mục 4 và `docs/LIMITATIONS.md` #36-38.

> Chi tiết từng phase con: xem [docs/PIPELINE.md](docs/PIPELINE.md), [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) và [docs/ARCHITECT.md](docs/ARCHITECT.md).
> 
> **Khoảng trống đã biết** (tóm tắt - xem đầy đủ tại [docs/LIMITATIONS.md](docs/LIMITATIONS.md)): `mu` vẫn tắt; `f1/f2` mới được backfill tới Phase 4, chưa nối vào cVAE; toàn bộ pipeline vẫn là FEM tuyến tính. Ở 8D in-distribution, cVAE chưa thắng retrieval về accuracy, manufacturability hoặc aesthetic; bằng chứng lợi thế generative so với retrieval rõ nhất ở sign-flip OOD (ν₁₂ dương 6/6 mục tiêu, MAE 0,19 so với 0,40), nhưng ở OOD SIMP chạy từ đầu vẫn tốt hơn cVAE (LIMITATIONS #32); không ngoại suy được biên độ ra ngoài dải train. Refine nhận thức nhị phân hóa nâng R²(FE) trong phân phối lên ~0,998 (LIMITATIONS #29). Không dùng hit-rate một mình làm bằng chứng vì base rate auxetic của dataset cao.

### Phạm vi Claim Khoa học (đọc trước khi trích dẫn)

Đã tách sang **[docs/LIMITATIONS.md § Phạm vi Claim Khoa học](docs/LIMITATIONS.md#phạm-vi-claim-khoa-học-đọc-trước-khi-trích-dẫn)** liệt kê claim được/chưa được ủng hộ bằng chứng, có trích số Giới hạn # cụ thể cho từng dòng.

---

## Bắt đầu

### Yêu cầu & cài đặt

**Python** ≥ 3.10. Toàn bộ dependency runtime, gồm `nlopt`, `torch`, `scikit-learn`, `pandas`, `Pillow` và `scikit-fem`, được liệt kê trong `requirements.txt`:

```bash
pip install -r requirements.txt
```

Hoặc cài package cùng các nhóm tùy chọn:

```bash
pip install -e ".[all]"
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
├── notebooks/                # Jupyter notebook phân tích và xác thực
├── tests/                     # Bộ kiểm thử PyTest (số lượng phụ thuộc phiên bản/dependency)
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

1. **LHS Screening (Phase 1)** - xác định `volfrac` là tham số chi phối (r≈0,87-0,96); ν₀ đã được thêm làm biến vật liệu nền.
2. **Multi-Batch Adaptive DOE (Phase 2)** - 8 lô, 7.920 mẫu, tỷ lệ hội tụ FE 100%, auxetic rate 91,9% sau rebuild.
3. **Dataset Build (Phase 3)** - dataset v2 hiện hành: train=57.216 / val=2.044 / test=2.044.
4. **CNN Surrogate (Phase 4)** - `surrogate_v2.pt`: R²(v12)=0,974, R²(v21)=0,964, R²(volfrac)=0,983.
5. **Conditional VAE (Phase 5)** - `cvae_v2_finetuned.pt` (khuyến nghị production): R²(FE,n=300)=0,9953 (oracle), hit rate single-shot=99,7%. Condition optional gồm `volfrac`/`void_size_frac` và ν₀; `best_of_n_eval.py` dùng chấm điểm tổng hợp accuracy/manufacturability/aesthetic. **2026-08-23 (nhánh `substrate-material`):** bộ hồi quy KAN-hóa + WIRE decoder (`decoder_type="conv"|"wire"`), ablation 2026-09-25 cho thấy KAN không thắng Linear khi đo bằng FE thật, production giữ Linear; tinh chỉnh latent nhận thức nhị phân hóa đạt R²(FE) ~0,998 - xem [docs/plan.md](docs/plan.md).
6. **Xác thực và so sánh (Phase 6-8)** - FE độc lập khớp tới 1e-9; thư viện 24 thiết kế đạt 87,5% manufacturable. Retrieval vẫn là baseline mạnh trong phân phối; lợi thế cVAE được ủng hộ rõ nhất trên sign-flip OOD.

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

Trạng thái kiểm thử: chạy `pytest tests/ -q` sau khi cài đủ dependency ML/analysis. Không ghi cố định số test pass ở đây vì bộ test thay đổi theo từng lần bổ sung tính năng.

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

Danh sách đầy đủ 21 mục (song ngữ Việt/English) đã được tách sang **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** cùng với mục "Phạm vi Claim Khoa học". Các điểm đáng chú ý nhất:

- R²/hit-rate của Phase 5 chỉ đáng tin ở cỡ mẫu lớn (n≈300-789); ở n=24 CI rất rộng.
- Manufacturability của đầu ra gốc (không lọc) rất thấp; cần `force_periodic()`/`--require-manufacturable`.
- Phạt `mu` trong mục tiêu auxetic đang tắt (`mu=0.0`).
- `f1, f2` (Pha B) chưa nối làm condition cho cVAE - xem `docs/archive/PROJECT_PLAN.md` Nhóm 2.
- Test tự động chưa phủ hết đường I/O nặng (screening loop, seeds, visualize và FE call thật trong `multi_batch/runner.py::evaluate_single`).
- Kết quả xác thực composite score chỉ ủng hộ một phần: Spearman trung bình so với Pareto front = 0,693 (trung vị 0,714), chưa đạt ngưỡng trung bình 0,7 đặt trước.
- Ở 8D in-distribution, retrieval không bị suy yếu đáng kể ngay cả với 500 mẫu tra cứu (R²=0,984); không dùng giả thuyết curse-of-dimensionality làm claim chính.
- Toàn bộ pipeline dùng FEM tuyến tính (giả định biến dạng nhỏ) - xem mục 14.
- `hit_rate` là metric yếu do base rate ~92% auxetic của dataset; chưa có bằng chứng cVAE vượt trội baseline nearest-neighbor trong-phân-phối (mục 17-18, phát hiện 2026-08-02).

---

## Tài liệu
- [`docs/PIPELINE.md`](docs/PIPELINE.md) - chi tiết từng bước pipeline (Phase 1-5.1): lệnh chạy, số liệu R²/hit-rate, lịch sử phát hiện+sửa bug
- [`docs/CLI_GUIDE.md`](docs/CLI_GUIDE.md) - tổng hợp mọi lệnh dòng lệnh (test suite, `simp`, `simp-analysis`, từng phase 1-5, sự cố thường gặp)
- [`docs/archive/PROJECT_PLAN.md`](docs/archive/PROJECT_PLAN.md) - roadmap ưu tiên theo effort/phụ thuộc thật (Giai đoạn A vật liệu nền, B xếp hạng đa mục tiêu, F nhiệt/CTE, v.v.)
- [`docs/ARCHITECT.md`](docs/ARCHITECT.md) - kiến trúc hệ thống: bản đồ module, luồng dữ liệu 8-phase, điểm mở rộng cho roadmap, kiến trúc đích giả định
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) - phạm vi claim khoa học + 38 mục giới hạn đã biết (song ngữ)
- [`docs/PHYSICS_AND_ML.md`](docs/PHYSICS_AND_ML.md) - bản chất toán học/cơ học/vật lý của SIMP + đồng nhất hóa, và vai trò cụ thể của ML/DL (surrogate, cVAE, differentiable-physics) trong pipeline
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) - nhật ký các phát hiện/sửa lỗi và đột phá chính thay đổi kết quả dự án
- `outputs/figures/` - năm figure đã dựng cho bài báo (kiến trúc, data hygiene, OOD, Pareto và uốn tấm)
- `docs/COMPOSITE_AUXETIC_PLAN.md` - **chưa tồn tại** (chỉ mới ở giai đoạn khảo sát ý tưởng, chưa viết thành file; xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) mục 2026-08-22 cho nội dung khảo sát thật)
- [`docs/LITERATURE_REVIEW_GAP_ANALYSIS.md`](docs/LITERATURE_REVIEW_GAP_ANALYSIS.md) - phân tích khoảng trống nghiên cứu
- `CHANGELOG.md` - lịch sử thay đổi theo phiên bản
- Pilot damping/continuation có kiểm soát (N=400/config, Wilson CI) cho thấy `use_sqrt` (damping η=0.5) + `penal_init=2.0` gần gấp đôi yield `reentrant_bowtie` (46%→76-80%) - **kết quả tốt, đã kiểm chứng thống kê, nhưng CHƯA được áp dụng vào cấu hình production**; xem [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).
- `outputs/{phase3,phase4,phase5}/` - báo cáo/kết quả từng phase (`evaluation_report.json`, `fe_verification_report.json`, `self_play/`, v.v.)
- `notebooks/01-06_*.ipynb`, `gamma_sweep_analysis.ipynb` - notebook phân tích Phase 1-5 và tổng kết end-to-end
- `notebooks/09_cheap_physical_properties.ipynb` - tính chất suy từ Q (E/G/B, proxy ấn lõm, tốc độ sóng quasi-static) trên dataset A4: kiểm tra cận vật lý, đo tính chất nào mang thông tin mới, biên Pareto ν ↔ độ cứng riêng

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
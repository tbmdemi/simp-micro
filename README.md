<!--
  Title:    SIMP Analyst — SIMP Topology Optimization for Periodic Material Microstructure Design
  Type:     Project README
  Language: Tiếng Việt (with English technical terms)
-->

# SIMP Analyst 🧊✨

**Tối ưu hóa hình dạng (topology optimization) cho thiết kế micro-cấu trúc vật liệu tuần hoàn với hệ số Poisson âm (auxetic).**

> **S**olid **I**sotropic **M**aterial with **P**enalization — implementation Python thuần của thuật toán SIMP,
> kết hợp phân tích phần tử hữu hạn (FEA) + đồng nhất hóa (homogenization) + tối ưu hóa hình dạng,
> nhắm đến mục tiêu thiết kế các ô cơ sở (unit cell) có tính chất cơ học đặc biệt, đặc biệt là **hành vi auxetic** (ν < 0).

---

## Mục lục

- [Tổng quan dự án](#tổng-quan-dự-án)
- [Tính năng chính](#tính-năng-chính)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
  - [Chạy tối ưu hóa SIMP](#chạy-tối-ưu-hóa-simp)
  - [Screening Pipeline (Phase 1)](#screening-pipeline-phase-1)
  - [Phân tích kết quả (Analysis)](#phân-tích-kết-quả-analysis)
  - [Makefile](#makefile)
- [Kiến trúc phần mềm](#kiến-trúc-phần-mềm)
  - [SIMP Core (`simp/`)](#simp-core)
  - [Pipeline (`pipeline/`)](#pipeline)
  - [Analysis (`analysis/`)](#analysis)
- [Seeds — Mẫu khởi tạo](#seeds--mẫu-khởi-tạo)
- [Objective Functions — Hàm mục tiêu](#objective-functions--hàm-mục-tiêu)
- [Homogenization & Periodic Boundary Conditions](#homogenization--periodic-boundary-conditions)
- [Convergence — Tiêu chí hội tụ](#convergence--tiêu-chí-hội-tụ)
- [Kết quả đầu ra](#kết-quả-đầu-ra)
- [HTML Reports & Dashboards](#html-reports--dashboards)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)
- [License](#license)

---

## Tổng quan dự án

**Input_SIMP_Analyst** (gọi tắt là **SIMPAnalyst**) là một dự án nghiên cứu và kỹ thuật nhằm **thiết kế ngược** micro-cấu trúc vật liệu
thông qua tối ưu hóa hình dạng. Mục tiêu là tìm ra sự phân bố vật liệu trong một ô cơ sở (unit cell) tuần hoàn sao cho
vật liệu tương đương (homogenized material) có các tính chất đàn hồi mong muốn — đặc biệt là **hệ số Poisson âm (auxetic)**.

Dự án được xây dựng bằng **Python 3.10+**, kế thừa tinh thần của mã MATLAB 99-dòng kinh điển (Sigmund, 2001)
và mở rộng với:
- Điều kiện biên tuần hoàn (PBC) cho ô cơ sở
- Đồng nhất hóa dựa trên năng lượng (energy-based homogenization) — Xia & Breitkopf (2015)
- 3 hàm mục tiêu khác nhau cho các chiến lược tối ưu khác nhau
- 10 mẫu seed khởi tạo khác nhau để khám phá không gian thiết kế
- Pipeline screening tự động với LHS (Latin Hypercube Sampling) và phân tích tương quan Spearman
- Hệ thống báo cáo HTML tự chứa (self-contained), hỗ trợ biểu đồ Chart.js

---

## Tính năng chính

| Tính năng | Mô tả |
|-----------|-------|
| **SIMP Loop** | Vòng lặp tối ưu hoàn chỉnh: FE → homogenization → objective → filter → OC update → convergence check |
| **Periodic BC** | Áp điều kiện biên tuần hoàn (null-space projection) cho ô cơ sở, đảm bảo tính tuần hoàn của trường chuyển vị |
| **Đồng nhất hóa (Homogenization)** | Tính tensor độ cứng tương đương `Q` (3×3) và độ nhạy `dQ/dx` bằng phương pháp năng lượng |
| **3 Objective Functions** | `auxetic` (ν₁₂ = −Q₁₂/Q₂₂), `first` (Q₁₂ − β·(Q₁₁+Q₂₂)), `second` (Q₁₂ + penalty) |
| **10 Seed Patterns** | Đa dạng mẫu khởi tạo: void tròn, vuông, lục giác, hình chữ thập, grid, v.v. |
| **LHS Screening** | Sinh mẫu Latin Hypercube, chạy batch, phân tích Spearman → xác định top-3 tham số ảnh hưởng nhất |
| **Image Metrics** | Phân tích chất lượng ảnh kết quả: binary rate, edge density, noise ratio, symmetry |
| **HTML Reports** | Báo cáo tự chứa (single-file) với biểu đồ Chart.js, dashboard, bảng phân loại |
| **CLI & Library** | Vừa dùng được qua dòng lệnh vừa import như một Python library |

---

## Cấu trúc thư mục

```
Input_SIMP_Analyst/
├── simp/                          # ★ Core SIMP optimization package
│   ├── __init__.py                # Package metadata (version 1.1.0)
│   ├── main.py                    # CLI entry point
│   ├── run.py                     # Entry — run with default params
│   ├── runner.py                  # Main optimization loop orchestrator
│   ├── config.py                  # SimpConfig dataclass với validation
│   ├── core/                      # Các thuật toán SIMP core
│   │   ├── fem.py                 # FEM mesh: node numbering, DOF mapping, sparse index vectors
│   │   ├── filter.py              # Cone-shaped density filter (chống checkerboard)
│   │   ├── pbc.py                 # Periodic Boundary Conditions (null-space projection)
│   │   ├── solver.py              # Sparse FE solver with PBC (direct LU + CG fallback)
│   │   ├── oc.py                  # Optimality Criteria update (bisection on Lagrange multiplier)
│   │   └── convergence.py         # Convergence detection (design change + objective stability)
│   ├── materials/
│   │   └── isotropic.py           # 4-node quad element stiffness matrix (plane stress)
│   ├── objectives/
│   │   ├── auxetic.py             # Auxetic: c = ν₁₂ = −Q₁₂/Q₂₂
│   │   ├── first_obj.py           # Type 1: c = Q₁₂ − β^loop · (Q₁₁ + Q₂₂)
│   │   └── second_obj.py          # Type 2: c = Q₁₂ + penalty for low axial stiffness
│   ├── homogenization/
│   │   └── compute.py             # Energy-based homogenization: Q + dQ
│   ├── seeds/                     # 10 initial void pattern generators
│   │   ├── circle.py
│   │   ├── square.py
│   │   ├── hourglass.py
│   │   ├── four_circle.py
│   │   ├── hexagonal.py
│   │   ├── nine_circle.py
│   │   ├── cross_rectangular.py
│   │   ├── grid_circular_voids.py
│   │   ├── small_square_cross.py
│   │   └── circle_half_quarter.py
│   └── io/
│       ├── logger.py              # CSV logging per iteration
│       └── visualizer.py          # Density field PNG export
│
├── pipeline/                      # Screening pipeline
│   ├── __init__.py
│   ├── params.py                  # Parameter space definitions (PARAM_SPACE, FIXED_PARAMS)
│   └── phase1_screening.py        # Phase 1: LHS screening, Spearman correlation, batch runner
│
├── analysis/                      # Post-processing analysis
│   ├── __init__.py
│   ├── cli.py                     # CLI: analysis entry point
│   ├── dataset.py                 # Xử lý dataset: đọc CSV/json → DataFrame
│   ├── image.py                   # Image metrics: binary rate, edge density, noise, symmetry
│   └── report.py                  # Generate self-contained HTML report
│
├── html/                          # Pre-built HTML resources
│   ├── index.html                 # Main report
│   ├── dashboards/
│   │   └── phase1_screening_dashboard.html
│   ├── guides/
│   │   ├── optimization_pipeline.html
│   │   └── simp_guide_and_roadmap.html
│   └── reports/
│       └── auxetic_report.html
│
├── outputs/                       # Kết quả sinh ra từ quá trình chạy
│   ├── report_simp_analysis.html  # Self-contained analysis report
│   ├── figures/                   # Biểu đồ phân tích (.png)
│   └── pipeline/phase1/           # Kết quả Phase 1 screening
│       ├── circle/auxetic/
│       ├── circle/first/
│       ├── circle/second/
│       ├── square/auxetic/
│       └── ... (10 seeds × 3 objectives)
│
├── tests/                         # Unit tests
├── pyproject.toml                 # Package metadata + tool config
├── Makefile                       # Convenience commands
├── requirements.txt               # Python dependencies
└── CHANGELOG.md
```

---

## Cài đặt

### Yêu cầu hệ thống

- **Python** ≥ 3.10
- **pip** (khuyên dùng phiên bản mới nhất)
- **Git** (để clone repository)

### Các bước cài đặt

```bash
# Clone repository
git clone https://github.com/your-org/Input_SIMP_Analyst.git
cd Input_SIMP_Analyst

# (Khuyên dùng) Tạo virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Cài đặt package ở chế độ development (editable)
pip install -e .
```

Sau khi cài đặt, bạn có thể dùng lệnh `simp` và `simp-analysis` từ terminal.

### Cài đặt optional dependencies

```bash
# Cài thêm dependencies cho analysis (pandas, scikit-image, seaborn, ...)
pip install -e ".[analysis]"

# Cài thêm dependencies cho development (pytest, flake8, black, mypy, ...)
pip install -e ".[dev]"

# Cài tất cả
pip install -e ".[all]"
```

Hoặc dùng Makefile:

```bash
make install           # pip install -e .
make install-core      # core dependencies only
make install-analysis  # analysis dependencies
make install-dev       # all dev dependencies
```

---

## Sử dụng

### Chạy tối ưu hóa SIMP

**Câu lệnh cơ bản nhất:**

```bash
python -m simp.run
```

Chạy với mesh 100×100, seed `circle`, objective `auxetic` — kết quả lưu vào `outputs/simp_results_circle/`.

**Tùy chỉnh tham số qua CLI:**

```bash
python -m simp.main --nelx 80 --nely 60 --volfrac 0.35 --seed hexagonal --objective second --output_dir outputs/my_run
```

**Hoặc dùng console script (nếu đã cài `pip install -e .`):**

```bash
simp --nelx 80 --nely 60 --volfrac 0.35 --seed hexagonal --objective second
```

**Tất cả CLI options** (xem chi tiết trong [`simp/README.md`](simp/README.md)):

| Option | Default | Mô tả |
|--------|---------|-------|
| `--nelx`, `--nely` | 100, 100 | Kích thước lưới phần tử |
| `--volfrac` | 0.4 | Tỉ lệ thể tích mục tiêu |
| `--penal` | 3.0 | Hệ số penalization SIMP |
| `--rmin` | 3.0 | Bán kính filter |
| `--ft` | 2 | Loại filter (1 = sensitivity, 2 = density) |
| `--seed` | circle | Tên seed khởi tạo |
| `--objective` | auxetic | Hàm mục tiêu: `auxetic`, `first`, `second` |
| `--void_size_frac` | 0.4 | Kích thước void ban đầu |
| `--rotation_deg` | 0.0 | Góc xoay seed |
| `--max_iter` | 200 | Số vòng lặp tối đa |
| `--save_every` | 1 | Lưu ảnh PNG mỗi N vòng lặp |

**Sử dụng programmatic (Python API):**

```python
from simp.runner import run_simp

params = {
    'nelx': 120,
    'nely': 120,
    'volfrac': 0.35,
    'penal': 3.0,
    'rmin': 2.5,
    'seed': 'hexagonal',
    'objective': 'auxetic',
    'void_size_frac': 0.45,
    'max_iter': 300,
    'save_every': 5,
}

result = run_simp(params)
print(f'ν₁₂ = {result["v12"]:.4f}, ν₂₁ = {result["v21"]:.4f}')
print(f'Converged: {result["converged"]} after {result["n_iters"]} iterations')

# Kết quả trả về
xPhys   = result['xPhys']      # (nely, nelx) numpy array — density field
Q       = result['Q']           # 3×3 homogenized stiffness tensor
history = result['history']     # dict — convergence history
```

### Screening Pipeline (Phase 1)

Pipeline Phase 1 thực hiện **LHS (Latin Hypercube Sampling) Screening** để xác định tham số nào có ảnh hưởng nhất đến kết quả tối ưu (phân tích tương quan Spearman).

**Chạy một combo (seed + objective):**

```bash
python -m pipeline.phase1_screening --objective auxetic --seed circle --n_samples 50
```

**Quét toàn bộ 30 combo (10 seeds × 3 objectives):**

```bash
python -m pipeline.phase1_screening --all --n_samples 30
```

Kết quả được lưu vào `outputs/pipeline/phase1/` dưới dạng CSV + JSON cho mỗi combo.

**Tham số được screening:**

| Tham số | Khoảng | Áp dụng cho |
|---------|--------|-------------|
| `volfrac` | (0.2, 0.6) | Tất cả |
| `penal` | (1.0, 5.0) | Tất cả |
| `rmin` | (1.0, 6.0) | Tất cả |
| `move` | (0.05, 0.3) | Tất cả |
| `void_size_frac` | (0.2, 0.7) | Tất cả |
| `rotation_deg` | (0.0, 90.0) | Tất cả |
| `beta` | (0.3, 1.5) | `first` objective |
| `beta_second` | (0.5, 2.5) | `second` objective |

### Phân tích kết quả (Analysis)

```bash
# Chạy analysis CLI
python -m analysis.cli

# Hoặc console script
simp-analysis
```

Module `analysis/` cung cấp:
- `dataset.py` — Đọc và xử lý dữ liệu từ CSV/JSON kết quả SIMP
- `image.py` — Tính các chỉ số chất lượng ảnh (binary rate, edge density, noise ratio, symmetry L/R)
- `report.py` — Tạo báo cáo HTML tự chứa với bảng phân loại và chỉ số hình ảnh

### Makefile

```bash
make test              # Chạy pytest với coverage
make lint              # flake8 + mypy
make format            # black + isort
make run-simp ARGS="--nelx 60 --nely 60 --volfrac 0.3"
make run-analysis      # Run analysis CLI
make clean             # Xóa build artifacts
```

---

## Kiến trúc phần mềm

<a name="simp-core"></a>
### SIMP Core (`simp/`)

Vòng lặp tối ưu SIMP được tổ chức theo kiến trúc modular, mỗi module chịu trách nhiệm một giai đoạn trong pipeline:

```
                    ┌──────────────┐
                    │   Seeding    │  ← seeds/*.py: tạo mật độ ban đầu x
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  FE Solver   │  ← core/fem.py + core/solver.py + core/pbc.py
                    │  (with PBC)  │    K · u = f  (trong không gian PBC-reduced)
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │Homogenization│  ← homogenization/compute.py
                    │   Q = f(u)   │    Tính tensor Q (3×3) + độ nhạy dQ
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │   Objective  │  ← objectives/*.py
                    │  c = f(Q)    │    c = ν₁₂, Q₁₂ - β·tr(Q), ...
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │    Filter    │  ← core/filter.py
                    │  x̃ = H · x   │    Cone-shaped density filter
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  OC Update   │  ← core/oc.py
                    │ x_new = OC() │    Optimality Criteria bisection
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  Convergence │  ← core/convergence.py
                    │   Check?     │    Δx < tol_change OR
                    │              │    Δobj < tol_obj (×window_size)?
                    └──────┬───────┘
                           ▼
                     Done / Next iter
```

**Luồng chi tiết** (do `runner.py` điều phối):

1. **Seed generation**: Tạo trường mật độ ban đầu `x` từ một trong 10 seed patterns
2. **FEM with PBC**: Lắp ráp ma trận độ cứng toàn cục `K`, áp PBC → giải tìm chuyển vị `u`
3. **Homogenization**: Tính tensor độ cứng tương đương `Q` (3×3) từ trường chuyển vị
4. **Objective**: Tính giá trị hàm mục tiêu `c` và độ nhạy `dc/dx`
5. **Filter**: Lọc trường độ nhạy/mật độ bằng cone-shaped filter (chống checkerboard)
6. **OC update**: Cập nhật mật độ bằng Optimality Criteria (bisection tìm λ)
7. **Convergence check**: Kiểm tra hội tụ qua 3 tiêu chí (change, objective stability, max iter)
8. Lặp lại từ bước 2 nếu chưa hội tụ

### Pipeline (`pipeline/`)

Module pipeline thực hiện **screening tự động**:

- **Phase 1** (`phase1_screening.py`): Sinh `N` mẫu LHS từ không gian tham số (6–8 biến), chạy SIMP cho mỗi mẫu, phân tích tương quan Spearman để tìm top-3 tham số ảnh hưởng nhất đến objective. Có thể chạy đơn lẻ hoặc quét toàn bộ 30 combo.

### Analysis (`analysis/`)

Module phân tích hậu kỳ:

- **dataset.py**: Đọc CSV/JSON → pandas DataFrame
- **image.py**: Tính các chỉ số từ ảnh kết quả:
  - `binary_rate`: Tỉ lệ pixel đã phân cực về 0 hoặc 1
  - `edge_density`: Mật độ biên giữa vật liệu và void
  - `noise_ratio`: Tỉ lệ nhiễu (pixel không ổn định)
  - `symmetry_lr`: Đối xứng trái-phải
- **report.py**: Tạo HTML self-contained với:
  - Summary cards (tổng số mẫu, số auxetic, số conventional)
  - Bảng phân loại (màu xanh cho auxetic, vàng cho conventional)
  - Bảng chỉ số chất lượng ảnh
  - Tất cả CSS/JS inline, không phụ thuộc CDN

---

## Seeds — Mẫu khởi tạo

10 seed patterns giúp khám phá đa dạng không gian thiết kế:

| Seed | Mô tả | Số void |
|------|-------|---------|
| `circle` | Một lỗ tròn ở tâm | 1 |
| `square` | Một lỗ vuông ở tâm | 1 |
| `hourglass` | Hai lỗ tam giác đối xứng (hình đồng hồ cát) | 2 |
| `four_circle` | Bốn lỗ tròn đối xứng | 4 |
| `hexagonal` | Một lỗ lục giác | 1 |
| `nine_circle` | 3×3 lỗ tròn dạng grid | 9 |
| `cross_rectangular` | Lỗ hình chữ thập | 1 |
| `grid_circular_voids` | Grid đều các lỗ tròn | Nhiều |
| `small_square_cross` | Hình chữ thập vuông nhỏ ở tâm | 1 |
| `circle_half_quarter` | Lỗ tròn tâm + 4 phần tư lỗ tròn ở góc | 5 |

Các seed được parameter hóa bởi `void_size_frac` (kích thước void) và `rotation_deg` (góc xoay), cho phép tạo ra vô số biến thể khởi tạo.

---

## Objective Functions — Hàm mục tiêu

### 1. Auxetic Objective (`auxetic`)

$$
c = \nu_{12} = -\frac{Q_{12}}{Q_{22}}
$$

- **Mục tiêu**: Cực tiểu hóa hệ số Poisson (làm âm hơn)
- **Cách hoạt động**: Trực tiếp tối ưu tỉ số `−Q₁₂/Q₂₂`
- **Độ nhạy**: Tính theo quotient rule từ `dQ/dx`
- **Phù hợp nhất** cho việc tìm kiếm thiết kế auxetic thuần túy

### 2. First Objective (`first`)

$$
c = Q_{12} - \beta^{\text{loop}} \cdot (Q_{11} + Q_{22})
$$

- **Mục tiêu**: Cực đại hóa `Q₁₂` đồng thời triệt tiêu độ cứng dọc trục
- **Điểm đặc biệt**: Hệ số `β^loop` giảm dần (decay) theo số vòng lặp — ban đầu penalty nặng lên `Q₁₁+Q₂₂`, sau đó nhẹ dần
- **Hội tụ ổn định**, thích hợp để thăm dò không gian thiết kế

### 3. Second Objective (`second`)

$$
c = Q_{12} + \text{penalty}(\text{nếu } Q_{11} < \delta \text{ hoặc } Q_{22} < \delta)
$$

- **Mục tiêu**: Cực đại hóa `Q₁₂` một cách trực diện
- **Penalty**: Chỉ kích hoạt khi độ cứng dọc trục xuống dưới ngưỡng `δ = 0.1 · volfrac · E₀`
- **Tập trung vào shear**, có thể cho kết quả agressive hơn `first`

---

## Homogenization & Periodic Boundary Conditions

### Homogenization

Module `homogenization/compute.py` thực hiện:

- **Tính tensor độ cứng tương đương** `Q` (3×3) từ trường chuyển vị trên unit cell:
  $$
  Q_{ij} = \frac{1}{|Y|} \int_Y (\boldsymbol{\varepsilon}^0_i - \boldsymbol{\varepsilon}(u_i))^T \cdot \mathbb{C} \cdot (\boldsymbol{\varepsilon}^0_j - \boldsymbol{\varepsilon}(u_j)) \, dY
  $$
- **Tính độ nhạy** `dQ/dx` — cần thiết cho gradient-based optimization

3 trường hợp tải (load case) được giải: epsilon_xx, epsilon_yy, epsilon_xy.

### Periodic Boundary Conditions (PBC)

Module `core/pbc.py` áp PBC bằng phương pháp **null-space projection**:

1. Xác định master/slave DOFs trên các biên đối diện
2. Xây dựng ma trận chiếu `P` sao cho `u = P · u_reduced`
3. Bài toán FE được giải trong không gian reduced: `(P^T · K · P) · u_reduced = P^T · f`
4. Kết quả `u` được khôi phục từ `u_reduced = P · u_reduced`

Phương pháp này hiệu quả vì giảm kích thước hệ phương trình (loại bỏ DOFs dư thừa do PBC).

---

## Convergence — Tiêu chí hội tụ

Vòng lặp tối ưu dừng khi **bất kỳ** điều kiện nào sau đây được thỏa mãn:

1. **Design change** < `tol_change` (mặc định 0.01):
   $$
   \max |x_{new} - x_{old}| < \text{tol\_change}
   $$

2. **Objective stability**: Biến thiên tương đối của objective < `tol_obj` (0.05) trong `window_size` (20) vòng lặp liên tiếp

3. **Max iterations** đạt `max_iter` (mặc định 200)

Module `core/convergence.py` triển khai class `ConvergenceChecker` quản lý cả 3 tiêu chí này.

---

## Kết quả đầu ra

Sau khi chạy tối ưu, thư mục `output_dir` chứa:

| File | Mô tả |
|------|-------|
| `iteration_00001.png` ... | Ảnh xám trường mật độ (đen = void, trắng = solid) tại mỗi vòng lưu |
| `iteration_data.csv` | Lịch sử hội tụ: Iteration, Poisson_v12, Poisson_v21, Objective, Volume_Fraction |
| (Thư mục con `sample_*/`) | Kết quả từng mẫu trong Phase 1 screening |

---

## HTML Reports & Dashboards

Dự án bao gồm các báo cáo HTML đã được xây dựng sẵn:

| HTML | Mô tả |
|------|-------|
| [`html/reports/auxetic_report.html`](html/reports/auxetic_report.html) | Báo cáo kết quả auxetic |
| [`html/dashboards/phase1_screening_dashboard.html`](html/dashboards/phase1_screening_dashboard.html) | Dashboard Phase 1 screening |
| [`html/guides/optimization_pipeline.html`](html/guides/optimization_pipeline.html) | Hướng dẫn pipeline tối ưu (tiếng Việt) |
| [`html/guides/simp_guide_and_roadmap.html`](html/guides/simp_guide_and_roadmap.html) | Guide và roadmap SIMP (tiếng Việt) |
| [`outputs/report_simp_analysis.html`](outputs/report_simp_analysis.html) | Báo cáo phân tích tổng hợp |

Các báo cáo được thiết kế **self-contained** (tất cả CSS/JS inline), có thể mở trực tiếp bằng `file://` mà không cần server.

---

## Testing

```bash
# Chạy toàn bộ test suite
python -m pytest tests/ -v

# Với coverage report
python -m pytest tests/ -v --cov=simp --cov=analysis --cov-report=term-missing

# HTML coverage report
python -m pytest tests/ -v --cov=simp --cov=analysis --cov-report=html
# open htmlcov/index.html

# Hoặc dùng Makefile
make test
make test-coverage
```

---

## Dependencies

### Core dependencies

- **Python** ≥ 3.10
- **numpy** ≥ 1.24
- **scipy** ≥ 1.10
- **matplotlib** ≥ 3.7

### Analysis dependencies (optional)

- pandas ≥ 2.0
- scikit-image ≥ 0.20
- Pillow ≥ 10.0
- seaborn ≥ 0.13

### Dev dependencies (optional)

- pytest ≥ 7.0 + pytest-cov
- flake8, black, isort
- mypy

Xem đầy đủ trong [`pyproject.toml`](pyproject.toml) và [`requirements.txt`](requirements.txt).

---

## Tài liệu tham khảo

1. Sigmund, O. (2001). *A 99 line topology optimization code written in Matlab.* Structural and Multidisciplinary Optimization, 21(2), 120–127.
2. Andreassen, E., et al. (2011). *Efficient topology optimization in MATLAB using 88 lines of code.* Structural and Multidisciplinary Optimization, 43(1), 1–16.
3. Xia, L., & Breitkopf, P. (2015). *Design of materials using topology optimization and energy-based homogenization.* Archives of Computational Methods in Engineering, 22(2), 229–260.
4. Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods, and Applications.* Springer.

---

## License

MIT License — xem file [`LICENSE`](LICENSE) (hoặc tham khảo trong `simp/__init__.py`).

---

> ⚡ **SIMPAnalyst** — từ tư duy thiết kế đến micro-cấu trúc auxetic.
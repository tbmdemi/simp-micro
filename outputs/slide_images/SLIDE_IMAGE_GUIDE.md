# Hướng dẫn chọn ảnh cho Slide Báo cáo Phase 1

> **Dự án:** SIMP Analyst - Topology Optimization cho Microstructure Auxetic  
> **Mục đích:** Cung cấp các ảnh đại diện (representative topologies) từ Phase 1 LHS Screening  
> **Tổng số ảnh đã thu thập:** 18 ảnh tại thư mục `outputs/slide_images/`

---

## 1. Ảnh đại diện theo 5 tiêu chí chính

Đây là 5 topology đại diện quan trọng nhất cho slide báo cáo, đã được phân tích trong
[Phase 1 Screening Report](../docs/phase1_screening_report.md).

| # | Tiêu chí | File ảnh | Seed | ν₁₂ | Objective | Iterations | Ghi chú |
|---|----------|----------|------|-----|-----------|------------|---------|
| 1 | 🏆 **Tốt nhất** (ν₁₂ gần auxetic nhất) | `01_best_v12_hexagonal.png` | hexagonal | **0.052** | 78.44 | 121 | Gần ν=0 nhất; cấu trúc lục giác + kết nối tinh vi |
| 2 | ❌ **Tệ nhất** (ν₁₂ cao nhất) | `02_worst_v12_square.png` | square | **0.520** | 105.73 | 10 | Stuck sớm, volfrac quá cao, mất cấu trúc |
| 3 | 🔧 **Phức tạp nhất** (edge density cao) | `03_most_complex_nine_circle.png` | nine_circle | 0.422 | 105.36 | 150 | Checkerboard, penal thấp, không kịp hội tụ |
| 4 | ✅ **Đơn giản nhất** (converge nhanh) | `04_simplest_circle.png` | circle | 0.115 | 28.99 | **22** | Vành tròn đơn giản, đối xứng hoàn hảo |
| 5 | ⚖️ **Ổn định nhất** (objective thấp nhất) | `05_most_stable_hourglass.png` | hourglass | 0.300 | **3.76** | 21 | Objective thấp nhất toàn bộ Phase 1 |

---

## 2. Chi tiết ảnh theo chất lượng hình ảnh

Các chỉ số được tính bằng module `analysis/image.py`:
- **binary_rate**: Tỉ lệ pixel đã phân cực về đen (<32) hoặc trắng (>224). 1.0 = lý tưởng
- **edge_density**: Mật độ biên giới vật liệu/void (Sobel). Cao = nhiều chi tiết/nhiễu
- **noise_ratio**: Tỉ lệ gray pixel không ổn định. Thấp = tốt
- **symmetry_lr**: Đối xứng trái-phải. 1.0 = đối xứng tuyệt đối

| File Ảnh | Tiêu chí | binary_rate | edge_density | noise_ratio | symmetry_lr |
|----------|----------|-------------|-------------|-------------|-------------|
| `01_best_v12_hexagonal.png` | 🏆 Tốt nhất ν₁₂ | 0.8257 | 0.0078 | 0.0004 | **0.9585** |
| `02_worst_v12_square.png` | ❌ Tệ nhất ν₁₂ | 0.6985 | 0.0058 | 0.0002 | 0.8539 |
| `03_most_complex_nine_circle.png` | 🔧 Phức tạp | **0.5143** | 0.0082 | 0.0003 | **0.4992** |
| `04_simplest_circle.png` | ✅ Đơn giản | **0.9366** | 0.0085 | 0.0004 | 0.9522 |
| `05_most_stable_hourglass.png` | ⚖️ Ổn định | **0.8585** | 0.0089 | 0.0004 | 0.9500 |

### Phân tích nhanh cho slide:

- **Best binary rate**: `07_perfect_binary_circle.png` (1.0) - hoàn hảo cho slide về quá trình hội tụ
- **Worst binary rate**: `09_worst_binary_small_square.png` (0.23) - minh họa cho "bad convergence"
- **Best symmetry**: `08_perfect_symmetry_circle.png` (1.0) - minh họa periodic cell lý tưởng
- **Worst symmetry**: `10_worst_symmetry.png` (0.41) - mất đối xứng do checkerboard
- **Most complex structure**: `11_most_complex_grid.png` - nhiều chi tiết nhất

---

## 3. Gợi ý bố trí slide

### Slide 1: Tổng quan Phase 1 Pipeline
> Dùng 5 ảnh đại diện (01-05) xếp thành hàng ngang với nhãn

### Slide 2: Chiến lược Objective Functions
> Dùng `06_first_obj_best_circle.png` (first obj converge nhanh)  
> + `13_second_obj_circle.png` (second obj giá trị âm lớn)

### Slide 3: Seed Comparison - Khởi tạo vs Kết quả
> Dùng cặp ảnh seed (initial) + final cho cùng 1 seed:
> - `seed_hexagonal_initial.png` → `01_best_v12_hexagonal.png`
> - `seed_circle_initial.png` → `04_simplest_circle.png`

### Slide 4: Chất lượng hội tụ
> Dùng `07_perfect_binary_circle.png` (hội tụ lý tưởng)  
> vs `09_worst_binary_small_square.png` (hội tụ kém)

### Slide 5: Phân tích tương quan Spearman
> Dùng heatmap từ `outputs/figures/heatmap_auxetic.png`

---

## 4. Các ảnh bổ sung có sẵn

| File | Mô tả |
|------|-------|
| `seed_hexagonal_initial.png` | Khởi tạo seed hexagonal (lục giác) |
| `seed_circle_initial.png` | Khởi tạo seed circle (tròn) |
| `seed_hourglass_initial.png` | Khởi tạo seed hourglass (đồng hồ cát) |
| `seed_square_initial.png` | Khởi tạo seed square (vuông) |
| `seed_nine_circle_initial.png` | Khởi tạo seed nine_circle (3×3 lỗ tròn) |
| `06_first_obj_best_circle.png` | First objective - hội tụ nhanh (9 iters) |
| `07_perfect_binary_circle.png` | Binary rate = 1.0 (hoàn hảo) |
| `08_perfect_symmetry_circle.png` | Đối xứng = 1.0 (hoàn hảo) |
| `09_worst_binary_small_square.png` | Binary rate = 0.23 (kém) |
| `10_worst_symmetry.png` | Đối xứng = 0.41 (kém) |
| `11_most_complex_grid.png` | Phức tạp nhất - edge density = 0.025 |
| `12_best_auxetic_square_v12_005.png` | Best ν₁₂ = 0.053 (square/auxetic) |
| `13_second_obj_circle.png` | Second objective obj = -636 |

### Biểu đồ phân tích (từ `outputs/figures/`)

| File | Mô tả |
|------|-------|
| `heatmap_auxetic.png` | Spearman correlation heatmap cho auxetic objective |
| `heatmap_first.png` | Spearman correlation heatmap cho first objective |
| `heatmap_second.png` | Spearman correlation heatmap cho second objective |
| `barplot_importance.png` | Bar chart mức độ quan trọng các tham số |
| `analysis_binary_rate.png` | Phân phối binary rate |
| `analysis_edge_density.png` | Phân phối edge density |
| `shear_nu12_nu21_curves.png` | Đường cong ν₁₂ và ν₂₁ |
| `stiff_nu12_nu21_curves.png` | Đường cong stiffness vs ν₁₂ |

---

## 5. Script để tự động tạo ảnh cho slide khác

Nếu bạn muốn tìm thêm ảnh theo tiêu chí khác, chạy:

```bash
cd /home/tbm/Documents/Input_SIMP_Analyst
python3 << 'EOF'
"""Tìm ảnh Phase 1 theo tiêu chí tùy chỉnh."""
import sys
sys.path.insert(0, '.')
from analysis.image import analyze_image
from pathlib import Path
import json, csv

PHASE1 = Path("outputs/pipeline/phase1")

def find_images_by_criteria(seed=None, objective=None, 
                            min_binary=0, max_iter=150):
    """Tìm các ảnh thỏa mãn điều kiện."""
    results = []
    for s_dir in sorted(PHASE1.iterdir()):
        if seed and s_dir.name != seed:
            continue
        if not s_dir.is_dir() or s_dir.name.startswith('_'):
            continue
        for o_dir in sorted(s_dir.iterdir()):
            if objective and o_dir.name != objective:
                continue
            if not o_dir.is_dir():
                continue
            for smp_dir in sorted(o_dir.iterdir()):
                if not smp_dir.is_dir():
                    continue
                pngs = sorted(smp_dir.glob("iteration_*.png"))
                if len(pngs) < 2:
                    continue
                final_img = pngs[-1]
                # Read CSV
                csv_path = smp_dir / "iteration_data.csv"
                if not csv_path.exists():
                    continue
                lines = csv_path.read_text().strip().split('\n')
                if len(lines) < 2:
                    continue
                parts = lines[-1].split(',')
                try:
                    iters = int(parts[0])
                    v12 = float(parts[1]) if parts[1] else None
                    obj = float(parts[2]) if parts[2] else None
                except:
                    continue
                # Compute metrics
                metrics = analyze_image(str(final_img))
                if metrics["binary_rate"] >= min_binary and iters <= max_iter:
                    results.append({
                        "path": str(final_img),
                        "seed": s_dir.name,
                        "objective": o_dir.name,
                        "sample": smp_dir.name,
                        "iterations": iters,
                        "v12": v12,
                        "objective_value": obj,
                        **metrics
                    })
    
    # Sort by desired metric
    results.sort(key=lambda x: x.get("binary_rate", 0), reverse=True)
    return results

# Example: find top-5 highest binary rate auxetic samples
best = find_images_by_criteria(objective="auxetic", min_binary=0.9)
for i, r in enumerate(best[:5]):
    print(f"{i+1}. {r['seed']}/{r['sample']} bin={r['binary_rate']:.4f} sym={r['symmetry_lr']:.4f}")
    print(f"   {r['path']}")
EOF
```

---

*Script này sẽ giúp bạn tìm thêm ảnh cho slide nếu cần thay đổi tiêu chí.*

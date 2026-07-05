# Pareto Front: Auxetic Behavior vs. Eigenfrequency

## Giải thích bài toán

### Bối cảnh

Trong thiết kế micro-cấu trúc (unit cell) auxetic bằng topology optimization (SIMP), ta tìm kiếm sự phân bố vật liệu tối ưu trong ô cơ sở tuần hoàn sao cho vật liệu tương đương (homogenized) có các tính chất mong muốn.

**Auxetic behavior** (ν₁₂ < 0) — khả năng co giãn âm — là mục tiêu chính. Tuy nhiên, các cấu trúc auxetic thường hy sinh độ cứng tổng thể, dẫn đến **tần số riêng thấp**, dễ bị kích thích dao động.

### Trade-off (sự đánh đổi)

| Tham số | Auxetic behavior tốt hơn | Tần số cao hơn |
|---------|------------------------|-----------------|
| Poisson ratio ν₁₂ | Càng thấp (≈ 0 hoặc âm) càng tốt | Càng cao (≈ 0.5 đẳng hướng) càng tốt |
| Độ cứng hiệu dụng E_eff | Thấp (dễ biến dạng) | Cao (chống biến dạng) |
| Cấu trúc vi mô | Liên kết chéo, xoắn, re-entrant | Dạng tổ ong, khung cứng |

**Tính auxetic đòi hỏi cấu trúc mềm → tần số thấp.**
**Tần số cao đòi hỏi cấu trúc cứng → mất tính auxetic.**

### Công thức

#### Poisson ratio (ν₁₂)
Từ tensor độ cứng đồng nhất hóa **Q** (3×3), ta tính:

```
ν₁₂ = Q₁₂ / Q₂₂
```

- ν₁₂ < 0 → auxetic (co giãn âm)
- ν₁₂ = 0 → không có hiệu ứng shear coupling
- ν₁₂ > 0 → vật liệu thông thường

#### Tần số riêng thứ nhất (f₁)
Trong xấp xỉ đồng nhất hóa cho unit cell 2D, tần số uốn cơ bản:

```
f₁ = 1/(2L) · √( E_eff / ρ_eff )
```

với:
- **L**: kích thước ô cơ sở
- **E_eff** = min(Q₁₁, Q₂₂, Q₃₃): độ cứng theo hướng yếu nhất
- **ρ_eff** = volfrac · ρ₀: mật độ hiệu dụng
- **E_eff** ∝ volfrac^penal · E₀: tỉ lệ với mật độ (SIMP scaling)

Từ đó:
```
f₁ ∝ √( volfrac^(penal-1) · E₀ · ν₁₂ / ρ₀ )
```

### Đường Pareto

Pareto front là tập hợp các thiết kế **không bị dominated**: không thể cải thiện auxetic behavior (giảm ν₁₂) mà không làm giảm tần số (giảm f₁), và ngược lại.

Trên đồ thị:
- **Trục x**: Poisson ratio ν₁₂ (càng thấp → càng auxetic)
- **Trục y**: Tần số chuẩn hóa f₁ (càng cao → càng cứng)
- **Đường màu đen**: Pareto frontier — các điểm tối ưu Pareto
- **Điểm màu vàng**: Các thiết kế Pareto-optimal

**Vùng Trade-off**: ν₁₂ thấp (0.05–0.15) và f₁ thấp (1–2) — các thiết kế auxetic mạnh nhưng mềm.
**Vùng Stiff**: ν₁₂ cao (0.4–0.5) và f₁ cao (4–5) — các thiết kế cứng, không auxetic.

### Kết quả thực nghiệm

Từ 1500 mẫu Phase 1 screening (10 seeds × 3 objectives × 50 samples), ta tìm được 49 mẫu Pareto-optimal:

| Seed | ν₁₂ | f₁ | volfrac | penal |
|------|:----:|:--:|:-------:|:-----:|
| circle_half_quarter | **0.0459** | 1.19 | 0.559 | 2.14 |
| hexagonal | 0.0521 | 1.32 | 0.581 | 3.28 |
| circle | 0.0564 | 1.38 | 0.581 | 3.28 |
| cross_rectangular | 0.0629 | 1.40 | 0.559 | 2.14 |
| cross_rectangular | 0.0831 | 1.50 | 0.521 | 4.05 |
| square | 0.0907 | 1.68 | 0.559 | 2.14 |
| small_square_cross | 0.0910 | 1.75 | 0.581 | 3.28 |

### Ứng dụng

Pareto front này giúp nhà thiết kế chọn điểm làm việc phù hợp:
- **Cần auxetic mạnh**: chọn thiết kế ν₁₂ ≈ 0.05 (circle_half_quarter), chấp nhận f₁ thấp
- **Cần cân bằng**: chọn điểm giữa Pareto front (ν₁₂ ≈ 0.1, f₁ ≈ 2)
- **Cần tần số cao**: chọn ν₁₂ > 0.3, không còn auxetic rõ rệt

### Phương pháp tính

1. **Phase 1**: LHS sampling + SIMP optimization → 1500 designs
2. **Proxy cho f₁**: từ Q tensor (ν₁₂, obj_val, volfrac) → f_proxy
3. **Pareto filter**: thuật toán tìm non-dominated set
4. **Visualization**: scatter plot + Pareto frontier

---

*File: outputs/pareto/pareto_front_auxetic_vs_frequency.png*

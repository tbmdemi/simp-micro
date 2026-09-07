# Dòng chảy toán học của bài toán tối ưu SIMP

Tài liệu này mô tả **chuỗi công thức toán học** đứng sau pipeline tối ưu hóa hình
dạng (topology optimization) cho thiết kế **ô cơ sở tuần hoàn** với tính chất cơ
học mục tiêu (vật liệu auxetic — hệ số Poisson âm). Nội dung chỉ bàn về mô hình
toán; cách triển khai trong code nằm ở `README.md`.

---

## 1. Phát biểu bài toán

Thiết kế ô cơ sở tuần hoàn (unit cell) trên lưới `nelx × nely` phần tử, mỗi phần
tử mang một mật độ vật liệu $x_e \in [0, 1]$. Bài toán tối ưu tổng quát:

$$
\min_{\mathbf{x}} \; c(\mathbf{x}) \qquad \text{s.t.} \qquad
\frac{1}{N}\sum_{e=1}^{N} x_e = f, \qquad 0 \le x_e \le 1,
$$

trong đó $f$ là tỉ lệ thể tích mục tiêu (`volfrac`), $N = \text{nelx}\times\text{nely}$
là số phần tử, và $c(\mathbf{x})$ là hàm mục tiêu xây dựng từ ten-xơ độ cứng đồng
nhất hóa $\mathbf{Q}$ (mục 5). Ràng buộc thể tích được thỏa mãn qua hệ số Lagrange
trong bước cập nhật OC (mục 7).

---

## 2. Nội suy vật liệu SIMP

Mô đun đàn hồi hiệu dụng của phần tử $e$ được nội suy theo mật độ bằng luật phạt
SIMP (Solid Isotropic Material with Penalization):

$$
E_e(x_e) = E_{\min} + \big(\rho_0\, x_e^{\,p}\big)\,(E_0 - E_{\min}),
$$

với $E_0$ là mô đun Young của vật liệu đặc, $E_{\min}$ là mô đun của lỗ rỗng
(giá trị nhỏ tránh suy biến ma trận độ cứng), $p \ge 3$ là hệ số phạt
(`penal`), và $\rho_0$ là hằng số tỉ lệ (mặc định $\rho_0 = 1$). Số mũ $p$ lớn
làm mật độ trung gian $0 < x_e < 1$ trở nên "tốn vật liệu nhưng ít đóng góp độ
cứng", ép nghiệm về thiết kế gần nhị phân $x_e \in \{0, 1\}$.

### 2.1. Ma trận độ cứng phần tử (bài toán ứng suất phẳng)

Vật liệu nền đẳng hướng, đàn hồi tuyến tính, trạng thái **ứng suất phẳng**
(plane stress). Ma trận đàn hồi (constitutive matrix):

$$
\mathbf{D} = \frac{E_0}{1-\nu^2}
\begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \dfrac{1-\nu}{2}
\end{bmatrix},
$$

với $\nu$ là hệ số Poisson của vật liệu nền ($-1 < \nu < 0.5$).

Ma trận độ cứng phần tử tứ giác 4 nút (8 bậc tự do) được tính bằng tích phân số
Gauss 2×2:

$$
\mathbf{k}_e = \int_{\Omega_e} \mathbf{B}^{\mathsf{T}} \mathbf{D}\, \mathbf{B}
\, d\Omega_e \;\approx\;
\sum_{g=1}^{4} w_g\,
\mathbf{B}(\xi_g, \eta_g)^{\mathsf{T}}
\mathbf{D}\,
\mathbf{B}(\xi_g, \eta_g)\;|\det \mathbf{J}_g|,
$$

trong đó $\mathbf{B}$ là ma trận biến dạng–chuyển vị của phần tử chuẩn
(isoparametric Q4), $\mathbf{J}$ là ma trận Jacobi, điểm Gauss
$(\xi_g, \eta_g) = (\pm 1/\sqrt{3}, \pm 1/\sqrt{3})$, trọng số $w_g = 1$.
Do $E_0$ đã nằm trong $\mathbf{D}$, ma trận phần tử $\mathbf{k}_e$ chứa sẵn $E_0$
(xem mục 4 — cần chuẩn hóa lại khi đồng nhất hóa).

---

## 3. Phân tích phần tử hữu hạn trên ô cơ sở (PBC)

### 3.1. Lắp ráp ma trận độ cứng toàn cục

Từ trường mật độ $\mathbf{x}$ và luật nội suy (2), ma trận độ cứng toàn cục được
lắp ráp:

$$
\mathbf{K}(\mathbf{x}) = \sum_{e=1}^{N} E_e(x_e)\;\mathbf{k}_e^{\text{norm}}
= \sum_{e=1}^{N} \frac{E_e(x_e)}{E_0}\;\mathbf{k}_e,
$$

với $\mathbf{k}_e^{\text{norm}} = \mathbf{k}_e / E_0$ là ma trận phần tử chuẩn hóa
(chứa hình học + $\nu$, không chứa $E_0$). Ma trận $\mathbf{K}$ được đối xứng hóa
$\mathbf{K} \leftarrow (\mathbf{K} + \mathbf{K}^{\mathsf{T}})/2$ để loại sai số số học.

### 3.2. Điều kiện biên tuần hoàn (null-space projection)

Ô cơ sở tuần hoàn yêu cầu trường chuyển vị lặp lại trên các biên đối diện. Các
ràng buộc tuần hoàn (master–slave) được gom thành một phép chiếu tuyến tính
$\mathbf{P}$ ($\text{ndof} \times \text{ndof}_{\text{reduced}}$) lên không gian con
thỏa mãn tuần hoàn (cơ sở của không gian null của ma trận ràng buộc):

$$
\mathbf{u} = \mathbf{P}\,\mathbf{u}_r,
\qquad
\mathbf{K}_{\text{pbc}} = \mathbf{P}^{\mathsf{T}} \mathbf{K}\, \mathbf{P}.
$$

Hệ FE với PBC trở thành bài toán trên không gian con rút gọn, không còn bậc tự do
dư thừa ở biên.

### 3.3. Ba trường hợp tải biến dạng đơn vị

Đồng nhất hóa cần giải ô cơ sở dưới 3 trường biến dạng đơn vị (kí hiệu Voigt
$[\varepsilon_{xx},\; \varepsilon_{yy},\; \gamma_{xy}]$):

| Case | Biến dạng áp đặt |
|------|------------------|
| $i = 1$ | $\varepsilon_{xx} = 1$ |
| $i = 2$ | $\varepsilon_{yy} = 1$ |
| $i = 3$ | $\gamma_{xy} = 1$ |

Trường chuyển vị áp đặt tương ứng (trên miền chuẩn hóa $[0,1]^2$,
$\mathbf{u}^0(\mathbf{x}) = \boldsymbol{\varepsilon}^0 \cdot \mathbf{x}$):

$$
\mathbf{u}^0_1 = (x, 0), \qquad
\mathbf{u}^0_2 = (0, y), \qquad
\mathbf{u}^0_3 = \left(\tfrac{y}{2},\; \tfrac{x}{2}\right).
$$

Phân tích chuyển vị tuần hoàn tách thành **trường áp đặt** $\mathbf{u}^0$ và
**trường dao động** (fluctuation) $\boldsymbol{\chi}$. Trường dao động thỏa mãn:

$$
\mathbf{K}_{\text{pbc}}\,\boldsymbol{\chi} = -
\mathbf{P}^{\mathsf{T}}\,\mathbf{K}\,\mathbf{u}^0,
$$

(2 bậc tự do đầu bị ghim để loại chuyển vị vật cứng). Chuyển vị tổng:

$$
\mathbf{u} = \mathbf{u}^0 + \boldsymbol{\chi}.
$$

---

## 4. Đồng nhất hóa dựa trên năng lượng

Ten-xơ độ cứng đồng nhất hóa $\mathbf{Q}$ ($3 \times 3$, thứ tự Voigt
$[11, 22, 12]$) được tính bằng phương pháp năng lượng (energy-based
homogenization — Andreassen et al. 2014, công thức (6)):

$$
Q_{ij} = \frac{1}{|\Omega|}\sum_{e=1}^{N}
\big(\mathbf{u}_e^{i}\big)^{\mathsf{T}}\,
\mathbf{k}_e^{\text{pen}}\,
\big(\mathbf{u}_e^{j}\big),
\qquad
\mathbf{k}_e^{\text{pen}} = \frac{E_e(x_e)}{E_0}\,\mathbf{k}_e,
$$

với $\mathbf{u}_e^{i}$ là vector chuyển vị phần tử (8 thành phần) của case $i$.
Miền chuẩn hóa $[0,1]^2$ có $|\Omega| = 1$ nên thừa số $1/|\Omega|$ triệt tiêu.
Lưu ý chia cho $E_0$: vì $\mathbf{k}_e$ đã chứa $E_0$, phép chia này bảo đảm
$Q_{ij}$ tỉ lệ **bậc nhất** theo $E_0$ (đúng thứ nguyên vật lý).

**Đạo hàm theo mật độ** (dùng cho độ nhạy, mục 5) — chỉ phần tử $e$ tham gia:

$$
\frac{\partial Q_{ij}}{\partial x_e} =
\big(\mathbf{u}_e^{i}\big)^{\mathsf{T}}\,
\frac{\partial \mathbf{k}_e^{\text{pen}}}{\partial x_e}\,
\big(\mathbf{u}_e^{j}\big),
\qquad
\frac{\partial \mathbf{k}_e^{\text{pen}}}{\partial x_e} =
\frac{\rho_0\, p\, x_e^{\,p-1}(E_0 - E_{\min})}{E_0}\;\mathbf{k}_e.
$$

(Đạo hàm theo chuyển vị triệt tiêu nhờ tính dừng của nghiệm FE — adjoint tự
nhiên, không cần biến phụ.)

---

## 5. Hàm mục tiêu auxetic và độ nhạy

### 5.1. Từ hệ số Poisson đến thành phần $Q_{12}$

Hệ số Poisson hiệu dụng được lấy từ ten-xơ tuân thủ
$\mathbf{S} = \mathbf{Q}^{-1}$:

$$
\nu_{12} = -\frac{S_{12}}{S_{11}}, \qquad \nu_{21} = -\frac{S_{12}}{S_{22}}.
$$

Khi ô cơ sở **trực hướng theo trục** (không ghép cắt–pháp, $Q_{13} = Q_{23} = 0$),
nghịch đảo ma trận $2\times 2$ con cho phép rút gọn:

$$
S_{12} = -\frac{Q_{12}}{Q_{11}Q_{22} - Q_{12}^{2}},
\qquad
S_{11} = \frac{Q_{22}}{Q_{11}Q_{22} - Q_{12}^{2}}
\;\;\Longrightarrow\;\;
\nu_{12} = \frac{Q_{12}}{Q_{22}}.
$$

Vì $Q_{22} > 0$ (ma trận độ cứng xác định dương), $\text{sign}(\nu_{12}) =
\text{sign}(Q_{12})$: **$Q_{12} < 0$ tương đương hệ số Poisson âm (auxetic)**.
Dùng $Q_{12}$ làm proxy thay vì $\nu_{12}$ trực tiếp vì đạo hàm
$\partial \nu_{12}/\partial x_e \propto \partial(Q_{12}/Q_{22})$ gần triệt tiêu
khi $Q_{12}/Q_{22} \approx Q_{11}/Q_{11}$ ở gần đẳng hướng, làm gradient mất tác
dụng trong OC update.

### 5.2. Hàm mục tiêu

$$
c(\mathbf{Q}) = Q_{12} - \mu\,(Q_{11} + Q_{22})
+ \beta \sum_{i \in \{1,2\}}
\frac{\big(\delta - Q_{ii}\big)^{2}_{+}}{\delta^{2}},
$$

với $(z)_{+} = \max(z, 0)$, ngưỡng độ cứng
$\delta = 0.1\, f\, E_0$ (10% độ cứng nền theo tỉ lệ thể tích), $\beta$ là trọng
số phạt, và $\mu$ là tham số cân bằng độ cứng (mặc định $\mu = 0$). Số hạng phạt
ngăn thiết kế **sụp đổ** (collapse): nếu $Q_{11}$ hoặc $Q_{22}$ rơi xuống dưới
$\delta$, mục tiêu bị phạt bình phương.

**Độ nhạy** theo mật độ vật lý $\hat{\mathbf{x}}$:

$$
\frac{\partial c}{\partial \hat{x}_e} =
\frac{\partial Q_{12}}{\partial \hat{x}_e}
- \mu\left(\frac{\partial Q_{11}}{\partial \hat{x}_e}
+ \frac{\partial Q_{22}}{\partial \hat{x}_e}\right)
- 2\beta \sum_{i \in \{1,2\}}
\frac{(\delta - Q_{ii})_{+}}{\delta^{2}}
\frac{\partial Q_{ii}}{\partial \hat{x}_e},
$$

trong đó $\partial Q_{ij}/\partial \hat{x}_e$ lấy từ mục 4 (xem chuỗi đạo hàm
hợp trong mục 6.3 để chuyển về biến thiết kế $\mathbf{x}$).

### 5.3. Biến thể chuẩn hóa (tùy chọn)

Thay vì $Q_{12}$ thô, cực tiểu hệ số ghép chuẩn hóa:

$$
c = \frac{Q_{12}}{\sqrt{Q_{11} Q_{22}}}.
$$

Do ma trận con $\begin{bmatrix} Q_{11} & Q_{12} \\ Q_{12} & Q_{22} \end{bmatrix}$
xác định bán dương (sinh từ dạng toàn phương năng lượng biến dạng), bất đẳng thức
Cauchy–Schwarz $Q_{12}^{2} \le Q_{11}Q_{22}$ kéo theo $c \in [-1, 1]$ — mục tiêu bị
chặn tự nhiên bởi chính vật lý. Phạt stiffness $\delta$ vẫn được giữ nguyên.

---

## 6. Bộ lọc và phép chiếu

Bộ lọc bảo đảm tính khả thi chế tạo (tránh checkerboard) và áp đặt độ dài đặc
trưng tối thiểu.

### 6.1. Bộ lọc mật độ (ft = 2)

Trọng số hình nón theo khoảng cách tâm phần tử:

$$
H_{ef} = \max\big(0,\; r_{\min} - \text{dist}(e, f)\big),
$$

mật độ lọc (mật độ vật lý khi không có projection):

$$
\tilde{x}_e = \frac{\sum_f H_{ef}\, x_f}{\sum_f H_{ef}}.
$$

### 6.2. Bộ lọc độ nhạy (ft = 1)

Biến thể lọc gradient (Sigmund 2001), áp lên độ nhạy đã nhân với biến thiết kế:

$$
\widetilde{\frac{\partial c}{\partial x_e}} =
\frac{\sum_f H_{ef}\, x_f \frac{\partial c}{\partial x_f}}
{x_e \sum_f H_{ef}}.
$$

### 6.3. Phép chiếu Heaviside làm mượt (tùy chọn)

Chiếu mật độ đã lọc về biên 0–1 sắc nét với độ dài đặc trưng tối thiểu
$\approx r_{\min}$:

$$
\hat{x}_e = \frac{\tanh(\beta_{\text{proj}}\,\eta)
+ \tanh\big(\beta_{\text{proj}}(\tilde{x}_e - \eta)\big)}
{\tanh(\beta_{\text{proj}}\,\eta)
+ \tanh\big(\beta_{\text{proj}}(1 - \eta)\big)},
\qquad \eta = 0.5,
$$

đạo hàm:

$$
\frac{d\hat{x}_e}{d\tilde{x}_e} = \frac{\beta_{\text{proj}}
\big(1 - \tanh^{2}(\beta_{\text{proj}}(\tilde{x}_e - \eta))\big)}
{\tanh(\beta_{\text{proj}}\,\eta)
+ \tanh\big(\beta_{\text{proj}}(1 - \eta)\big)}.
$$

**Chuỗi đạo hàm hợp** (từ mục tiêu về biến thiết kế):

$$
\frac{dc}{dx_e} =
\frac{\partial c}{\partial \hat{x}_e}
\cdot \frac{d\hat{x}_e}{d\tilde{x}_e}
\cdot \frac{\partial \tilde{x}_e}{\partial x_e},
$$

trong đó $\partial \tilde{x}_e / \partial x_e$ là ma trận lọc $H_{ef}/H_s$
(ft = 2) hoặc quy tắc ft = 1 ở trên. Khi tắt projection ($\beta_{\text{proj}} \to 0$),
$d\hat{x}/d\tilde{x} = 1$ và chuỗi rút về đúng SIMP cổ điển.

---

## 7. Cập nhật tiêu chí tối ưu (OC)

### 7.1. Một ràng buộc (thể tích)

Bài toán con Lagrange của (1): $\min c(\mathbf{x})$ với
$\sum_e x_e = N f$. Điều kiện tối ưu KKT cho phần tử trong vùng $0 < x_e < 1$:

$$
B_e(\lambda) = -\frac{\partial c / \partial x_e}{\lambda\; \partial V / \partial x_e} = 1
\qquad \forall e,
$$

với $V = \sum_e x_e$, $\partial V/\partial x_e = 1$ và $\lambda$ là hệ số Lagrange.
Cập nhật nhân theo heuristic OC chuẩn:

$$
x_e^{\text{new}} = \operatorname{clip}\!\Big(
x_e\, B_e(\lambda)^{\eta},\;\;
x_e - m,\;\; x_e + m,\;\; [x_{\min},\, 1]
\Big),
$$

trong đó số mũ giảm chấn $\eta = 1$ (MATLAB reference) hoặc $\eta = 1/2$
($\sqrt{\cdot}$, Sigmund 2001), $m$ là giới hạn bước (`move`), và
$x_{\min} = 0.001$ (sàn dương — tại $x_e = 0$ chính xác, độ nhạy
$\propto x_e^{p-1}$ triệt tiêu, khiến bản cập nhật nhân không thể thoát khỏi
trạng thái "hấp thụ" về 0). Hệ số $\lambda$ được tìm bằng **tìm kiếm nhị phân**
sao cho $\text{mean}(\tilde{\mathbf{x}}^{\text{new}}) = f$ (hoặc
$\text{mean}(\hat{\mathbf{x}}^{\text{new}}) = f$ nếu có projection — ràng buộc
nhắm vào trường vật lý thực dùng ở FE, không phải trường trung gian).

### 7.2. Ràng buộc độ cứng (chế độ gate)

Khi có ràng buộc cứng $Q_{11} \ge \delta$, $Q_{22} \ge \delta$, vòng nhị phân
thể tích được nối với cổng nhị phân $g = \mathbf{1}[Q_{11} \ge \delta \wedge
Q_{22} \ge \delta]$: chỉ khi ràng buộc độ cứng thỏa mãn thì thể tích mới được
phép tăng lên $\text{volfrac}$; nếu vi phạm, vật liệu được phân bổ lại để hồi
phục độ cứng. $Q$ được đánh giá tại $\mathbf{x}$ cũ của vòng lặp (xấp xỉ bậc 0,
chấp nhận được với bước $m$ nhỏ).

### 7.3. Hai hệ số Lagrange (chế độ dual, tùy chọn)

Mở rộng nhiều ràng buộc (Bruyneel & Duysinx 2005): thể tích và độ cứng
$g = \min(Q_{11}, Q_{22}) \ge \delta$ mỗi ràng buộc một multiplier:

$$
B_e(\mu_1, \mu_2) =
\frac{-(\partial c / \partial x_e)}
{\max\big(\mu_1\,\partial V/\partial x_e
- \mu_2\,\partial g/\partial x_e,\; \varepsilon\big)},
\qquad
x_e^{\text{new}} = \operatorname{clip}\!\big(x_e\, B_e^{\eta}\big).
$$

Ràng buộc độ cứng tại $\mathbf{x}^{\text{new}}$ được xấp xỉ tuyến tính bậc 1:

$$
g(\mathbf{x}^{\text{new}}) \approx g(\mathbf{x}) +
\sum_e \frac{\partial g}{\partial x_e}\,\big(x_e^{\text{new}} - x_e\big),
$$

dấu trừ ở mẫu số phản ánh chiều ngược nhau của hai ràng buộc: $\mu_1$ tăng thì
giảm vật liệu (kiểm soát thể tích), $\mu_2$ tăng thì **tăng** vật liệu ở nơi
$\partial g/\partial x_e$ lớn (hồi phục độ cứng). Vòng lặp lồng nhau: bisect ngoài
trên $\mu_1$ (khớp thể tích), bisect trong trên $\mu_2$ (khớp xấp xỉ độ cứng);
nếu ràng buộc không vi phạm tại $\mathbf{x}$ thì $\mu_2 = 0$ (complementary
slackness).

---

## 8. Tiêu chí hội tụ

Vòng lặp dừng khi một trong các điều kiện sau đạt:

1. **Thay đổi thiết kế nhỏ** — cửa sổ trượt:
   $$
   \Delta_k = \max_e \big|x_e^{(k)} - x_e^{(k-1)}\big| < \text{tol\_change}
   $$
   trong `window_change` vòng liên tiếp.
2. **Mục tiêu ổn định** — thay đổi tương đối:
   $$
   \frac{\big|c^{(k)} - c^{(k-1)}\big|}{\max(|c^{(k-1)}|, \varepsilon)}
   < \text{tol\_obj}
   $$
   trong `window_obj` vòng liên tiếp.
3. Đạt số vòng lặp tối đa `max_iter`.

---

## 9. Sơ đồ dòng chảy tổng thể

```
 ┌────────────────────────────────────────────────────────────┐
 │ 1. Khởi tạo: x ← seed (mật độ ban đầu)                     │
 │ 2. Dựng: k_e (mục 2.1), K, P (PBC), H/Hs (bộ lọc)          │
 └──────────────────────────┬─────────────────────────────────┘
                            ▼
   ┌──────────────  VÒNG LẶP TỐI ƯU (k = 1..max_iter)  ──────────────┐
   │                                                                  │
   │  ┌─ a) Nội suy SIMP: E_e = E_min + (ρ₀ x_e^p)(E_0 − E_min)      │
   │  │   → lắp ráp K(x), chiếu PBC: K_pbc = Pᵀ K P                  │
   │  │                                                               │
   │  ├─ b) FE: giải K_pbc χ⁽ⁱ⁾ = −Pᵀ K u⁰⁽ⁱ⁾  (3 case)             │
   │  │   → u⁽ⁱ⁾ = u⁰⁽ⁱ⁾ + χ⁽ⁱ⁾                                      │
   │  │                                                               │
   │  ├─ c) Đồng nhất hóa: Q_ij, ∂Q_ij/∂x_e  (mục 4)                │
   │  │                                                               │
   │  ├─ d) Mục tiêu: c(Q), dc/dx̂  (mục 5)                           │
   │  │                                                               │
   │  ├─ e) Chuỗi đạo hàm hợp + lọc độ nhạy → dc/dx (mục 6)          │
   │  │                                                               │
   │  ├─ f) OC update: bisect λ → x_new (mục 7)                     │
   │  │                                                               │
   │  ├─ g) Lọc mật độ / projection → x̃, x̂ (mục 6)                 │
   │  │                                                               │
   │  └─ h) Kiểm tra hội tụ (mục 8) → chưa hội tụ quay lại (a)      │
   │                                                                  │
   └─────────────── Kết thúc: trả x̂*, Q*, ν₁₂ = −S₁₂/S₁₁ ─────────────┘
```

---

## Kí hiệu

| Kí hiệu | Ý nghĩa | Tham số tương ứng |
|---------|---------|-------------------|
| $x_e$ | Mật độ thiết kế phần tử $e$ | `x` |
| $\tilde{x}_e$ | Mật độ sau bộ lọc | `x_tilde` |
| $\hat{x}_e$ | Mật độ vật lý (sau projection) | `xPhys` |
| $p$ | Hệ số phạt SIMP | `penal` |
| $E_0$, $E_{\min}$ | Mô đun Young vật liệu đặc / lỗ rỗng | `E0`, `Emin` |
| $\nu$ | Hệ số Poisson vật liệu nền | `nu` |
| $\rho_0$ | Hằng số tỉ lệ mật độ | `rho0` |
| $\mathbf{k}_e$, $\mathbf{K}$ | Độ cứng phần tử / toàn cục | `KE`, `K_global` |
| $\mathbf{P}$ | Phép chiếu PBC | `pbc` |
| $\mathbf{u}^0$, $\boldsymbol{\chi}$, $\mathbf{u}$ | Chuyển vị áp đặt / dao động / tổng | `U0`, `U`, `U_total` |
| $\mathbf{Q}$, $\mathbf{S}$ | Ten-xơ độ cứng / tuân thủ đồng nhất hóa | `Q`, `S = inv(Q)` |
| $f$ | Tỉ lệ thể tích mục tiêu | `volfrac` |
| $\delta$ | Ngưỡng độ cứng tối thiểu $= 0.1 f E_0$ | `delta` |
| $r_{\min}$ | Bán kính bộ lọc | `rmin` |
| $m$ | Giới hạn bước cập nhật OC | `move` |
| $\lambda$, $\mu_1$, $\mu_2$ | Hệ số Lagrange | — |
| $H_{ef}$, $H_s$ | Trọng số lọc / tổng trọng số | `H`, `Hs` |
| $\beta_{\text{proj}}$, $\eta$ | Độ dốc / ngưỡng chiếu Heaviside | `beta_proj`, `eta` |

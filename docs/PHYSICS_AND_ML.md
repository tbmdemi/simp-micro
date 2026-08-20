# Bản chất Toán học - Cơ học - Vật lý - Logic, và vai trò của ML/DL

> Tài liệu này giải thích **vì sao** pipeline được xây dựng theo cách nó đang có: nền tảng cơ học/vật lý của bài toán (SIMP, đồng nhất hóa, mục tiêu auxetic), và **vai trò cụ thể** của Machine Learning/Deep Learning trong từng giai đoạn - không phải mô tả API. Xem [README.md](../README.md) cho tổng quan dự án, [PIPELINE.md](PIPELINE.md) cho lệnh chạy từng phase, [LIMITATIONS.md](LIMITATIONS.md) cho giới hạn/claim khoa học.

---

## 1. Bài toán vật lý: vật liệu auxetic và tối ưu hóa topology

**Vật liệu auxetic** là vật liệu có **hệ số Poisson âm** (ν < 0): khi kéo dãn theo một phương, nó **nở ra** theo phương vuông góc thay vì co lại như vật liệu thông thường (ν > 0). Hành vi này không đến từ thành phần hóa học mà từ **hình học vi cấu trúc** (microstructure geometry) của ô đơn vị (unit cell) tuần hoàn - ví dụ hình lỗ nơ (bowtie), hình lục giác lõm (re-entrant hexagon). Bài toán trung tâm của dự án:

> Cho một vật liệu nền đẳng hướng thông thường (ν₀ > 0, ví dụ thép/nhựa), **tìm cách phân bố vật liệu/lỗ rỗng bên trong một ô đơn vị 2D** sao cho vật liệu đồng nhất hóa (homogenized material) có ν₁₂, ν₂₁ càng âm càng tốt.

Đây là bài toán **tối ưu hóa topology** (topology optimization) kinh điển, giải bằng phương pháp **SIMP** (Solid Isotropic Material with Penalization).

## 2. SIMP: từ biến liên tục đến topology rời rạc

Ý tưởng cốt lõi của SIMP: thay vì tối ưu hóa trực tiếp trên biến nhị phân "có vật liệu / không có vật liệu" (bài toán tổ hợp NP-khó), ta cho phép mỗi phần tử lưới FE có một **mật độ liên tục** `x_e ∈ [x_min, 1]` (`x_min = 0.001`, xem `simp/core/oc.py`), rồi phạt các giá trị trung gian để nghiệm hội tụ về gần 0/1:

```
E(x_e) = E_min + x_e^p · (E0 - E_min)          (nội suy SIMP, p = hệ số phạt "penal")
```

- `p > 1` (mặc định `penal=3.0`) làm vật liệu ở mật độ trung gian (x≈0.5) "không đáng giá" về mặt cơ học so với chi phí thể tích nó chiếm - đẩy nghiệm về 0 hoặc 1.
- `x_min = 0.001` (không phải 0 tuyệt đối) là **quyết định toán học có chủ đích**: độ nhạy `dQ/dx ∝ x^(p-1)` triệt tiêu chính xác tại `x=0`, khiến phần tử "chết" vĩnh viễn dưới luật cập nhật nhân (không có cách nào hồi phục dù ràng buộc khác yêu cầu) - đây từng là một bug thật đã fix (xem [LIMITATIONS.md](LIMITATIONS.md) mục 16).

**Vòng lặp tối ưu hóa** (xem sơ đồ trong README):

```
Seed Generation → FE Analysis → Homogenization → Objective & Sensitivity →
Density/Sensitivity Filtering → OC Update → Convergence Check → Repeat
```

1. **Seed Generation** (`simp/seeds/`): khởi tạo hình học ban đầu (11 loại - vòng tròn, lục giác, nơ, v.v.) làm điểm xuất phát cho tối ưu hóa cục bộ (local optimizer, không phải global search).
2. **FE Analysis** (`simp/core/fem.py`, `solver.py`): giải phương trình cân bằng `K(x)·U = F` bằng phương pháp phần tử hữu hạn (phần tử vuông 2D, tích phân Gauss), với 3 trường hợp tải đơn vị (unit strain cases) để trích ra đủ 3 hàng của tensor độ cứng 3×3.
3. **Homogenization** (`simp/homogenization/compute.py`) - xem mục 3.
4. **Objective & Sensitivity** (`simp/objectives/auxetic.py`) - xem mục 4.
5. **Density/Sensitivity Filtering** (`simp/core/filter.py`): lọc hình nón (cone filter) bán kính `rmin` để chống hiện tượng checkerboard (dao động lưới ảo, không phải vật lý thật) và đảm bảo tính khả chế tạo tối thiểu (độ dày nét ≥ rmin).
6. **OC Update** (`simp/core/oc.py`): thuật toán **Optimality Criteria** cổ điển (Bendsøe & Sigmund) - cập nhật `x_new` theo tỉ lệ `sqrt(-dc/dv/λ)` với tìm kiếm nhị phân trên hệ số Lagrange `λ` để thỏa ràng buộc thể tích `mean(x) = volfrac`, giới hạn bước di chuyển bởi `move`.
7. **Convergence Check** (`simp/core/convergence.py`): dừng khi thay đổi thiết kế < `tol_change`, hoặc mục tiêu ổn định trong `window_size` vòng lặp, hoặc chạm `max_iter`.

## 3. Đồng nhất hóa dựa trên năng lượng (Energy-based Homogenization)

Để biết một ô đơn vị hữu hạn, không đồng nhất (có lỗ rỗng) "tương đương" với vật liệu đồng nhất nào ở quy mô vĩ mô, ta dùng **lý thuyết đồng nhất hóa** (homogenization theory, Xia & Breitkopf 2015):

```
Q_ij = 1/|Ω| Σ_e (u_e^i)ᵀ · k_e · (u_e^j),   k_e = E(x_e)/E0 · KE
```

trong đó `u_e^i` là trường chuyển vị **tổng** (biến dạng đơn vị vĩ mô `u0` + dao động cục bộ `u_fluctuation` do vi cấu trúc gây ra) dưới trường hợp tải đơn vị thứ `i` (Andreassen et al. 2014, eq. 6). Điều kiện biên bắt buộc là **tuần hoàn** (periodic boundary conditions, PBC): chuyển vị trên biên trái phải khớp biên phải, biên trên khớp biên dưới (sai khác đúng bằng biến dạng vĩ mô áp vào) - triển khai bằng phương pháp **null-space projection** (`simp/core/pbc.py`): tìm cơ sở không gian null của ma trận ràng buộc PBC rồi giải FE trực tiếp trên không gian con đó (không cần nhân tử Lagrange).

Từ tensor độ cứng `Q` (3×3, ký hiệu Voigt `[11,22,12]`), suy ra hệ số Poisson **chính xác** bằng nghịch đảo ma trận đầy đủ `S = Q⁻¹`:

```
ν₁₂ = -S₁₂/S₁₁          ν₂₁ = -S₁₂/S₂₂
```

**Vì sao không dùng công thức tắt trực hướng `ν₁₂ = Q₁₂/Q₂₂`?** Công thức này chỉ đúng khi ô đơn vị **không bị xoay** (`Q₁₃ = Q₂₃ = 0`, không có coupling cắt-pháp). Dùng nhầm công thức tắt là nguyên nhân gốc của bug "0 mẫu auxetic" ở lần chạy Phase 1 đầu tiên (xem `EXPERIMENT_LOG.md`) - một ví dụ cụ thể cho thấy sai lệch toán học nhỏ (bỏ qua 1 giả định) có thể làm sập toàn bộ pipeline downstream.

## 4. Hàm mục tiêu auxetic và độ nhạy (sensitivity)

```
c = Q₁₂ − μ·(Q₁₁ + Q₂₂) + penalty(Q₁₁, Q₂₂; δ)
```

- **`Q₁₂` làm proxy cho `ν₁₂` âm**: vì `ν₁₂ = -S₁₂/S₁₁` và `S₁₁ > 0` luôn đúng (dương xác định), dấu của `ν₁₂` trùng dấu của `S₁₂`, và trong chế độ gần-trực-hướng `S₁₂ ≈ -Q₁₂/(Q₁₁Q₂₂)` nên minimize `Q₁₂` ⟺ đẩy `ν₁₂` xuống âm. Dùng `Q₁₂` (không phải `ν₁₂` trực tiếp) làm mục tiêu tối ưu vì nó là hàm **trơn, khả vi trực tiếp theo `x`** qua `dQ` (đạo hàm giải tích có sẵn từ homogenization) - còn `ν₁₂` cần thêm một bước nghịch đảo ma trận mới lấy được đạo hàm, tốn kém hơn trong vòng lặp OC chạy hàng nghìn lần.
- **`μ` (mặc định `0.0`, đang tắt)**: dự định tạo thêm áp lực đẩy `Q₁₂` xuống sâu hơn thay vì dừng ở gần 0, nhưng công thức hiện tại có sai sót khái niệm chưa thiết kế lại - xem [LIMITATIONS.md](LIMITATIONS.md) mục 4. Toàn bộ dataset production hiện có đều dùng `μ=0`.
- **`penalty`**: kích hoạt khi `Q₁₁` hoặc `Q₂₂` giảm dưới ngưỡng `δ = 0.1·volfrac·E₀`, phạt bậc 2 chuẩn hóa theo `δ²` - ngăn tối ưu hóa "ăn gian" bằng cách làm vật liệu quá mềm/sụp cấu trúc để đạt `Q₁₂` âm giả tạo (không có ý nghĩa cơ học/chế tạo thật).

**Sensitivity (đạo hàm `dc/dx`)** không tính bằng sai phân số hay autodiff, mà bằng **công thức giải tích trực tiếp** dựa trên tính **tự-adjoint** (self-adjoint) của bài toán năng lượng đàn hồi tuyến tính - đúng kiểu công thức sensitivity chuẩn trong 99-line/88-line SIMP kinh điển (Sigmund 2001; Andreassen et al. 2011). Đây là lý do pipeline chạy được hàng nghìn vòng lặp OC cho mỗi mẫu trong ngân sách thời gian hợp lý.

## 5. Từ 1 vi cấu trúc tối ưu đến bài toán thiết kế ngược (inverse design)

SIMP + OC ở trên giải quyết bài toán **thuận** (forward): cho trước 1 seed hình học + 1 bộ tham số (`volfrac`, `penal`, `rmin`, ...), tối ưu hóa cục bộ ra **một** vi cấu trúc auxetic. Đây **không phải** bài toán thiết kế ngược thật: không có cách nào nhắm thẳng tới một cặp `(ν₁₂, ν₂₁)` mục tiêu cụ thể bằng cách chỉnh tham số đầu vào một cách trực tiếp - phải chạy rất nhiều lần với tham số khác nhau rồi "trúng" gần đúng.

**Đây chính là chỗ ML/DL vào cuộc** - không phải để thay thế vật lý, mà để **học ánh xạ ngược** (target property → geometry) từ dữ liệu do chính engine vật lý SIMP sinh ra, đóng vai trò "hàm khuếch đại" biến hàng nghìn lần chạy SIMP (Phase 1-3, tốn hàng giờ/ngày) thành một mô hình suy luận tức thời (Phase 4-5).

## 6. Vai trò của Deep Learning trong pipeline

### 6.1. Phase 4 - CNN Surrogate: xấp xỉ vật lý khả vi, tốc độ cao

`SurrogateCNN` (4 khối Conv+BN+ReLU+MaxPool → global average pool → FC) học ánh xạ **thuận** `density field (64×64) → (ν₁₂, ν₂₁, volfrac)`, xấp xỉ lại chính thứ mà FE+homogenization tính ra (R²≈0,97-0,98 trên dataset v2). Vai trò của nó **không phải** để thay thế FE trong đánh giá cuối cùng (không đáng tin ở vùng ngoài phân phối train - xem [LIMITATIONS.md](LIMITATIONS.md)), mà để:

- Cung cấp một **hàm mất mát khả vi, rẻ** (`property_consistency_loss` trong `losses.py`) để huấn luyện Phase 5 - CNN thường/thuận thì `torch.autograd` lan truyền ngược qua nó gần như miễn phí, trong khi lan truyền ngược qua chính solver FE thật (giải hệ phương trình tuyến tính lớn) đắt hơn nhiều bậc.
- Đây là ví dụ kinh điển của **surrogate modeling**: một mạng nơ-ron học lại một hàm vật lý tốn kém để dùng nó ở nơi cần tính đạo hàm/lặp lại hàng vạn lần.

### 6.2. Phase 5 - Conditional VAE: mô hình sinh cho thiết kế ngược

`CVAE` (Encoder 4 khối Conv giữ nguyên feature map không gian + FC → `(μ, logσ²)`; Decoder đối xứng bằng `ConvTranspose2d`) học phân phối `p(density field | condition=[ν₁₂, ν₂₁])`. Đây là bài toán sinh có điều kiện (**conditional generative modeling**) kinh điển: latent `z ~ N(0,I)` + condition vector → decoder → ảnh mật độ. Loss:

```
L = BCE(reconstruction) + β·KL(q(z|x,c) ‖ N(0,I)) + γ·property_consistency_loss + regularizers
```

- **BCE reconstruction**: buộc decoder tái tạo đúng ảnh gốc (autoencoding thông thường).
- **KL term** (`β` tăng dần theo lịch "KL-warmup"): buộc không gian latent gần phân phối chuẩn, để có thể **sample** `z` ngẫu nhiên lúc suy luận (không chỉ encode lại ảnh có sẵn) - đây là điểm khác biệt cốt lõi giữa VAE và autoencoder thường, cho phép **sinh ra thiết kế mới** chưa từng có trong training set (novelty, [LIMITATIONS.md](LIMITATIONS.md) mục 18).
- **`γ`·property-consistency**: buộc ảnh sinh ra, khi đưa qua surrogate Phase 4, phải dự đoán ra đúng `condition` đã yêu cầu - đây là cơ chế "closed-loop" khiến mô hình sinh học cách **bám mục tiêu vật lý**, không chỉ tái tạo hình ảnh đẹp mắt.

### 6.3. "Surrogate exploitation" - khi ML tự đánh lừa chính nó, và cách vật lý sửa nó

Phát hiện quan trọng nhất của dự án (về mặt phương pháp luận ML): khi tăng `γ` để ép mô hình bám mục tiêu chặt hơn, decoder học được cách sinh ra ảnh khiến **surrogate CNN** dự đoán đúng property mong muốn - nhưng khi đem chính ảnh đó giải lại bằng **FE thật**, `R²` âm sâu (dự đoán sai hoàn toàn). Đây là hiện tượng kinh điển trong ML: một mạng được tối ưu hóa để đánh lừa một "giám khảo" cố định (surrogate đóng băng) sẽ tìm ra vùng không gian mà giám khảo đó có điểm mù, giống tấn công đối kháng (adversarial example) - ở đây giám khảo không bị tấn công chủ đích, mà bị khai thác như một tác dụng phụ của gradient descent.

Hai hướng khắc phục **thuần ML** (self-play adversarial retraining trong `self_play.py`, ensemble nhiều surrogate độc lập để tăng độ khó "đánh lừa đồng thời") **không giải quyết được tận gốc**. Giải pháp thật sự đến từ việc **đưa lại vật lý vào vòng lặp**: `real_physics.py` triển khai một lớp PyTorch tùy chỉnh (`autograd.Function`) chạy **FE-solve + homogenization thật** ngay trong training loop, dùng đúng công thức đạo hàm giải tích tự-adjoint đã có sẵn ở Phase 1-3 (không phải autodiff số qua solver, không cần adjoint solve riêng - `dQ/dx` từ `compute_homogenized_tensor()` đã đủ, chỉ cần lan truyền tiếp qua phép nghịch đảo ma trận 3×3 bằng quy tắc chuỗi giải tích: `dS/dx = -S·(dQ/dx)·S`). Kết quả: `R²(FE)` từ âm sâu lên ≈0,995-0,999 (n≈300-789) - **differentiable-physics** ở đây không phải một mẹo kỹ thuật, mà là hệ quả trực tiếp của việc tái sử dụng chính cấu trúc toán học (tự-adjoint) làm nền tảng cho cả forward SIMP lẫn backward training.

Đây là minh chứng cụ thể cho một nguyên tắc chung: **một surrogate/loss xấp xỉ chỉ đáng tin trong vùng nó được huấn luyện để xấp xỉ đúng** - khi optimizer (dù là OC cổ điển hay SGD của mạng nơ-ron) được thưởng bởi chính hàm xấp xỉ đó, nó sẽ tìm mọi cách khai thác sai số xấp xỉ nếu có đường đi tới đó; cách duy nhất để đóng đường đó lại là đưa chính "oracle" (ở đây là vật lý thật, khả vi) vào bên trong vòng lặp tối ưu, không chỉ dùng để đánh giá sau cùng.

### 6.4. Best-of-N: dùng FE thật làm bộ lọc cuối, không phải chỉ để "cho chắc"

Ngay cả với checkpoint differentiable-physics tốt nhất, `best_of_n_eval.py` vẫn sinh N ứng viên/điều kiện rồi để **FE thật** (không phải surrogate, không phải mô hình sinh tự đánh giá) chọn người thắng cuộc - lấy cảm hứng từ pipeline Deep-DRAM đã công bố (Pahlavani et al. 2024). Về logic: mô hình sinh (cVAE) đóng vai trò **đề xuất** (proposal distribution) hiệu quả về mặt tính toán so với random search/gradient-based search thuần túy trên không gian pixel, còn **quyết định cuối cùng luôn thuộc về vật lý xác định** (deterministic FE solve) - ML không bao giờ là trọng tài cuối cùng trong pipeline này, kể cả khi nó đã học rất tốt.

### 6.5. Vai trò ML ở các phần khác

- **Phase 1 (screening)**: không dùng ML - phân tích thống kê cổ điển (tương quan Spearman, LHS) để xác định tham số nào đáng quét kỹ ở Phase 2. ML chỉ vào cuộc **sau khi** đã có đủ hiểu biết thống kê về không gian tham số.
- **Phase 2 (adaptive DOE)**: không dùng mạng nơ-ron, dùng **ước lượng mật độ hạt nhân** (KDE) để phát hiện vùng thưa dữ liệu trong không gian thuộc tính - một kỹ thuật thống kê phi tham số, không phải deep learning, nhưng cùng họ "học từ dữ liệu" để quyết định lô tiếp theo nên lấy mẫu ở đâu (active learning theo nghĩa cổ điển, khác active learning ở Phase 7).
- **Phase 7 (active-learning loop)**: dùng đúng checkpoint cVAE hiện có để sinh mẫu mới, verify bằng FE thật, rồi fine-tune tiếp - về mặt lý thuyết đây là active learning cho mô hình sinh (không phải mô hình phân loại/hồi quy kinh điển). Kết quả thực nghiệm: **không cải thiện** thêm vì checkpoint đã gần mức trần hiệu năng khả dĩ với kiến trúc/dữ liệu hiện tại (xem [LIMITATIONS.md](LIMITATIONS.md)) - một kết quả âm tính có giá trị: cho thấy nút thắt hiện tại không phải "thiếu dữ liệu thêm ở vùng đã biết" mà có thể là giới hạn kiến trúc/mục tiêu proxy.

### 6.6. Vật liệu nền (ν0) như một trục thiết kế - cùng 1 lợi ích vật lý, hiện rõ khác nhau ở từng tầng ML (Giai đoạn A, 2026-08-18 → 2026-08-19)

Xuyên suốt mục 1-5, `nu` (ν0, hệ số Poisson của vật liệu **nền** đẳng hướng - khác `ν₁₂/ν₂₁` là kết quả homogenized của cả unit cell) luôn được giả định **cố định** (`0,3`, kiểu thép/nhựa điển hình). Giai đoạn A đặt lại câu hỏi: nếu vật liệu nền không cố định - unit cell auxetic cần thiết kế riêng cho từng vật liệu nền mục tiêu (composite, polymer khác, v.v.) - kiến trúc pipeline hiện tại có mở rộng được không, và lợi ích có đo được không?

**Hạ tầng cần đúng 1 thay đổi khái niệm, không phải kiến trúc mới:** `Material.__init__(E0, Emin, nu)` (`simp/materials/isotropic.py`) đã validate `nu` ở biên từ trước (khoảng hợp lệ `(-1, 0.5)`, đúng nguyên tắc "validate ở biên" của project) - phần thiếu chỉ là để `nu` **biến thiên per-sample** thay vì 1 hằng số toàn cục, xuyên suốt cả pipeline forward (Phase 1-3) lẫn training loop differentiable-physics (Phase 5). Điểm kỹ thuật đáng chú ý nhất: cache mesh trong `real_physics.py` trước đây gộp chung `(nelx, nely, E0, Emin, nu)` làm 1 key - tách đúng phần đắt (`edofMat`/PBC, chỉ phụ thuộc lưới) khỏi phần rẻ (dựng `Material`, ~97µs, ~0,1-0,2% chi phí 1 lần FE-solve) hóa ra không mất tác dụng tăng tốc như lo ngại ban đầu trong roadmap - một ví dụ nhỏ cho việc **đo thật** (`timeit`) thay vì suy diễn trước khi quyết định kiến trúc.

**Kết quả thực nghiệm phân kỳ giữa 2 tầng ML - điểm thú vị nhất của Giai đoạn A:**

| Tầng | Thêm ν0 làm input/condition | Lợi ích đo được? |
|---|---|---|
| Phase 4 - CNN Surrogate (`SurrogateCNN(include_nu0=True)`) | ν0 nối vào `fc_in` sau global-average-pool, cùng cấp seed one-hot | **Không** - Δv12 R²≈0,0000, Δv21 R²≈-0,0003 (trong nhiễu train-to-train), cùng dataset |
| Phase 5 - cVAE (`CVAEDataset(include_nu0=True)`, real-physics fine-tune) | ν0 thành 2 cột `[nu0, nu0_mask]` trong condition, dùng ĐÚNG per-sample khi tính `real_physics_loss` | **Có, nhất quán** - ΔR²(FE)=+0,0067 (0,9776→0,9843), Δfrac_manufacturable=+0,053 (0,312→0,365) |

Đây **không phải** một kết quả mâu thuẫn - nó củng cố đúng luận điểm ở mục 6.3 (surrogate exploitation): `property_consistency_loss` của surrogate là 1 xấp xỉ không hoàn hảo, còn `real_physics_loss` (differentiable-physics, mục 6.3) giải FE thật với đúng vật liệu per-sample mỗi bước gradient. Khi tín hiệu học "ν0 ảnh hưởng thế nào tới hình học tối ưu" đến từ vật lý thật (per-sample, chính xác), nó truyền đạt được cho decoder; khi đến từ 1 mạng nơ-ron xấp xỉ (surrogate, vốn cũng chỉ mới thấy ~16% mẫu có ν0 biến thiên trong dataset A4), tín hiệu quá yếu để đo được ở R². Nói cách khác: **thêm 1 trục vật lý mới không tự động cải thiện mọi tầng của pipeline ML - lợi ích chỉ chắc chắn xuất hiện ở tầng gần vật lý xác định nhất.**

Quy trình phát hiện lại nhấn mạnh 1 bài học đã có ở mục 6.3/6.4: hạ tầng nối `nu0_col` qua nhiều call site (`load_frozen_surrogate`, `property_consistency_loss`, `evaluate_density_field` trong `best_of_n_eval.py`) chỉ thực sự đúng khi có 1 lần chạy **end-to-end** với checkpoint `include_nu0=True` thật - unit test dùng surrogate/stub mặc định (`include_nu0=False`) không phơi ra được 3 bug chặn cứng đã tìm thấy ở A7 (chi tiết: [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md) mục 2026-08-19), trong đó nghiêm trọng nhất là verify FE dùng sai vật liệu (`nu=0,3` cố định thay vì đúng ν0 mục tiêu) - lỗi loại này không crash, chỉ âm thầm làm sai con số kết luận, nguy hiểm hơn lỗi crash rõ ràng.

## 7. Tổng kết: logic điều phối Vật lý ↔ ML xuyên suốt dự án

| Câu hỏi | Vật lý/Toán trả lời | ML/DL trả lời |
|---|---|---|
| Một hình học cụ thể có auxetic không, chính xác bao nhiêu? | FE + homogenization (nghịch đảo ma trận đầy đủ) - **luôn là nguồn sự thật cuối cùng** | Surrogate CNN chỉ xấp xỉ, không thay thế |
| Làm sao tối ưu **một** hình học từ 1 điểm khởi tạo? | SIMP + OC + sensitivity giải tích | Không cần ML |
| Làm sao đi ngược từ target `(ν₁₂,ν₂₁)` ra hình học **tức thời**, không cần chạy lại SIMP hàng nghìn vòng lặp? | Cung cấp dữ liệu training (Phase 1-3) + đạo hàm giải tích để huấn luyện đúng | cVAE (Phase 5) học ánh xạ ngược từ dữ liệu, `real_physics.py` giữ nó trung thực với vật lý trong lúc học |
| Ứng viên do mô hình sinh ra có đáng tin không? | FE thật verify lại (`best_of_n_eval.py`) | cVAE chỉ đề xuất (proposal), không tự quyết |
| Sao biết mô hình không chỉ "học thuộc lòng" tập train? | - | Novelty/diversity eval (`analysis/scripts/novelty_diversity_eval.py`) + so sánh baseline nearest-neighbor retrieval |

Nguyên tắc thiết kế xuyên suốt: **ML/DL được dùng ở đúng chỗ nó có lợi thế tuyệt đối (tốc độ suy luận, học ánh xạ ngược khó biểu diễn giải tích), nhưng không bao giờ được phép tự làm giám khảo cho chính nó** - mọi con số claim khoa học cuối cùng đều phải quy về nghiệm FE + homogenization xác định, độc lập kiểm chứng được (kể cả bằng một implementation FE hoàn toàn khác, `scikit-fem`, xem [LIMITATIONS.md](LIMITATIONS.md) mục 15).

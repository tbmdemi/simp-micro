# Nhật ký Thử nghiệm & Lỗi đã Sửa

Tài liệu này chỉ giữ lại **những phát hiện thực sự thay đổi kết quả dự án** - bug làm sai vật lý/kết quả trên diện rộng, hoặc đột phá phương pháp. Các tính năng thử nghiệm/pilot không được áp dụng, kết quả không đáng kể, hoặc việc vệ sinh dữ liệu thuần túy không được liệt kê ở đây - xem lịch sử git/`CHANGELOG.md` nếu cần. "Hệ thống hoạt động thế nào ở trạng thái hiện tại" thuộc về [`README.md`](README.md).

---

## 2026-08-24 - Roadmap modules MNO/KINN/ICKAN và benchmark GPU

Đã bổ sung các module độc lập cho ba task còn thiếu:

- `pipeline/mno/`: MNO nhận `images` và dự đoán đúng 18 trường dịch chuyển.
	Backend `mamba_ssm` được dùng khi cài được; môi trường `simp` hiện chưa có
	package này nên phép đo dùng fallback GRU hai chiều. Smoke benchmark GPU với
	lưới 16x16 cho output `(1,18,16,16)` trong khoảng 0,0004 s. Đây chỉ là đo
	forward, chưa phải nghiệm thu tốc độ production vì chưa có dataset FE chứa
	trường `displacements` 18 kênh.
- `pipeline/kinn/`: KINN dùng `EfficientKANLinear`, deep-energy loss lấy đạo
	hàm không gian bằng PyTorch autograd, và adapter solver tùy chọn JAX-CG/
	SciPy. Smoke test truyền gradient hữu hạn; `jax`/`jax-amg` chưa có trong
	môi trường nên chưa thể nghiệm thu KINN phi tuyến + AMG.
- `pipeline/ickans/`: constitutive energy model với trọng số dương và
	quadratic curvature floor. Hessian tangent smoke test cho eigenvalue nhỏ
	nhất khoảng 0,0246 > 0 trên 4 mẫu. Đây là nghiệm thu tính xác định dương,
	chưa phải KPI R2 >= 0,985 trên dữ liệu composite thực.

Benchmark các task Phase I hiện có:

- WIRE checkpoint `cvae_wire_v2.pt`: sinh 64x64 khoảng 0,09 s và 256x256
	khoảng 0,0045 s trong một lần đo GPU; output nằm trong `[0,1]`.
- WIRE best-of-N report hiện có: single-shot 4,17%, best-of-N 16,67%,
	R2(FE) = -10,29, nên chưa đạt `passes_all >= 75%`.
- ConvKAN report hiện có: R2 `[v12=0,9636, v21=0,9574, volfrac=0,9161]`,
	dưới KPI 0,985.
- 35 test hẹp cho Phase 4/5, tandem và roadmap modules pass; sau khi format
	Black, toàn bộ suite đạt 633 passed với 4 cảnh báo edge-case đã biết.

Sau đó KINN được nối thêm bằng `pipeline.phase5_cvae.losses.kinn_prior_loss()`;
hook này tạo lưới tọa độ cùng kích thước density và giữ gradient về đầu ra
generator. ICKAN sau khi đổi sang signed features `[x,-x]` đạt MSE giảm đều
trong 200 epoch nhưng R2 smoke = -0,069; eigenvalue tangent nhỏ nhất = 0,00102
vẫn dương. Vì vậy DoN R2 >= 0,985 chưa đạt, dù điều kiện ổn định vật lý đạt ở
smoke test.

Giới hạn tái lập: `base` không có PyTorch đầy đủ; mọi lệnh compute trên được
chạy bằng `conda activate simp`. Không đạt KPI thực nghiệm chỉ được ghi nhận,
không nâng thành claim khoa học.

## Bảng tổng hợp các bug lớn đã sửa

| Bug | Ảnh hưởng | Cách sửa |
|---|---|---|
| Chuyển vị FE trong đồng nhất hóa | Trường dao động χ dùng trực tiếp làm chuyển vị tổng thay vì `U0 + χ` - âm thầm sai tensor `Q` ở **mọi** lần chạy | `compute_homogenized_tensor()`: `U_total = U0 + U` |
| Công thức tắt trực hướng ν₁₂ khi xoay | `ν₁₂ = Q₁₂/Q₂₂` chỉ đúng khi `Q₁₃=Q₂₃=0`; sai ở các góc xoay → **0 mẫu auxetic** ở lần screening đầu tiên | Dùng nghịch đảo đầy đủ ma trận 3×3 (`S=Q⁻¹`) |
| **`dQ` (sensitivity đồng nhất hóa) bị hoán vị pixel** | `reshape()` mặc định `order='C'` thay vì `'F'` → tương đương TRANSPOSE sensitivity map cho lưới vuông. Vô hại với seed đối xứng, nhưng sai hướng gradient cho 3 seed bất đối xứng (`hourglass`,`hexagonal`,`reentrant_bowtie`) - 2/3 từng hội tụ **sai dấu hoàn toàn** | Thêm `order='F'`, kiểm chứng finite-difference + rebuild dataset |
| **Q bị sai scale (thiếu chia `E0`, thừa chia `nelx*nely`)** | `Q_cũ = Q_đúng × E0/nele` - không ảnh hưởng `ν₁₂/ν₂₁` (bất biến scale), nhưng khiến ngưỡng phạt stiffness `delta` trong objective hiệu lực **~46% thay vì 10% thiết kế** suốt lịch sử dự án | `k_e = E_penal/E0`, bỏ chia dư `/(nelx*nely)` - kiểm chứng bằng test "ô đặc hoàn toàn → Q=D" |
| **Ràng buộc cứng stiffness (Q₁₁,Q₂₂≥δ) trong `oc_update()` bị "mồ côi"** | Cơ chế khớp thiết kế tham chiếu MATLAB `topK_Hourglass.m`, có sẵn trong `oc_update()` nhưng `runner.py` chưa từng truyền `Q`/`delta` vào - vô hình suốt cả dự án, bị bug scale Q ở trên vô tình "che" tác dụng phụ (phạt mềm luôn kích hoạt quá mạnh thay thế cho nó) | Nối `Q=Q, delta=stiffness_delta(...)` vào lời gọi `oc_update()` |
| **Biến thiết kế `x` clip về sàn `0.0` thay vì `0.001` (Sigmund 2001)** | `dQ/dx ∝ x^(penal-1)` → tại `x=0` CHÍNH XÁC, độ nhạy=0 tuyệt đối; vì cập nhật OC là phép NHÂN, `x=0` là trạng thái hấp thụ vĩnh viễn không lối thoát, không liên quan độ mạnh/yếu ràng buộc nào. Gây sụp cấu trúc hoàn toàn (`vol→0`) trên `hexagonal`/`reentrant_bowtie` ở một số điều kiện | Hằng số `X_MIN = 0.001` khớp đúng reference |
| `converged=True` không đảm bảo hội tụ thật | Bật cả khi chỉ chạm `max_iter`, 76,3% dataset thuộc diện này | Cột `n_iters < max_iter` làm tiêu chí hội tụ thật |
| Nhãn "dao động" (limit-cycle) không bị lọc | OC rơi vào chu kỳ dao động cuối vòng lặp, ảnh lưu là 1 lát cắt ngẫu nhiên chưa ổn định - một số mẫu **sai cả dấu** auxetic | Metric `osc_score` đo trực tiếp trên lịch sử `Poisson_v12`, ngưỡng 0,15 |
| `val_loss` không an toàn để chọn checkpoint cVAE khi `gamma>0` | `beta` (KL) ramp tuyến tính khiến `val_loss` tăng cơ học, thiên vị chọn checkpoint SỚM chưa train đủ - tái hiện độc lập 2 lần | `--select-by fe_r2` (chọn theo R² FE thật đo định kỳ) |

---

## Đột phá & phát hiện lớn theo thời gian

### Surrogate exploitation (2026-07-22→23) → differentiable-physics (2026-07-24)

**Phát hiện:** R² qua surrogate CNN đóng băng tăng đơn điệu theo trọng số `gamma` (0,63→0,86), nhưng kiểm chứng bằng FE thật cho R² **âm sâu ở mọi mức** (−1,2 đến −2,4), khoảng cách nới rộng khi gamma tăng - decoder học đánh lừa surrogate thay vì sinh hình học đúng (**surrogate exploitation**). Hai biện pháp né tránh đều thất bại: self-play (fine-tune surrogate định kỳ) cho R² phẳng/nhiễu qua 8 vòng; ensemble surrogate không hội tụ trong ngân sách thử.

**Phát hiện phụ có giá trị ứng dụng cao:** `force_periodic()` - ép tuần hoàn bằng 1 phép gán pixel biên (không cần gradient/train lại) - tăng `passes_all` từ 1,7%→19,5% (>10×) với chi phí property gần như bằng 0 (mean |Δv12|=0,019). Giữ mặc định BẬT trong toàn bộ inference.

**Đột phá (2026-07-24):** `real_physics_prior_loss()` - FE-solve thật + đạo hàm giải tích trực tiếp trong training loop (không qua surrogate có thể bị đánh lừa) - chạy huấn luyện thật lần đầu. R²(FE) tăng dựng đứng qua epoch (0,11→0,41→0,92), kiểm chứng ở quy mô đầy đủ (n=789, toàn bộ test set):

| Chế độ | R²(FE) [95% CI] | hit rate single-shot |
|---|---|---|
| oracle (N=30) | **0,9988** [0,9985, 0,9990] | 98,7% |
| thực dụng (K=10) | **0,9920** [0,9899, 0,9937] | 98,7% |

`hit_rate_single_shot` nhảy từ ~35-40% (mọi checkpoint trước) lên 98,7%. Đây là lần đầu tiên surrogate-exploitation được sửa **tận gốc ở decoder**, không phải né ở inference-time. `cvae_realphysics.pt`/`cvae_v2_finetuned.pt` là checkpoint khuyến nghị hiện hành.

**Bài học phương pháp luận quan trọng (áp dụng cho mọi số liệu tương lai):** đánh giá lại ở quy mô lớn (n=24→300→789) cho thấy oracle mode giữ kết luận nhưng khiêm tốn hơn khi n tăng; chế độ "thực dụng" K=10 **sụp đổ về CI âm/cắt-0** ở quy mô đầy đủ cho MỌI checkpoint TRƯỚC differentiable-physics - khuyến nghị cũ dựa trên n=24 ("K=10 giữ gần hết R² gain") đã SAI, phải rút lại. Số liệu càng bé, càng dễ ngộ nhận.

**Việc CHƯA làm:** train from-scratch (chỉ mới fine-tune 2-stage, xác nhận BẮT BUỘC 2 giai đoạn - train thẳng real-physics từ epoch 1 thất bại, R²(FE) âm sâu suốt 60 epoch, xem mục dataset v2 dưới); đối chiếu FE bên ngoài codebase.

---

### Bug hoán vị `dQ` - rebuild dataset (2026-07-24)

Bug ở bảng trên gây hội tụ dưới tiềm năng thật cho 3 seed bất đối xứng - kiểm chứng ở 4 quy mô tăng dần (1 điểm/seed → 4 combo LHS → pilot 90 mẫu → rebuild đầy đủ 2.160 mẫu) trước khi tích hợp, tránh kết luận vội trên nhiễu. Kết quả rebuild: `hourglass` v12 trung bình −0,625 (cũ −0,115, ×5,4), `hexagonal` −0,349 (×4,1), `reentrant_bowtie` −0,172 (×11,4). **Tác động toàn dataset (7.920 mẫu): auxetic rate 82,1%→91,9%.**

Cùng đợt: audit phát hiện `converged=True` không đáng tin (chỉ 23,7% hội tụ thật) và 23,9% dataset hình học rời rạc; dọn 33,6% mẫu nhãn dao động (limit-cycle, phát hiện từ `osc_score`, gốc rễ là OC dao động qua lại giữa 2 giá trị mỗi vòng lặp cuối - ảnh lưu là 1 lát cắt ngẫu nhiên, một số **sai cả dấu** auxetic). Ablation 3-seed độc lập sau đó xác nhận phần lớn gain R² Phase 4 (0,48→0,92) đến từ bản vá `dQ`, KHÔNG phải bộ lọc `osc_score` này - vẫn giữ bộ lọc vì giá trị vệ sinh nhãn độc lập với R².

---

### Dataset v2 - bug `beta=0,8` ẩn (2026-07-25)

Sinh dataset v2 mở rộng, `reentrant_bowtie` sụp về nghiệm suy biến gần như ngay lập tức (13 vòng, v12≈0, 100% rời rạc). Root-cause: script sinh dữ liệu hardcode `beta=0,8` (không rõ nguồn gốc) suốt cả phiên trước, trong khi **production pipeline gốc chưa bao giờ set `beta`** → luôn chạy mặc định `beta=1,0` của `run_simp()`. Xác nhận trực tiếp: `beta=0,8` → kẹt 13 vòng; `beta=1,0` → chạy đủ 150 vòng, v12=-0,063 (âm thật). Bài học: một tham số hardcode sai trong script phụ có thể âm thầm khác hẳn hành vi production thật mà không ai nhận ra cho tới khi so sánh trực tiếp.

Kết quả cuối: dataset v2 = 57.216 mẫu train (production hiện hành). `surrogate_v2.pt` R²(v12/v21/volfrac) = 0,974/0,964/0,983. Yield sạch sau rebuild: `hourglass`=95,5%, `hexagonal`=96,4%, `reentrant_bowtie`=42,4% (dưới mốc lịch sử 77,2% - xem phần "vẫn treo" ở cuối tài liệu này và các ghi chú pilot liên quan trong chính log này cho một hướng khắc phục đã kiểm chứng nhưng chưa áp dụng).

*(Lưu ý vận hành: batch raw đầu tiên ~10.700 mẫu/~3,5h compute từng bị MẤT vì ghi vào `/tmp` (tmpfs, xóa qua reboot) - từ đó mọi output cần giữ ghi vào `outputs/`, không dùng `/tmp`.)*

---

### 2026-07-29 - Bug scaling `Q` (A1) làm lộ ra bug thứ hai nghiêm trọng hơn: sụp cấu trúc vĩnh viễn

**A1 - bug scaling Q** (xem bảng tổng hợp): `Q_cũ = Q_đúng × E0/nele` - đã kiểm chứng bằng test chuẩn "ô đặc hoàn toàn → Q=D" (rtol<1e-6). Không ảnh hưởng nhãn `ν₁₂/ν₂₁` của dataset hiện có (bất biến scale), nhưng khiến ngưỡng phạt stiffness `delta=0,1·volfrac·E0` (so trực tiếp với Q, KHÔNG bất biến scale) hiệu lực **~46% thay vì 10% thiết kế** suốt lịch sử dự án - phạt mềm vô tình đóng vai trò lưới bảo vệ chống sụp cấu trúc.

**Sau khi sửa A1 đúng scale, bug thứ hai lộ ra:** `hexagonal`/`reentrant_bowtie` sụp cấu trúc HOÀN TOÀN (`vol→0,004-0,006`, không còn vật liệu) ở một số điều kiện trong dải volfrac production. Truy đến 2 lớp nguyên nhân độc lập:

1. **Ràng buộc cứng bị "mồ côi"** (xem bảng tổng hợp) - nối lại giúp một phần nhưng CHƯA đủ.
2. **`X_MIN=0,0` thay vì `0,001`** (xem bảng tổng hợp) - nguyên nhân gốc rễ thật sự. Trace trực tiếp: `Q11/Q22` rơi từ ~175 (khỏe mạnh) xuống ĐÚNG 0,000 ở vòng 11 và đứng yên suốt 139 vòng còn lại dù ràng buộc cứng báo vi phạm mọi vòng - xác nhận đây là trạng thái hấp thụ toán học, không phải vấn đề độ mạnh ràng buộc.

**Kết quả sau khi sửa CẢ HAI:** tỷ lệ sụp cấu trúc trên mẫu thử **2/12 (16,7%) → 0/400** (đo lại ở quy mô lớn hơn nhiều). `reentrant_bowtie` yield **77,2% (lịch sử) → 87,0%** - giải quyết dứt điểm câu hỏi mở tồn đọng từ 2026-07-25. `hourglass` không bị ảnh hưởng. 407 test (bao gồm 2 test hồi quy mới cho từng cơ chế) pass.

**`hexagonal` còn khoảng hở đã CHẤP NHẬN** (không phải bug chưa sửa): yield 93,9%→80,5% (xác nhận thật ở n=400, CI 95% không chồng lấn lịch sử). Đã thử 3 hướng khắc phục, **CẢ 3 đều phản tác dụng hoặc vô ích**:
1. Tăng `delta` (10%→25%) - sạch giảm còn 7,5-37,5% (bùng nổ dao động).
2. Tăng `beta` (1,0→4,0) - sạch giảm còn 0-10% (bộ tối ưu bỏ hẳn mục tiêu auxetic).
3. `use_sqrt` (damping η=0,5) + `penal_init=2,0` (continuation nhẹ) - kỹ thuật đã kiểm chứng giúp `reentrant_bowtie` gần gấp đôi yield trên code CŨ (pilot được ghi ở các mục liên quan trong log này) - thử lại trên `hexagonal` với code đã fix hôm nay: **cũng làm tệ hơn** (82,5%→70-75%, driven bởi rời rạc tăng 3→10-11 và `hit_cap` tăng mạnh - continuation cần nhiều vòng lặp hơn để hoàn thành, 150 vòng không đủ ngân sách cho seed này).

Xác nhận cấu hình mặc định (`delta=10%, beta=1,0`, không damping/continuation) đã là lựa chọn tốt nhất trong mọi hướng đã thử. Nguyên nhân nhiều khả năng: bug A1 từng vô tình làm "regularizer" cho riêng seed mỏng/khó nhất này suốt lịch sử dự án; sau khi sửa đúng, không có cách chỉnh tham số đơn giản nào phục hồi lại mức ổn định đó trong ngân sách tính toán hiện tại (150 vòng lặp) - hạn chế thật của thiết kế, không phải lỗi chưa sửa. Xem [README.md](README.md) mục Giới hạn Đã biết #16.

**Ý nghĩa:** dataset 57k mẫu sản xuất hiện có sinh bằng code CŨ (trước các fix này) - KHÔNG bị ảnh hưởng, không cần rebuild. Nhưng bất kỳ lần sinh dữ liệu MỚI nào dùng code hiện tại đều cần các fix này (mặc định, không phải flag thử nghiệm) để tránh hỏng nặng trên `hexagonal`.

---

### 2026-07-30 - Hướng thứ 4 đã thử cho `hexagonal`: OC hai-multiplier (dual-constraint) - THẤT BẠI, đã tài liệu hóa nguyên nhân

**Giả thuyết ban đầu:** cơ chế `stiff_ok` hiện tại (`oc_update()`, `stiffness_mode='gate'` - xem `simp/core/oc.py`) AND một điều kiện nhị phân vào bisection thể tích DUY NHẤT. Vì `Q` được evaluate MỘT LẦN tại `x` cũ và giữ NGUYÊN suốt cả 100 vòng bisection nhị phân, khi vi phạm (`stiff_ok=False`) điều kiện `vol>volfrac and stiff_ok` luôn False bất kể `lmid` - bisection sụp về `lmid->0`, TOÀN BỘ ràng buộc thể tích bị bỏ qua trong vòng lặp đó. Giả thuyết: đây là nguyên nhân khiến tăng `delta`/`beta` ở trên gây dao động bùng nổ (tăng `delta` → `stiff_ok=False` xảy ra thường xuyên hơn → cơ chế "bỏ ràng buộc thể tích" kích hoạt thường xuyên hơn).

**Đã implement `oc_update_dual()`** (nhánh `OC-2-multilayer`, xem `simp/core/oc.py`): multiplier RIÊNG cho volume (`mu1`) và stiffness (`mu2`), nested bisection (outer `mu1` khớp volume, inner `mu2` khớp xấp xỉ tuyến tính bậc 1 của `Q11`), theo đúng công thức OC nhiều ràng buộc kinh điển (Bruyneel & Duysinx 2005). 5 unit test mới + 411/411 test cũ pass, smoke-test tích hợp qua `run_simp()` thật chạy thông.

**Pilot N=100/config (bắt cặp, hexagonal, dải hẹp lịch sử 0,50-0,58/0,28-0,38) - KẾT QUẢ TỆ HƠN HẲN, không phải cải thiện:**

| Config | Yield sạch | 95% CI | Suy biến (vol→0) | Dao động |
|---|---|---|---|---|
| `gate` (baseline) | 78,0% | [68,9%, 85,0%] | 0,0% | 13,0% |
| `dual` (bản đầu, chưa cắt target) | **18,0%** | [11,7%, 26,7%] | **64,0%** | 74,0% |

`gate` khớp mốc lịch sử 80,5% (CI chồng lấn) → pilot setup đáng tin.

**Nguyên nhân gốc #1 (đã trace bằng debug trực tiếp trên 1 mẫu suy biến thật):** mục tiêu tuyến tính hóa của ràng buộc stiffness (`slack0 = delta - Q11`) có thể VƯỢT XA mức đạt được trong 1 vòng move-limit (`max_khả_thi ≈ sum(max(dQ11/dx,0)) * move`) - đo được thật: `slack0=9,05` nhưng `max_khả_thi=4,21` (chưa bằng nửa). Inner bisection cho `mu2` đuổi theo mục tiêu BẤT KHẢ THI, bị đẩy tới trần `mu2_max=1e9`; với `mu2` khổng lồ, `denom = mu1*dv - mu2*dg` ÂM ở gần như MỌI phần tử (`dQ11/dx > 0` ở 2498/2500 phần tử - trường độ nhạy RỘNG KHẮP, không cục bộ như giả định ban đầu), công thức suy biến thành "chỉ theo dấu `dc`" - XÓA SỔ hoàn toàn ảnh hưởng của `mu1`, outer bisection mất quyền kiểm soát thể tích.

**Fix #1 đã thử:** cắt mục tiêu về mức khả thi trước khi bisect (`slack_target = min(slack0, 0,9*max_khả_thi)`) - đảm bảo tồn tại 1 `mu2` HỮU HẠN thỏa mãn xấp xỉ. **VẪN THẤT BẠI** trên cùng mẫu debug (vol vẫn sụp về 0,001 trong 15 vòng) - vì `max_khả_thi` bản thân đã yêu cầu gần như TOÀN BỘ domain nhảy lên `x+move` (trường `dQ11/dx` rộng khắp, không thưa), nên target dù đã cắt còn 90% vẫn tái hiện gần hệt cơ chế suy biến cũ.

**Nguyên nhân gốc #2 (so sánh trực tiếp trace `gate` vs `dual` trên cùng mẫu):** cả hai chế độ đều trải qua giai đoạn thể tích undershoot mạnh ở vòng 1-5 (0,925→0,74→0,555→0,371→0,187 - HÀNH VI BÌNH THƯỜNG của OC move-limited, giống hệt nhau ở cả 2 chế độ vì ràng buộc chưa active). Khác biệt xảy ra từ vòng 5-6: `gate` HỒI PHỤC (vol 0,222→...→0,529 ở vòng 10, hội tụ êm từ vòng 13), còn `dual` tiếp tục SỤP (0,214→0,106→0,009→0,002...) và bị KẸT VĨNH VIỄN ở sàn `X_MIN` (trạng thái hấp thụ toán học đã biết, xem docstring `X_MIN` trong `oc.py` - `dQ/dx∝x^(penal-1)→0` khi `x→X_MIN`). Nghi ngờ: mẫu mật độ sinh ra bởi "flood theo dấu `dc`" của `dual` (kết hợp với trường `dg` gần-đồng-nhất) tạo ra phân bố RỜI RẠC/KHÔNG LIÊN THÔNG khiến `Q11` đồng nhất hóa sụp về ~0 dù thể tích trung bình (0,214) chưa hẳn quá thấp - CHƯA xác nhận đầy đủ (cần audit hình học trực tiếp, chưa làm do giới hạn thời gian phiên này).

**Kết luận:** dừng hướng OC hai-multiplier bản nested-bisection thô ở đây - đã fix được 1 lớp lỗi (target bất khả thi) nhưng lộ ra lớp lỗi sâu hơn (trường độ nhạy stiffness rộng khắp domain, không cục bộ, khiến 2 multiplier tranh chấp gần như toàn bộ design freedom thay vì bổ trợ nhau). Đây là đúng loại vấn đề mà OC bisection thủ công KHÔNG được thiết kế để xử lý bền vững - literature dùng MMA (Method of Moving Asymptotes, Svanberg 1987) chính vì moving-asymptote cung cấp cơ chế trust-region ngầm mà bisection multiplier trần trụi không có. Code giữ trên nhánh `OC-2-multilayer` (KHÔNG merge, KHÔNG đổi default `stiffness_mode='gate'`) làm tài liệu tham khảo, tránh ai thử lại mù quáng cùng hướng.

---

### 2026-07-30 - ĐỘT PHÁ: MMA (nlopt.LD_MMA) thay OC cho `hexagonal` - 78,0%→**100,0%** yield sạch, đã kiểm chứng

Follow-on trực tiếp của mục thất bại ngay phía trên (OC hai-multiplier tự viết) - kết luận ở đó là bisection multiplier thủ công không có cơ chế trust-region để xử lý ràng buộc stiffness một cách bền vững, khuyến nghị đầu tư vào MMA thật. Đã triển khai ngay trong cùng phiên.

**Implement:** `simp/mma_runner.py::run_simp_mma()` - dùng `nlopt.LD_MMA` (package mới, thêm vào `requirements.txt`/`pyproject.toml`) thay hoàn toàn cho `oc_update()`. Khác biệt kiến trúc quan trọng: `oc_update()` được gọi 1 LẦN MỖI vòng lặp SIMP (runner.py tự quản `while` loop, tự giải FE rồi gọi OC lấy `xnew`); `nlopt.LD_MMA` NGƯỢC LẠI - tự quản TOÀN BỘ vòng lặp bên trong 1 lệnh `opt.optimize()`, gọi lại callback (tự giải FE+homogenization THẬT tại mỗi điểm) nhiều lần tới khi hội tụ. 2 ràng buộc stiffness (`Q11≥δ`, `Q22≥δ`) truyền RIÊNG BIỆT cho nlopt - không cần gộp thành "thành phần chặt hơn" như `oc_update_dual()` phải làm, MMA tự xử lý đồng thời đúng kiểu KKT. TÍNH NĂNG THỬ NGHIỆM: chỉ hỗ trợ `ft=2`, chưa hỗ trợ Heaviside projection, `penal` cố định (không continuation - không có "vòng lặp SIMP" ngoài để continuation theo).

**Bug bắt được khi tích hợp:** seed thô có pixel `=0.0` tuyệt đối (vùng void) - dưới sàn `X_MIN=0.001` - `nlopt.optimize()` raise `invalid_argument` vì điểm khởi tạo vi phạm bound (khác `oc_update()`, chỉ áp `X_MIN` cho `xnew` SAU vòng đầu, không đòi hỏi `x0` hợp lệ). Fix: `np.clip(x0, X_MIN, 1.0)` trước khi đưa vào nlopt - có test hồi quy (`test_run_simp_mma_respects_bounds`).

**Smoke-test xác nhận trước khi pilot:** chạy trực tiếp trên ĐÚNG mẫu (`volfrac=0,5243`, `void_size_frac=0,3325`) đã làm `oc_update_dual` sụp cấu trúc vĩnh viễn (`vol→0,001`) ở mục trên - MMA hội tụ THẬT (62 evals, `converged=True`), `volfrac_achieved=0,5243` (khớp mục tiêu gần tuyệt đối), `v12=-0,714` (auxetic mạnh), không dao động, không sụp.

**Pilot N=100/config (bắt cặp, cùng dải hẹp lịch sử 0,50-0,58/0,28-0,38, cùng tiêu chí "yield sạch"):**

| Config | Yield sạch | 95% CI | Suy biến | Dao động | Rời rạc | `hit_cap` | \|vol_đạt−mục tiêu\| trung bình | Thời gian/mẫu |
|---|---|---|---|---|---|---|---|---|
| `gate` (OC, baseline) | 78,0% | [68,9%, 85,0%] | 0,0% | 13,0% | 12,0% | 70,0% | 0,80% | 19,0s |
| **`mma` (nlopt.LD_MMA)** | **100,0%** | **[96,3%, 100,0%]** | **0,0%** | **0,0%** | **0,0%** | **12,0%** | **0,0017%** | **13,2s** |

95% CI của `mma` **không hề chồng lấn** CI của `gate` - khác biệt có ý nghĩa thống kê rõ ràng, không phải nhiễu mẫu nhỏ. `v12` của `mma` LUÔN âm trên cả 100 mẫu (min −1,09, max −0,66 - kể cả mẫu "tệ nhất" vẫn auxetic mạnh), so với `gate` có mẫu `v12=+0,08` (sai dấu). `hit_cap` (chạm trần `max_iter=150` không có nghĩa hội tụ thật, xem bảng bug ở đầu file) giảm từ 70,0%→12,0% - `mma` hội tụ THẬT ở phần lớn mẫu, không chỉ "chạy hết ngân sách". Nhanh hơn baseline 30% dù giải FE+homogenization thật ở MỌI callback (không cận approximation như `oc_update()`).

**Tại sao MMA thắng nơi OC hai-multiplier thất bại:** cùng bản chất vấn đề (2 ràng buộc non-linear tổng quát, trường độ nhạy stiffness rộng khắp domain) - nhưng MMA dùng "moving asymptotes" (biên xấp xỉ lồi tự thích ứng theo lịch sử hội tụ: mở rộng nếu dao động, thu hẹp nếu đơn điệu) cung cấp cơ chế trust-region NGẦM mà bisection nested multiplier tự viết hoàn toàn thiếu - đúng như tài liệu Svanberg 1987 thiết kế để giải.

**Giới hạn đã biết / CHƯA làm** (do phạm vi phiên này):
1. Chỉ kiểm chứng trên seed `hexagonal`, dải hẹp lịch sử, N=100 - CHƯA test `hourglass`/`reentrant_bowtie` (đang tốt sẵn với `gate`, cần xác nhận `mma` không làm tệ hơn trước khi cân nhắc đổi default toàn cục) hoặc dải volfrac rộng của `hourglass`.
2. CHƯA tích hợp vào `analysis/scripts/generate_production_batch.py` (script sinh dataset production vẫn gọi `simp.runner.run_simp` trực tiếp).
3. CHƯA hỗ trợ Heaviside projection, `move_min`/`penal_init` continuation (không áp dụng được với kiến trúc nlopt quản lý toàn bộ vòng lặp).
4. Ràng buộc volume trong `mma` là BẤT ĐẲNG THỨC (`mean(xPhys)<=volfrac`), không phải bisection nhắm CHÍNH XÁC như `gate` - thực nghiệm cho thấy tại optimum gần như luôn dùng hết ngân sách (sai số trung bình 0,0017%), nhưng đây là khác biệt phương pháp luận cần lưu ý nếu volfrac_achieved là tiêu chí quan trọng cho use-case khác.
5. Dataset 57k mẫu sản xuất hiện có sinh bằng `gate`/OC - KHÔNG bị ảnh hưởng bởi thay đổi này (chưa đổi bất kỳ default nào).

Code: `simp/mma_runner.py` (mới), `requirements.txt`/`pyproject.toml` (+`nlopt`), `tests/test_mma_runner.py` (5 test). Đã commit vào `main` (nhánh gốc `OC-2-multilayer` không còn tồn tại). Script pilot một lần `analysis/scripts/pilot_mma.py` đã xóa 2026-08-15 sau khi kết quả được ghi đầy đủ vào log này.

---

### 2026-07-30 (tiếp) - Mở rộng kiểm chứng MMA sang `hourglass`/`reentrant_bowtie`: KHÔNG phải chiến thắng toàn cục - `reentrant_bowtie` thất bại nặng với MMA

Theo đúng khuyến nghị "chưa vội đổi default toàn cục" ở mục trên - pilot N=100/config, bắt cặp, dùng ĐÚNG dải tham số production của từng seed (`hourglass`: dải RỘNG 0,3-0,7/0,1-0,4; `reentrant_bowtie`: dải hẹp 0,50-0,58/0,28-0,38, giống `hexagonal`).

| Seed | `gate` (baseline) | `mma` | Kết luận |
|---|---|---|---|
| `hexagonal` | 78,0% [68,9%,85,0%] | **100,0%** [96,3%,100,0%] | Thắng rõ, CI không chồng lấn (xem mục trên) |
| `hourglass` | 80,0% [71,1%,86,7%] | **87,0%** [79,0%,92,2%] | Cải thiện, CI chồng lấn 1 phần (không mạnh bằng hexagonal) nhưng KHÔNG regression; rời rạc 19%→**0%**, dao động 14%→3% |
| `reentrant_bowtie` | **90,0%** [82,6%,94,5%] | **0,0%** [0,0%,3,7%] | **THẤT BẠI NẶNG** - CI không chồng lấn, nhưng lần này `mma` TỆ HƠN HẲN |

**Đã trace nguyên nhân `reentrant_bowtie` thất bại (không phải ngẫu nhiên - lặp lại y hệt trên nhiều mẫu độc lập, xem `outputs/pilot_mma_reentrant_bowtie/simp_runs/mma_*/`):** lịch sử hội tụ MMA cho THẤY một mẫu hình cực kỳ nhất quán - 4 bước đầu (eval 1-4) đi theo quỹ đạo hợp lý (volume tăng dần êm từ giá trị seed ~0,19-0,21), rồi **bước thứ 5 luôn nhảy vọt sai hướng** (`v12` đổi dấu dương +0,38-0,40, volume nhảy lên ~0,46-0,47), tiếp theo **bước thứ 6 bùng nổ lên gần đặc hoàn toàn** (`vol≈0,96-0,98` - GẤP ĐÔI mục tiêu thật ~0,5-0,58!), rồi kẹt ở vùng gần-đặc đó suốt phần còn lại ngân sách (không bao giờ phục hồi về tốt hơn điểm ở bước 5). `nlopt` đúng là trả về điểm TỐT NHẤT nó tìm được (bước 5), nhưng bước 5 TỰ NÓ đã là 1 optimum cục bộ sai dấu (`v12>0`, không auxetic) - `mma` với quỹ đạo "mượt"/gradient-thuần đã tìm THẲNG tới đáy cục bộ gần seed (không auxetic) mà `gate`/OC (nhờ chính sự "không mượt" - dao động, move-limit, đôi khi cả những sai số approximation của nó) tình cờ có khả năng thoát khỏi để tới đáy auxetic sâu hơn ở seed khó này, đạt 90% ổn định qua nhiều năm.

**Đã thử 1 fix nhanh:** `opt.set_initial_step()` (nlopt hỗ trợ giới hạn bước đầu cho thuật toán gradient-based, tương tự khái niệm `move` của OC) - thêm làm param tùy chọn `mma_initial_step` (mặc định `None`=tắt, không đổi hành vi). Test đơn lẻ: `step=0,2` VÀ `step=0,05` cho `v12` âm mạnh (-1,03/-0,95) trên mẫu đã thất bại, nhưng `step=0,1` VÀ `step=0,02` vẫn thất bại - **không đơn điệu theo step, không phải fix đáng tin**. Pilot xác nhận N=30 với `step=0,2`: **33,3% sạch (10/30)** - cải thiện thật so với 0,0% mặc định, nhưng CÒN XA dưới `gate` (90,0%) - kiểu thất bại lưỡng cực rõ (`v12` hoặc mạnh âm ~-0,8 đến -1,1, hoặc mạnh dương ~+0,24 đến +0,34, gần như không có vùng giữa) - `set_initial_step` chỉ giảm xác suất rơi vào đáy sai, không loại bỏ được nó.

**Kết luận:** giữ nguyên `mma_initial_step` như 1 tham số tùy chọn (mặc định tắt, không hại) trong `simp/mma_runner.py` để tham khảo cho hướng tương lai, nhưng **KHÔNG khuyến nghị dùng `mma` cho `reentrant_bowtie`** ở dạng hiện tại dưới bất kỳ cấu hình nào đã thử - `gate`/OC vẫn là lựa chọn tốt nhất đã biết cho seed này (90,0%, mốc mới nhất, tốt hơn cả 87,0% ghi nhận trước đó). **Quyết định: dùng optimizer KHÁC NHAU theo từng seed** (`mma` cho `hexagonal`/`hourglass`, `gate` cho `reentrant_bowtie`), không phải 1 lựa chọn toàn cục - xem `analysis/scripts/generate_production_batch.py` (đã nối `SEED_OPTIMIZER` dispatch, CHƯA kích hoạt chạy sinh dataset thật, đó là quyết định riêng cần xác nhận).

**Bài học phương pháp luận:** kiểm chứng "không regression" trên TẤT CẢ seed liên quan trước khi tổng quát hóa 1 kết quả tốt - nếu chỉ dựa vào kết quả `hexagonal` (100,0%!) mà vội đổi default toàn cục, sẽ vô tình phá hỏng `reentrant_bowtie` (90,0%→0,0%, tệ hơn CẢ 3 hướng phá hoại đã thử trước đó cho `hexagonal`). Optimizer "tốt hơn về mặt lý thuyết" (MMA hội tụ đúng KKT, có trust-region) không tự động tốt hơn trong THỰC HÀNH cho mọi bài toán non-convex - đôi khi chính sự "không hoàn hảo" của 1 heuristic cũ (dao động, move-limit) lại có tác dụng khám phá không gian nghiệm tình cờ hữu ích mà 1 phương pháp "sạch" hơn về toán học không có.

---

### 2026-07-30 (tiếp) - Xác nhận N=400 cho `hexagonal`/`hourglass`: kết quả N=100 KHÔNG phải may mắn mẫu nhỏ

Theo đúng chuẩn nghiêm ngặt dự án đã dùng (báo cáo pilot 2026-07-26, script/report một lần đã xóa sau khi ghi kết quả vào log này - quy ước chung của dự án cho pilot ngoài production, xem `LIMITATIONS.md` mục 4: luôn lên N=400 trước khi cân nhắc đổi default) - nhân đôi rồi gấp 4 lần N cho 2 seed đã thắng/cải thiện ở pilot N=100. Bỏ qua `reentrant_bowtie` - N=100 đã cho CI cực hẹp [0,0%,3,7%], scale thêm không đổi kết luận, lãng phí compute.

| Seed | `gate` N=400 | `mma` N=400 | So với N=100 |
|---|---|---|---|
| `hexagonal` | 76,5% [72,1%,80,4%] | **99,8%** [98,6%,100,0%] | Khớp N=100 (78,0%/100,0%), CI hẹp hơn, vẫn không chồng lấn |
| `hourglass` | 76,8% [72,4%,80,6%] | **88,0%** [84,4%,90,8%] | Khớp N=100 (80,0%/87,0%) - **CI giờ KHÔNG chồng lấn** (N=100 còn chồng lấn 1 phần: gate hi=86,7% vs mma lo=79,0%) |

**Kết luận:** cả 2 kết quả N=100 đều được xác nhận đầy đủ ở N=400, không phải nhiễu mẫu nhỏ - `hourglass` đặc biệt đáng chú ý vì N=100 chưa đủ mạnh để khẳng định chắc (CI chồng lấn), N=400 mới thực sự khóa chặt kết luận "MMA cải thiện thật, không regression". `SEED_OPTIMIZER` dispatch trong `generate_production_batch.py` (mma cho hexagonal/hourglass, gate cho reentrant_bowtie) giờ có bằng chứng thống kê đầy đủ ở cả 2 quy mô N.

---

### 2026-07-31 - Phase 7 (roadmap): active-learning loop implement xong, chạy thật KHÔNG cải thiện checkpoint đã tối ưu

Roadmap 7.1-7.3+7.5 (thêm candidate verify tốt vào dataset → train lại surrogate → fine-tune generative model → quyết định dừng/tiếp) trước đó CHƯA có code thật - `self_play.py` (đã có sẵn) giải quyết vấn đề KHÁC (mining mẫu đối kháng để vá surrogate, đã bị real-physics loss thay thế), không phải active-learning thật.

**Implement:** `pipeline/phase5_cvae/active_learning.py` (mới) - mỗi vòng: (1) `find_weak_targets()` chạy `coverage_eval.py` (roadmap 7.4, đã có sẵn) trên checkpoint hiện tại, xếp hạng target auxetic theo `abs_error` giảm dần (target FE-solve toàn lỗi coi là tệ nhất, +inf); (2) `generate_and_verify_candidates()` sinh ứng viên cho các target yếu nhất, chấm FE THẬT trên toàn bộ (không chỉ mẫu trúng); (3) `filter_good_candidates()` giữ mẫu FE thành công VÀ `check_manufacturability().passes_all`; (4) fine-tune surrogate qua `--adversarial-npz` (cơ chế có sẵn từ `self_play.py`, chỉ khác nội dung npz); (5) fine-tune cVAE qua `--resume-from`/`--surrogate-path`/`--select-by fe_r2`; (6) đo lại coverage, ghi `outputs/phase5/active_learning/summary.json`; (7) `should_stop_loop()` - dừng khi dead_zone không giảm HOẶC hit_rate cải thiện <2 điểm % HOẶC hết `--max-rounds` (mặc định 3). 12 test mới (`tests/test_phase5_active_learning.py`), 430/430 test toàn repo pass.

**Bug bắt được lúc smoke-test:** gọi `phase5_cvae/train.py` fine-tune cVAE thiếu `--real-physics-subsample/--real-physics-every/--real-physics-workers` (README §5 dùng `subsample=8 every=2 workers=8` cho fine-tune 35-epoch ~15-18 phút đã kiểm chứng) → rơi về default tốn kém nhất của `train.py` (full-batch, mọi batch, tuần tự) → 33+ phút vẫn chưa xong 1 epoch. Đã sửa bằng cách truyền đúng 3 cờ này (mặc định mới của `active_learning.py`, khớp README).

**Chạy thật ở quy mô production** (`--n-samples 15 --grid-size 8 --n-weak-targets 10 --ft-epochs 10 --cvae-epochs 20 --max-rounds 3`, bắt đầu từ `cvae_realphysics.pt`):

| | Round 0 (baseline) | Round 1 |
|---|---|---|
| hit_rate | 1,000 | 1,000 |
| mean_abs_error | **0,0246** | **0,0686** (~gấp 2,8 lần, TỆ ĐI) |
| dead_zone_targets | 0 | 0 |
| n_good_candidates thêm vào | - | 23 (x40 oversample) |

Loop dừng đúng sau round 1 theo điều kiện 7.5 (`dead_zone_targets không giảm (0->0)`) - nhưng dead_zone=0 ngay từ round 0 vì `grid_size=8` chỉ cho ra 5 target auxetic trong `V12_TRAIN_RANGE`, cả 5 đã hit sẵn - lưới quá thô để tìm ra "vùng yếu" thật trên 1 checkpoint đã gần mức trần (`cvae_realphysics.pt`: R²(FE,n=789)=0,999, hit-rate 98,7%, xem mục "ĐỘT PHÁ 2026-07-24"). Fine-tune thêm 20 epoch trên 1 batch nhỏ (23 mẫu thật, oversample x40) từ 1 checkpoint đã hội tụ rất chặt làm `mean_abs_error` tệ đi gần 3 lần thay vì cải thiện.

**Quyết định:** KHÔNG thay `cvae_al_round1.pt` vào production - `cvae_realphysics.pt` vẫn là checkpoint tốt nhất đã biết. Kết luận Phase 7 tại đây (không thử lưới mịn hơn) - cùng bài học phương pháp luận với hướng OC hai-multiplier/hexagonal đã ghi ở trên: 1 kỹ thuật hợp lý về mặt lý thuyết (active-learning nhắm vùng yếu) không tự động cải thiện 1 hệ thống đã tối ưu tốt - đặc biệt khi phép đo "vùng yếu" (lưới coverage thô, 8 điểm) không đủ độ phân giải để tìm ra lỗ hổng thật. Code giữ nguyên trên nhánh `OC-2-multilayer` (không merge, không đổi checkpoint production) làm hạ tầng active-learning tái dùng được nếu sau này có checkpoint mới kém tối ưu hơn (vd sau khi mở rộng multi-param Pha B) thật sự cần vùng yếu để vá.

---

### 2026-07-31 - Phase 8.1 (roadmap): đối chiếu FE độc lập bằng scikit-fem - engine hiện tại được XÁC NHẬN đúng vật lý, không chỉ nhất quán nội bộ

Theo dõi trực tiếp Giới hạn Đã biết #15 (README): R²=0,999 của physics oracle (`cvae_realphysics.pt`) từ trước tới nay dùng CHUNG 1 engine FE/homogenization nội bộ (`simp/core/solver.py` + `simp/homogenization/compute.py`) cho cả training loss lẫn evaluation - "nhất quán nội bộ, đã kiểm chứng gradient", nhưng chưa từng đối chiếu với 1 implementation FE độc lập bên ngoài codebase.

**Implement:** `analysis/scripts/skfem_homogenization.py` - giải LẠI đúng bài toán homogenization tuần hoàn 2D (Q4 bilinear, plane-stress, SIMP `E(x)=Emin+x^penal(E0-Emin)`, 3 macro-strain đơn vị Voigt xx/yy/xy) bằng thư viện `scikit-fem` (`MeshQuad`, `ElementVector(ElementQuad1)`, `BilinearForm` tự viết qua `sym_grad`/`ddot`/`trace`/`eye` của skfem, tận dụng `skfem.models.elasticity.plane_stress()` có sẵn cho Lamé parameters). Periodic BC (ghép DOF biên trái/phải/trên/dưới, nối các góc, cố định 1 góc để bỏ rigid-body translation) VÀ công thức homogenize Q đều **tự viết mới hoàn toàn** trên DOF layout của skfem - **không tái dùng** `simp/core/pbc.py`/`simp/homogenization/compute.py`, vì đúng 2 file đó là nơi từng có các bug nghiêm trọng nhất dự án (A1 Q-scaling, hoán vị dQ, dấu RHS) - tái dùng lại sẽ chỉ kiểm tra được phần lắp ráp FE cơ bản, không bắt được lỗi logic ở đúng chỗ rủi ro nhất.

**Bug tự bắt được qua chính cổng kiểm chứng (trước khi so sánh với engine cũ):** công thức Q ban đầu copy y hệt dạng viết trong `compute.py` (`Q = Σ k_e · u^T·KE(E0)·u`, có hệ số `1/E0`) nhưng áp dụng sai ngữ cảnh - `compute.py` dùng 1 `KE` CỐ ĐỊNH tham chiếu E0 rồi nhân hệ số `k_e=E/E0` mỗi phần tử, trong khi implementation mới lắp ráp `K` trực tiếp bằng modulus THẬT per-element (`assemble_K()` dùng `E_elem` thật, không phải E0 tham chiếu) - 2 cách tính `K` khác nhau này chỉ tương đương nếu KHÔNG chia thêm `/E0` lần nữa. Bắt được ngay qua cổng phân tích (ô đặc hoàn toàn `x=1,penal=1` phải cho `Q=D` dạng đóng): kết quả lệch đúng 1 hệ số 198≈E0 - sửa 1 dòng (`compute_q_skfem()` bỏ `/E0`), pass ngay `rtol=1e-6` ở cả 2 độ phân giải lưới (10x10, 20x20). Đây chính là lý do cổng kiểm chứng phân tích BẮT BUỘC chạy trước khi tin bất kỳ số liệu so sánh nào - nếu bỏ qua bước này, bug trên sẽ lẫn vào kết quả so sánh cuối và bị hiểu nhầm là "solver cũ sai" thay vì "implementation mới sai".

**Chạy so sánh trên 24 mẫu THẬT** (không qua cVAE - lấy trực tiếp từ `outputs/phase3/test.npz`, có `penal` thật từng mẫu, chọn stratified theo 8 bin `v12` để trải đều `[-0,82, -0,04]`, không dồn 1 vùng property space):

| | Kết quả |
|---|---|
| n_samples | 24 |
| mean \|Δv12\| | 3,7×10⁻⁹ |
| max \|Δv12\| | 6,8×10⁻⁹ |
| mean \|Δv21\| | 5,3×10⁻⁹ |
| max \|Δv21\| | 1,2×10⁻⁸ |
| R²(v12, engine cũ vs skfem) | 1,000000 |

**Kết luận:** 2 engine hoàn toàn độc lập (khác thư viện, khác đánh số DOF, khác code lắp ráp ma trận, khác code periodic BC, cùng chung lý thuyết homogenization - điều không thể tránh khỏi ở bất kỳ implementation nào) khớp nhau tới mức sai số làm tròn dấu phẩy động (~1e-9, KHÔNG phải trùng hợp - phù hợp với sai khác round-off giữa 2 sparse solver khác nhau: `splu` trực tiếp ở engine cũ vs `spsolve` ở skfem), trên toàn bộ dải v12 kể cả mẫu auxetic mạnh (-0,82) lẫn gần trung tính (-0,04). Đây là bằng chứng độc lập mạnh nhất có thể có cho tới nay rằng engine FE nội bộ **đúng vật lý**, không chỉ "nhất quán nội bộ" - Giới hạn Đã biết #15 coi như đã giải quyết ở mức độ hợp lý cho 1 dự án ở quy mô này (chưa đối chiếu với phần mềm thương mại như ANSYS/Abaqus, nhưng scikit-fem là 1 implementation FEM độc lập thật, không phải bản sao/wrapper của code hiện tại).

Code: `analysis/scripts/skfem_homogenization.py` (engine độc lập, có thể tái dùng), `analysis/scripts/run_independent_fe_check.py` (script so sánh), `tests/test_skfem_homogenization.py` (7 test, gồm cổng phân tích ô đặc). 437/437 test toàn repo pass. Kết quả đầy đủ: `outputs/phase5/reports/independent_fe_crosscheck.json`. Đã commit vào `main`.

---

### 2026-08-02 - Phản hồi audit khoa học: novelty/diversity metric + baseline comparison lần đầu tiên - phát hiện hit_rate là metric yếu, retrieval baseline ngang cVAE trên test set hiện tại

Hai báo cáo audit bên ngoài (không lưu trong repo) chỉ ra 2 khoảng trống cụ thể chưa có bằng chứng trong pipeline: (1) chưa đo cVAE có sinh thiết kế "mới" hay chỉ tái tạo gần dữ liệu train, (2) chưa có baseline nào ngoài so sánh optimizer nội bộ (OC vs MMA) - không biết nếu bỏ cVAE đi thì kết quả còn giữ được không.

**1. Novelty/diversity (`analysis/scripts/novelty_diversity_eval.py`, mới):** với mỗi target lấy từ `test.npz` (seed=123, giống `best_of_n_eval.py`), sinh K mẫu bằng `cvae_v2_finetuned.pt`, đo khoảng cách L2 (pixel space, density field thô [0,1], không binarize) tới ảnh train GẦN NHẤT trong toàn bộ 57.216 mẫu (không subsample, dùng ma trận-hoá `||a-b||²=||a||²+||b||²-2a·b` theo chunk để tránh cấp phát ma trận khổng lồ). Có đường tham chiếu: khoảng cách NN thật-tới-thật (300 mẫu train ngẫu nhiên, tìm hàng xóm trong phần còn lại của train set, loại bỏ chính nó).

Chạy `n_conditions=24, n_samples=20` (480 mẫu sinh ra):

| | median | p5 | p95 |
|---|---|---|---|
| Novelty (sinh ra → train gần nhất) | 20,68 | 15,16 | 22,75 |
| Reference (thật → thật gần nhất) | 1,58 | 0,004 | 12,12 |
| Diversity (trong cùng 1 condition, K=20 mẫu) | 14,18 | 12,60 | 16,58 |

**0/480 mẫu sinh ra (0,0%) có novelty thấp hơn P5 của đường tham chiếu** → không có bằng chứng "gần như sao chép" theo tiêu chí này. Sanity-check riêng: ảnh sinh ra có tỉ lệ pixel "xám" (0,05-0,95) = 24,5% so với 30,9% ở ảnh train thật - không lệch nhiều, nên khoảng cách lớn không phải do decoder sinh ảnh mờ/blur (mới confound tiềm ẩn), mà là khác biệt cấu trúc thật.

**2. Baseline comparison (`analysis/scripts/baseline_comparison.py`, mới):** CÙNG tập 24 target (seed=123, giống hệt `best_of_n_eval.py` để so sánh apples-to-apples), 3 phương án: (a) random - chọn bừa 1 ảnh train, dùng nhãn (v12,v21) thật có sẵn; (b) nearest-neighbor retrieval - tra bảng train.npz tìm mẫu có (v12,v21) THẬT gần target nhất (không dùng mô hình sinh); (c) `best_of_n()` gọi thẳng (không viết lại), cVAE `cvae_v2_finetuned.pt`, N=10 mẫu/condition, FE thật trên tất cả.

| Phương án | hit_rate | R² |
|---|---|---|
| random | 1,000 | **-3,304** |
| **nearest-neighbor retrieval** | 1,000 | **1,000** |
| cVAE best-of-10 | 1,000 | 0,973 |

**Phát hiện 1 - `hit_rate` (định nghĩa hiện tại trong toàn bộ Phase 5: chỉ đúng dấu ν₁₂) là metric yếu:** vì dataset gốc ~92% mẫu đã auxetic (README §2), random guess cũng đạt hit_rate=1,000 ở n=24. R² mới là con số phân biệt được các phương án - mọi số hit_rate đã công bố trước đây (98,7% single-shot, v.v.) cần đọc kèm R²/CI, không tự đứng một mình. Đúng câu hỏi audit report mục 20.3 ("metric bị thiên lệch").

**Phát hiện 2 - nearest-neighbor retrieval (tra bảng, không cần mô hình sinh) đạt R²=1,000, bằng hoặc nhỉnh hơn cVAE best-of-10 (0,973) trên chính test set này** (`mean_condition_dist`=0,0021 - dataset đủ dày nên hầu như luôn có mẫu train rất gần bất kỳ target nào rút từ cùng phân phối test/train). **Diễn giải đúng phạm vi:** phép test này CHỈ chứng minh (hoặc không chứng minh) lợi thế accuracy trên target trong-phân-phối (in-distribution) - test set và train set cùng 1 quy trình sinh dữ liệu (Phase 2 DOE), nên "tìm hàng xóm gần" gần như luôn khả thi. Đây KHÔNG phải bằng chứng cVAE vô dụng, mà là bằng chứng cụ thể rằng **lợi thế thật sự của cVAE (nếu có) phải nằm ở khả năng nội suy ngoài phân phối/mật độ train, hoặc ở việc không cần lưu trữ+tra cứu 57k mẫu** - trục này CHƯA được đo (cần thử nghiệm E trong audit report: out-of-distribution test).

**Giới hạn chưa giải quyết được phát hiện khi implement:** không có baseline "SIMP-only nhắm 1 target Poisson ratio cụ thể" - `simp/objectives/auxetic.py` hiện chỉ cực trị hóa Q12 (`compute_auxetic_q12_objective`), không có số hạng bám mục tiêu (vd `(v12-target)²`). Cần thêm objective mới mới làm được so sánh này - ngoài phạm vi lần audit-response này, ghi vào Giới hạn Đã biết #18.

Code: `analysis/scripts/novelty_diversity_eval.py`, `analysis/scripts/baseline_comparison.py` (cả 2 tái dùng `best_of_n_eval.py`/`dataset.py`/`adversarial_dataset.py`/`manufacturability.py` có sẵn, không viết lại logic FE/sampling). Kết quả đầy đủ: `outputs/phase5/reports/novelty_diversity_cvae_v2_finetuned.json`, `outputs/phase5/reports/baseline_comparison_cvae_v2_finetuned.json`. Trên nhánh `main`, uncommitted.

### 2026-07-25 - Tham số input optional (`volfrac`, `void_size_frac`) + chấm điểm toàn diện (Pha A, branch `feature/optional-multi-condition`)

Yêu cầu: thêm tham số input ngoài `v12/v21` nhưng phải **optional** (không bắt buộc); đồng thời chấm điểm lời giải toàn diện hơn - chính xác/ổn định ưu tiên cao, khả năng chế tạo ưu tiên trung bình, thẩm mỹ ưu tiên thấp (trục hoàn toàn mới, chưa từng có trong codebase).

**Nghiên cứu trước khi code:** kiểm tra trực tiếp `outputs/phase3/train.npz` phát hiện `volfrac`/`void_size_frac` (2 trong 5 tham số DOE gốc: `volfrac,penal,rmin,move,void_size_frac`) đã nằm sẵn trong field `params`/`param_names` của MỌI file npz - `CVAEDataset` thậm chí đã load `volfrac_achieved` nhưng vứt bỏ trước khi vào `condition` (`dataset.py`), và `train.py::run_epoch` từng nhận nó qua biến `_volfrac` rồi bỏ luôn. Tức là thêm 2 tham số này vào condition **không tốn backfill dữ liệu gì cả** - khác hẳn `f1=E₁₁/E₀, f2=E₂₂/E₀` (roadmap gốc, [Giới hạn #5](docs/LIMITATIONS.md#giới-hạn-đã-biết--known-limitations)): tuy `simp/runner.py::run_simp()` đã trả về `Q` (tensor 3×3 đầy đủ, đủ để suy f1/f2) ở MỌI lần chạy, `pipeline/phase2_multi_batch/runner.py::evaluate_single()` chỉ trích `v12/v21/obj_value` rồi vứt `Q` - nên f1/f2 cần backfill (1 FE-solve/mẫu trên ảnh cuối đã lưu, rẻ, không phải chạy lại DOE) + train lại Phase 4 surrogate + kiểm tra multicollinearity (f1/f2/v12/v21 cùng suy từ 1 `Q`). Do ngân sách được yêu cầu giữ chặt (50% giờ còn lại, tối đa 10%/tuần), chia 2 pha: **Pha A** (volfrac/void_size_frac, làm ngay) và **Pha B** (f1/f2, để lại làm sau).

**Thiết kế "optional" - presence-mask + condition-dropout, không phải chỉ default value:** nếu chỉ set sentinel=0 khi không chỉ định, model không phân biệt được với giá trị thật bằng 0. `condition` mở từ 2 lên 6 chiều `[v12,v21,volfrac,volfrac_mask,void_size_frac,void_size_frac_mask]` (`--extended-condition`), lúc train áp `apply_condition_dropout()` (kiểu classifier-free-guidance: zero value+mask ngẫu nhiên theo từng chiều optional, độc lập từng mẫu, `--optional-dropout-p` mặc định 0,5) để model học xử lý cả 2 trường hợp. `volfrac` có loss riêng gần như miễn phí (`volfrac_consistency_loss` - suy trực tiếp từ `recon.mean()`, không qua surrogate/FE, chỉ tính trên mẫu mask=1); `void_size_frac` không có loss riêng ở Pha A (không có công thức rẻ tương tự), chỉ học ngầm qua reconstruction.

**Chấm điểm toàn diện trong `best_of_n_eval.py`:** `argmin(|Δv12|)` (thuần túy chính xác) đổi thành `argmax(composite_score)` với `composite_score = 0,6·accuracy_score + 0,3·manuf_score + 0,1·aesthetic_score` (đúng thứ tự ưu tiên yêu cầu). `manuf_score` graded (trung bình 3 cờ con, không chỉ nhị phân `passes_all` như trước). `aesthetic_score` (module mới `aesthetics.py`: đối xứng + tỉ lệ chu vi/diện tích) cố tình giữ rẻ vì là trục ưu tiên thấp nhất - không cần model riêng.

**Kiểm chứng:** 414/414 test pass (thêm ~35 test), smoke-test thật trên data thật (train 2 epoch không NaN, `sample.py`/`best_of_n_eval.py` có/không tham số optional, cả checkpoint cũ `condition_dim=2` lẫn mới `=6`). 1 test cũ (`TestBestOfNOracleSelectionLogic`) phải cố định `w_accuracy=1,w_manuf=0,w_aesthetic=0` vì assert đúng winner theo argmin thuần túy, độc lập với manuf/aesthetic đo trên ảnh model chưa train thật.

**Trạng thái:** chưa commit/merge, đang ở branch `feature/optional-multi-condition` - người yêu cầu muốn thử nghiệm riêng trước khi đưa vào `FixLoss` vì chưa chắc chắn thành công. Pha B (f1/f2) chưa bắt đầu.

### 2026-08-03 - Audit kiểu phản biện (advisor test) trên Pha A sau khi rebase branch lên `main`: phát hiện + sửa 2 bug thật, 1 trong đó nằm ngay trên lệnh khuyến nghị của README

Bối cảnh: `feature/optional-multi-condition` không có commit riêng (chỉ là `main` cũ + 1 stash Pha A chưa apply) nên đã fast-forward lên `main` mới nhất (26 commit, dataset v2 + active-learning + MMA...) rồi `git stash pop` - chỉ conflict ở 2 đoạn `README.md` (2 bên thêm nội dung khác vị trí, không mâu thuẫn thật, merge tay). Sau đó chủ động test như advisor sẽ thử: chạy lại toàn bộ pipeline, thử mọi tổ hợp cờ CLI, không chỉ đọc code.

**Bug 1 (test isolation): `tests/test_mma_runner.py` xoá/ghi đè thư mục output đã commit vào git.** `run_simp_mma()` mặc định `output_dir=f'outputs/simp_results_{seed_name}'` khi không truyền `output_dir`, và làm `shutil.rmtree(output_dir)` trước khi ghi lại. 2 test dùng `seed='hexagonal'` không truyền `output_dir` - mà `outputs/simp_results_hexagonal/` (khác `simp_results_circle/`, đã có trong `.gitignore`) đang được TRACK trong git (34 file, lọt vào từ 1 lần chạy pilot trước đó). Hệ quả: chạy `pytest` xong `git status` bẩn (`metadata.json` bị ghi đè timestamp/git_hash mới) - đúng kiểu thứ advisor thấy ngay khi `git status` sau khi chạy test suite. Sửa: thêm `'output_dir': str(tmp_path / 'run')` vào params của 2 test đó (giữ nguyên 3 test còn lại - 2 test dùng seed `circle` (đã gitignore) không sửa, 2 test `rejects_ft1`/`rejects_projection` raise lỗi TRƯỚC khi chạm tới output_dir nên không cần sửa).

**Bug 2 (nghiêm trọng hơn nhiều - đúng lệnh khuyến nghị trong README crash ngay lập tức):** README mục 5.1 khuyến nghị fine-tune Pha A bằng đúng lệnh `train.py --extended-condition --lambda-volfrac 1.0 --resume-from outputs/phase5/cvae_realphysics.pt`. Chạy thử y hệt lệnh này crash ngay: `RuntimeError: size mismatch for encoder.fc_mu.weight/encoder.fc_logvar.weight/decoder.fc.weight` - vì `cvae_realphysics.pt` có `condition_dim=2` còn model mới tạo với `--extended-condition` có `condition_dim=6`, mà 3 layer fc này (`model.py`: `Encoder.fc_mu/fc_logvar`, `Decoder.fc`) nhận trực tiếp `condition` nối vào input nên shape phụ thuộc `condition_dim`; `model.load_state_dict(strict=True)` (mặc định) từ chối nạp. Tức là TOÀN BỘ workflow fine-tune Pha A như tài liệu hoá chưa từng chạy được thật - chỉ smoke-test trước đó (memory `phase5-optional-multiparam`) train from-scratch, không qua đường `--resume-from` checkpoint condition_dim khác.

Sửa bằng "mở rộng" 3 layer đó thay vì crash/reset toàn bộ model: hàm mới `resize_condition_dim_weights()` (`train.py`) giữ NGUYÊN cột ứng với base features (ảnh phẳng/latent z, không đổi) VÀ 2 cột đầu của condition (v12/v21, đúng thứ tự `build_condition_vector`), chỉ random-init (init mặc định `nn.Linear`) các cột condition MỚI (volfrac/mask/void_size_frac/mask) - giữ được toàn bộ CNN encoder/decoder backbone (chiếm hầu hết tham số) + phần lớn trọng số 2 layer fc thay vì train lại từ đầu. Verify bằng script tay (so `old_w[:, :base_dim+2]` == `new_w[:, :base_dim+2]` bit-for-bit, cột mới có `std()>0`) VÀ bằng cách chạy lại đúng lệnh README thật (1 epoch, GPU thật, dataset v2 thật) - chạy xong, checkpoint lưu đúng `condition_dim=6`. Thêm 3 test regression (`TestResizeConditionDimWeights` trong `test_phase5_train.py`) khoá lại hành vi: no-op khi cùng dim, giữ đúng cột base+v12/v21, và `load_state_dict` vào model mới không crash.

**Kiểm chứng thêm (không phát hiện bug, nhưng đã test thật để loại trừ):** `sample.py`/`best_of_n_eval.py` chạy được cả 2×2 tổ hợp có/không `--volfrac`/`--void-size-frac`, checkpoint cũ `condition_dim=2` + cờ optional in cảnh báo và BỎ QUA đúng như thiết kế (không crash), giá trị optional ngoài khoảng `[0,1]` (vd `--volfrac 1.5/-0.5`) không crash (target mềm, không phải ràng buộc cứng, nhất quán với cách v12/v21 cũng không validate range), thiếu `--v12/--v21` báo lỗi argparse rõ ràng, `--volfrac` không kèm `--v12/--v21` bị chặn bởi validation có sẵn (`parser.error`, đúng docstring), `best_of_n_eval.py` sweep mode mặc định (không `--v12/--v21`) lẫn `--require-manufacturable` đều hoạt động đúng trên checkpoint `condition_dim=6`, `n-samples=1` không crash (min-max normalization có fallback khi range=0). `aesthetics.py` test tay trên ảnh toàn rỗng/toàn đặc/bàn cờ/nhiễu ngẫu nhiên - không NaN, không chia-cho-0. `apply_condition_dropout()`/`volfrac_consistency_loss()` test tay ở `p=0,0`/`p=1,0`/mask toàn 0 - không leak giá trị thật khi mask=0, không NaN khi mask toàn 0.

**Kết quả:** 483/483 test pass (480 trước audit + 3 test mới). Không còn bug đã biết nào chặn workflow Pha A end-to-end (train fine-tune → sample → best_of_n_eval với composite scoring) theo đúng lệnh README.

### 2026-08-04 - Thử nghiệm E (audit report): cVAE vs retrieval baseline NGOÀI phân phối train - kết quả phân cực, không phải chiến thắng toàn diện

Bối cảnh: mục 2026-08-02 để lại câu hỏi treo lớn nhất chưa trả lời - so sánh trong-phân-phối (n=24, cùng phân phối test/train) cho thấy retrieval (R²=1,000) ngang hoặc nhỉnh hơn cVAE best-of-10 (R²=0,973), nhưng phép test đó KHÔNG đo được điều retrieval thật sự yếu: ngoại suy ra ngoài vùng dữ liệu đã thấy. Đây là "thử nghiệm E" mà audit report gốc đề xuất.

**Thiết kế target OOD:** kiểm tra trực tiếp `train.npz` (57.216 mẫu) cho thấy v₁₂/v₂₁ nằm hoàn toàn trong `[-1,9527, -0,0023]` - TOÀN BỘ dataset là auxetic, KHÔNG một mẫu nào có v₁₂≥0. Từ đó chọn 12 target OOD chia 2 nhóm có lý do rõ ràng: (a) `non_auxetic` (sign-flip, v₁₂ mục tiêu = +0,1..+0,8) - vùng mật độ train tuyệt đối bằng 0, retrieval về mặt cấu trúc KHÔNG THỂ trả lời đúng dấu; (b) `extreme_auxetic` (v₁₂ mục tiêu = -2,1..-2,8) - vượt qua min train, đòi hỏi ngoại suy cường độ. `v21=target v12` cho đa số target (đơn giản, dễ giải thích - train có corr(v12,v21)≈0,05, gần như độc lập, không có hướng "tự nhiên" nào khác để suy ra v21).

Script mới `analysis/scripts/ood_baseline_comparison.py` - tái dùng `nearest_neighbor_baseline()` (từ `baseline_comparison.py`) và `best_of_n()` (từ `best_of_n_eval.py`, gọi qua `custom_condition` từng target một), KHÔNG viết lại logic FE/sampling. Chạy trên `cvae_realphysics.pt` (checkpoint hiện dùng), N=10 mẫu/condition, seed=123 (khớp mọi benchmark trước).

| Nhóm | retrieval R² | retrieval MAE | cVAE best-of-10 R² | cVAE MAE |
|---|---|---|---|---|
| Tổng hợp (n=12) | 0,057 | 1,053 | **0,418** | **0,844** |
| `non_auxetic` (sign-flip, n=6) | -3,035 | 0,402 | **-0,285** | **0,191** |
| `extreme_auxetic` (n=6) | -61,514 | 1,703 | -38,729 | 1,496 |

**Phát hiện chính:** trên nhóm sign-flip, retrieval bị buộc luôn trả về mẫu gần 0 nhất trong train (`v12=-0,0023` cố định cho MỌI target dương, vì đó là biên train) - sai dấu 100%. cVAE, ngược lại, tự sinh cấu trúc MỚI đạt đúng DẤU DƯƠNG cho 5/6 target (vd target=+0,10 → cVAE đạt +0,10; target=+0,30 → +0,20; nhưng target=+0,80 → chỉ +0,30, hụt biên độ rõ khi target đi xa) - đây là bằng chứng thật, cụ thể, đo được về khả năng ngoại suy HƯỚNG (không chỉ tra bảng), điều mà retrieval về mặt cấu trúc không bao giờ làm được dù có bao nhiêu dữ liệu train.

Trên nhóm extreme_auxetic thì KHÔNG có tin tốt: cVAE bão hòa quanh v₁₂≈-0,83 đến -1,10 bất kể target đẩy xa tới đâu (-2,1 hay -2,8 cho kết quả gần như nhau) - không ngoại suy được cường độ auxetic vượt ngoài phạm vi đã thấy, dù đỡ tệ hơn retrieval (retrieval còn tệ hơn nữa, có 3/6 target rơi đúng vào cùng 1 mẫu train biên `v12=-0,0930` bất kể target cách xa bao nhiêu).

**Kết luận cẩn trọng (khớp tinh thần "known_gap" của baseline_comparison.py - không phóng đại):** cVAE CÓ generalization thật ngoài phân phối, thể hiện rõ nhất và có thể giải thích được qua khả năng đổi dấu - đây là điểm retrieval không bao giờ vượt qua được về nguyên tắc. Nhưng lợi thế này có giới hạn biên độ rõ ràng, và hoàn toàn không giúp được cho việc ngoại suy cường độ (magnitude) vượt xa phạm vi train. Không nên diễn giải kết quả này thành "cVAE tổng quát hoá tốt ra OOD" một cách chung chung - chỉ đúng cho trục sign-flip, không đúng cho trục magnitude.

Code: `analysis/scripts/ood_baseline_comparison.py` (mới, tái dùng hạ tầng có sẵn). Kết quả đầy đủ: `outputs/phase5/reports/ood_baseline_comparison_cvae_realphysics.json`. Đã cập nhật `docs/LIMITATIONS.md` mục #19 (VI+EN) và mục tóm tắt claim đầu file. Trên nhánh `main`, uncommitted.

### 2026-08-05 - Backfill f1=E₁₁/E₀, f2=E₂₂/E₀ (Pha B roadmap gốc) - phát hiện landmine manifest.csv giữa chừng, đổi phương án, Phase 4 surrogate 5-chiều đạt R² cùng bậc v12/v21

Bối cảnh: `Q` (tensor độ cứng đồng nhất hóa 3×3 đầy đủ) được `run_simp()` tính ở MỌI lần chạy FE nhưng chưa từng lưu xuống đĩa - chỉ `v12/v21/obj_value` được giữ (xem [Giới hạn #5](docs/LIMITATIONS.md#giới-hạn-đã-biết--known-limitations)). Kế hoạch ban đầu: đọc `outputs/phase3/manifest.csv` (10.269 dòng, có `image_path`), chạy lại FE trên ảnh gốc độ phân giải cao, join kết quả vào `train/val/test.npz` theo `image_path`.

**Landmine phát hiện giữa chừng:** `manifest.csv` (10.269 dòng) KHÔNG khớp số lượng thật của pool đã dùng để build `train/val/test.npz` hiện tại - suy ngược từ `split_report.json` (`n_train_after_augment`=57.216, augment x6 cố định → 9.536 mẫu gốc + val 2.044 + test 2.044 = **13.624**, khớp CHÍNH XÁC với `outputs/phase3/dataset_64.npz` có sẵn (n=13.624) nhưng KHÔNG khớp `manifest.csv` - chênh 3.355 dòng, không rõ nguồn gốc chênh lệch (có thể `manifest.csv` là bản trung gian/cũ, pool gốc `outputs/phase3_v2_raw/` đã không còn trên đĩa để đối chiếu lại). Join theo `image_path` từ `manifest.csv` sẽ SAI LỆCH ngầm mà không phát hiện được (thiếu ~1/4 dữ liệu, không rõ mẫu nào). Quan trọng hơn: 5/6 mẫu trong `train.npz` là bản tăng cường đối xứng (xoay/lật) CHỈ tồn tại dưới dạng mảng numpy trong `.npz` - không có file PNG gốc nào trên đĩa để chạy FE độ phân giải cao hơn, kể cả khi giải quyết được vấn đề join.

**Đổi phương án:** viết `analysis/scripts/backfill_f1_f2_npz.py` - chạy FE TRỰC TIẾP trên ảnh 64×64 đã lưu sẵn trong `{train,val,test}.npz` (dùng `penal` thật từ `params[:,1]`), không cần join gì cả - an toàn tuyệt đối về mặt liên kết dữ liệu, đổi lại chấp nhận thêm 1 lớp resize (64×64→50×50 lưới FE, sau khi ảnh gốc đã qua 1 lần resize xuống 64×64 khi build dataset) làm nhiễu tăng nhẹ so với phương án đọc ảnh gốc (sanity check ban đầu trên `manifest.csv` cho sai số median chỉ 0,0006-0,0018; phương án cuối cho median 0,0018 nhưng đuôi dài hơn, max tới 0,99 ở 3,2% mẫu train - xem phân bố đầy đủ trong `docs/PIPELINE.md § 3`).

12 worker song song, ~14 phút cho 57.216 mẫu train (val/test ~2 phút mỗi tập), 0 lỗi FE. Merge vào `outputs/phase3/{split}_ext.npz` (giữ nguyên mọi field cũ + f1, f2, `dv12_backfill_qc`).

**Mở rộng Phase 4 surrogate** (`dataset.py::AuxeticDataset(include_f1f2=...)`, `model.py::SurrogateCNN(n_outputs=...)`, `train.py --include-f1f2` - tương thích ngược hoàn toàn, mặc định TẮT, mọi test cũ (28 test) vẫn pass). Train 5-target (`[v12,v21,volfrac,f1,f2]`, trọng số loss `[1,1,0.3,1,1]`) trên GPU, ~14s/epoch, early-stop epoch 42 (val_loss=0,00249). Kết quả R²(test, n=2044): **v12=0,970, v21=0,960, volfrac=0,980, f1=0,933, f2=0,961** - f1/f2 học được TỐT dù dữ liệu backfill nhiễu hơn, không có dấu hiệu nhiễu resize cản trở việc học (nhiễu trung bình hoá qua N lớn).

**Multicollinearity** (đo trực tiếp trên 57.216 mẫu, không cần chờ model): corr(v12,f2)=0,692, corr(v21,f1)=0,645 - hợp lý về vật lý (v12≈Q₁₂/Q₂₂=Q₁₂/(f2·E₀) theo công thức rút gọn trong `objectives/auxetic.py` docstring, tương tự v21↔f1), nhưng KHÔNG đủ mạnh (chưa tới 0,7) để coi f1/f2 là dư thừa/suy được tuyến tính từ v12/v21.

**Chưa làm tiếp** (ngoài phạm vi hôm nay): nối f1/f2 làm condition cho cVAE Phase 5 - mới dừng ở Phase 4 surrogate. Checkpoint: `outputs/phase4/surrogate_f1f2.pt`. Code: `analysis/scripts/backfill_f1_f2_npz.py`, `merge_f1f2_into_npz.py` (mới); `pipeline/phase4_surrogate/{dataset,model,train}.py` (sửa, tương thích ngược). Đã cập nhật `docs/PIPELINE.md § 3`, `README.md`, `docs/LIMITATIONS.md` mục #5 (VI+EN). Trên nhánh `main`, uncommitted.

### 2026-08-05 (tiếp) - A/B test `objective_variant='normalized'` (ứng viên thay thế `mu` penalty) - kết quả tiêu cực rõ ràng trên cả 3 seed, KHÔNG nên bật

Bối cảnh: `mu` penalty tuyến tính (`c = Q12 - mu*(Q11+Q22)`) đã bị tắt từ trước (mu=0.0) vì sai sót khái niệm - `runner.py:305` ghi rõ thực nghiệm cũ cho thấy mu>0 không cải thiện. Code đã có sẵn 1 ứng viên thay thế CHƯA TỪNG test: `compute_auxetic_normalized_objective()` (`simp/objectives/auxetic.py:100-157`, `c = Q12/√(Q11·Q22)`) - nhờ bất đẳng thức Cauchy-Schwarz trên ma trận PSD `Q`, tỉ số này tự nhiên bị chặn [-1,1] mà không cần tham số tùy chỉnh nào, đúng chỗ `mu` bị coi là sai khái niệm. Wired sẵn qua `objective_variant='normalized'` trong `run_simp()`, nhưng docstring ghi rõ "CHƯA áp dụng cho dataset hiện có, cần A/B test trước" - không cần thiết kế công thức mới, chỉ cần đo.

Viết `analysis/scripts/pilot_normalized_objective.py` (cùng pattern `pilot_mma.py`), so `q12` (mặc định) vs `normalized` trên CÙNG dải tham số production, n=50 mẫu/config, 3 seed khó (`hexagonal`, `hourglass`, `reentrant_bowtie`) - tổng 300 lần chạy SIMP, ~12s/mẫu đơn luồng, 10 worker song song, tổng ~10 phút.

| Seed | q12 sạch | normalized sạch | q12 osc_unstable | normalized osc_unstable | q12 disconnected | normalized disconnected |
|---|---|---|---|---|---|---|
| hexagonal | 84,0% | **8,0%** | 10% | **92%** | 8% | **82%** |
| hourglass | 72,0% | **26,0%** | 18% | **72%** | 26% | 26% |
| reentrant_bowtie | 92,0% | **32,0%** | 8% | **58%** | 8% | 30% |

**Kết quả rõ ràng, nhất quán trên cả 3 seed: `normalized` làm yield sụp đổ nặng** (giảm 60-76 điểm phần trăm tuyệt đối). Điều bất ngờ: KHÔNG phải do sai dấu auxetic - `reentrant_bowtie` với `normalized` đạt 100% đúng dấu v12<0 (còn tốt hơn `q12`'s 92%), `hourglass` giữ nguyên 84%. Nguyên nhân thật sự là **mất ổn định hội tụ** (`osc_unstable` tăng 5-9 lần ở cả 3 seed) - nhãn dao động limit-cycle cuối vòng lặp (xem [Giới hạn #13](docs/LIMITATIONS.md#giới-hạn-đã-biết--known-limitations)), cộng thêm rời rạc hoá nặng riêng ở `hexagonal` (8%→82%, kèm `Q11/Q22` trung bình tăng vọt 22→33 - dấu hiệu optimizer "lách" bằng cách làm cứng đều cả 2 trục thay vì đẩy Q12 âm thật).

**Diễn giải:** chuẩn hoá theo `√(Q11·Q22)` tuy đúng và có chặn-biên đẹp về mặt lý thuyết, nhưng đưa vào 1 phép chia phi tuyến trong gradient (`dc` có thêm số hạng `-Q12*dP/(2*denom³)`) làm cảnh quan gradient khó hội tụ hơn nhiều trong khung OC + continuation hiện tại (không phải vấn đề riêng của bất kỳ seed nào - nhất quán cả 3). **Cả 2 hướng thay thế `mu` (phạt tuyến tính VÀ chuẩn hoá tỉ lệ) giờ đều đã bị loại bằng thực nghiệm trực tiếp** - giữ nguyên mặc định `mu=0.0`/`objective_variant='q12'`. Muốn tiếp tục hướng này cần ý tưởng khác hẳn (không phải đổi công thức objective nữa), ví dụ đổi cơ chế tối ưu (MMA đã thắng lớn ở `hexagonal`/`hourglass` cho vấn đề khác - có thể đáng thử kết hợp) hoặc continuation schedule khác cho riêng `normalized`.

Manifest đầy đủ: `outputs/pilot_normalized_{hexagonal,hourglass,reentrant_bowtie}/manifest.csv`. Đã cập nhật `docs/LIMITATIONS.md` mục #4 (VI+EN), đã commit vào `main`. Script pilot một lần đã xóa 2026-08-15 sau khi kết quả được ghi đầy đủ vào log này.

---

### 2026-08-09 - Ảnh hưởng đặc trưng hình học (số cạnh/thanh, độ dày) lên v12 - phát hiện Simpson's paradox rõ ràng giữa số liệu gộp và từng họ seed

Bối cảnh: yêu cầu "đánh giá ảnh hưởng các kết quả, ví dụ số lượng cạnh/độ dày tới hệ số Poisson's" khác với `analysis/sensitivity/` đã có (đo tham số SIMP ĐẦU VÀO như `volfrac`/`rmin` → objective) - đây là đo đặc trưng TOPOLOGY xuất hiện (emergent) trong ảnh kết quả → v12, chưa có trong pipeline.

Viết `notebooks/02_geometric_feature_influence.ipynb` (mới, hàm dùng chung đặt trong `notebooks/utils.py` theo convention hiện có - không thêm dependency, chỉ numpy/scipy/networkx đã có sẵn trong `requirements.txt`, không cần `scikit-image`). Trích đặc trưng từ ảnh binarize (ngưỡng 0,5, đúng convention `manufacturability.py`): pad tuần hoàn (wrap) trước khi skeleton hóa để tránh endpoint giả tại biên ô đơn vị, Zhang-Suen thinning vector hóa bằng numpy (không loop pixel), rồi rút gọn skeleton pixel-graph thành graph topology thật (contract chuỗi pass-through) để đếm `n_edges` (số thanh), `n_junctions`, `n_endpoints`; độ dày thanh = 2×distance-transform tại pixel skeleton. Đã sanity-check bằng overlay skeleton lên vài mẫu (hourglass/hexagonal/reentrant_bowtie/circle) - skeleton bám đúng trục trung tuyến của vật liệu, junction/endpoint hợp lý về mặt trực quan.

Chạy trên 6.001 mẫu lấy phân tầng theo `seed_class` (gộp `train+val+test_ext.npz`, phân tích mô tả post-hoc nên không có rủi ro leakage). Ba phép kiểm định: Pearson/Spearman, partial correlation kiểm soát `volfrac`, và kiểm định nhị phân Mann-Whitney U (chia nhóm cao/thấp theo median) - đúng yêu cầu gốc "câu nhị phân".

**Kết quả chính - đảo chiều tương quan (Simpson's paradox) giữa số liệu GỘP và từng HỌ seed:**
- Gộp toàn bộ: `corr(n_edges, v12) = +0,168` (p=2×10⁻³⁹) - có vẻ như càng nhiều thanh thì v12 càng ít âm.
- Nhưng tách theo `seed_class`, dấu ĐẢO NGƯỢC ở hầu hết các họ: `hourglass` (n=3.121) **-0,384**, `circle` -0,374, `square` -0,366, `four_circle` -0,342, `cross_rectangular` -0,324 - càng nhiều thanh thì v12 càng ÂM hơn (auxetic mạnh hơn), ngược hẳn xu hướng gộp. Ngoại lệ: `hexagonal` (n=933) gần như không có quan hệ (+0,040, p=0,22, không có ý nghĩa thống kê).
- Độ dày thanh (`mean_thickness_px`) cũng đảo chiều tương tự: âm ở `hourglass`/`hexagonal`/`circle_half_quarter` (thanh mỏng hơn → auxetic mạnh hơn) nhưng dương ở `circle`/`square`/`cross_rectangular`/`small_square_cross` (+0,44 đến +0,50).
- Tất cả kiểm định Mann-Whitney trong từng họ lớn đều có p<0,001 (nhiều trường hợp p≈0, underflow) - hiệu ứng có thật về mặt thống kê, không phải nhiễu ngẫu nhiên, nhưng độ mạnh ở mức trung bình (|r|≈0,2-0,4 trong từng họ, rank-biserial tương ứng), KHÔNG phải yếu tố chi phối chính (volfrac/họ seed vẫn quyết định phần lớn - hồi quy đa biến gộp chỉ đạt R²=0,042).

**Diễn giải:** số liệu gộp gây hiểu lầm vì mỗi họ seed chiếm vùng giá trị v12 và cấu trúc rất khác nhau (`hourglass` áp đảo 52% mẫu); kết luận đúng phải đọc theo từng họ, không đọc số gộp. Cơ chế vật lý hợp lý: trong các họ dạng "khung xoay/gấp khúc" (hourglass, circle-based), thêm thanh phụ (junction) thường tạo thêm điểm xoay/uốn góp phần tăng hiệu ứng auxetic, còn với `hexagonal` (cơ chế re-entrant hình học đã cố định bởi góc cell) topology phụ không ảnh hưởng nhiều - khớp với ghi nhận trước đó rằng `hexagonal` có hành vi yield khác biệt các seed khác ([[hexagonal_yield_oc_dual_multiplier]]).

Code: `notebooks/02_geometric_feature_influence.ipynb` (mới), hàm dùng chung trong `notebooks/utils.py` (`extract_geometric_features`, `skeleton_topology`, `zhang_suen_thin`, `analyze_geometric_group`, `binary_split_test`, `partial_corr`, `load_geometric_analysis_sample`). Output: `outputs/phase3/reports/geometric_feature_influence_n6001.json`, `geometric_features_raw_n6001.csv`, hình `outputs/figures/geometric_feature_influence_n_edges_vs_v12_n6001.png`. Trên nhánh `main`, uncommitted.

---

### 2026-08-18 - Xác minh lại A3 (hội tụ FE theo ν0 vật liệu nền) - claim cũ không có bằng chứng, số liệu thật khác đáng kể về mặt định lượng

Bối cảnh: chuẩn bị làm A4 (docs/archive/PROJECT_PLAN.md Nhóm 1 - thêm ν0 làm input phụ cho CNN surrogate), phát hiện comment `SEED_NU_RANGE` trong `analysis/scripts/generate_production_batch.py` mô tả 1 batch A3 rất cụ thể (n=20/seed, "ranh giới hội tụ sắc tại ν0≈0.30" cho `reentrant_bowtie`, `corr(nu,clean)=0,76`) nhưng rà soát toàn repo (output dir, `EXPERIMENT_LOG.md`, git log/stash) **không tìm thấy bất kỳ dữ liệu/log nào chứng minh batch đó từng chạy thật** - vi phạm quy ước log-mọi-pilot của project.

Viết lại `analysis/scripts/pilot_nu_convergence.py` (cùng pattern `pilot_normalized_objective.py`) - sweep ν0 đều trong `PARAM_SPACE['nu']=(0.2, 0.4)`, dùng đúng dải volfrac/void_size_frac và optimizer production thật (`SEED_PARAM_RANGES`/`SEED_OPTIMIZER`) cho 3 seed `hourglass`/`hexagonal`/`reentrant_bowtie`, n=50/seed (150 mẫu, 10 worker, 210s).

| Seed | %sạch | corr(nu,clean) | Dạng quan hệ |
|---|---|---|---|
| hourglass | 88,0% (44/50) | 0,058 | phẳng, không phụ thuộc ν0 |
| hexagonal | 100,0% (50/50) | không tính được (0 mẫu hỏng) | hoàn toàn không phụ thuộc ν0 |
| reentrant_bowtie | 76,0% (38/50) | 0,386 | dốc MƯỢT: 45%(ν0∈[0,20,0,25)) → 75% → 80% → 94%(ν0∈[0,35,0,40)) |

**Kết luận:** hướng định tính của claim cũ đúng (reentrant_bowtie nhạy nhất với ν0, hexagonal gần như miễn nhiễm), nhưng phần định lượng sai đáng kể - không có "ranh giới sắc" nào (ν0∈[0,25,0,30) vẫn đạt 75% sạch, không phải 0% như claim cũ), và mọi hệ số tương quan claim cũ đều bị thổi phồng (0,76 vs 0,386 thật cho reentrant_bowtie; 0,45 vs 0,058 thật cho hourglass). **Quyết định: KHÔNG thu hẹp `SEED_NU_RANGE` theo seed cho A4** - giữ dải đầy đủ (0.2, 0.4) cho cả 3 seed, vì (a) không có cơ sở thật để thu hẹp, và (b) mục đích A4 là dạy surrogate quan hệ ν0→tính chất - thu hẹp riêng `reentrant_bowtie` sẽ tạo lỗ hổng dữ liệu đúng ở vùng ν0 thấp, đúng nơi cần học quan hệ này nhất; 45% sạch ở vùng thấp vẫn đủ dùng, quality filter sẵn có sẽ lọc phần hỏng.

**Bài học quy trình:** mọi claim thực nghiệm trong comment code PHẢI có manifest/log đối chiếu được - comment mô tả số liệu mà không kèm đường dẫn kết quả kiểm chứng được là dấu hiệu cảnh báo, cần chạy lại trước khi dùng để quyết định downstream (ở đây là A4).

Manifest đầy đủ: `outputs/pilot_nu_convergence/manifest.csv` (gitignored, scratch). Đã sửa `analysis/scripts/generate_production_batch.py::SEED_NU_RANGE`. Script pilot một lần đã xóa sau khi kết quả được ghi đầy đủ vào log này. Trên nhánh `main`, uncommitted.

---

### 2026-08-18 (tiếp) - A4: thêm ν0 làm input phụ cho CNN surrogate, retrain, so R² với baseline - ĐẠT sàn, không thấy lợi ích accuracy rõ rệt

Bối cảnh: A3 (mục trên) xác nhận không có vách cắt hội tụ cứng theo ν0, mở đường cho A4 (docs/archive/PROJECT_PLAN.md Nhóm 1, bắt buộc trước A6) - dạy CNN surrogate quan hệ hình học+ν0→tính chất, vì khi ν0 biến thiên quan hệ này không còn là hàm 1-1 của ảnh mật độ.

**Sinh dữ liệu:** `generate_production_batch.py --n-raw 3000 --vary-nu --run-dir outputs/phase3_a4_nu_raw` (dải đầy đủ ν0∈(0.2,0.4), không thu hẹp theo seed - xem quyết định A3) - 3000 mẫu, 2635 sạch (87,8%), 4122s (~69 phút, 10 worker).

**Plumbing (tương thích ngược hoàn toàn, mặc định TẮT):** `pipeline/phase4_surrogate/model.py::SurrogateCNN(include_nu0=...)` - concat ν0 vào `fc_in` cùng seed one-hot sau GAP, `forward(image, seed_vec, nu0=None)`; `dataset.py::AuxeticDataset(include_nu0=...)` - trả về ν0 như phần tử thứ 4 (KHÔNG lẫn vào `targets`, vì ν0 là input không phải mục tiêu dự đoán); `train.py`/`evaluate.py` wiring + `--include-nu0` flag; `pipeline/phase3_dataset/build_npz.py` thêm field `nu` (fallback 0.3). 10 test mới (`test_phase4_dataset.py`, `test_phase4_model.py`, `test_phase4_train.py`), 552/552 test toàn repo pass.

**Build dataset:** `analysis/scripts/assemble_phase3_a4.py` (mới) - gộp `outputs/phase3/dataset_64.npz` (13.624 mẫu, CHÍNH pool đã tạo baseline `surrogate_v2.pt`, ν0=0.3 fallback) + batch mới (2635 mẫu, ν0 thật) = 16.259 mẫu sạch → train 68.286 (sau augment x6, LỚN HƠN baseline 57.216 vì gộp thêm chứ không thay thế) / val 2439 / test 2439. Chủ động dùng `dataset_64.npz` thay vì `manifest_quality.csv` (chỉ 5.215 mẫu sạch) để tránh nhiễu "ít dữ liệu hơn" lẫn vào so sánh hiệu ứng ν0.

**Train 2 model trên CÙNG dataset A4 để tách bạch hiệu ứng ν0 khỏi hiệu ứng cỡ dữ liệu** (60 epoch, không early-stop, val_loss vẫn cải thiện tới cuối - GPU, ~17s/epoch):

| Model | R²(v12) | R²(v21) | R²(volfrac) |
|---|---|---|---|
| baseline `surrogate_v2.pt` (57.216 mẫu, ν0=0,3 cố định) | 0,974 | 0,964 | 0,983 |
| `surrogate_a4_control.pt` (68.286 mẫu, KHÔNG ν0) | **0,9816** | **0,9734** | 0,9889 |
| `surrogate_a4_nu0.pt` (68.286 mẫu, CÓ ν0 làm input) | **0,9816** | **0,9731** | 0,9887 |

**Kết luận:** sàn cứng ĐẠT (CLAUDE.md: R² không được thấp hơn baseline) - cả 2 model A4 đều vượt 0,974/0,964 rõ ràng, chủ yếu nhờ dataset lớn hơn (68.286 vs 57.216). Nhưng so `control` vs `nu0` (cùng dataset, chỉ khác việc có đưa ν0 vào model hay không): **khác biệt R² không đáng kể** (Δv12=0,0000, Δv21=-0,0003 - trong nhiễu train-to-train, không phải tín hiệu thật) - thêm ν0 làm input KHÔNG cải thiện đo được độ chính xác trên phân bố dữ liệu hiện tại, vì mẫu ν0 biến thiên chỉ chiếm ~16% dataset (2635/16259). Đây KHÔNG phải thất bại của A4: mục tiêu chính là hạ tầng (surrogate giờ NHẬN ν0 làm input, sẵn sàng cho A6 conditioning) chứ không phải tăng R² trên phân bố hiện tại vốn vẫn chủ yếu ν0=0,3; muốn đo lợi ích accuracy thật cần tăng tỉ trọng mẫu ν0 biến thiên trong dataset ở lần lặp sau.

**Đánh đổi 60/40 (CLAUDE.md):** không có đánh đổi thật ở đây - sàn đạt mà không phải hy sinh gì, vì `control` cũng vượt sàn nhờ dataset lớn hơn. Việc thêm ν0 (dù chưa đo được lợi ích) được giữ lại vì đây là mục tiêu hạ tầng bắt buộc của Giai đoạn A (A4 → A6), không phải một lựa chọn accuracy/performance.

**Checkpoint:** `outputs/phase4/surrogate_a4_nu0.pt` (khuyến nghị dùng cho A6), `surrogate_a4_control.pt` (đối chứng, giữ để tái kiểm chứng sau). Chưa promote đè `surrogate_v2.pt`/`outputs/phase3/` production - cần quyết định riêng có "chốt" dataset A4 làm production mới hay không (E0 CHƯA làm, A5 - real_physics.py per-sample nu - vẫn chặn A6 dùng differentiable-physics training như hiện tại dùng cho `cvae_realphysics.pt`). Code: `analysis/scripts/assemble_phase3_a4.py` (mới). Trên nhánh `main`, uncommitted.

---

### 2026-08-18 (tiếp) - A5: `real_physics.py` nhận `nu`/`E0` per-sample - rủi ro cache trong plan KHÔNG xảy ra

Bối cảnh: A4 (mục trên) đã cho surrogate Phase 4 nhận ν0 làm input; A5 (docs/archive/PROJECT_PLAN.md Nhóm 1, chặn A6) là bước tiếp theo - `RealPhysicsNu.forward`/`solve_nu_with_grad` trong `pipeline/phase5_cvae/real_physics.py` trước đó chỉ nhận `nu`/`E0`/`Emin` là 3 scalar áp CHUNG cho cả batch, không cho phép mỗi sample có ν0 riêng khi training differentiable-physics.

**Rủi ro nêu trong plan:** `_get_mesh` cache theo key `(nelx,nely,E0,Emin,nu)` - lo ngại per-sample nu sẽ làm mỗi sample phải build mesh riêng, mất tác dụng tăng tốc cache.

**Kiểm tra lại bằng đo thật (không suy diễn):** `Material(E0,Emin,nu).__init__` chỉ tích phân Gauss 2×2 ra ma trận `KE` 8×8, đo `timeit` = **~97 µs/lần construct**, so với FE-solve ~50-100 ms/mẫu (đã benchmark trước đó) thì chiếm ~0,1-0,2% - không đáng kể. Phần thật sự đắt (`build_dof_mesh`/`build_pbc` dựng `edofMat`/`iK`/`jK`/`pbc`) chỉ phụ thuộc `(nelx,nely)`, KHÔNG phụ thuộc vật liệu.

**Sửa:** tách `_MESH_TOPOLOGY_CACHE` (key `(nelx,nely)`, cache đúng phần đắt) khỏi việc dựng `Material` (rẻ, dựng mới mỗi lần gọi `_get_mesh`, cho phép nu/E0 khác nhau từng sample mà không tốn cache riêng). `RealPhysicsNu.forward` giờ nhận `E0`/`nu` là scalar (tương thích ngược, broadcast qua `_broadcast_per_sample()`) HOẶC mảng/list/tensor độ dài batch (mỗi sample 1 giá trị). `Emin`/`penal`/`rho0` vẫn dùng chung cho cả batch - không phải trục biến thiên của Giai đoạn A.

`pipeline/phase5_cvae/losses.py::real_physics_loss` (call site production duy nhất) KHÔNG cần sửa gì - hàm này chỉ pass-through `fe_params.get("nu"/"E0", ...)` thẳng vào `RealPhysicsNu.apply`, nên tự động hỗ trợ per-sample ngay khi `fe_params["nu"]` được set thành mảng ở A6 (hiện `CVAEDataset` chưa có field `nu0` - nối field đó là phạm vi A6, chưa làm ở đây).

7 test mới trong `tests/test_phase5_real_physics.py` (`TestPerSampleMaterial`): forward/backward khớp đúng khi gọi trực tiếp `solve_nu_with_grad` riêng lẻ với nu/E0 khác nhau từng sample, gradient không lẫn giữa các sample, multiprocessing (`n_workers>0`) khớp tuần tự với per-sample nu, và xác nhận `_MESH_TOPOLOGY_CACHE` không phình theo số giá trị nu (chỉ theo `(nelx,nely)`). **559/559 test toàn repo pass** (từ 552, +7 test mới).

**Còn lại của Giai đoạn A:** A6 (nối ν0/E0 làm condition cho `CVAEDataset`/cVAE, tái dùng pattern `extended_condition`) giờ không còn bị chặn kỹ thuật bởi A5. Code: `pipeline/phase5_cvae/real_physics.py`, `tests/test_phase5_real_physics.py`. Trên nhánh `substrate-material`, uncommitted.

---

### 2026-08-18 (tiếp) - A6: nối ν0 làm condition optional cho cVAE - HOÀN THÀNH hạ tầng toàn bộ Nhóm 1 (Giai đoạn A)

Bối cảnh: A5 (mục trên) đã gỡ điểm nghẽn kỹ thuật cuối cùng (per-sample ν0 trong `real_physics.py`); A6 là mảnh ghép cuối để CNN surrogate (A4) và differentiable-physics (A5) thực sự dạy được cVAE quan hệ ν0→hình học, thay vì chỉ có hạ tầng nằm im.

**Thiết kế:** `CVAEDataset(include_nu0=...)` thêm 2 cột `[nu0, nu0_mask]` vào CUỐI condition vector, ĐỘC LẬP với `extended_condition` (không gộp chung 1 cờ như volfrac/void_size_frac) - lý do: `outputs/phase3/*.npz` (dataset production hiện tại) KHÔNG có field `nu` (sinh trước A4), gộp chung sẽ phá vỡ mọi lệnh `--extended-condition` hiện có. condition_dim giờ composable ∈ {2,4,6,8} tùy tổ hợp 2 cờ. `include_nu0=True` mà npz thiếu field `nu` → raise `ValueError` rõ ràng (validate ở biên), không âm thầm coi ν0=0.3.

**Điểm quan trọng nhất (tận dụng đúng A5):** `losses.py::real_physics_loss` thêm tham số `nu0_col` - khi có, trích ν0 THẬT per-sample từ `condition[:, nu0_col]` (nếu mask=1) làm FE-solve dùng đúng vật liệu của từng mẫu, thay vì `fe_params['nu']=0.3` cố định cho cả batch như trước (sẽ tính SAI vật lý cho mọi mẫu có ν0≠0.3 nếu không sửa - đây chính là lý do A5 phải làm trước A6). mask=0 (dropout/không chỉ định) fallback về `fe_params['nu']` mặc định, không dùng giá trị cột đã bị zero (0.0 không có nghĩa vật lý "ν0=0", chỉ có nghĩa "không biết"). `apply_condition_dropout()` tổng quát hóa nhận `optional_pairs` thay vì hardcode `((2,3),(4,5))`, để nu0 (cột (2,3) hoặc (6,7) tùy `extended_condition`) cũng được dropout đúng kiểu classifier-free-guidance.

**Bug phát hiện khi triple-check (không phải do A6 gây ra trực tiếp nhưng cùng root cause):** `evaluate.py`/`best_of_n_eval.py` suy `extended_condition` từ `condition_dim == 6` - biểu thức này BỎ SÓT `include_nu0` khi `condition_dim` ∈ {4,8}, sẽ tạo `CVAEDataset` sai `condition_dim`, crash shape-mismatch ngay khi decode/encode. Thêm `dataset.py::condition_flags_from_dim()` làm NGUỒN DUY NHẤT suy `(extended_condition, include_nu0)` từ `condition_dim`, dùng lại ở cả 2 file - tránh lệch nhau lần thứ 3 nếu condition_dim mở rộng tiếp (vd Nhóm 6 - CTE). Phát hiện thêm 1 bug CÙNG LOẠI ở `self_play.py::verify_round` (tồn tại từ trước, kể cả với `condition_dim=6`, không phải do A6) - đã tách task riêng (`spawn_task`) thay vì sửa lẫn vào commit A6, giữ diff tập trung.

**CLI mới:** `train.py --include-nu0` (+ `--data-dir` để trỏ dataset có field `nu`, vd `outputs/phase3_a4/`, vì `outputs/phase3/` mặc định không có), `sample.py --nu0`, `best_of_n_eval.py --nu0 --data-dir`.

**Test:** 33 test mới (`tests/test_phase5_dataset.py` `TestIncludeNu0`/mở rộng `TestBuildConditionVector`, `tests/test_phase5_losses.py` `TestRealPhysicsLossNu0Col`/`TestRealPhysicsPriorLossNu0Col`, `tests/test_phase5_train.py` `TestRunEpochIncludeNu0`/mở rộng `TestApplyConditionDropout`, `tests/test_phase5_sample.py` `TestNu0ConditionCli`, `tests/test_phase5_best_of_n_eval.py` `TestBestOfNConditionDimFlags`, `tests/test_phase5_evaluate.py` case condition_dim=8) - trong đó có test chứng minh trực tiếp `real_physics_loss` dùng ĐÚNG ν0 per-sample (so target = giá trị giải bằng `solve_nu_with_grad` với đúng ν0 từng mẫu, loss phải ~0; và test đối chứng KHÔNG truyền `nu0_col` cho loss KHÁC 0 rõ rệt trên cùng input, xác nhận test đầu không đúng do trùng hợp). **593/593 test toàn repo pass** (từ 559).

**Kết luận Nhóm 1 (Giai đoạn A):** hạ tầng đã THÔNG SUỐT từ Phase 1 (`PARAM_SPACE['nu']`) → Phase 2/3 (`build_npz.py` field `nu`) → Phase 4 (`include_nu0` surrogate, A4) → Phase 5 (`real_physics.py` per-sample A5 + `CVAEDataset`/cVAE condition A6). Việc còn lại KHÔNG phải viết thêm code hạ tầng, mà là chạy thí nghiệm thật: train 1 checkpoint `--include-nu0` trên `outputs/phase3_a4/`, đo R²(FE) so với baseline - CHƯA làm ở đây (ngoài phạm vi 1 lần sửa hạ tầng, cần thời gian train + đánh giá riêng). E0 vẫn ngoài phạm vi toàn bộ Nhóm 1 (không có field ở bất kỳ tầng nào của pipeline - `FIXED_PARAMS['E0']=199.0` cố định từ Phase 1).

Code: `pipeline/phase5_cvae/dataset.py`, `train.py`, `losses.py`, `sample.py`, `best_of_n_eval.py`, `evaluate.py`. Trên nhánh `substrate-material`, uncommitted.

---

### 2026-08-19 - Giai đoạn A hoàn thành: đo lợi ích thật của ν0 (chạy thí nghiệm cVAE `--include-nu0` đầu tiên) - CÓ lợi ích đo được, cùng 3 bug hạ tầng chặn thí nghiệm bị phát hiện + sửa

Bối cảnh: A6 (mục trên) đã hoàn thành hạ tầng ν0 xuyên suốt Phase 1→5 nhưng chưa từng chạy thí nghiệm thật - PROJECT_PLAN.md ghi rõ "việc còn lại là chạy thí nghiệm thật: train 1 checkpoint `--include-nu0` trên `outputs/phase3_a4/`, đo R²(FE) so với baseline". Lần chạy đầu tiên này phát hiện hạ tầng A4-A6 **chưa từng được thực thi end-to-end** - lộ ra 3 bug chặn cứng, không phải do A6 gây ra trực tiếp mà do chưa ai thực sự chạy qua đường này.

**Bug 1 - `losses.py::load_frozen_surrogate()` bỏ qua field `include_nu0`:** dựng `SurrogateCNN(include_nu0=False mặc định)` bất kể checkpoint thật (`surrogate_a4_nu0.pt`, từ A4) có `include_nu0=True` - crash `size mismatch` ngay ở `fc.0.weight` (268 vs 267, thiếu đúng 1 chiều ν0) khi vừa load. Sửa: đọc `ckpt.get("include_nu0", False)` giống hệt pattern `n_outputs` đã sửa 2026-08-15.

**Bug 2 - `property_consistency_loss()`/`cvae_loss()` không truyền ν0 vào surrogate:** sau khi sửa Bug 1, surrogate load được nhưng `forward()` raise `ValueError` vì `include_nu0=True` bắt buộc nhận kwarg `nu0`, trong khi `property_consistency_loss()` gọi `surrogate(recon, seed_vec)` thiếu tham số này. Sửa: thêm `nu0_col` (cùng quy ước với `real_physics_loss.nu0_col` đã có từ A6) vào `property_consistency_loss()`/`cvae_loss()`, nối qua `run_epoch()` - mask=0/không truyền `nu0_col` fallback về 0.3 (khớp `OLD_NU_FALLBACK`).

**Bug 3 - `best_of_n_eval.py` verify FE dưới vật liệu SAI:** `evaluate_density_field()` luôn dùng `FE_PARAMS['nu']=0.3` CỐ ĐỊNH cho MỌI condition khi đo R²(FE) chính thức, bất kể target ν0 thật khác 0.3 - hình học ĐÚNG cho ν0 mục tiêu vẫn bị chấm sai vì bị verify dưới vật liệu khác nó được thiết kế cho. Đây là bug nghiêm trọng nhất trong 3 bug vì nó làm SAI chính con số dùng để kết luận thí nghiệm, không chỉ crash. Sửa: thêm `nu0_col` override `fe_params['nu']` theo đúng ν0 từng condition trước khi gọi `evaluate_density_field()` (cả nhánh best-of-N và single-shot), cùng quy ước `nu0_col` với 2 bug trên.

Cả 3 bug đều **cùng 1 root cause**: A4/A5/A6 mở rộng hạ tầng nhận ν0 nhưng chỉ nối tới đúng 1-2 call site đã kiểm chứng bằng test đơn vị (thường dùng surrogate/dummy KHÔNG có `include_nu0=True` nên không phơi ra bug) - chưa từng có 1 lần chạy end-to-end thật với checkpoint `include_nu0=True` thật để bắt các call site còn sót. 5 test hồi quy mới thêm (`tests/test_phase5_losses.py::TestLoadFrozenSurrogate`/`TestPropertyConsistencyLoss`, `tests/test_phase5_train.py::TestRunEpochIncludeNu0`, `tests/test_phase5_best_of_n_eval.py::TestBestOfNConditionDimFlags`) dùng surrogate/stub `include_nu0=True` THẬT để đóng đúng lỗ hổng coverage này. 599/599 test toàn repo pass.

**Quy trình 2-stage** (base rồi fine-tune real-physics, bắt buộc theo kinh nghiệm 2026-07-25 "train from-scratch với real-physics thất bại"), trên `outputs/phase3_a4/` (68.286 mẫu train, ν0∈[0.20,0.40] thật cho ~16% mẫu, còn lại ν0=0.3 fallback từ pool cũ):

| Checkpoint | Surrogate dùng | Base R²(FE, 8 cond) | Fine-tune R²(FE, 8 cond) |
|---|---|---|---|
| `cvae_a4_control_base.pt` → `cvae_a4_control_finetuned.pt` (condition_dim=2, KHÔNG ν0) | `surrogate_a4_control.pt` | 0,042 | 0,9596 |
| `cvae_a4_nu0_base.pt` → `cvae_a4_nu0_finetuned.pt` (condition_dim=4, CÓ ν0) | `surrogate_a4_nu0.pt` | 0,600 | 0,9546 |

**Đo chính thức (`best_of_n_eval.py --n-conditions 300 --n-samples 30`, oracle, cùng test set `outputs/phase3_a4/test.npz`, SAU khi sửa Bug 3 để verify đúng ν0 từng condition):**

| Checkpoint | R²(FE, n=300) | hit_rate single-shot | hit_rate best-of-N | frac_manufacturable |
|---|---|---|---|---|
| `cvae_a4_control_finetuned.pt` (không ν0) | 0,9776 | 0,9967 | 1,000 | 0,312 |
| `cvae_a4_nu0_finetuned.pt` (có ν0) | **0,9843** | 0,9967 | 1,000 | **0,365** |

**Kết luận:** thêm ν0 làm condition cho cVAE cho lợi ích đo được, nhất quán (không chỉ trong nhiễu train-to-train) trên CẢ 2 trục: chính xác Poisson (ΔR²=+0,0067) VÀ khả năng chế tạo (Δfrac_manufacturable=+0,053) - khác với A4 (surrogate Phase 4), nơi thêm ν0 KHÔNG cho lợi ích đo được vì lúc đó dữ liệu ν0 biến thiên chỉ ~16%. Ở tầng cVAE, differentiable real-physics loss (fine-tune) dùng ĐÚNG ν0 per-sample (nhờ A5/A6) để tính gradient - đây có thể là lý do lợi ích xuất hiện rõ hơn ở cVAE so với surrogate thuần túy (property-consistency loss của surrogate không có cùng độ chính xác vật lý). Sàn cứng CLAUDE.md (R² không thấp hơn baseline) không áp dụng trực tiếp ở đây vì đây là 2 checkpoint MỚI trên dataset MỚI (không so được thẳng với `cvae_v2_finetuned.pt` train trên dataset ν0=0.3 cố định) - phép so sánh hợp lệ duy nhất là control-vs-nu0 CÙNG dataset, đã thực hiện đúng ở trên.

**Giai đoạn A (Nhóm 1, PROJECT_PLAN.md) coi là HOÀN THÀNH đầy đủ về khoa học** (hạ tầng + lợi ích đo được), không chỉ hạ tầng như trước 2026-08-19. Checkpoint: `outputs/phase5/cvae_a4_nu0_finetuned.pt`. Chưa promote đè `cvae_v2_finetuned.pt` làm checkpoint production mặc định (cần quyết định riêng có "chốt" dataset A4 ν0-biến thiên làm production hay không - ngoài phạm vi 1 lần đo lợi ích). Báo cáo đầy đủ: `outputs/phase5/self_play/best_of_n_a4_{control,nu0}.json`. Code: `pipeline/phase5_cvae/losses.py`, `train.py`, `best_of_n_eval.py`, `tests/test_phase5_{losses,train,best_of_n_eval}.py`. Trên nhánh `substrate-material`, uncommitted.

### 2026-08-20 - Pilot "curse of dimensionality" (Nhóm 4.2): retrieval KHÔNG suy yếu rõ rệt ở 4D - giả thuyết chưa được ủng hộ ở mức mở rộng hiện có

Bối cảnh: `PROJECT_PLAN.md` Nhóm 4.2 nêu giả thuyết - retrieval thắng cVAE trong-phân-phối ở 2D (`v12,v21`, R²=1,000 vs 0,973, `LIMITATIONS.md` mục 18) là vì dataset đủ dày (`mean_condition_dist`=0,002), nhưng khi Giai đoạn A thêm ν0 làm chiều điều kiện thứ 3-4 thì khoảng cách nearest-neighbor của retrieval "được dự đoán sẽ tăng nhanh hơn sai số cVAE" - đây được ghi rõ là "chưa kiểm chứng, hiện chỉ là giả thuyết". Đây là bài kiểm định trực tiếp đầu tiên.

**Thiết kế:** script mới `analysis/scripts/curse_of_dimensionality_comparison.py` (tái dùng `best_of_n()`, `_hit_rate_and_r2()` có sẵn, không viết lại logic FE/sampling). Lấy 24 target `(v12,v21,ν0)` là joint sample THẬT từ `outputs/phase3_a4/test.npz` (không bịa như thiết kế OOD 2026-08-04, vì ở đây đang đo hiệu ứng MẬT ĐỘ trong-phân-phối, không phải khả năng ngoại suy - cần giữ đúng tương quan dữ liệu thật). So sánh CÙNG 1 tập target giữa 2 không gian điều kiện: "2D" (`cvae_a4_control_finetuned.pt`, condition_dim=2) và "4D" (`cvae_a4_nu0_finetuned.pt`, condition_dim=4, + ν0). Nearest-neighbor ở 4D dùng khoảng cách Euclid **chuẩn hoá z-score** (bắt buộc - std(ν0)≈0,023 nhỏ hơn ~9 lần std(v12,v21)≈0,20 trên `outputs/phase3_a4/train.npz`, đo trực tiếp; nếu dùng khoảng cách thô thì chiều ν0 gần như không ảnh hưởng kết quả nearest-neighbor, làm sai lệch phép đo thành do scale chứ không phải do số chiều).

| Không gian | Retrieval R² | Retrieval mean_dist (z-score) | cVAE best-of-10 R² |
|---|---|---|---|
| 2D (v12,v21) | 1,000 | 0,0060 | 0,975 |
| 4D (+ν0) | 1,000 | **0,0077 (+28%)** | 0,975 |

**Kết quả:** retrieval KHÔNG suy yếu ở 4D - vẫn R²=1,000 tuyệt đối, khoảng cách nearest-neighbor chỉ tăng nhẹ 28% (không phải "dốc đứng" như giả thuyết mô tả). cVAE R² không đổi giữa 2 không gian (0,975 cả hai, dưới retrieval ở cả hai). Nguyên nhân nhiều khả năng: `outputs/phase3_a4/train.npz` vẫn rất dày (68.286 mẫu) ngay cả sau khi thêm 1 chiều ν0 hẹp (dải chỉ [0,20; 0,40], std thật 0,023) - phép "tăng cấp số nhân theo chiều" trong giả thuyết cần dải giá trị rộng hơn nhiều hoặc số chiều cao hơn nhiều (6D với volfrac/void_size_frac qua `--extended-condition`) mới bộc lộ rõ, và **hiện KHÔNG có checkpoint condition_dim∈{6,8} nào đã train** để test trực tiếp (đã kiểm tra toàn bộ `outputs/phase5/*.pt`) - đây là việc lớn hơn nhiều (cần train mới 2-stage, không phải chạy script).

**Kết luận cẩn trọng:** Nhóm 4.2 CHƯA đóng được ở mức 4D hiện có - bằng chứng "curse of dimensionality" chưa xuất hiện rõ, khác kỳ vọng ban đầu của roadmap. Không nên diễn giải kết quả này thành "retrieval luôn thắng bất kể số chiều" (mới test 1 điểm 4D, dải ν0 hẹp) cũng không nên coi giả thuyết đã bị bác bỏ hẳn - chỉ là CHƯA đo được ở quy mô hiện có. Muốn kiểm định đầy đủ giả thuyết trung tâm của Nhóm 4.2 cần: (a) train checkpoint `--extended-condition` (condition_dim=6/8) trên dataset có volfrac/void_size_frac biến thiên đủ rộng, HOẶC (b) thiết kế lại phép đo bằng cách chủ động làm thưa tập train (subsample) để mô phỏng mật độ 6D mà không cần train mới.

Code: `analysis/scripts/curse_of_dimensionality_comparison.py` (mới). Kết quả đầy đủ: `outputs/phase5/reports/curse_of_dimensionality_comparison.json`. Trên nhánh `substrate-material`, uncommitted.

### 2026-08-20 (tiếp) - Train checkpoint 8D (`--extended-condition --include-nu0`) + đo lại 2D→4D→8D: khoảng cách retrieval NHẢY VỌT ở 8D (bằng chứng ủng hộ 1 nửa giả thuyết), nhưng cVAE KHÔNG giữ vững độ chính xác tốt hơn retrieval - kết quả trái kỳ vọng ở nửa còn lại

Bối cảnh: pilot ở trên (4D) không thấy "curse of dimensionality" rõ rệt. `outputs/phase3_a4/` hoá ra đã có sẵn `volfrac_achieved` (0,30-0,72) và `void_size_frac` (0,10-0,55, qua `params`/`param_names`) với dải biến thiên đủ rộng - không cần backfill gì để test 8D thật (`v12,v21,volfrac,void_size_frac,nu0` = 5 biến vật lý + 3 mask).

**Train 2-stage** (`--extended-condition --include-nu0 --data-dir outputs/phase3_a4`, đúng recipe A7): Stage 1 base early-stop epoch 22 (~15 phút, val_loss=760,14) → `cvae_a4_full8d_base.pt`. Stage 2 fine-tune real-physics (`--lambda-real-physics 20.0 --real-physics-subsample 8 --real-physics-every 2 --real-physics-workers 8`, đúng cấu hình đã kiểm chứng cho fine-tune 35-epoch trước đây) early-stop epoch 19 (~15 phút, val_loss=787,32, `real_physics` loss giảm đều 0,0106→0,0006, không NaN/instability) → `cvae_a4_full8d_finetuned.pt` (condition_dim=8, verify trực tiếp qua checkpoint metadata).

Chạy lại `curse_of_dimensionality_comparison.py` (mở rộng thêm `space_8d`, CÙNG 24 target, cùng seed=123):

| Không gian | Retrieval R² | Retrieval mean_dist (z-score) | cVAE best-of-10 R² |
|---|---|---|---|
| 2D (v12,v21) | 1,000 | 0,0060 | 0,975 |
| 4D (+ν0) | 1,000 | 0,0077 (+28%) | 0,975 |
| 8D (+volfrac,void_size_frac) | 0,998 | **0,0841 (+992% so 4D)** | **0,943** |

Kiểm tra thêm (không chỉ dựa vào mean, tránh outlier đánh lừa): median `condition_dist_z` cũng nhảy tương tự (4D: 0,0026 → 8D: 0,0733, ~28 lần) và **min** ở 8D (0,029) đã cao hơn cả **max** ở 2D/4D (0,025/0,042) - tức toàn bộ phân phối dịch chuyển, không phải do vài outlier.

**Diễn giải 2 chiều, không phóng đại theo hướng nào:**
- **Nửa ĐƯỢC ủng hộ:** khoảng cách nearest-neighbor của retrieval THẬT SỰ nhảy vọt ở 8D (~30 lần so với 4D) - đây chính là tín hiệu "curse of dimensionality" mà 4D chưa thấy được. Dataset 68k mẫu vẫn dày ở 4D nhưng bắt đầu loãng rõ rệt ở 8D.
- **Nửa KHÔNG được ủng hộ:** hệ quả về ĐỘ CHÍNH XÁC không đi theo hướng giả thuyết cần - retrieval R² chỉ giảm rất nhẹ (1,000→0,998, vẫn gần như hoàn hảo dù khoảng cách xa hơn nhiều), trong khi **cVAE R² giảm NHIỀU HƠN** (0,975→0,943). Nếu chỉ nhìn con số này, kết luận đảo ngược hoàn toàn kỳ vọng: retrieval "chịu đựng" tốt hơn cVAE khi thêm chiều, không phải ngược lại.

**2 nghi vấn confound quan trọng, CHƯA loại trừ được (khác với nghi vấn cVAE thắng/thua thật):**
1. **cVAE 8D có thể chưa train đủ chín:** checkpoint 4D/2D đã qua nhiều vòng tinh chỉnh lịch sử (gamma sweep, nhiều lần fine-tune) trước khi có số liệu chính thức `R²(FE,n=300)`; checkpoint 8D hôm nay chỉ mới 1 lần train 2-stage (22+19=41 epoch tổng), CHƯA qua quy trình chọn checkpoint kỹ như các checkpoint production khác - R²=0,943 có thể phản ánh model chưa đủ trưởng thành, không phải giới hạn kiến trúc.
2. **Nhầm lẫn giữa "curse of dimensionality" và "bài toán khó hơn":** thêm volfrac/void_size_frac làm target không chỉ tăng SỐ CHIỀU mà còn tăng SỐ RÀNG BUỘC đồng thời cVAE phải thỏa mãn cùng lúc (đúng v12/v21 VÀ đúng volfrac VÀ đúng void_size_frac VÀ đúng ν0) - đây là 1 bài toán sinh khó hơn về bản chất, độc lập với hiệu ứng mật độ dữ liệu mà giả thuyết Nhóm 4.2 muốn đo. Phép đo hiện tại KHÔNG tách được 2 hiệu ứng này.
3. Cỡ mẫu n=24 (giống mọi benchmark khác trong dự án) - CI rộng, chưa đủ để khẳng định chênh lệch 0,975 vs 0,943 có ý nghĩa thống kê hay không.

**Kết luận cẩn trọng:** Nhóm 4.2 vẫn CHƯA đóng được, nhưng theo hướng khác pilot 4D - giờ có bằng chứng thật về hiệu ứng mật độ (retrieval distance tăng mạnh), nhưng bằng chứng đó KHÔNG tự động chuyển thành lợi thế accuracy cho cVAE trong lần đo này, và có 2 confound hợp lý (undertraining + task khó hơn) chưa loại trừ được trước khi kết luận bất cứ điều gì chắc chắn. Không dùng số liệu 8D này để viết vào bài báo ở dạng hiện tại - cần ít nhất: (a) đưa checkpoint 8D qua cùng quy trình tinh chỉnh/chọn lựa như 2D/4D trước khi so sánh công bằng, (b) tách riêng phép đo mật độ (chỉ tính khoảng cách, không cần sinh mẫu) khỏi phép đo độ khó bài toán sinh.

Code: `analysis/scripts/curse_of_dimensionality_comparison.py` (mở rộng thêm `space_8d`). Checkpoint mới: `outputs/phase5/cvae_a4_full8d_{base,finetuned}.pt`. Kết quả đầy đủ: `outputs/phase5/reports/curse_of_dimensionality_comparison.json` (ghi đè, có cả 3 không gian). Trên nhánh `substrate-material`, uncommitted.

### 2026-08-20 (tiếp) - Sửa bug `self_play.py` không hỗ trợ checkpoint mở rộng (LIMITATIONS.md mục 22)

`verify_round()` từng luôn dựng `CVAEDataset(test.npz)` với `condition_dim=2` mặc định bất kể checkpoint thật, và hardcode `PHASE3_DIR=outputs/phase3` (không có field `nu`) - chấm checkpoint `include_nu0`/`extended_condition` sẽ crash hoặc âm thầm sai. Sửa bằng đúng pattern đã dùng ở `best_of_n_eval.py` (A6/A7): đọc `condition_dim` trực tiếp từ checkpoint, suy `(extended_condition, include_nu0)` qua `condition_flags_from_dim()`, dựng `CVAEDataset` đúng cờ, thêm tham số `data_dir`/`--data-dir`.

**Phát hiện thêm trong lúc sửa (chưa có trong mục 22 gốc):** cùng lúc đó, `evaluate_density_field()` trong `verify_round()` cũng bị đúng bug Bug 3 của A7 (verify dưới `FE_PARAMS['nu']` cố định thay vì ν0 thật từng target) - vá bằng `nu0_col` cùng quy ước với `best_of_n_eval.py`/`losses.py::real_physics_loss`.

**Chủ động không mở rộng phạm vi:** vòng lặp round-trip ĐẦY ĐỦ (`run()`, bước 2/4 gọi subprocess `phase4_surrogate/train.py`/`phase5_cvae/train.py`) vẫn CHƯA truyền `--data-dir`/`--extended-condition`/`--include-nu0` - chỉ sửa đường "chấm điểm 1 checkpoint có sẵn" (`verify_round()`), không sửa đường "chạy self-play từ đầu trên checkpoint mở rộng" (việc lớn hơn, ngoài phạm vi bug #22, đã ghi rõ trong docstring `run()` để không quên).

3 test mới (`TestVerifyRoundConditionDimFlags`): không crash ở condition_dim=4, verify đúng ν0 per-target (không phải hằng số 0,3), và regression condition_dim=2 không đổi hành vi. 602/602 test toàn repo pass.

Code: `pipeline/phase5_cvae/self_play.py`, `tests/test_phase5_self_play.py`. Trên nhánh `substrate-material`, uncommitted.

### 2026-08-20 (tiếp) - Phát hiện + sửa bug thật trong `analysis/pareto/frontier.py::is_pareto_efficient()` (chưa từng có test), rồi hoàn thành Nhóm 3.2

Bối cảnh: bắt tay vào Nhóm 3.2 (`PROJECT_PLAN.md`) - đối chiếu composite score (`best_of_n_eval.py`) với Pareto front độc lập. Viết vòng lặp xếp tầng Pareto (NSGA-II style: lặp lại `is_pareto_efficient()`, mỗi lần loại 1 tầng) trên 30 ứng viên/target - **treo vô hạn ngay lần chạy đầu** (~1 giờ không ra kết quả trên tập 24 target thật, dù đo trực tiếp 1 condition×3 mẫu chỉ mất 0,6s - loại trừ được nghi ngờ "máy chậm").

**Cô lập bằng test tối giản:** `is_pareto_efficient(np.array([[1.,1.],[2.,2.]]), maximize=True)` (2 điểm, điểm sau lấn át điểm trước hoàn toàn) trả về `[False, False]` - SAI, phải là `[False, True]`. Thử thêm `is_pareto_efficient(np.array([[1.,2.]]), maximize=True)` (1 điểm) → `[False]` - cũng SAI, phải `[True]` (1 điểm luôn Pareto-efficient tầm thường).

**Root cause:** bản gốc dùng vòng lặp có tác dụng phụ, đánh giá `is_efficient[i] = np.any(costs[i] > costs[i+1:], axis=0).all()` - tại `i = n_points-1` (phần tử CUỐI CÙNG của mảng), `costs[i+1:]` rỗng. `np.any()` trên tập rỗng đúng ngữ nghĩa numpy là `False`, nhưng `.all()` của giá trị `False` đó lại tiếp tục là `False` - trong khi ngữ nghĩa ĐÚNG cần ở đây là "không có điểm nào phía sau lấn át tôi → tôi hiệu quả → **True**". Hệ quả: **phần tử cuối cùng của MỌI mảng đầu vào luôn bị đánh dấu sai là không-Pareto-efficient**, bất kể giá trị thật - kể cả khi đó là điểm tốt nhất tuyệt đối. Khi dùng lặp lại để xếp tầng (loại tầng 1, tìm tầng 2 trên phần còn lại...), điểm cuối cùng còn sót lại ở mỗi tầng không bao giờ được xếp hạng → `remaining` không bao giờ rỗng → vòng lặp vô hạn.

**Mức độ nghiêm trọng:** đây là bug production thật (`analysis/pareto/frontier.py`, dùng bởi `analysis/pareto/runner.py` cho Pareto analysis Phase 1), **chưa từng có 1 test nào** cho module này trước đây. May mắn: kiểm tra `outputs/` không có thư mục `pareto/` nào - module này **chưa từng chạy thật cho kết quả nào đã công bố** trong README/docs/notebooks - không có claim khoa học cũ nào bị ảnh hưởng, chỉ là hạ tầng nằm im bị lỗi từ đầu.

**Sửa:** thay toàn bộ thuật toán bằng phiên bản O(n²) tường minh, không phụ thuộc thứ tự duyệt hay trường hợp biên mảng rỗng - với mỗi điểm i, kiểm tra trực tiếp có tồn tại điểm j≠i thỏa `costs[j]>=costs[i]` ở MỌI trục và `costs[j]>costs[i]` ở ÍT NHẤT 1 trục hay không (đúng định nghĩa Pareto dominance, vectorized, không có tác dụng phụ). Verify bằng tay 6 trường hợp (điểm tốt nhất ở giữa/cuối mảng, 1 điểm, 2 điểm lấn át, 2 điểm không lấn át nhau, minimize mode) - khớp kỳ vọng ở mọi trường hợp.

**Test mới:** `tests/test_pareto_frontier.py` (8 test, module CHƯA từng có test trước đây) - bao gồm test trực tiếp regression cho bug biên (điểm tốt nhất ở hàng cuối, 1 điểm, 2 điểm lấn át) và test lặp lại xếp tầng trên n=30 ngẫu nhiên xác nhận vòng lặp hội tụ. 612/612 test toàn repo pass.

**Sau khi sửa, hoàn thành Nhóm 3.2** (`notebooks/08_composite_score_pareto_validation.ipynb`, checkpoint `cvae_v2_finetuned.pt`, n=24 target × 30 mẫu/target, seed=123, chạy lại mất 62s - đúng như ước tính ban đầu, xác nhận bug ở trên là nguyên nhân duy nhất gây treo):

| Chỉ số | Giá trị | Mục tiêu (đặt trước khi chạy) |
|---|---|---|
| Spearman trung bình (composite score vs -tầng Pareto) | **0,693** | ≥ 0,7 — **KHÔNG đạt** (sát ngưỡng) |
| Spearman trung vị | **0,714** | ≥ 0,7 — **đạt** |
| Spearman min/max (24 target) | 0,432 / 0,889 | - (không đặt DoN riêng) |
| Tỉ lệ ứng viên thắng nằm ở tầng Pareto 1 | 1,000 (tất yếu toán học, không phải phát hiện thực nghiệm - xem giải thích dưới) | - (không đặt DoN riêng) |

**Đối chiếu tiêu chí đã đặt TRƯỚC khi chạy (Spearman trung bình ≥ 0,7):** **KHÔNG đạt** theo trung bình (0,693 < 0,7, sát ngưỡng), **đạt** theo trung vị (0,714). Phân phối trải liên tục 0,43-0,89 trên 24 target, không có nhóm outlier tách biệt - vài target tương quan yếu (0,43-0,56) kéo trung bình xuống dưới ngưỡng.

`frac_winner_on_pareto_front1=1,0` là **hệ quả tất yếu của toán học đa mục tiêu** (argmax của tổng có trọng số DƯƠNG luôn nằm trên Pareto front - nếu 1 điểm khác trội hơn cả 3 trục thì tổng có trọng số của nó cũng phải cao hơn, mâu thuẫn giả thiết argmax), KHÔNG phải bằng chứng thực nghiệm mới - không nên trích dẫn như 1 phát hiện.

**Kết luận cẩn trọng:** composite score có tương quan dương rõ ràng, mức trung bình-khá với cấu trúc Pareto thật (24/24 target dương, phần lớn >0,6) - ủng hộ MỘT PHẦN cho việc đây là cách tổng hợp hợp lý, không tùy tiện. Nhưng chưa đạt ngưỡng "rất mạnh" tự đặt ra theo tiêu chí trung bình - không đủ để tuyên bố "đã xác nhận" phương pháp luận mà không dè dặt. Nghi vấn nguyên nhân (chưa kiểm chứng): `accuracy_score` chuẩn hóa min-max NGAY TRONG pool đang xét (rank-based, phụ thuộc phân phối `|Δv12|` của batch đó) có thể tạo nhiễu khác với cấu trúc dominance tuyệt đối mà Pareto front đo. Chưa đủ cơ sở đổi trọng số mặc định 0,6/0,3/0,1 chỉ từ 1 lần đo này.

Code: `analysis/pareto/frontier.py` (sửa bug), `tests/test_pareto_frontier.py` (mới), `pipeline/phase5_cvae/best_of_n_eval.py` (thêm `return_all_scores=True`, backward-compatible, có test riêng ở `tests/test_phase5_best_of_n_eval.py`), `notebooks/08_composite_score_pareto_validation.ipynb` (mới). Kết quả đầy đủ: `outputs/phase5/reports/composite_score_pareto_validation.json`. Trên nhánh `substrate-material`, uncommitted.

### 2026-08-20 (tiếp) - Loại trừ confound #1 của pilot 8D: tune lại checkpoint bằng `--select-by fe_r2` xác nhận R² thấp trước đó là do landmine chọn checkpoint theo `val_loss`, KHÔNG phải giới hạn kiến trúc

Bối cảnh: pilot "curse of dimensionality" 8D (mục trên) đo được cVAE R²=0,943 - thấp hơn 4D (0,975) và thấp hơn retrieval ở 8D (0,998) - nhưng nêu rõ 2 nghi vấn confound chưa loại trừ, trong đó nghi vấn #1 là checkpoint `cvae_a4_full8d_finetuned.pt` chỉ mới train 1 lần bằng `--select-by val_loss` (mặc định) - đúng tổ hợp đã 2 lần xác nhận CHỌN NHẦM checkpoint trong lịch sử dự án (`LIMITATIONS.md` mục 12, cảnh báo tự in ra ngay khi chạy `train.py` không kèm `--select-by fe_r2`).

**Thử nghiệm loại trừ:** fine-tune lại TỪ CÙNG checkpoint base (`cvae_a4_full8d_base.pt`, không train lại từ đầu), CÙNG mọi tham số real-physics, chỉ đổi `--select-by fe_r2 --fe-eval-every 2 --n-fe-eval-conditions 8` (chọn checkpoint theo R² FE thật đo định kỳ, thay vì val_loss). Chạy đủ 35/35 epoch (không early-stop), R²(FE, n=8 condition validation) dao động mạnh giữa các epoch sau khi đạt đỉnh (0,977→0,79→0,94→0,68→0,83→0,87 ở epoch 24-34) - đúng bằng chứng trực tiếp cho thấy val_loss KHÔNG phản ánh đúng epoch nào thực sự tốt, và việc chọn theo val_loss (thay vì R2 thật) có thể vô tình giữ lại 1 trong các epoch tệ này. `--select-by fe_r2` giữ đúng epoch tốt nhất (24): **R²(FE)=0,9774** → `cvae_a4_full8d_finetuned_v2.pt`.

**Đo lại `curse_of_dimensionality_comparison.py` (CKPT_8D trỏ sang checkpoint mới, cùng 24 target, cùng seed):**

| Không gian | Retrieval R² | cVAE R² (checkpoint cũ, val_loss) | cVAE R² (checkpoint mới, fe_r2) |
|---|---|---|---|
| 8D | 0,998 | 0,943 | **0,997** |

**Kết luận:** nghi vấn confound #1 ĐƯỢC XÁC NHẬN ĐÚNG - cVAE R² thấp ở pilot 8D lần đầu là do checkpoint chưa được tune đúng cách (dính landmine `val_loss`), KHÔNG phải giới hạn kiến trúc/năng lực cVAE ở 8D. Sau khi chọn checkpoint đúng bằng R² FE thật, cVAE (0,997) gần như ngang bằng retrieval (0,998) ở 8D - đảo ngược hoàn toàn kết luận trước đó ("cVAE giảm nhiều hơn retrieval khi thêm chiều").

**Ý nghĩa cho Nhóm 4.2:** với bằng chứng mới này, retrieval vẫn chưa "thua" cVAE ở 8D theo accuracy (cả 2 gần như hoàn hảo, 0,997 vs 0,998) - luận điểm trung tâm "curse of dimensionality khiến cVAE thắng vì retrieval kém đi" **vẫn CHƯA được chứng minh về accuracy**, dù đã có bằng chứng thật về mật độ (retrieval distance +992% ở 8D). Confound #2 (thêm biến vừa tăng chiều vừa tăng ràng buộc) vẫn còn treo - nhưng ít nhất giờ đã tách được: sự sụt giảm R² quan sát trước đó là hiện tượng huấn luyện (training artifact), không phải hiện tượng khoa học cần giải thích.

Checkpoint mới: `outputs/phase5/cvae_a4_full8d_finetuned_v2.pt` (R²(FE,n=8)=0,9774, `--select-by fe_r2`). Code: `analysis/scripts/curse_of_dimensionality_comparison.py` (CKPT_8D cập nhật trỏ sang `_v2`). Kết quả đầy đủ: `outputs/phase5/reports/curse_of_dimensionality_comparison.json` (ghi đè). Trên nhánh `substrate-material`, uncommitted.

### 2026-08-20 (tiếp) - Thử hướng lập luận mới cho Nhóm 4.2 (manufacturability thay vì accuracy) ở 8D - kết quả NGƯỢC giả thuyết đề xuất, retrieval THẮNG cVAE cả về khả năng chế tạo

Bối cảnh: sau khi accuracy ở 8D gần như hòa (retrieval 0,998 vs cVAE 0,997, mục trên), có đề xuất hướng lập luận thay thế: retrieval buộc phải chọn mẫu có khoảng cách z-score xa (+992% so 4D) để khớp điều kiện 8D, có thể phải "hy sinh" tính toàn vẹn cấu trúc (liên thông, kích thước nét in, đối xứng) - trong khi cVAE (có `force_periodic()` + composite scoring ưu tiên manuf/aesthetic) giữ được cả hai.

**Lưu ý về giả thuyết trước khi đo (tránh lặp lại lỗi diễn giải có lợi):** retrieval KHÔNG sinh cấu trúc mới - nó trả về NGUYÊN XI 1 ảnh THẬT đã có sẵn trong `train.npz` (đã được SIMP tối ưu thật từ trước). Khoảng cách z-score xa chỉ ảnh hưởng độ khớp ĐIỀU KIỆN của mẫu được chọn, không ảnh hưởng gì đến chính cấu trúc vật lý của ảnh đó - nên giả thuyết "z-score xa → cấu trúc sụp đổ" không có cơ sở cơ chế rõ ràng, cần đo thật thay vì giả định.

**Thiết kế:** script mới `analysis/scripts/manufacturability_retrieval_vs_cvae_8d.py` - tái dùng CHÍNH KẾT QUẢ cVAE đã có sẵn từ `curse_of_dimensionality_comparison.py` (checkpoint `_v2.pt` đã tune đúng, không chạy lại model), chỉ viết thêm phần retrieval: lấy ảnh THẬT của nearest-neighbor 8D (z-score) cho CÙNG 24 target, áp `force_periodic()` (đúng pipeline hậu xử lý cVAE đang dùng, để so sánh công bằng), rồi chấm `check_manufacturability()`/`aesthetic_score()` - tái dùng nguyên hàm có sẵn trong `manufacturability.py`/`aesthetics.py`, không viết lại logic.

| Phương án | manuf_score (ứng viên được chọn) | aesthetic_score | Ghi chú |
|---|---|---|---|
| Retrieval (ảnh thật) | **0,889** | 0,913 | `passes_all` nghiêm ngặt: **20/24 (83,3%)** |
| cVAE (8D, đã tune) | 0,667 | 0,883 | `frac_manufacturable` trong pool N=10 mẫu/condition: 26,3% |

**Kết quả NGƯỢC hoàn toàn giả thuyết đề xuất: retrieval THẮNG cVAE cả về manuf_score lẫn aesthetic_score ở 8D**, không hề "sụp đổ cấu trúc" dù khoảng cách z-score xa. Điều này khớp đúng cơ chế đã nêu trước khi đo: retrieval trả về ảnh THẬT (đã qua tối ưu SIMP thật, vốn có tỉ lệ manufacturable cao hơn dataset thô lọc sẵn theo cách khác), còn cVAE dù có `force_periodic()` vẫn chỉ đạt tỉ lệ manufacturable tự nhiên thấp trong pool sinh ra (26,3%, khớp với con số nền `frac_manufacturable≈0,25-0,35` đã ghi ở [Giới hạn #2](docs/LIMITATIONS.md#giới-hạn-đã-biết--known-limitations)) - composite scoring chỉ CHỌN ứng viên tốt nhất trong N mẫu đã sinh, không thể tạo ra tính liên thông nếu không mẫu nào trong pool có sẵn.

**Ý nghĩa cho Nhóm 4.2:** hướng lập luận "cVAE thắng nhờ bảo toàn chất lượng cấu trúc" **KHÔNG thành lập** - dữ liệu đo được đi ngược hoàn toàn. Retrieval ở 8D hiện đang thắng cVAE trên CẢ accuracy (gần hòa, 0,998 vs 0,997) LẪN manufacturability (0,889 vs 0,667) - luận điểm trung tâm "cần generative vì retrieval kém đi ở nhiều chiều" vẫn chưa có bằng chứng thực nghiệm nào ủng hộ ở mức 8D hiện tại, dù bằng chứng về mật độ dữ liệu (khoảng cách z-score) là thật.

Code: `analysis/scripts/manufacturability_retrieval_vs_cvae_8d.py` (mới). Kết quả đầy đủ: `outputs/phase5/reports/manufacturability_retrieval_vs_cvae_8d.json`. Trên nhánh `substrate-material`, uncommitted.

### 2026-08-20 (tiếp) - Fast pilot "low-data regime" (chỉ retrieval, không train lại cVAE) - retrieval KHÔNG bế tắc khi khan hiếm dữ liệu, khai tử hướng lập luận này ở 8D mà không cần tốn giờ GPU nào

Bối cảnh: sau khi Hướng "manufacturability" (mục trên) cho kết quả ngược kỳ vọng, đề xuất tiếp theo là kiểm tra xem retrieval có "bế tắc" khi tập tra cứu bị thu nhỏ hay không (mô phỏng kịch bản miền vật lý mới khan hiếm dữ liệu) - trước khi quyết định có đáng đầu tư train lại cVAE trên từng mức dữ liệu nhỏ (1,5-3 giờ GPU, "bản công bằng") hay không. Chạy bản rẻ trước (chỉ retrieval, ~1,4 giây): giữ nguyên checkpoint cVAE 8D hiện tại làm mốc tham chiếu (R²=0,997, không train lại), chỉ giới hạn tập tra cứu của retrieval xuống Ndb=[500, 1.000, 5.000, 10.000, full=68.286], tính lại nearest-neighbor 8D z-score TRÊN ĐÚNG Ndb mẫu đó (chuẩn hoá lại theo chính subset, không phải toàn bộ train).

| Ndb | R²(retrieval) | MAE | mean_dist_z |
|---|---|---|---|
| 500 | 0,9837 | 0,0189 | 0,3040 |
| 1.000 | 0,9850 | 0,0173 | 0,2316 |
| 5.000 | 0,9950 | 0,0091 | 0,1428 |
| 10.000 | 0,9983 | 0,0055 | 0,1116 |
| Full (68.286) | 0,9977 | 0,0058 | 0,0841 |

**Kết quả rõ ràng (Kịch bản B, "bất lợi" theo đúng tiêu chí đã đặt trước khi chạy):** ngay cả khi chỉ còn 500/68.286 mẫu (~0,7% dữ liệu gốc), retrieval vẫn đạt R²=0,984 ở không gian 8D - khoảng cách z-score tăng ~3,6 lần (0,084→0,304) nhưng accuracy chỉ mất 1,4 điểm phần trăm. Retrieval KHÔNG "bế tắc" khi khan hiếm dữ liệu như giả thuyết kỳ vọng.

**Quyết định:** theo đúng tiêu chí đã thống nhất trước khi chạy pilot rẻ này - khai tử hướng lập luận "low-data regime" ở dạng đo trong-phân-phối (in-distribution) này, KHÔNG đầu tư bản "công bằng" (train lại cVAE trên từng mức Ndb, 1,5-3 giờ GPU) vì kết quả retrieval đã đủ rõ để không cần đối chứng tốn kém. Tiết kiệm được toàn bộ ngân sách GPU dự kiến cho hướng này.

**Ý nghĩa tổng hợp cho Nhóm 4.2 sau 3 thí nghiệm liên tiếp hôm nay (accuracy, manufacturability, low-data):** ở mức 8D hiện tại (in-distribution, joint sample thật từ test set), **cVAE không thắng rõ retrieval trên bất kỳ trục nào đã đo** - chỉ có bằng chứng mật độ dữ liệu (khoảng cách retrieval tăng) là thật nhưng chưa chuyển hóa thành lợi thế nào. Luận điểm trung tâm thật sự vững của dự án vẫn là Nhóm 4.1 (sign-flip OOD, R²=0,418 vs 0,057, 2026-08-04) - nơi retrieval THẬT SỰ sụp đổ vì bị ép ngoại suy ra ngoài phạm vi dữ liệu đã thấy, khác hẳn các thí nghiệm hôm nay (vẫn trong-phân-phối, chỉ đổi mật độ/số chiều).

Code: `analysis/scripts/retrieval_low_data_pilot_8d.py` (mới). Kết quả đầy đủ: `outputs/phase5/reports/retrieval_low_data_pilot_8d.json`. Trên nhánh `substrate-material`, uncommitted.

### 2026-08-24 - Benchmark ConvKAN Phase 4 và mất ổn định số học khi train WIRE

**ConvKAN:** đã train `SurrogateCNN(use_kan=True)` trên dataset production v2
(57.216 train / 2.044 validation), RTX 3050 6 GB, batch 128. Training dừng
early-stopping ở epoch 44/60 sau khoảng 12 phút; val loss tốt nhất = 0,00292.
Đánh giá độc lập trên `outputs/phase3/test.npz` cho kết quả:

| Target | R² | MAE | Mục tiêu R² (DoN Task 3) |
|---|---:|---:|---:|
| v12 | 0,9636 | 0,0235 | ≥ 0,985 |
| v21 | 0,9574 | 0,0255 | ≥ 0,985 |
| volfrac_achieved | 0,9161 | 0,0183 | ≥ 0,985 |

Checkpoint đã export cho Phase 5 tại
`outputs/phase4/surrogate_convkan_for_phase5_v2.pt`. ConvKAN hoạt động đúng và
tương thích metadata, nhưng **chưa đạt DoN cũ R²≥0,985** cho v12/v21; không
được promote thay checkpoint production hiện tại. Trong quá trình evaluate đã
bắt và sửa một bug wiring: evaluator Phase 4 không đọc `use_kan`, nên dựng
Linear rồi crash khi load các khóa `base_weight`/`spline_weight` của KAN.

**WIRE pilot:** cấu hình hidden=64, batch=16, 1 epoch chạy được, không OOM.
Full run đầu tiên (`lr=1e-3`, 30 epoch) thất bại ở epoch 5: KL/train loss tăng
đến khoảng 2,17×10¹³, sau đó output decoder không còn hữu hạn và BCE CUDA báo
`input_val >= zero && input_val <= one`. Đây là gradient explosion trong giai
đoạn KL warmup, không phải thiếu VRAM; checkpoint run lỗi không được dùng làm
kết quả.

**Biện pháp đang thử:** thêm gradient clipping `max_norm=1.0` sau backward và
giảm learning rate xuống `1e-4`, chạy lại với `CUDA_LAUNCH_BLOCKING=1` để bắt
lỗi đồng bộ. Test training sau thay đổi: 26 passed. Rerun WIRE đã khởi động
nhưng tại thời điểm ghi mục này chưa có epoch hoàn chỉnh để kết luận; cần chờ
metric finite trước khi full training/FE verification.

Code/report liên quan: `pipeline/phase4_surrogate/model.py`,
`pipeline/phase4_surrogate/evaluate.py`, `pipeline/phase4_surrogate/train.py`,
`pipeline/phase5_cvae/train.py`, `outputs/phase4/evaluation_convkan_v2.json`.

### 2026-08-24 - WIRE checkpoint train được nhưng thất bại khi kiểm chứng FE

Checkpoint `outputs/phase5/cvae_wire_v2.pt` đã train đủ 30/30 epoch trên RTX
3050 6 GB với `wire_hidden_dim=64`, batch=16, `lr=1e-4` và gradient clipping
`max_norm=1.0`. Run ổn định số học sau khi sửa gradient explosion, nhưng
validation loss không dự báo được chất lượng FE thật.

Benchmark best-of-N dùng 24 target auxetic, 10 mẫu/target, force-periodic bật,
lọc manufacturability bật; report đầy đủ tại
`outputs/phase5/wire_best_of_n_result.json`:

| Chỉ số | WIRE v2 | Mục tiêu (DoN Task 2) |
|---|---:|---:|
| FE calls | 170 | - |
| hit rate single-shot (= `passes_all` single-shot) | 0,042 (1/24) | ≥ 0,75 |
| hit rate best-of-N | 0,167 (4/24) | - (không đặt DoN riêng) |
| R²(FE, best-of-N) | -10,2888 | - (không đặt DoN riêng) |
| mean frac manufacturable | 0,042 | - (không đặt DoN riêng) |

Kết quả **không đạt DoN** `passes_all≥75%` và regression rất lớn so với
checkpoint cVAE production. Checkpoint WIRE này chỉ giữ làm research artifact,
**không promote** và không dùng làm nền cho Tandem L-BFGS. Nguyên nhân hiện
chưa được quy kết chắc chắn: WIRE đã học reconstruction/property trên surrogate
(property loss cuối khoảng 0,01) nhưng không được train với real-physics loss,
đồng thời decoder INR có thể sinh hình học không liên thông dù đã force-periodic.
Không nên kết luận WIRE thất bại về mặt kiến trúc từ một run hidden=64 duy nhất;
cần ablation có kiểm soát (real-physics loss, sampling/threshold và hidden size)
trước khi đầu tư thêm compute.

### 2026-08-22 - Thêm công cụ trích xuất hằng số kỹ thuật + minh họa định tính synclastic/anticlastic; khảo sát hướng "composite auxetic" cho bài báo #2

**Không phải thí nghiệm khoa học mới** (không có claim/số liệu FE mới) - đây là hạ tầng/tài liệu bổ sung, ghi lại ở đây để tránh khoảng trống giữa code và log.

**`compute_elastic_constants(Q)`** (`simp/objectives/auxetic.py`) - trích xuất `E_x, E_y, G_xy, nu_12, nu_21, B_eff` (mô-đun khối hiệu dụng 2D plane-stress) từ tensor đồng nhất hóa `Q`, gọi lại `compute_nu12()`/`compute_nu21()` có sẵn thay vì chép công thức (tránh 2 nguồn tính lệch nhau). Test `test_solid_cell_recovers_isotropic_constants` xác nhận bằng số: ô cơ sở đặc hoàn toàn (x=1) cho đúng `Ex=Ey=E0`, `Gxy=E0/(2(1+nu))`, `B_eff=E0/(2(1-nu))` - khớp nghịch đảo giải tích ma trận D đẳng hướng.

**`reconstruct_3d_bent_surface()` + `_bending_curvature_ratio()`** (`simp/io/visualizer.py`) - hình minh họa 3D bề mặt uốn tấm dựa trên quan hệ chuẩn Kirchhoff-Love `κ_yy = -nu* · κ_xx`, phân biệt định tính synclastic (nu*<0, auxetic, 2 độ cong cùng dấu) vs anticlastic (nu*>0, vật liệu thường, ngược dấu). **Cảnh báo quan trọng ghi rõ trong docstring hàm:** đây CHỈ là minh họa hình học/giải tích bậc 2, KHÔNG PHẢI kết quả mô phỏng FE uốn tấm thật - dự án hiện chỉ có homogenization TRONG mặt phẳng (`simp/homogenization/compute.py`), chưa có solver uốn ngoài mặt phẳng nào. `kappa_xx` là độ cong giả định tùy chọn chỉ để hình có độ dốc hợp lý, không đo từ mô phỏng nào. **Nếu dùng hình sinh từ hàm này (`outputs/figures/fig5_synclastic_bending.png`) trong bài báo/luận văn, PHẢI chú thích rõ là minh họa định tính, không phải số liệu định lượng** - tránh lặp lại kiểu nhầm lẫn giữa minh họa và kết quả đã từng là bài học của dự án (xem `LIMITATIONS.md` mục 24 về module chưa từng chạy thật).

### 2026-08-22 (tiếp) - Dựng Figure 5 hoàn chỉnh cho Bài báo #1 (Nhóm 3.2 Pareto validation + minh họa uốn tấm), thêm vào `notebooks/08_composite_score_pareto_validation.ipynb`

Theo kế hoạch 5-figure cho Bài báo #1 (đề xuất chuẩn Q1, CMAME/Materials & Design): Figure 5 gộp 2 panel - (a) phân bố Spearman 24 target (dữ liệu thật, không chạy lại - đọc trực tiếp `outputs/phase5/reports/composite_score_pareto_validation.json` đã có từ Nhóm 3.2), (b) 2 mặt uốn 3D minh họa đối chiếu anticlastic (ν*=+0.3, giá trị `nu` mặc định vật liệu nền của dự án) vs synclastic (ν*=-0.625, trung bình v12 đo THẬT trên seed `hourglass` sau rebuild dQ - không phải số tròn tùy chọn).

**Sửa 1 điểm trong kế hoạch gốc trước khi dựng:** đề xuất ban đầu có Figure 3 (đối chiến 8D in-distribution) framing theo hướng "cVAE thắng vì database-free" - mâu thuẫn trực tiếp với quyết định đã chốt ở `PROJECT_PLAN.md` mục 3 ("Nhóm 4.2 KHÔNG dùng làm luận điểm chính" vì cVAE không thắng rõ retrieval trục nào ở 8D). Đã báo lại cho user, chưa dựng Figure 3 theo khung sai đó.

Code: cell mới trong `notebooks/08_composite_score_pareto_validation.ipynb` (sau cell `results001`), tái dùng `_bending_curvature_ratio()` từ `simp/io/visualizer.py` (không chép công thức). Output: `outputs/figures/fig5_pareto_validation_and_bending.png`. Trên nhánh `substrate-material`, uncommitted.

### 2026-08-22 (tiếp) - Dựng nốt Figure 1, 2, 3, 4 cho Bài báo #1 - toàn bộ 5 figure hoàn thành, 100% dữ liệu/hình thật (không mô phỏng/giả lập)

Tiếp tục kế hoạch 5-figure. Cả 4 figure còn lại đều dùng dữ liệu/ảnh THẬT đọc trực tiếp từ report JSON, manifest CSV, hoặc chạy lại đúng hàm production có sẵn (`best_of_n`, `solve_fe`, `nearest_neighbor_baseline`) - không có số nào giả lập/minh họa tùy ý, khác với đoạn code mẫu gốc trong đề xuất ban đầu (vốn dùng `iterations`/`normal_converge` GIẢ LẬP cho Figure 2b).

**Phát hiện hạ tầng quan trọng:** python mặc định (`/home/tbm/miniconda3/bin/python3`) có `torch` bị hỏng (namespace package rỗng, `ModuleNotFoundError: No module named 'torch.utils'`) - đây là lý do `pytest tests/` báo lỗi collection ở các file phase4/5 trong phiên làm việc trước. **Đã tìm ra:** conda env `simp` (`/home/tbm/miniconda3/envs/simp`) có torch 2.6.0+cu124 hoạt động đầy đủ + CUDA khả dụng - dùng env này (`conda activate simp`) để chạy bất kỳ việc gì cần torch (inference cVAE, `pytest tests/test_phase4*`/`test_phase5*`). Ghi lại ở đây để không phải dò lại.

**Figure 1 (kiến trúc framework):** sơ đồ khối 2 pha (Offline Training có đường gradient nét đứt qua Differentiable Homogenization Solver; Online Inference qua Manufacturability filter + Composite score) - dựng bằng `matplotlib.patches` (FancyBboxPatch/FancyArrowPatch), không cần Inkscape/Illustrator vì tái dùng được mỗi khi pipeline đổi. Không có script lưu lại trong repo (thuần schematic, không đọc dữ liệu) - `outputs/figures/fig1_framework_architecture.png`.

**Figure 2 (data hygiene):** (a) tỷ lệ auxetic pre/post dQ-fix tự tính lại từ `outputs/phase3/manifest_pre_dqfix_backup.csv` (82,08%) và `outputs/phase3/manifest_quality.csv` (91,91%) - KHÔNG hard-code, khớp số đã công bố (`LIMITATIONS.md` mục 10). (b) quỹ đạo v12 THẬT của osc_score filter, đọc trực tiếp `iteration_data.csv` của 2 mẫu thật: `outputs/multi_batch/batch_3/circle_half_quarter/sample_0401` (osc_score=0,314, bị lọc, dao động rõ 0,135↔0,186 suốt >15 iteration cuối) vs `outputs/multi_batch/batch_1/circle/sample_0009` (osc_score=0,0008, giữ lại, hội tụ mượt). `outputs/figures/fig2_data_hygiene.png`.

**Figure 3 (8D in-distribution, ĐÃ SỬA khung diễn giải):** giữ đúng góp ý đã báo trước - đổi từ "cVAE thắng vì database-free" sang "gần hòa, đây là lý do bài báo dựa vào OOD (Figure 4) làm luận điểm trung tâm". 3 panel đọc trực tiếp từ report JSON có sẵn (`curse_of_dimensionality_comparison.json`, `manufacturability_retrieval_vs_cvae_8d.json`, `retrieval_low_data_pilot_8d.json`) - không chạy lại gì. `outputs/figures/fig3_8d_indistribution_comparison.png`.

**Figure 4 (OOD sign-flip, generative advantage trung tâm):** (a) scatter 12 target thật từ `ood_baseline_comparison_cvae_realphysics.json` (R²=0,418 cVAE vs 0,057 retrieval, khớp `LIMITATIONS.md` Giới hạn #19). (b) gallery ảnh THẬT - chạy lại `best_of_n(cvae_realphysics.pt, custom_condition=...)` (dùng conda env `simp`) cho 2 target đại diện (sign-flip v12=+0,5 → cVAE đạt +0,28 đúng dấu, retrieval kẹt ở -0,002; extreme-magnitude v12=-2,5 → cVAE đạt -0,85, retrieval chỉ -0,093) + ảnh nearest-neighbor thật từ `train.npz`. (c) trường ứng suất von Mises THẬT - giải FE bằng `solve_fe`/`build_pbc`/`build_dof_mesh` (`simp/core/`) trên chính ảnh cVAE sinh cho target sign-flip, B-matrix tại điểm Gauss trung tâm (công thức Q4 chuẩn, tự suy từ `Material._compute_element_stiffness()`). **Lưu ý kỹ thuật phát hiện giữa chừng:** thử tính "ν* cục bộ" bằng trung bình strain_yy dưới tải ε_xx=1 áp đặt PBC ra ~0 - KHÔNG phải bug, mà vì phương pháp homogenization của dự án áp đặt trực tiếp biến dạng vĩ mô (ε_yy=0 cho case này), nên trung bình strain_yy BẮT BUỘC=0 theo điều kiện biên - nu12 phải suy từ nghịch đảo tensor Q (`compute_nu12`), không phải đo strain trung bình; đã bỏ hướng này, chỉ giữ von Mises (không phụ thuộc giả định sai). `outputs/figures/fig4_ood_generalization.png`, ảnh trung gian lưu ở `outputs/figures/fig4_assets/` (hiện untracked (`??`) như mọi file khác trong `outputs/figures/` - không bị `.gitignore` chặn, chỉ đơn giản là chưa `git add`).

Trên nhánh `substrate-material`, toàn bộ 5 figure + notebook + log này đang uncommitted.

### 2026-08-22 (tiếp) - Sửa Figure 1: đường mũi tên gradient đi lệch, trông như xuất phát từ box "Density image x"

User phát hiện qua ảnh chụp: mũi tên nét đứt (adjoint gradient) route bằng `arc3` cong lên gần đáy box "Density image x", khiến nhìn như nối từ box đó thay vì từ "Physics loss" về "Decoder". Sửa lần 1 bằng path elbow đi VÒNG PHÍA TRÊN (qua khoảng trắng dưới tiêu đề section) - user hỏi lại "sao không nối ở dưới". Lý do ban đầu: đáy "Physics loss" (y=4.3) cách đường phân cách 2 section (y=4.15) chỉ 0.15 đơn vị, không đủ chỗ. **Sửa lần 2 (bản cuối):** giãn thêm khoảng trống bằng cách dời đường phân cách xuống y=3.3 và dịch toàn bộ hàng box ONLINE INFERENCE xuống 0.8 đơn vị, rồi route elbow đi VÒNG PHÍA DƯỚI trong khoảng trống mới (y=3.7, giữa đáy Physics loss và đường phân cách) - nối rõ đáy Physics loss → đáy Decoder, không cắt qua box nào, không lấn sang Online Inference. `outputs/figures/fig1_framework_architecture.png` (ghi đè).

2 test class mới (`TestElasticConstants`, `TestReconstructBentSurface`, `tests/test_core_smoke.py`).

**Tài liệu mới (untracked, chưa commit):**
- `docs/ARCHITECT.md`, `docs/CLI_GUIDE.md` - tài liệu kiến trúc/hướng dẫn CLI tham chiếu, không phải log thí nghiệm.
- **`docs/COMPOSITE_AUXETIC_PLAN.md` chưa từng được tạo trên đĩa** - nội dung khảo sát dưới đây chỉ ghi lại ở log này, không có file riêng. Khảo sát 2 hướng mở rộng "composite auxetic" cho bài báo #2: Hướng A (vật liệu nền đa pha/multiscale, chỉ đụng `simp/materials/`) vs Hướng B (multi-material trong unit cell, đụng cả `solver.py`/`oc.py`/format dữ liệu Phase 3-5). Khuyến nghị Hướng A mức "Thấp" (công thức đóng Halpin-Tsai/Rule of Mixtures) - rủi ro kiến trúc thấp nhất, tái dùng trực tiếp hạ tầng Nhóm 1 (A1-A7). **Chưa quyết định, chưa viết code** - cần xác nhận với GS hướng dẫn (chuyên ngành composite/FGM mechanics) trước khi chọn hướng. Đã nối 1 dòng tham chiếu vào `PROJECT_PLAN.md` mục 3 (bảng "Ngoài phạm vi Bài báo #1"), cùng nhóm với Nhóm 6 (nhiệt/CTE) - không chen vào trước khi Nhóm 4.1/4.2 (đã xong) được viết thành bài báo #1.

Trên nhánh `substrate-material`, uncommitted.

### 2026-08-23 - KAN regression head (Task 1): baseline 0.85 không tái lập; real-physics là chìa khóa property fidelity

**Phát hiện 1 (đột xuất): `outputs/phase5/evaluation_report.json` (2026-07-29, R²=0,8575/0,8492) KHÔNG tái lập với code hiện tại.** Cùng checkpoint `cvae_v2_finetuned.pt` + cùng test set, đo lại bằng `property_accuracy()` hiện hành: R²=0,35 (surrogate v2) và −1,81 (surrogate v1 - đúng file mà report tháng 7 dùng). Đã đối chiếu git: `property_accuracy()` không đổi hành vi cho condition_dim=2, forward surrogate không đổi cho n_outputs=3 (không include_nu0), `test.npz`/surrogate v1 không đổi ngày file - chưa tìm ra gốc rễ khác biệt, khả năng cao report cũ sinh bằng code path khác (script/notebook thời đó). **Hệ quả:** con số "baseline 0.85" trong mọi so sánh cũ (kể cả DoN Task 1) là sai lệch; mọi so sánh R² từ nay phải đo lại cùng code hiện tại. Baseline Linear cũ đo lại: 0,29 (base) / 0,35 (finetuned).

**Phát hiện 2 (phương pháp): chọn checkpoint theo `val_loss` có thể cho property R² ÂM sâu, dù reconstruction TỐT NHẤT.** KAN retry (lr 5e-4, patience 25) đạt recon tốt nhất trong 3 run (val 774) nhưng R² = −1,34 (tệ hơn dự đoán hằng số) - reconstruction tốt hoàn toàn không tương quan với property fidelity trên kiến trúc KAN. Xác nhận định lượng thêm cho dòng "val_loss không an toàn" ở bảng trên.

**Phát hiện 3 (tích cực): KAN + real-physics fine-tune VƯỢT baseline Linear (đo cùng code hiện tại).** KAN base (chọn val_loss) R²=0,19. Fine-tune 50 epoch, resume từ KAN base, `--select-by fe_r2 --fe-eval-every 10 --lambda-real-physics 1.0 --real-physics-subsample 8 --real-physics-every 50 --lr 3e-4` → R²=0,58 (v12 0,74, v21 0,42); R²(FE thật) tốt nhất trong train = 0,7865. **Vòng 2 (60 epoch, resume từ v1, `--lambda-real-physics 2.0 --real-physics-every 20 --lr 2e-4`): R²=0,64 (v12 0,82, v21 0,45), R²(FE thật) tốt nhất = 0,8889** - vượt Linear finetuned cũ (0,35) gần 2×, tiến độ hội tụ (0,58 → 0,64). KAN không phá recipe real-physics đã kiểm chứng (mục 2026-07-24) - đây là hướng đưa KAN lên ngang/vượt Linear. Checkpoints: `cvae_kan_best.pt` (base), `cvae_kan_realphysics.pt` (v1), `cvae_kan_realphysics_v2.pt` (v2, khuyến nghị).

### 2026-09-11 - Exp 3: MLP + Physics ablation

Đã thêm nhánh MLP (`--use-mlp-head`) và chạy đủ 150 epoch với seed `123`,
batch size `512`, learning rate `5e-4`, KL warmup 30, `gamma=1`,
`lambda_real_physics=20`, FE subsample 8, `lambda_volfrac=3`, và FE-eval
mỗi 10 epoch trên 24 condition. Đánh giá cuối dùng cùng test loader,
surrogate `surrogate_for_phase5_v2.pt` và seed prior cố định cho cả hai model.

| Model | R² v12 | R² v21 | R² volfrac | R² FE(v12), n=24 |
|---|---:|---:|---:|---:|
| MLP + Physics, `exp3_mlp_physics_150.pt` | 0,432 | 0,087 | -2,483 | -9,834 |
| KAN + Physics v2, `cvae_kan_realphysics_v2.pt` | **0,836** | **0,476** | **-0,221** | **0,581** |

MLP checkpoint được chọn theo FE-R² là epoch 20 (`-6,2465`); epoch 150
đạt `-7,4893`. KAN v2 là checkpoint cũ, chưa retrain sau khi thêm decoder
symmetry và volfrac loss, nên kết quả ủng hộ KAN nhưng chưa đủ để khẳng định
ảnh hưởng nhân quả của riêng KAN. Cần chạy lại KAN cùng code/recipe để đóng
ablation 2x2.

### 2026-09-23 - WIRE thiếu enforce_symmetry; physics-guided latent refinement (Task 4) benchmark đầu tiên trên checkpoint thật

**Phát hiện (root-cause WIRE thất bại KPI, không phải bug resize/mismatch với surrogate như nghi ngờ ban đầu):** đối chiếu checkpoint metadata + `cvae_wire_v2_history.json` (cả 30 epoch có `real_physics: NaN`) xác nhận `cvae_wire_v2.pt` train với `--lambda-real-physics 0.0` (mặc định, chưa từng dùng real-physics loss), `wire_hidden_dim=64` (không phải 128), `lr=1e-4`, chỉ 30 epoch - ngân sách nhỏ hơn hẳn KAN/conv. Đồng thời phát hiện `WireContinuousDecoder` **không có cơ chế `enforce_symmetry`** như `Decoder` (conv) đang có (`model.py`, `image = 0.5*(image+image.transpose(-1,-2))`) dù `CVAE.__init__` nhận tham số này cho mọi decoder_type - nhánh `wire` âm thầm bỏ qua, mất hẳn 1 prior hình học có lợi. **Đã sửa:** thêm `enforce_symmetry` cho `WireContinuousDecoder` + nối từ `CVAE.__init__` (trước đây bị bỏ qua) - tương thích ngược qua `--disable-symmetry` (flag đã có, giờ áp dụng đúng cho cả 2 decoder_type). 3 test mới (`tests/test_phase5_model.py::TestWireDecoder`), 638/638 pass. Đã launch retrain `cvae_wire_realphysics_v3.pt` (`--wire-hidden-dim 128 --lambda-real-physics 1.0 --real-physics-every 10 --real-physics-subsample 8 --epochs 60 --select-by fe_r2 --fe-eval-every 5`, chạy nền, ETA ~2,5-3 giờ đo bằng calib 1 epoch=3m22s) - kết quả sẽ ghi bổ sung khi xong.

**Physics-guided latent refinement (Task 4, `pipeline/phase5_cvae/tandem_lbfgs.py`):** thêm `guidance_source="real_physics"` - dùng gradient GIẢI TÍCH từ `RealPhysicsNu`/`losses.real_physics_loss` (FE thật) thay cho surrogate CNN đã biết "exploitable" (cảnh báo đầu `losses.py`), tái dùng nguyên hàm có sẵn, không viết lại logic FE. `surrogate_model` giờ optional. 9 test mới (`tests/test_tandem_lbfgs.py`), 643/643 pass toàn repo.

**Benchmark đầu tiên trên checkpoint thật** (`pipeline/phase5_cvae/benchmark_physics_guided_refinement.py`, mới - so sánh z ngẫu nhiên (baseline, đúng hành vi `model.generate()` hiện tại) với z SAU KHI tối ưu bằng `guidance_source="real_physics"`, cùng 1 z khởi tạo, 1-mẫu-1-lần-sinh KHÔNG best-of-N để cô lập đúng tác dụng refinement): trên `cvae_kan_realphysics_v2.pt`, 24 condition test set (seed=123, khớp `best_of_n_eval.py`), 30 bước L-BFGS -

| | R²(v12) | MAE(v12) |
|---|---:|---:|
| Baseline (z ngẫu nhiên) | -16,00 | 0,587 |
| Refined (physics-guided, 30 bước) | **-7,60** | **0,355** |

*(DoN gốc Task 4 = sai số ≤8% so với target OOD cực đoan ν\*=-2,5 - không ghi được trực tiếp vào bảng trên vì khác đơn vị đo (%sai số so target, không phải R²/MAE) VÀ khác kịch bản (OOD cực đoan, không phải 24 condition trong-phân-phối đo ở đây) - xem đoạn dưới.)*

Cải thiện thật, đo bằng FE thật độc lập với chính loss dùng để tối ưu (không tự chấm điểm bằng hàm mình vừa minimize) - MAE giảm ~40%. R² tuyệt đối vẫn rất âm vì đây là so sánh 1-mẫu-1-lần-sinh KHÔNG lọc/best-of-N (khắt khe hơn nhiều so với pipeline production dùng best-of-30 + force_periodic), và target lấy trong-phân-phối (KHÔNG phải kịch bản OOD cực đoan ν*=-2,5 mà DoN gốc Task 4 nhắm tới) - chưa đóng được DoN "sai số ≤8% so với target OOD", cần thí nghiệm riêng cho đúng kịch bản đó. Kết quả này là tín hiệu quyết định cho bước tiếp theo (latent diffusion prior, xem `docs/plan.md` Giai đoạn 2): refinement đơn giản đã cải thiện đáng kể mà không cần train model mới, ủng hộ hướng tiếp tục đầu tư vào physics-guided sampling.

**Retrain WIRE (`cvae_wire_realphysics_v3.pt`) DỪNG GIỮA CHỪNG (2026-09-23, người dùng cần tắt máy) - CHƯA có kết luận.** Chạy nền được 40/60 epoch (dừng an toàn bằng `TaskStop`, không kill giữa lúc ghi checkpoint - đã load lại kiểm chứng OK) trước khi phải dừng. Checkpoint trên đĩa ứng với epoch 25 (điểm cải thiện R²(FE) gần nhất trước lúc dừng): `fe_r2=-10,7690` - **chưa vượt** baseline cũ đang thất bại (`cvae_wire_v2.pt`, R²(FE)=-10,29) sau 40 epoch quan sát được (dao động -10,77 đến -11,22 suốt epoch 5-40, không thấy xu hướng cải thiện rõ). Không kết luận được symmetry-fix + real-physics có sửa được WIRE hay không từ dữ liệu này - có thể cần nhiều epoch hơn (LR mới giảm tới 4.8e-05/60 epoch tại lúc dừng, chưa vào vùng hội tụ cuối như KAN v2 từng cần) hoặc recipe 2-giai-đoạn (base rồi fine-tune real-physics) giống KAN thay vì 1 giai đoạn từ đầu. Không có file lịch sử `_history.json` (chỉ ghi lúc vòng lặp kết thúc tự nhiên) - số liệu epoch-by-epoch nằm trong log đã sao lưu `outputs/phase5/cvae_wire_realphysics_v3_partial_log.txt`.

**Resume round 2 (2026-09-23, máy bật lại):** `--resume-from outputs/phase5/cvae_wire_realphysics_v3.pt --decoder-type wire --wire-hidden-dim 128 --epochs 35 --lr 1e-4 --lambda-real-physics 2.0 --real-physics-every 10 --real-physics-subsample 8 --fe-eval-every 5 --select-by fe_r2 --output-name cvae_wire_realphysics_v3_round2.pt`. **Cố ý đổi `--output-name`** (không ghi đè `cvae_wire_realphysics_v3.pt`) vì `train.py` khởi tạo lại `best_val=-inf` mỗi lần chạy bất kể `--resume-from` - nếu dùng cùng tên file, epoch fe-eval ĐẦU TIÊN của round 2 (dù tệ hơn epoch 25 cũ) sẽ ghi đè mất checkpoint tốt hiện có. Cũng tăng `lambda_real_physics` 1.0→2.0 và giảm `lr` 1.5e-4→1e-4 (giống pattern KAN v1→v2) để đẩy mạnh hơn tín hiệu real-physics ở vòng 2.

**KẾT QUẢ ROUND 2 (chạy đủ 35/35 epoch) - CẢI THIỆN trên tập FE-eval nhỏ (8 condition) NHƯNG THẤT BẠI trên benchmark chính thức (24 condition) - phát hiện quan trọng về overfitting lên validation subset.**

R²(FE) đo trên 8 condition validation dùng trong lúc train cải thiện đều đặn tới cuối: epoch 15=-10,42 → epoch 25=-10,35 → epoch 30=-10,18 → **epoch 35=-10,10** (tốt nhất trong TOÀN BỘ 2 round, vượt cả round 1 lẫn baseline gốc `cvae_wire_v2.pt`=-10,29).

**Nhưng chạy `best_of_n_eval.py` chính thức (24 condition, 30 mẫu/condition, seed=123 - đúng benchmark dùng để báo cáo số liệu, KHÁC 8 condition dùng nội bộ lúc train) cho kết quả TỆ HƠN baseline gốc trên mọi trục:**

| | `cvae_wire_v2.pt` (baseline gốc, thất bại) | `cvae_wire_realphysics_v3_round2.pt` (mới) | Mục tiêu (DoN Task 2) |
|---|---:|---:|---:|
| hit_rate single-shot (= `passes_all` single-shot) | 4,17% | **0,00%** | ≥ 75% |
| hit_rate best-of-N | 16,67% | **0,00%** | - (không đặt DoN riêng) |
| R²(FE, best-of-N) | -10,29 | **-13,87** | - (không đặt DoN riêng, chỉ có mốc định tính "dương") |
| frac_manufacturable | 4,17% | 2,78% | - (không đặt DoN riêng) |

**Kết luận trung thực:** cải thiện R² đo trên 8 condition validation KHÔNG chuyển hóa thành cải thiện thật trên benchmark 24-condition đầy đủ - đây là bằng chứng cụ thể của việc **overfit lên chính tập validation nhỏ dùng để chọn checkpoint** (8 condition cố định, lặp lại `--fe-eval-every 5` suốt cả round 1+2 ~75 epoch hiệu dụng - đủ để mô hình "học" đặc thù của đúng 8 condition đó thay vì tổng quát hóa). Việc thêm `enforce_symmetry` + bật `lambda_real_physics` **KHÔNG sửa được** WIRE decoder trong ngân sách đã thử (75 epoch hiệu dụng, 2 round) - Task 2 (WIRE INR decoder) vẫn **CHƯA đạt KPI**, thậm chí kết quả chính thức tệ hơn baseline gốc. File kết quả đầy đủ: `outputs/phase5/self_play/best_of_n_result_wire_realphysics_v3_round2.json`.

**Khuyến nghị cho lần thử tiếp theo (chưa làm):** (1) tăng `--n-fe-eval-conditions` lên gần 24 (hoặc dùng đúng 24 condition benchmark) để validation-during-train không lệch khỏi benchmark cuối; (2) thử lại recipe 2-giai-đoạn ĐÚNG kiểu KAN (base thuần property-consistency trước, fine-tune real-physics sau) thay vì bật real-physics từ epoch 1; (3) cân nhắc việc thiếu `enforce_symmetry` không phải nguyên nhân chính - có thể vấn đề nằm sâu hơn ở kiến trúc Gabor/INR không phù hợp với bài toán density-field nhị phân hoá cứng (khác domain ảnh liên tục mà WIRE gốc thiết kế cho).

**Latent dataset cho Giai đoạn 2 (Latent diffusion prior, `docs/plan.md`) - script mới `build_latent_dataset.py`, phát hiện đáng chú ý.** Mã hoá cả train/val/test bằng encoder đóng băng của `cvae_kan_realphysics_v2.pt` (deterministic, `z=mu`, không sample qua reparameterization). **`mu` thật lệch rõ so với N(0,1) prior mặc định của `CVAE.generate()`:** mean≈0,35-0,36, std≈1,28-1,32 (nhất quán cả 3 split, không phải nhiễu 1 lần đo) - xác nhận bằng số liệu thật giả thuyết nền tảng của Giai đoạn 2 (aggregate posterior KHÔNG khớp Gaussian chuẩn, sample ngẫu nhiên từ N(0,1) đang lấy mẫu ở vùng latent space không đại diện đúng cho dữ liệu thật). 3 test mới (`tests/test_phase5_build_latent_dataset.py`), 646/646 pass. File latent: `outputs/phase5/cvae_kan_realphysics_v2_latents_{train,val,test}.npz`.


### 2026-09-25 - plan.md v3 P1.0/P1.1: bug `load_cvae` làm sai benchmark 09-23; refinement đạt ở single-shot, thua ở best-of-N/OOD vì khai thác vật liệu xám

**Bug loader (đã sửa, `LIMITATIONS.md` mục 28):** `adversarial_dataset.load_cvae()` bỏ qua `use_kan`/`enforce_symmetry` từ khi mặc định `CVAE(enforce_symmetry=True)` được thêm (2026-09-11) → checkpoint KAN cũ bị ép đối xứng qua đường chéo lúc đánh giá, checkpoint Linear crash. Benchmark refinement 2026-09-23 (KAN v2, 24 condition) đo lại đúng: R²(v12) baseline **0,058** (không phải −16,00), refined **0,92** (không phải −7,60), MAE giảm 68% [CI95 57-76%]. Chỉ 2,5% ảnh train đối xứng qua đường chéo và 45% mẫu có |v12−v21|>0,1 → mọi checkpoint train với mặc định đối xứng mới không biểu diễn được dữ liệu dị hướng (ứng viên nguyên nhân Exp 3 MLP/WIRE v3 thất bại, chưa kiểm chứng).

**Harness (P1.0):** `benchmark_physics_guided_refinement.py` thêm `--targets/--target-v21/--n-samples/--force-periodic/--projection-betas` + CI paired bootstrap (`bootstrap_ci.bootstrap_paired_mae_reduction`) + chỉ số "guarded". Với loader cũ tái lập bit-for-bit baseline 09-23 (MAE 0,5868); refined lệch nhẹ (0,384 vs 0,355) do L-BFGS trên GPU không tất định (2 lần chạy liên tiếp lệch R² ~0,006).

**P1.1a - refinement single-shot, 100 condition, 30 bước:** `cvae_v2_finetuned` (production) R²(v12) 0,874→**0,976**, MAE giảm **57,7%** [49,4; 64,6] - ĐẠT tiêu chí ≥20%. `cvae_realphysics` 19,4% [0,2; 35,0] (sát ngưỡng). KAN v2 57,1% [49,5; 64,6]. Phụ: R²(v12) single-shot FE thật của `cvae_v2_finetuned` = 0,874 trong khi `property_accuracy()` (surrogate) báo 0,27 - thước đo surrogate xếp hạng sai.

**P1.1c - khả thi vật lý (giải tích):** vật liệu trực hướng 2D cần ν12·ν21<1 → target đối xứng ν12=ν21≤−1 không tồn tại. Dữ liệu khớp (max v12·v21=0,43; mẫu v12=−1,95 đi với v21=−0,04). Quét OOD phải dùng target dị hướng.

**P1.1b/d - KHÔNG ĐẠT (best-of-30 + `force_periodic`, `cvae_v2_finetuned`):** (b) refine sau best-of-30 làm tệ hơn: MAE(v12) 0,0152→0,0246 (−62,5% [−98,3; −33,9]). (d) OOD dị hướng (v21=−0,05, 10 lặp/target): best-of-30 không refine đạt sai số tương đối v12 trung vị 0,5%/0,9%/4,1% tại −1,0/−1,5/−1,75 nhưng **bão hòa ~−1,95** (= min tập train): 11%/16%/23% tại −2,0/−2,25/−2,5; refine làm tệ hơn ngoài dải (36-44%). **Nguyên nhân đã xác nhận:** loss liên tục cuối ~1e-10 (khớp target gần tuyệt đối kể cả ν*=−2,5) nhưng MSE sau force_periodic+nhị phân hóa+resize nearest 1e-3-1e-2 → refine khai thác **vật liệu xám** (mật độ trung gian), cùng họ surrogate exploitation. Hướng sửa (P1.1e): đưa force_periodic + Heaviside projection khả vi (β tăng dần) + resize nearest khớp PIL tuyệt đối (`pipeline/phase5_cvae/heaviside.py`) vào objective.


**P1.1e - ĐỘT PHÁ: refine nhận thức nhị phân hóa** (objective: force_periodic → Heaviside β {1,4,16,64}, L-BFGS khởi động lại mỗi β → resize nearest khớp PIL tuyệt đối; 32 bước; `cvae_v2_finetuned`): (a) single-shot `IN100`: MAE(v12) −91,8% [89,0; 93,9], R²(v12) 0,874→**0,998**; (b) sau best-of-30: MAE −62,6% [50,7; 72,1], R² 0,986→0,997 (trước đó làm TỆ hơn 62%); (d) OOD sai số tương đối v12 trung vị −2,0: 12,5% (bản thường) / 4,9% (guarded = chỉ nhận refine khi FE verify tốt hơn), −2,25: 22%/15%, −2,5: 32%/22% - vẫn bão hòa ngoài dải train, không claim ngoại suy biên độ. Chi phí ~77 FE-solve/condition. `pipeline/phase5_cvae/heaviside.py` (mới, 8 test), `tandem_lbfgs(projection_betas=, periodic=)`, 674/674 test pass.

**P1.2 - Ablation KAN vs Linear có kiểm soát** (cùng recipe 2 giai đoạn production: base γ=20 50 epoch → fine-tune real-physics λ=20 35 epoch, `--disable-symmetry`, chọn theo `fe_r2` trên 24 condition validation; 2 seed fine-tune; đánh giá `IN100` + refine P1.1e): theo tiêu chí đặt trước (KAN thắng khi CI paired > 0 trên cả 2 seed) **KAN không thắng ở chế độ nào**. Best-of-30 (production): **Linear thắng cả 2 seed** (KAN MAE kém hơn 25% [−56; −1] / 20% [−42; −2]). Có refine: cả 2 head R²(v12) 0,994-0,999. KAN có xu hướng tốt hơn ở single-shot (+18% [4; 30] seed 7, không có ý nghĩa ở seed 123). KAN 4,79M tham số vs Linear 1,49M. Giữ Linear cho production; luận điểm "KAN vượt Linear ~2×" của plan v2 bị bác bỏ bằng thước đo FE thật.

**P1.3 - WIRE lần cuối, ĐÓNG:** base 25 epoch → fine-tune real-physics 25 epoch với Heaviside trong loss train (`train.py --rp-projection-beta-max 64 --rp-periodic`, β tăng hình học, dùng chung `heaviside.project_for_fe`), `--disable-symmetry`. `IN24`: hit-rate single-shot 12,5% (tiêu chí ≥30%), R²(FE, best-of-30) −7,19 (tiêu chí >0), manufacturable 4,6%; có refine P1.1e R² −1,20. Tốt hơn mọi WIRE trước (0-4,2% / −10,3 đến −13,9) nhưng không đạt tiêu chí đặt trước → không đầu tư thêm, ghi kết quả âm tính.
### 2026-09-30 - Tính chất "rẻ" suy từ Q: E/G/B, proxy ấn lõm, tốc độ sóng quasi-static (dataset A4)

**Code:** `real_physics.solve_elastic_with_grad()` trả về v12/v21/E_x/E_y/G_xy/B_eff kèm gradient giải tích theo pixel, chỉ cần 1 lần FE (dS = −S·dQ·S; B_eff = 1/(S00+S11+2S01)). `solve_nu_with_grad()` giờ là lát cắt ν của hàm này, không đổi hành vi. `auxetic.compute_wave_speeds()` giải bài toán Christoffel trên Q, chuẩn hóa theo sqrt(E0/ρs). Thêm 14 test (FD gradient cho 4 mô-đun, đối chiếu `compute_elastic_constants`, ô đặc đẳng hướng closed-form); tổng 693/693 pass.

**Backfill:** `analysis/scripts/backfill_elastic_props_npz.py` chạy trên toàn bộ `outputs/phase3_a4/{train,val,test}.npz` (73.164 mẫu, penal và ν0 thật từng mẫu, ~25 phút/12 worker) và ghi ra `{split}_props.npz` (Q thô + 13 cột). Không có lỗi FE nào. Sanity |Δv12| median 0,0012, 1,9–2,7% mẫu lệch >0,1 do resize 64→50 (khớp backfill f1/f2 2026-08-05).

**Phân tích** (`notebooks/03_cheap_physical_properties.ipynb`, val+test n=4778 sau QC; không dùng train vì augment ×6 làm rò rỉ CV):
- 0% vi phạm cận Voigt và Hashin–Shtrikman (xác nhận FE đúng). Median B_eff/B_HS = 0,17: auxetic kém hiệu quả về độ cứng khối.
- R² (5-fold CV) dự đoán từ 5 điều kiện hiện có: E/B/M/c_qL đạt 0,92–0,97, gần như dư thừa nếu thêm làm condition. **G_xy 0,84, c_qT 0,80**: mang nhiều thông tin mới nhất, ứng viên ưu tiên cho condition/refinement.
- Proxy ấn lõm M_y = E_y/(1−ν12ν21): khi cố định mật độ, tương quan với ν đổi dấu giữa các nhóm (+0,54…−0,32), nên KHÔNG có quy luật chung "auxetic cứng hơn". Riêng cụm ν21 < −1,2 có M_y/ρ ≈ 0,6 (gấp 2–3 lần), do 1−ν12ν21 → 0.
- Biên Pareto ν12 ↔ E_x/ρ: độ cứng riêng giảm ~5,7× (0,53 → 0,09) khi ν12 đi từ −0,27 đến −1,19. Biên do hexagonal/hourglass chi phối.

**Chưa làm:** nối `solve_elastic_with_grad` vào refinement (target đa tính chất), solver dẫn nhiệt κ.

### 2026-10-06 - P1.6: baseline SIMP, xác nhận IN100-B, hội tụ lưới, retrieval verified, OOD đổi dấu

`pipeline/phase5_cvae/benchmark_simp_baseline.py` (mới, 7 test): ½‖ν−ν*‖² bằng MMA, thể tích ≤0,55, Q11/Q22 ≥ δ, Heaviside β 1→64, verify nhị phân hóa + FE độc lập giống cVAE, đa khởi tạo 4 seed. Ghép cặp 100 target IN100 với `p1_1e_*`. **ν12:** cVAE best-of-30+refine guarded (107 FE) MAE 0,0047 vs SIMP best-of-4 hội tụ (897 FE) 0,0071 (−34% [13; 50]). **Cặp (ν12,ν21): hòa** (CI chứa 0) - cVAE yếu ν21 (R² 0,838 vs 0,997). **Chế tạo được: SIMP 77-84% vs cVAE ~25%.** Phát hiện phụ: SIMP β=64 cũng có khoảng lệch xám↔nhị phân (2e-7 → 3,6e-2 trung vị/seed) - xác nhận luận điểm objective phải là thiết kế được verify áp dụng chung, không riêng mô hình sinh. Chi tiết + hệ quả cho bài: `docs/plan.md` mục P1.6a.

**Đính chính trong ngày (phát hiện khi vẽ `fig_stiffness`):** "hòa" ở cặp (ν12,ν21) do **1 condition #46** (target (−0,036; −1,25), ν21 thấp nhất IN100): cVAE sinh ô đứt (1 cột rỗng, E_x/E0≈2e-11). Bỏ #46: cVAE R²(ν21) 0,997, sai số cặp tốt hơn SIMP `full` 29% [9; 46]. Đề xuất báo cáo: đủ 100 + coi E<1e-3·E0 là thất bại + độ nhạy 99.

**IN100-B (seed 456, 4/100 trùng IN100) - refine tái lập trên tập mới:** single-shot MAE(ν12) −87,8% [82,3; 92,3] (R² 0,831→0,990); best-of-30 −65,8% [55,7; 74,7] (guarded R² 0,9955).

**Hội tụ lưới (`docs/paper1/scripts/mesh_convergence.py`, 30 thiết kế test):** cùng hình học, ν 50² vs 200² lệch trung vị 0,015 (3,8%), max 0,042, có hệ thống; 100² vs 200² còn 0,005. Lớn hơn ~3× MAE claim (0,0047). Thiết kế cuối verify trên 200² (`final_design_eval.py`): cVAE R²(ν12) 0,992 vs SIMP 0,985 - xếp hạng giữ.

**Retrieval verified (`retrieval_in100.py`) + lệch nhãn↔verify:** retrieval best-of-30 R²(ν12) 0,977 (không force_periodic) - thấp hơn cVAE best-of-30 0,986 → `LIMITATIONS.md` #18 "R²=1,000" là đo trên nhãn, đã đính chính. Phân tách lệch nhãn dataset↔verify (150 mẫu): khứ hồi 64→50 0,019 · +penal 3 0,024 · +nhị phân 0,032 · +force_periodic 0,055. **`force_periodic` có thể sai khái niệm** (lưới phần tử + PBC: cột 0 và 49 kề nhau, ảnh nào cũng lát được) - chờ quyết định (`LIMITATIONS.md` #34).

**OOD đổi dấu với checkpoint production (P1.6b, 28 mục tiêu ν>0):** refine giảm MAE 59-76%, cVAE+refine guarded MAE(ν12) 0,038/0,104/0,089 (đối xứng / ν21=0,1 / ν21=0,3) nhưng **SIMP từ đầu tốt hơn rõ** (0,005-0,040) → lợi thế OOD chỉ claim được so với retrieval.

**Chế tạo được sau refine (P1.6c):** 30% → 24% (**đính chính 2026-10-07:** không vững - 5 lần chạy −6 đến +9 điểm %, gộp 61 thêm / 44 mất, p=0,12). **Chi phí dataset (P1.6z):** ≈2,7-3,0M FE-solve (65-72 CPU-giờ), hòa vốn với SIMP sau ~3 500 target.

Code/test mới: `benchmark_simp_baseline.py` (+7 test), harness `--save-images` + cờ manufacturable (+1 test), `docs/paper1/scripts/{p1_6a_compare,mesh_convergence,final_design_eval,retrieval_in100,refine_examples,deformation_examples}.py`, 5 hình mới trong `make_figures.py`. 701/701 test pass. Chi tiết + số đầy đủ: `docs/plan.md` mục P1.6.


### 2026-10-07 - P1.7 lai cVAE → SIMP (KHÔNG ĐẠT), tái lập P1.6a trên IN100-B, ablation C5 `force_periodic`

**P1.7** (`pipeline/phase5_cvae/benchmark_hybrid.py`, 5 test; `run_simp_target` thêm `x0`/`betas`): SIMP khởi tạo từ thiết kế cVAE guarded, β {8,16,32,64} ×15 eval, guarded. Tiêu chí ghi trước 2026-10-06: trượt 2/4 - sai số cặp vs SIMP `full` +7% CI [−96; 62] (cận dưới < −10%, do #46 ô đứt), chế tạo 0,33 (< 0,70); FE 166 và suy biến 1% đạt → Khung A, không chạy xác nhận.

**IN100-B** (SIMP `full` mới, 889 FE/target): ν12 cVAE vs SIMP +16% [−21; 43] trên đủ 100 - không tái lập có ý nghĩa vì 2 ô cVAE đứt (#1, #57); bỏ 2 ô: +31% [6; 50].

**C5** (cùng z, chỉ khác `--force-periodic`, `docs/paper1/scripts/c5_force_periodic.py`): độ chính xác không đổi; kiểm tra cạnh khớp 1,00 → 0,30 khi bỏ fp; liên thông + nét tối thiểu 0,24 → 0,32. Đề xuất bỏ fp. Chi tiết: `docs/plan.md` mục P1.7; `LIMITATIONS.md` #32, #34, #36.

**Chẩn đoán + P1.8:** mọi ô cVAE đứt là mục tiêu dị hướng cực đoan r = max(ν21/ν12, ν12/ν21) ≥ 10, cả 30/30 ứng viên đều đứt (giới hạn phủ dữ liệu; C6 lọc lúc chọn → đóng). Xác nhận ghi trước trên IN100-C (seed 789, SIMP `full` 893 FE/target), tầng r < 10: sai số cặp cVAE thấp hơn SIMP 26,8% [7,9; 42,4] (đạt), ν12 +18,7% [−9,6; 39,7] (không đạt) → claim "ngang SIMP hội tụ với ~8× ít FE", không claim ν12 hơn. `docs/paper1/scripts/p1_8_anisotropy_strata.py`; `LIMITATIONS.md` #37.

**P1.9 + 2 đính chính:** (1) "refine làm giảm chế tạo được 30%→24%" KHÔNG vững - 5 lần chạy −6 đến +9 điểm %, gộp 61 thêm / 44 mất (p=0,12). (2) R² best-of-30 0,995 (n=300, Bảng 2 bài) là chọn thuần độ chính xác (tái lập 0,9953); pipeline composite 0,6/0,3/0,1 cho 0,983 (không fp 0,988). Single-shot không fp: MAE −92,5%, R² 0,9985. Nháp các phần mới EN+VI (Khung A): `docs/paper1/drafts/p1_10_sections_{en,vi}.tex`. `LIMITATIONS.md` #35 (đính chính), #38.

**P1.9 hoàn tất (C5 = bỏ fp, 14:43):** chạy lại 11 lần (`p1_9_r1..r8`, Pareto) → `p1_9_nofp_summary.json`. Thay đổi đáng kể so với có fp: OOD −2,0 sai số trung vị 12,5% → 1,2% (2 cụm, 4/10 lần 17-31%); so SIMP tầng r<10 ν12 +37-39% có ý nghĩa trên cả 3 tập (có fp: IN100-C không đạt); chế tạo guarded 0,24 → 0,32; Spearman Pareto 0,693 → 0,718. Không đổi: SIMP thắng OOD đổi dấu và chế tạo; refine liên tục sau best-of-30 vẫn làm tệ hơn. P1.8 giữ kết quả đăng ký trước (có fp, 1/2).

### 2026-10-08 - P1.10 số 200² không fp; K1 refine nhận thức chế tạo (ĐẠT tiêu chí, nhưng đánh đổi độ chính xác)

**200² không fp (`p1_10_final_design_eval_nofp.json`):** R²(ν12) cVAE 0,990 / SIMP full 0,985, MAE 0,014 / 0,016, E_x/E0 trung vị 0,078 / 0,070 - xếp hạng giữ nguyên như bản fp. Hình vẽ lại toàn bộ, PDF EN+VI build lại.

**Phát hiện bản lề 1 nút:** 16-26% thiết kế cVAE cuối (SIMP 7-13%) có 2 pixel rắn chỉ chạm góc. `check_connectivity` 8 hướng tính là nối; lưới 200² dựng bằng kron giữ nguyên điểm chạm 1 nút nên hội tụ lưới không bắt được. Ví dụ OOD ν*=+0,5 trong `fig_deformation` là 2 khối nối qua góc (E_x/E0 1,6e-2, không suy biến theo tiêu chí E<1e-3).

**K1 (tiêu chí ghi trước, `plan.md` mục K1):** objective refine thêm λ·(P_góc + P_mảnh) trên ảnh 64² sau Heaviside (`manuf_penalty.py`, `--corner-weight/--thin-weight`); định nghĩa chế tạo mới = liên thông 4 hướng + nét tối thiểu. λ chỉnh trên IN100-B (20 cond): 0/0,03/0,1/1 → chế tạo 0,45/0,65/0,80/0,85, chọn 0,1. Chạy IN100 + IN100-C (mỗi tập 15 phút): chế tạo 0,31 → 0,72 / 0,76, chạm góc 0,16 → 0,08 / 0,21 → 0,05, R²(ν12) 0,996 / 0,997, FE 112/target - đạt cả 5 tiêu chí. **Nhưng MAE ν12 tệ hơn 37-40% (CI không chứa 0)**, lợi thế sai số cặp so với SIMP ở tầng r<10 từ 41% [26; 54] còn 15% [−10; 34] (mất ý nghĩa). Pilot n=20 cho thấy độ chính xác tăng - không tái lập. Code mới: `manuf_penalty.py`, `check_connectivity(connectivity=4)`, `count_corner_contacts`, harness in tiến độ từng condition, `docs/paper1/scripts/k1_compare.py`; 716/716 test.

**K1 - phương án A (tác giả chọn 2026-10-08):** pipeline chính giữ λ=0, K1 là điểm vận hành thứ hai. Đường đánh đổi trên IN100-B đủ 100 target (`k1/sweep_in100b_l*.json`): λ = 0/0,03/0,1/0,3/1 → chế tạo (4 hướng) 0,33/0,61/0,79/0,81/0,83, MAE ν12 0,0059/0,0067/0,0066/0,0068/0,0077 (chỉ λ=1 tăng có ý nghĩa, +30% [6; 64]); SIMP full cùng tập 0,68 / 0,0073. λ=0,1 ở điểm gãy. Hình `fig_k1_tradeoff` dùng MAE ν12 (sai số cặp trên tập B bị target suy biến chi phối, không đơn điệu). Bản thảo EN+VI: Methods (phạt + bản lề + tiêu chí), Results (đoạn K1 + Hình 8), bảng dùng phương pháp, Discussion, Limitations, Conclusions, Abstract, chú thích `fig_deformation`; định nghĩa chế tạo các số cũ giữ 8 hướng. PDF 20 trang mỗi bản, build sạch.

### 2026-10-10 - N1 khe lưới verify ↔ hiện thực hóa; N2 thiết kế qua bộ lọc ĐẠT xác nhận đặt trước; thước đo chế tạo cũ chủ yếu đếm đảo rời

**N1 chẩn đoán (300 thiết kế IN100: cVAE λ=0 / K1 / SIMP full; tiêu chí ghi trước `plan.md` mục N1):** hiện thực hóa R(n, s) = làm mượt Gauss tuần hoàn σ → lấy mẫu song tuyến lưới n×n dịch s → ngưỡng 0,5 → FE (`pipeline/phase5_cvae/realization.py`). Độ nhạy dịch S (std ν12 qua 8 phép dịch lệch lưới) dự báo khe verify 50² ↔ R(200): Spearman 0,52 (H2 đạt); H1, H3 trượt (SIMP nhạy dịch hơn cVAE; S không khác giữa thiết kế chế tạo được / không). **Refine phần lớn là khớp lưới 50²:** MAE ν12 −68% trên lưới verify nhưng chỉ −26% trên kron 128² (0,0202 → 0,0149); lợi thế cVAE so với SIMP co lại và mất ý nghĩa trên R(200) σ=0,5 (SIMP +53% → +14%, CI chứa 0). **Hội tụ lưới (6 thiết kế, `n1/mesh_convergence.json`):** biên bậc thang (kron) hội tụ đều 50 → 400², lệch verify ~0,022 cùng chiều; biên bo σ=0,25 ở 400² về sát giá trị 50² (±0,002), chưa hội tụ (800² vượt RAM) → ν của thiết kế pixel chỉ xác định tới ~0,02 tùy biểu diễn biên. Khe 0,015 (`LIMITATIONS.md` #33) là hiệu ứng biên bậc thang, không do bản lề (p=0,13). Pilot objective N1 (verify + 4 dịch, IN100-B 20 condition): e_real σ=0,5 0,0173 → 0,0101 nhưng chế tạo 0,45 → 0,25 - mỗi objective chỉ cải thiện đúng loại hiện thực hóa nó tối ưu.

**N2 vòng 1 (tiêu chí ghi trước, pilot IN100-B 20 condition):** E1 robust formulation (co/giãn η 0,25/0,75 trên trường làm mượt σ=1) **không đạt** - chế tạo 0,45 < 0,60; refine sinh đốm vật liệu "miễn phí" ở các hàng/cột mà resize 64→50 bỏ đi, và luật guarded (chấm trên lưới 50²) loại 45% refine bền vững. E2 FE lưới 100² ở mức β cuối: e_mesh −35% nhưng e_real σ=0,5 tệ hơn 37% (CI > 0) - lần thứ 3 thấy đánh đổi "lưới mịn của biên bậc thang ↔ độ bền với biên thực tế". Tác giả chốt thước đo chính = **e_real** (MAE ν12 trên hiện thực hóa làm mượt 200², σ = 0,5 và 1,0).

**N2 vòng 2:** E1′ (robust trên thiết kế đã lọc) **không đạt** (e_real σ=0,5 +53%). **F - thiết kế qua bộ lọc** (ablation, không tiêu chí): x_phys = Heaviside(lấy mẫu 50² của ảnh decoder lọc Gauss tuần hoàn σ = 1 phần tử); FE verify, chọn best-of-30, guard, kiểm chế tạo và ảnh lưu đều trên CÙNG x_phys (`--design-filter-sigma 1.0`) - pilot trội A ở mọi thước đo, cùng chi phí → đăng ký xác nhận E3-F.

**E3-F (n=100 × 2 tập, guarded; `n2/final_e3f.log`, `e3f_scores_*.json`) - ✅ ĐẠT cả 5 tiêu chí đặt trước trên IN100, tái lập trên IN100-C:** so với A (λ=0): e_real σ=0,5 −47% [−0,0100; −0,0049] / −43% [−0,0098; −0,0040]; σ=1,0 −59%; chế tạo 0,39 → 0,78 / 0,37 → 0,81; e_verify (+32%) và e_mesh (−10%) không khác có ý nghĩa; FE/target 107 (1,00×). So với SIMP full: e_real σ=0,5 −53%, e_mesh −21% / −28% (CI < 0), chế tạo ngang (0,78 vs 0,72; 0,81 vs 0,80), ~107 vs ~900 FE/target. Giới hạn: co/giãn đều η 0,35/0,65 (e_ed) tệ hơn A 11-12% (CI > 0), ngang SIMP; e_real và e_shift cùng họ làm mượt Gauss với bộ lọc (vòng lặp một phần - trên thước đo không làm mượt F không tệ hơn A); r ≥ 10 vẫn ra ô đứt (IN100 #46).

**N2-C6 xóa đảo rời** (`n2/island_cleanup.py`; giữ thành phần rắn chính theo liên thông 4 hướng tuần hoàn): ν không đổi (|Δν12| = 0 trên 29/200 thiết kế F có đảo, trung vị 7 px). Chế tạo trước → sau, IN100 / IN100-C: A 0,39 → 0,84 / 0,37 → 0,85; F 0,78 → 0,92 / 0,81 → 0,94; SIMP 0,72 → 1,00 / 0,80 → 1,00. → **Thước đo chế tạo cũ chủ yếu đếm đảo rời** (xóa miễn phí); phần lớn lợi ích chế tạo của K1 đạt được bằng xóa đảo mà không mất độ chính xác; sau xóa đảo SIMP vẫn nhỉnh hơn F (−0,08 / −0,06, CI < 0).

**Độ nhạy σ (N2-S, pilot) + xác nhận σ = 1,5 (N2-E3-S15, n=100 × 2):** đơn điệu theo σ ở độ bền, lưới mịn và chế tạo; co/giãn đều tệ dần. σ = 1,5 đạt luật đặt trước (chế tạo thô +0,17 [+0,08; +0,26]) nhưng sau xóa đảo chỉ +0,04 [−0,03; +0,11], co/giãn đều tệ hơn SIMP 9-11% → khuyến nghị giữ σ = 1,0 + xóa đảo, báo σ = 1,5 là phân tích độ nhạy.

**Chờ tác giả (`plan.md` mục N3):** N3-D1 đổi pipeline chính Bài #1 sang F + xóa đảo, N3-D2 chọn σ, N3-D3 chạy E4 (r ≥ 10). Bản thảo EN+VI **chưa** sửa theo N1/N2. Code: `realization.py` (`realize_shifted`, `filtered_design`), cờ harness `--realization-shifts/--realization-sigma`, `--fe-upsample [--fe-upsample-last-only]`, `--robust-etas/--robust-sigma`, `--design-filter-sigma`; script chấm điểm `outputs/phase5/plan_v3/{n1,n2}/`; 732/732 test. Ghi chú viết báo (phương pháp, số, trích dẫn): `docs/paper1/ghi_chu_viet_bao.md`.

### 2026-10-10 (khuya) - Rà soát code: 3 lỗi ẩn ở train/resume và solver, KHÔNG ảnh hưởng kết quả đã có

**`train.py --resume-from`** dựng kiến trúc từ cờ CLI (mặc định KAN + ép đối xứng), không từ checkpoint: resume checkpoint Linear (mọi checkpoint production, lệnh `PIPELINE.md`, bước train lại của `self_play.py`) crash; nếu chỉ truyền `--use-mlp-head` thì load im lặng nhưng decoder bị ép đối xứng qua đường chéo suốt fine-tune - đo trên `cvae_v2_finetuned.pt`: ảnh lệch 0,126/pixel, bất đối xứng 0,253 → 0 (mục tiêu ν12 ≠ ν21 không thể đạt). `resize_condition_dim_weights` chỉ hiểu head KAN nên `--extended-condition --resume-from` checkpoint Linear crash. Đã sửa (kiến trúc lấy từ checkpoint; mặc định train mới = Linear, không đối xứng theo P1.2) và chạy thật cả 2 lệnh resume. **Ảnh hưởng quá khứ: không** - mọi checkpoint đang có đều lưu hoặc suy ra `enforce_symmetry=False`, `use_kan` khớp trọng số.

**Solver dự phòng CG** (`simp/core/solver.py`) gọi `cg(tol=)` mà SciPy 1.15 đã bỏ → luôn `TypeError`, tức thiết kế gần suy biến bị coi là FE thất bại thay vì giải CG. Đo 700 thiết kế thật (100 IN100 cuối × 2, 500 test): nhánh này kích hoạt 0 lần. Thay bằng `LinAlgError` tường minh; ν tính lại 100 thiết kế IN100 khớp JSON đã lưu tới 1e-13.

**Kiểm nhãn v21 dataset** (sanity check `backfill_f1_f2_npz.py` từng chỉ so v12): |v21 nhãn − v21 FE tính lại| TB 0,021 (train/val/test), cùng cỡ v12 (0,019) và khớp lệch nhãn↔verify đã biết (`LIMITATIONS.md` #18); kiểm tráo v12↔v21 cho 0,15 → không có landmine tráo nhãn.

**Mặc định `force_periodic` của `best_of_n_eval.py`/`sample.py`/`coverage_eval.py`** (vẫn BẬT, ngược quyết định C5): chạy so 40 condition × 30 mẫu trên `cvae_v2_finetuned.pt` - mặc định R²(ν12) best-of-30 0,974 / chế tạo 0,239 vs cấu hình bài (`--no-force-periodic --periodicity-tol 1.0`) 0,979 / 0,231 → không đo được tác hại; giữ nguyên, chờ tác giả quyết có đổi mặc định cho khớp C5. Chi tiết thay đổi code: `CHANGELOG.md`.

---

*Xem [`CHANGELOG.md`](CHANGELOG.md) cho lịch sử thay đổi theo phiên bản, và [`README.md`](README.md) cho trạng thái/cách hoạt động hiện tại của dự án.*

# Nhật ký Thử nghiệm & Lỗi đã Sửa

Tài liệu này chỉ giữ lại **những phát hiện thực sự thay đổi kết quả dự án** - bug làm sai vật lý/kết quả trên diện rộng, hoặc đột phá phương pháp. Các tính năng thử nghiệm/pilot không được áp dụng, kết quả không đáng kể, hoặc việc vệ sinh dữ liệu thuần túy không được liệt kê ở đây - xem lịch sử git/`CHANGELOG.md` nếu cần. "Hệ thống hoạt động thế nào ở trạng thái hiện tại" thuộc về [`README.md`](README.md).

---

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

Kết quả cuối: dataset v2 = 57.216 mẫu train (production hiện hành). `surrogate_v2.pt` R²(v12/v21/volfrac) = 0,974/0,964/0,983. Yield sạch sau rebuild: `hourglass`=95,5%, `hexagonal`=96,4%, `reentrant_bowtie`=42,4% (dưới mốc lịch sử 77,2% - xem phần "vẫn treo" ở cuối tài liệu này, và [`outputs/pilot_damping_decay/PILOT_REPORT.md`](outputs/pilot_damping_decay/PILOT_REPORT.md) cho một hướng khắc phục đã kiểm chứng nhưng chưa áp dụng).

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
3. `use_sqrt` (damping η=0,5) + `penal_init=2,0` (continuation nhẹ) - kỹ thuật đã kiểm chứng giúp `reentrant_bowtie` gần gấp đôi yield trên code CŨ ([`outputs/pilot_damping_decay/PILOT_REPORT.md`](outputs/pilot_damping_decay/PILOT_REPORT.md)) - thử lại trên `hexagonal` với code đã fix hôm nay: **cũng làm tệ hơn** (82,5%→70-75%, driven bởi rời rạc tăng 3→10-11 và `hit_cap` tăng mạnh - continuation cần nhiều vòng lặp hơn để hoàn thành, 150 vòng không đủ ngân sách cho seed này).

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

Code: `simp/mma_runner.py` (mới), `requirements.txt`/`pyproject.toml` (+`nlopt`), `tests/test_mma_runner.py` (5 test), `analysis/scripts/pilot_mma.py`. Trên nhánh `OC-2-multilayer`, uncommitted.

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

Theo đúng chuẩn nghiêm ngặt dự án đã dùng (`PILOT_REPORT.md` 2026-07-26: luôn lên N=400 trước khi cân nhắc đổi default) - nhân đôi rồi gấp 4 lần N cho 2 seed đã thắng/cải thiện ở pilot N=100. Bỏ qua `reentrant_bowtie` - N=100 đã cho CI cực hẹp [0,0%,3,7%], scale thêm không đổi kết luận, lãng phí compute.

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

Code: `analysis/scripts/skfem_homogenization.py` (engine độc lập, có thể tái dùng), `analysis/scripts/run_independent_fe_check.py` (script so sánh), `tests/test_skfem_homogenization.py` (7 test, gồm cổng phân tích ô đặc). 437/437 test toàn repo pass. Kết quả đầy đủ: `outputs/phase5/reports/independent_fe_crosscheck.json`. Trên nhánh `OC-2-multilayer`, uncommitted.

---

### 2026-08-02 - Phản hồi audit khoa học: novelty/diversity metric + baseline comparison lần đầu tiên - phát hiện hit_rate là metric yếu, retrieval baseline ngang cVAE trên test set hiện tại

Hai báo cáo audit bên ngoài (`AUDIT_REPORT_SCIENTIFIC_2026-08-02.md`, `AUDIT_REPORT_SCIENTIFIC_REVIEW_2026-08-02.md`) chỉ ra 2 khoảng trống cụ thể chưa có bằng chứng trong pipeline: (1) chưa đo cVAE có sinh thiết kế "mới" hay chỉ tái tạo gần dữ liệu train, (2) chưa có baseline nào ngoài so sánh optimizer nội bộ (OC vs MMA) - không biết nếu bỏ cVAE đi thì kết quả còn giữ được không.

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

Code: `analysis/scripts/pilot_normalized_objective.py` (mới). Manifest đầy đủ: `outputs/pilot_normalized_{hexagonal,hourglass,reentrant_bowtie}/manifest.csv`. Đã cập nhật `docs/LIMITATIONS.md` mục #4 (VI+EN). Trên nhánh `main`, uncommitted.

---

*Xem [`CHANGELOG.md`](CHANGELOG.md) cho lịch sử thay đổi theo phiên bản, và [`README.md`](README.md) cho trạng thái/cách hoạt động hiện tại của dự án.*

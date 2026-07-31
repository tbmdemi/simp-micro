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

*Xem [`CHANGELOG.md`](CHANGELOG.md) cho lịch sử thay đổi theo phiên bản, và [`README.md`](README.md) cho trạng thái/cách hoạt động hiện tại của dự án.*

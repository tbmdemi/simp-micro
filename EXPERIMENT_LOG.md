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

**`hexagonal` còn khoảng hở đã CHẤP NHẬN** (không phải bug chưa sửa): yield 93,9%→80,5% (xác nhận thật ở n=400, CI 95% không chồng lấn lịch sử). Thử tăng `delta` (10%→25%) và `beta` (1,0→4,0) để phục hồi ổn định - **cả 2 đều phản tác dụng mạnh** (sạch giảm còn 0-10%, do bùng nổ dao động/rời rạc, hoặc bộ tối ưu bỏ hẳn mục tiêu auxetic) - xác nhận cấu hình mặc định (`delta=10%, beta=1,0`, khớp thiết kế tham chiếu gốc) đã là lựa chọn tốt nhất trong các mức đã thử. Nguyên nhân nhiều khả năng: bug A1 từng vô tình làm "regularizer" cho riêng seed mỏng/khó nhất này suốt lịch sử dự án; sau khi sửa đúng, ngưỡng 10% intended không đủ mạnh để giữ ổn định như trước - hạn chế thật của thiết kế, không phải lỗi mới. Xem [README.md](README.md) mục Giới hạn Đã biết #16.

**Ý nghĩa:** dataset 57k mẫu sản xuất hiện có sinh bằng code CŨ (trước các fix này) - KHÔNG bị ảnh hưởng, không cần rebuild. Nhưng bất kỳ lần sinh dữ liệu MỚI nào dùng code hiện tại đều cần các fix này (mặc định, không phải flag thử nghiệm) để tránh hỏng nặng trên `hexagonal`.

---

*Xem [`CHANGELOG.md`](CHANGELOG.md) cho lịch sử thay đổi theo phiên bản, và [`README.md`](README.md) cho trạng thái/cách hoạt động hiện tại của dự án.*

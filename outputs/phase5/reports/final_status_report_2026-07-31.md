# Báo cáo tổng hợp trạng thái dự án - 2026-07-31

Gộp toàn bộ số liệu **đã kiểm chứng bằng FE thật** (không phải qua surrogate/proxy)
tính tới thời điểm này, theo khung roadmap 8 phase. Mọi số liệu dưới đây có nguồn
gốc trực tiếp từ `EXPERIMENT_LOG.md`/`README.md`/output thật trong `outputs/`, không
suy diễn. Mục đích: 1 tài liệu duy nhất để trình advisor hoặc làm nền cho luận văn/
báo cáo khoa học, nêu rõ cả kết quả đạt được lẫn giới hạn còn mở.

---

## 1. Tóm tắt kết quả theo phase

| Phase | Trạng thái | Chỉ số chính |
|---|---|---|
| 1-3 (screening, DOE, dataset) | ✅ Hoàn thành | 57.216 mẫu train (dataset v2), auxetic rate 91,9% |
| 4 (surrogate CNN) | ✅ Hoàn thành | R²(v12/v21/volfrac) = 0,974/0,964/0,983 (n=1.184 test) |
| 5 (cVAE inverse design) | ✅ Hoàn thành | R²(FE,n=789)=**0,999** (oracle)/0,992 (K=10); hit-rate single-shot **98,7%**; frac_manufacturable tự nhiên 0,35 |
| 6 (hậu xử lý & kiểm định FEA) | ✅ Hoàn thành | connectivity/min-feature/periodicity + `force_periodic()` |
| 7 (active-learning loop) | ✅ Kết luận | Implement+chạy production - KHÔNG cải thiện checkpoint đã tối ưu (xem §3) |
| 8.1 (đối chiếu FE độc lập) | ✅ Hoàn thành | 2 engine độc lập khớp tới **1e-9**, R²=1,000000 (n=24) |
| 8.2 (biến dạng lớn) | ⬜ Chưa làm | Tuỳ chọn - xem §5 |
| 8.3 (thư viện thiết kế) | ✅ Hoàn thành | 24 thiết kế, hit-rate 100%, 21/24 (87,5%) xác nhận manufacturable |
| 8.4 (báo cáo này) | ✅ Hoàn thành | - |
| 8.5 (chuẩn bị STL) | ⬜ Chưa làm | Ngoài phạm vi code, chờ kế hoạch chế tạo thật |

---

## 2. Kết quả cốt lõi: cVAE inverse design (Phase 5) đã được xác thực nghiêm ngặt

Checkpoint production: **`cvae_realphysics.pt`** (fine-tune 35 epoch, differentiable
real-physics loss, `--lambda-real-physics 20.0`).

- **R²(FE thật, n=789 - toàn bộ test set, không phải mẫu nhỏ n=24 như báo cáo lịch sử):**
  0,9988 (oracle, best-of-30) / 0,9920 (thực dụng, K=10 FE-verify).
- **hit_rate_single_shot = 98,7%** (nhảy từ ~35-40% ở mọi checkpoint trước
  differentiable-physics).
- **frac_manufacturable = 0,35** tự nhiên (không cần đánh đổi accuracy).
- Đây là câu trả lời **có** cho câu hỏi nghiên cứu cốt lõi: cVAE + differentiable-physics
  giải được bài toán thiết kế ngược auxetic metamaterial, với bằng chứng thống kê ở quy
  mô toàn bộ test set (không phải n=24 CI rộng như các báo cáo sơ bộ trước đây).

## 3. Phase 7 (active-learning): thử nghiệm trung thực, kết quả âm tính có ý nghĩa

Implement vòng lặp active-learning thật (tìm vùng yếu qua coverage grid → sinh+FE-verify
ứng viên → train lại surrogate+cVAE → đo lại → quyết định dừng), chạy ở quy mô production
(N=15 mẫu/target, grid_size=8, 20 epoch fine-tune).

| | Round 0 (baseline) | Round 1 |
|---|---|---|
| hit_rate | 1,000 | 1,000 |
| mean_abs_error | **0,0246** | **0,0686** (tệ hơn ~2,8 lần) |

**Kết luận:** checkpoint gốc đã ở gần mức trần (chỉ 5 target auxetic trong lưới coverage
8 điểm, cả 5 đã hit sẵn) - không có "vùng yếu" thật để active-learning nhắm vào. Fine-tune
thêm trên 1 batch nhỏ (23 mẫu) từ 1 checkpoint đã hội tụ rất chặt làm số liệu tệ đi thay vì
cải thiện. **Quyết định: giữ nguyên `cvae_realphysics.pt`, không promote checkpoint mới.**
Đây là 1 kết quả âm tính có giá trị phương pháp luận (cùng bài học với thử nghiệm OC
hai-multiplier ở Phase data-gen: 1 kỹ thuật hợp lý về lý thuyết không tự động cải thiện
1 hệ thống đã tối ưu tốt).

## 4. Phase 8.1: xác nhận độc lập độ đúng vật lý của FE solver

Đối chiếu engine FE/homogenization nội bộ (`simp/core/solver.py` +
`simp/homogenization/compute.py`) với 1 implementation **hoàn toàn độc lập** viết mới
trên thư viện `scikit-fem` (`analysis/scripts/skfem_homogenization.py` - khác mesh,
khác đánh số DOF, khác code lắp ráp, khác code periodic BC - chỉ chung lý thuyết
homogenization, điều không tránh khỏi ở bất kỳ implementation nào).

- Cổng kiểm chứng phân tích (ô đặc hoàn toàn ⇒ Q = D dạng đóng): **PASS** (rtol=1e-6).
- So sánh trên 24 mẫu thật trải khắp property space (v12 ∈ [-0,82, -0,04]):
  **mean|Δv12| = 3,7×10⁻⁹, max|Δv12| = 6,8×10⁻⁹, R² = 1,000000.**

Mức khớp này tương đương sai số làm tròn dấu phẩy động (2 sparse solver khác nhau),
không phải trùng hợp. Đây là bằng chứng độc lập mạnh nhất có thể có cho tới nay rằng
R²=0,999 ở Phase 5 phản ánh **vật lý đúng thật**, không chỉ "nhất quán nội bộ" (từng là
Giới hạn Đã biết #15).

## 5. Data-generation (OC vs MMA optimizer) - bối cảnh liên quan

Không thuộc roadmap 8-phase chính nhưng ảnh hưởng trực tiếp tới chất lượng dataset
tương lai. MMA (`nlopt.LD_MMA`) thắng rõ rệt trên 2/3 seed (N=400, CI không chồng lấn):

| Seed | `gate`/OC (baseline) | `mma` |
|---|---|---|
| `hexagonal` | 76,5% | **99,8%** |
| `hourglass` | 76,8% | **88,0%** |
| `reentrant_bowtie` | **90,0%** | 0,0% (thất bại nặng) |

Dispatch per-seed đã implement (`generate_production_batch.py`) nhưng **chưa chạy để
regenerate dataset 57k mẫu production** - quyết định riêng, chưa thực hiện.

## 6. Thư viện thiết kế đã tuyển chọn (Phase 8.3)

24 thiết kế, chọn stratified theo v12 từ target thật (`outputs/phase3/test.npz`), mỗi
thiết kế là best-of-30 (FE thật, `require_manufacturable=True`):

- **hit_rate = 100%** (24/24 đạt đúng dấu auxetic mong muốn).
- **21/24 (87,5%) xác nhận manufacturable thật sự** (double-check trực tiếp trên ảnh đã
  lưu, không chỉ dựa vào cờ filter lúc chọn).
- **3/24 fallback accuracy-only** (không tìm được ứng viên manufacturable nào trong 30
  mẫu) - cả 3 đều là target gần trung tính (v12 ∈ [-0,13, -0,07]) - dấu hiệu manufacturability
  khó hơn ở vùng auxetic yếu, đáng điều tra thêm nếu cần thư viện đầy đủ hơn.
- Vị trí: `outputs/phase5/design_library/` (24 PNG + `manifest.json`/`manifest.csv`).

**Lưu ý phạm vi:** lựa chọn "best" ở đây dùng accuracy (FE thật) + manufacturability -
KHÔNG dùng composite score 3 trục (accuracy/manuf/aesthetic) vì phần đó chỉ tồn tại trên
nhánh `feature/optional-multi-condition` chưa merge vào `OC-2-multilayer`.

---

## 7. Giới hạn còn mở (chưa giải quyết trong báo cáo này)

Xem đầy đủ tại README §Giới hạn Đã biết. Các mục **còn mở**, có ý nghĩa nếu hướng tới
công bố khoa học hoặc luận văn:

1. **Thành phần phạt `mu` trong objective auxetic đang tắt** (`mu=0,0`, sai sót khái
   niệm chưa thiết kế lại) - toàn bộ dataset hiện có dùng cấu hình này.
2. **`f1, f2` (mục tiêu độ cứng chuẩn hoá theo roadmap gốc) chưa khả dụng** - Pha B của
   multi-param conditioning bị hoãn.
3. **~24% dataset có hình học rời rạc, `converged=True` không đáng tin cho 76,3% mẫu**
   (audit 2026-07-24) - chưa dùng để lọc lại dataset trước khi train Phase 4/5.
4. **Toàn bộ pipeline dùng FEM tuyến tính (small-strain)** - chưa kiểm chứng auxetic
   effect có giữ được dưới biến dạng lớn/phi tuyến hay không (Phase 8.2, tuỳ chọn,
   CHƯA làm - đây là việc nặng nhất còn lại nếu cần, vì phải derive lại bài toán phi
   tuyến chứ không chỉ đổi thư viện như 8.1).
5. **`hexagonal` yield thấp hơn lịch sử đã CHẤP NHẬN** (80,5% so với 93,9%, đã thử 3
   hướng khắc phục đều phản tác dụng - xác nhận không có fix đơn giản).
6. **Đối chiếu FE độc lập (8.1) mới dùng `scikit-fem`**, chưa đối chiếu với phần mềm
   thương mại/được công nhận rộng rãi hơn (ANSYS/Abaqus/FEniCS/COMSOL) - đủ cho quy mô
   dự án hiện tại, nhưng nếu hướng tới công bố ở venue cao hơn có thể cần thêm.
7. **Phase 8.5 (chuẩn bị xuất STL/CAD) và bước chế tạo/thử nghiệm vật lý thật** hoàn
   toàn chưa bắt đầu - roadmap gốc có đề cập nhưng ngoài phạm vi 1 dự án thuần
   computational hiện tại.

---

## 8. Kết luận

Theo mục tiêu nghiên cứu cốt lõi (cVAE + differentiable-physics giải bài toán thiết kế
ngược auxetic metamaterial, đạt đúng target property, sản xuất được, và đúng vật lý đã
kiểm chứng độc lập) - dự án đã đạt **đầy đủ bằng chứng cần thiết**: R²=0,999 (n=789),
hit-rate 98,7%, đối chiếu FE độc lập khớp 1e-9, thư viện 24 thiết kế hit-rate 100%/87,5%
manufacturable xác nhận. Đây là mức độ chặt chẽ khoa học phù hợp để trình bày như 1 kết
quả proof-of-concept đã kiểm chứng nghiêm ngặt.

Nếu scope mở rộng hướng tới công bố/luận văn ở mức cao hơn, 3 mục §7.1/§7.2/§7.4 (mu
penalty, f1/f2, đối chiếu FE thương mại) là các khoảng trống đáng ưu tiên tiếp theo;
§7.4 biến dạng lớn (Phase 8.2) là khoảng trống nặng nhất về công sức nếu roadmap hướng
tới chế tạo/thử nghiệm vật lý thật trong tương lai.

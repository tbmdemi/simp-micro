# Kế hoạch dự án & Tech Stack

Trạng thái: v2, sau vòng research/đánh giá/phản biện bằng agent (2026-08-14) đối chiếu trực tiếp
với code + git log thật. Bản v1 có 2 sai lệch thực tế đã sửa: (1) nhánh
`feature/optional-multi-condition` **đã merge vào `main` từ 2026-07-25** (PR #13), không còn
"uncommitted"; (2) Giai đoạn D (động tuyến tính) bị đánh giá thấp effort - thực tế L-XL, không
phải "chi phí thấp". Chưa đụng vào code - tài liệu này chỉ để lên kế hoạch.

## 1. Tech stack hiện tại

| Lớp | Công nghệ | Vai trò |
|---|---|---|
| Ngôn ngữ | Python ≥3.10 | Toàn bộ pipeline |
| Số học/FE | NumPy, SciPy | Lắp ráp ma trận độ cứng, giải hệ tuyến tính FE |
| Tối ưu hóa | `nlopt` (LD_MMA), OC tự viết | Vòng lặp SIMP - MMA cho ràng buộc đa mục tiêu, OC cho bài toán đơn giản |
| Kiểm chứng FE độc lập | `scikit-fem` | Đối chiếu vật lý, không dùng chung engine với `simp/core/` |
| Học máy | PyTorch ≥2.0 | CNN surrogate (Phase 4) + cVAE (Phase 5) |
| Tiền xử lý/dataset | scikit-learn, pandas, Pillow | Chia train/val/test, thao tác ảnh mật độ |
| Trực quan hóa | Matplotlib | Biểu đồ phân tích, hình trong report |
| Kiểm thử | pytest | `tests/` - **483/483 pass** (verify 2026-08-14) |
| Đóng gói | `pyproject.toml` (setuptools) | Extras: `dev`, `analysis`, `ml`, `all` |
| CLI entry points | `simp`, `simp-analysis` | `simp/main.py`, `analysis/cli.py` |

**Kiến trúc pipeline (8 phase):** LHS Screening → Multi-Batch Adaptive DOE → Dataset Build → CNN Surrogate → Conditional VAE (differentiable real-physics fine-tune) → best-of-N + composite scoring → manufacturability/export. Checkpoint khuyến nghị: `outputs/phase5/cvae_v2_finetuned.pt` (R²(FE,n=789)=0.992-0.999, hit-rate ≥98%).

## 2. Mục tiêu kế hoạch này

Lấp khoảng trống A∩B∩C xác định ở gap analysis: **chưa có công trình nào dùng generative model điều kiện hóa theo cả hình học lẫn vật liệu nền, đồng thời xếp hạng đa mục tiêu.**

**Cập nhật quan trọng sau research:** Giai đoạn B (xếp hạng đa mục tiêu) **đã hoàn thành ~90%** trên `main` - composite score (0.6 accuracy + 0.3 manuf + 0.1 aesthetic, `best_of_n_eval.py:59-228`) và cơ chế presence-mask/condition-dropout (`dataset.py`, `train.py:118-157`) là code production, không phải draft. Việc còn thiếu thật sự nằm ở Giai đoạn A (vật liệu nền) - và effort ở đó lớn hơn v1 ước lượng vì đụng tới `real_physics.py` và Phase 4 surrogate, không chỉ `params.py`/`isotropic.py`.

## 3. Roadmap (ưu tiên theo effort thật + phụ thuộc thật, không theo A→B→C máy móc)

### Nhóm 0 - Dọn tài liệu (S, làm ngay)

| # | Việc |
|---|---|
| 0.1 | Đã sửa: xóa "uncommitted" khỏi mô tả `feature/optional-multi-condition` trong plan này |
| 0.2 | Đã sửa (2026-08-15): cập nhật memory note `phase5_optional_multiparam.md` - phản ánh đúng trạng thái đã merge PR #13, và liên kết rõ với tiến độ Pha B (f1/f2) đã backfill ở Phase 4 |

### Nhóm 1 - Giai đoạn A: vary vật liệu nền (L, effort lớn nhất, ưu tiên cao nhất)

| # | Việc | File | Effort |
|---|---|---|---|
| A0 | Ước lượng số điểm ν0 cần sample + cost sinh dữ liệu tương ứng (benchmark trên thời gian Phase 1 screening hiện có) | - | M |
| A1 | Thêm `nu0`/`E0` vào `PARAM_SPACE`, **và propagate xuống `pipeline/phase2_multi_batch/runner.py::DEFAULT_FIXED`** (độc lập với `params.py`, xem comment `params.py:20-24` - nếu bỏ qua, dataset thật vẫn cố định ν0) | `pipeline/params.py`, `pipeline/phase2_multi_batch/runner.py` | M |
| A2 | `Material` nhận `nu`/`E0` biến thiên theo sample (hạ tầng validate đã sẵn, không cần sửa nhiều) | `simp/materials/isotropic.py` | S |
| A3 | Sinh batch nhỏ dải hẹp ν0 (0.2–0.4 theo Nhóm 7) kiểm tra hội tụ FE trước khi mở rộng | `pipeline/phase1_screening/` | M |
| A4 | **Thêm ν0/E0 làm input phụ cho CNN surrogate, retrain, so R² với baseline (0.974/0.964).** Bắt buộc làm TRƯỚC A6 - nếu bỏ qua, surrogate cũ học sai vì quan hệ hình học→tính chất không còn là hàm 1-1 của ảnh mật độ khi ν0 biến thiên | `pipeline/phase4_surrogate/model.py`, `dataset.py` | M-L |
| A5 | Sửa `real_physics.py` để nhận `nu`/`E0` **per-sample** (hiện là 3 scalar áp cho cả batch, `RealPhysicsFELayer.forward` dòng 156-171); đánh giá lại chi phí mesh/PBC-cache vốn được key theo `(nelx,nely,E0,Emin,nu)` - sẽ mất tác dụng tăng tốc khi mỗi sample có ν0 riêng | `pipeline/phase5_cvae/real_physics.py` | L |
| A6 | Nối ν0/E0 làm condition optional cho cVAE (tái dùng đúng pattern `extended_condition`/dropout đã có trong `main`, KHÔNG cần tham chiếu nhánh riêng) | `pipeline/phase5_cvae/dataset.py`, `model.py`, `train.py` | M - phụ thuộc A4, A5 |

### Nhóm 2 - f1/f2 vào Phase 5 (S-M, độc lập, có thể làm song song Nhóm 1)

| # | Việc | File |
|---|---|---|
| 2.1 | Nối f1/f2 (đã backfill Phase 4, R²=0.933/0.961) làm condition/target phụ cho cVAE - lặp lại đúng pattern `extended_condition` đã dùng cho volfrac/void_size_frac | `pipeline/phase5_cvae/dataset.py`, `train.py` |

### Nhóm 3 - Xếp hạng đa mục tiêu: chỉ còn validate, không còn viết code (S)

| # | Việc | File |
|---|---|---|
| 3.1 | Trích số liệu baseline sẵn có (`LIMITATIONS.md` mục 1, R²(FE,n=789)=0.992-0.999) thay vì chạy lại từ đầu | - |
| 3.2 | Đối chiếu composite score với 4 paper A∩B (TO+Pareto) bằng Pareto front độc lập - **đặt tiêu chí dừng định lượng trước khi chạy** (ví dụ Spearman correlation ≥ 0.7), tránh diễn giải có lợi sau khi có kết quả (bài học từ bug chọn checkpoint theo `val_loss`, `LIMITATIONS.md` mục 12) | `notebooks/` |

### Nhóm 4 - Củng cố lập luận khoa học trước khi viết luận văn (S, độc lập, làm song song 1-2)

| # | Việc | Ghi chú |
|---|---|---|
| 4.1 | Trả lời trực diện câu hỏi "vì sao generative thay vì retrieval" - retrieval baseline hiện đang THẮNG cVAE trong-phân-phối (R²=1.000 vs 0.973, `LIMITATIONS.md` mục 18), chỉ thua rõ ở OOD. Thiết kế thí nghiệm: retrieval trên trục ν0 mới sẽ thất bại ngoài tập train, cVAE có thể nội suy - tái dùng thiết kế đã có ở `phase5_ood_baseline.md` | Rủi ro lập luận lớn hơn cả gap A∩B∩C nếu bỏ qua |
| 4.2 | **Kiểm định giả thuyết "curse of dimensionality"** - retrieval thắng ở 2 chiều điều kiện (v12,v21) là hiển nhiên vì dataset đủ dày (`mean_condition_dist`=0,002); nhưng khi Giai đoạn A thêm ν0/E0 (lên 4-6 chiều điều kiện), khoảng cách nearest-neighbor của retrieval được dự đoán sẽ tăng nhanh hơn sai số cVAE (dữ liệu cần tăng cấp số nhân theo số chiều để giữ cùng độ dày, cVAE thì không). Đo lại đúng phép so sánh retrieval-vs-cVAE (`analysis/scripts/baseline_comparison.py`) ở không gian điều kiện mở rộng - đây là luận điểm trung tâm để trả lời "vì sao cần inverse design" thay vì tra bảng, quan trọng hơn cả thí nghiệm sign-flip OOD đã có | Phụ thuộc Giai đoạn A (Nhóm 1) xong; chưa kiểm chứng, hiện chỉ là giả thuyết |
| 4.3 | (Tùy chọn) Thiết kế **hybrid retrieval-gated generative**: router theo khoảng cách (retrieval khi điều kiện nằm trong vùng dày, cVAE khi ngoài vùng dày/OOD) + latent warm-start (encode sample retrieval gần nhất thành z0 thay vì sample ngẫu nhiên từ prior) + gradient refinement khả vi qua `real_physics.py`. Biến câu hỏi "ai thắng ai" thành đóng góp hệ thống - kết hợp đúng chỗ tốt hơn dùng riêng lẻ, tránh phải tuyên bố cVAE "thắng" retrieval một cách gượng ép | Tái dùng `baseline_comparison.py` + `best_of_n_eval.py` + `real_physics.py`; rủi ro: warm-start "lười" có thể suy biến thành retrieval trá hình - theo dõi bằng `novelty_diversity_eval.py` |

### Nhóm 5 - Giai đoạn D (động tuyến tính): hoãn, effort thực tế L-XL không phải "thấp"

Đã xác nhận qua code: **không tồn tại** bất kỳ hạ tầng mass-matrix/eigenvalue/Bloch-Floquet nào (`grep` toàn repo cho `eigsh|frequency|bandgap|omega` → 0 kết quả vật lý thật). `simp/core/pbc.py` chỉ là projection tĩnh thực (real-valued) cho 3 load case biến dạng đơn vị - band-gap cần PBC **phức** (Bloch phase `e^{ik·L}`) và sweep qua Brillouin zone, gần như viết lại từ đầu chứ không phải "mở rộng".

| # | Việc | File | Effort |
|---|---|---|---|
| D0 | Xác nhận với GS liệu D có cần cho luận văn hiện tại hay để đề tài con - áp dụng logic tương tự Giai đoạn E | - | - |
| D1 | Lắp ma trận khối lượng M từ trường mật độ ρ | `simp/core/` | M |
| D2 | Giải trị riêng suy rộng `Kφ=ω²Mφ`; xử lý spurious low-frequency mode do `Emin=1e-9` ở vùng gần-rỗng (vấn đề kinh điển trong eigenfrequency TO - cần mode-tracking/MAC criterion) | `simp/core/` | M-L |
| D3 | Band-gap qua Bloch-Floquet - viết PBC phức mới, sweep Brillouin zone | `simp/core/pbc.py` (viết mới phần lớn) | L |
| D4 | Nếu đưa tần số/band-gap làm condition cVAE: kéo theo retrain Phase 4 + Phase 5 đầy đủ (effort ≈ Nhóm 1) | `pipeline/phase4_surrogate/`, `pipeline/phase5_cvae/` | L |

### Giai đoạn E - Động phi tuyến (nổ/va đập): vẫn ngoài phạm vi

Cần solver FE động tường minh hoàn toàn mới, không tái dùng `simp/core/`. Không đưa vào roadmap này trừ khi GS yêu cầu chính thức và đồng ý tách thành đề tài con.

### Nhóm 6 - Giai đoạn F: Nhiệt (CTE) - ưu tiên cùng cấp Nhóm 1 (quyết định 2026-08-14)

**Quyết định phạm vi (2026-08-14):** trong 3 domain vật lý mới (động tuyến tính, áp điện, nhiệt),
chọn **vật liệu nền + nhiệt** làm ưu tiên - cùng họ toán homogenization đã có, effort M-L, có sẵn
công thức tham khảo trực tiếp trong literature review (Nhóm 2: Sigmund & Torquato 1997 - 3-phase TO
cho CTE cực trị; Guo et al. 2024 - NTE bằng energy-based homogenization, cùng kỹ thuật PBC dự án
đang dùng). **Áp điện bị loại khỏi roadmap này** - cần constitutive model điện-cơ ghép đôi hoàn
toàn mới, effort nặng hơn cả Giai đoạn D, không tái dùng được `simp/core/`. **Động tuyến tính (D)
giữ nguyên trạng thái hoãn** như Nhóm 5 đã phân tích.

| # | Việc | File | Effort |
|---|---|---|---|
| F0 | Đọc kỹ công thức energy-based homogenization cho CTE trong Guo et al. (2024) - xác nhận mức độ tái dùng được với `simp/homogenization/compute.py` (cùng PBC, khác phương trình cấu thành: cần thêm hệ số giãn nở nhiệt α vào vật liệu, giải thêm 1 load case "nhiệt đơn vị" bên cạnh 3 load case biến dạng đơn vị hiện có) | - | S |
| F1 | Thêm `alpha` (hệ số giãn nở nhiệt vật liệu nền) vào `Material` - cùng cơ chế với `nu`/`E0` ở A2 | `simp/materials/isotropic.py` | S |
| F2 | Mở rộng homogenization: giải thêm load case nhiệt, tính tensor CTE hiệu dụng α* (không chỉ Q) | `simp/homogenization/compute.py`, `simp/core/pbc.py` | M-L |
| F3 | Đối tượng tối ưu mới hoặc mở rộng `simp/objectives/auxetic.py`: CTE cực trị/âm, có thể đồng thời với auxetic (2 tính chất) - tham khảo trực tiếp cách Sigmund & Torquato ràng buộc 3-pha | `simp/objectives/` | M |
| F4 | Sinh dataset nhỏ kiểm tra hội tụ trước khi tích hợp vào Phase 1-3 đầy đủ | `pipeline/phase1_screening/` | M |
| F5 | Thêm α (CTE) làm output/condition cho surrogate + cVAE - lặp lại đúng pattern đã dùng cho f1/f2 (Nhóm 2) | `pipeline/phase4_surrogate/`, `pipeline/phase5_cvae/` | M |

**Phụ thuộc với Nhóm 1:** F1 dùng chung cơ chế mở rộng `Material` với A2 - nên làm A2+F1 cùng lúc
(cả hai đều là "thêm 1 thuộc tính vật liệu nền"), tránh sửa `isotropic.py` 2 lần riêng biệt.

## 4. Rủi ro cần theo dõi

- **A3** có thể phá hội tụ FE nếu ν0 âm hoặc gần biên `(-1, 0.5)` - bắt đầu dải hẹp, mở rộng dần.
- **A5** là điểm rủi ro kỹ thuật lớn nhất của Nhóm 1 - mất tác dụng tăng tốc của mesh/PBC-cache có thể làm chậm đáng kể vòng lặp differentiable-physics training.
- **A4 trước A6** là ràng buộc thứ tự bắt buộc - bỏ qua sẽ khiến mọi số liệu R² dựa trên surrogate cũ sai lệch có hệ thống.
- **Nhóm 4** không phải việc phụ - nếu luận văn tập trung vào A∩B∩C mà chưa xử lý câu hỏi "vì sao không dùng retrieval", đây là lỗ hổng lập luận dễ bị phản biện chỉ ra hơn cả gap chính.
- **Nhóm 5 (D)** effort thực tế gần bằng cả Nhóm 1 nếu D4 được chọn - không nên coi là "mở rộng nhẹ" khi trình bày với GS.
- **Nhóm 6 (F)** F2 là bước rủi ro nhất - cần xác nhận công thức CTE homogenization của Guo et al. (2024) áp dụng đúng với cách project tính Q trước khi code, tránh vừa viết vừa suy diễn công thức sai (bài học từ lỗi hoán vị `dQ` từng gặp, LIMITATIONS.md mục 10).

## 5. Ước tính độ phức tạp, khả thi, ETA, khả năng ra báo

ETA tính theo person-week giả định làm bán thời gian (song song với việc học/nghiên cứu khác),
**không** tính thời gian chờ GS duyệt hoặc review. Đây là ước tính, không phải cam kết - dùng để
sắp xếp thứ tự ưu tiên, không dùng để báo cáo tiến độ cứng.

| Nhóm | Độ phức tạp | Khả thi (trong khung luận văn hiện tại) | ETA | Khả năng ra báo/đóng góp khoa học |
|---|---|---|---|---|
| 0 - Dọn tài liệu | Rất thấp (S) | Cao | ~1 buổi | Không - chỉ vệ sinh tài liệu |
| 1 (A) - Vary vật liệu nền | Cao (L, 7 sub-task) | Cao, nhưng A5 (real_physics.py per-sample) là điểm nghẽn kỹ thuật thật | 3–5 tuần | **Cao** - đây chính là mảnh lấp gap A∩B∩C, thành phần trung tâm của đóng góp |
| 2 - f1/f2 vào Phase 5 | Thấp (S-M) | Cao - chỉ lặp lại pattern `extended_condition` đã có | 3–5 ngày | Bổ trợ - làm đầy đủ hơn output, không tự thành đóng góp riêng |
| 3 - Validate xếp hạng đa mục tiêu | Thấp (S) | Cao - số liệu FE baseline đã có sẵn | ~1 tuần (chờ chạy Pareto đối chiếu) | Bổ trợ - củng cố phần B đã có, không phải kết quả mới |
| 4.1–4.2 - Củng cố lập luận retrieval-vs-generative | Thấp-Trung bình (S, nhưng 4.2 phụ thuộc Nhóm 1 xong) | Cao cho 4.1 (dùng lại thiết kế OOD sẵn có); 4.2 khả thi trung bình vì cần chờ Giai đoạn A | 4.1: ~1 tuần độc lập; 4.2: ~1 tuần sau khi Nhóm 1 xong | **Bắt buộc phải có** - không tự thành 1 kết quả để "khoe", nhưng thiếu nó thì toàn bộ lập luận vì sao dùng generative dễ bị phản biện bác |
| 4.3 - Hybrid retrieval-gated generative (tùy chọn) | Trung bình-Cao (M-L) | Trung bình - phụ thuộc 4.1/4.2 xong và Nhóm 1 ổn định | 2–3 tuần nếu làm | Cao nếu thành công - biến câu hỏi thắng/thua thành 1 đóng góp phương pháp riêng biệt, nhưng là hạng mục rủi ro-cao/lợi-ích-cao, không phải việc bắt buộc |
| 5 (D) - Động tuyến tính | Rất cao (L-XL) | Thấp trong khung thời gian hiện tại - chưa có hạ tầng mass-matrix/eigenvalue nào, D0 cần GS quyết định trước | 6–10+ tuần nếu làm đủ D0-D4 | Cao nếu làm trọn (auxetic + band-gap ít người kết hợp), nhưng effort/rủi ro vượt xa lợi ích trong khung luận văn - **đang hoãn**, không đưa vào ETA tổng |
| E - Động phi tuyến (nổ/va đập) | Rất cao, cần solver mới hoàn toàn | Rất thấp - không tái dùng được `simp/core/` | Không ước tính (ngoài phạm vi) | Ngoài phạm vi trừ khi GS yêu cầu tách đề tài con |
| 6 (F) - Nhiệt (CTE) | Trung bình-Cao (M-L) | Trung bình-Cao - F2 (verify công thức homogenization) là rủi ro chính, còn lại tái dùng hạ tầng sẵn có | 3–5 tuần | **Cao** - mở rộng multi-physics (auxetic + CTE) tự nhiên trên cùng nền A∩B∩C, cùng họ toán đã kiểm chứng |

**Thứ tự ưu tiên tổng hợp theo bảng trên:** Nhóm 1 (A) trước tiên vì là core của đóng góp; Nhóm 4.1
làm song song sớm vì rẻ và bắt buộc; Nhóm 2/3 chen vào bất cứ lúc nào vì rẻ và độc lập; Nhóm 6 (F)
sau khi Nhóm 1 ổn định (dùng chung hạ tầng `Material`); Nhóm 4.2/4.3 sau cùng vì phụ thuộc Nhóm 1;
Nhóm 5 (D) và Giai đoạn E giữ nguyên trạng thái hoãn/ngoài phạm vi.

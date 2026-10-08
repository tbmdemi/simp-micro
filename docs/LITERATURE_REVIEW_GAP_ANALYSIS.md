# Báo cáo: Khái quát tài liệu & Phân tích vùng tối

Nguồn: 97 bài báo, 9 nhóm chủ đề, tổng hợp qua literature review (Google Docs, 2026-08-14). Bối cảnh: GS hướng dẫn yêu cầu mở rộng review sang (1) thiết kế ngược vật liệu nói chung, (2) xếp hạng đa tính chất, (3) tính chất phụ thuộc vật liệu nền, sau đó thêm (4) tính chất động lực học (nổ/va đập) ngoài miền tĩnh học hiện tại.

## 0. Khái quát 97 bài báo (9 nhóm)

| Lớp | Nhóm | Số paper | Nội dung cốt lõi |
|---|---|---|---|
| 1 | Nhóm 1 | 12 | Cơ chế hình học sinh ra ν âm |
| 2 | Nhóm 2 | 12 | Thiết kế ngược vật liệu nói chung (đa domain) |
| 3 | Nhóm 3 | 11 | Nền tảng thuật toán topology optimization |
| 4 | Nhóm 4 | 12 | Kiến trúc học sâu sinh mẫu (VAE/GAN/diffusion) |
| 5 | Nhóm 5 | 9 | Homogenization, PBC, differentiable-physics |
| 6 | Nhóm 6 | 12 | Xếp hạng/tối ưu đa mục tiêu |
| 7 | Nhóm 7 | 9 | Vật liệu nền quyết định tính chất hiệu dụng |
| 8 | Nhóm 8 | 9 | Manufacturability, in 3D, sai số as-built |
| 9 | Nhóm 9 | 11 | Đáp ứng động lực học / nổ / va đập |
| **Tổng** | | **97** | |

### Chi tiết từng nhóm (tóm tắt)

- **Nhóm 1 - Cơ chế hình học (12).** Từ foam ν âm thực nghiệm đầu tiên → cellular solids → cơ chế cụ thể: chiral honeycomb (ν gần -1 dải biến dạng rộng), rotating-square, buckling từ lỗ tròn (liên quan trực tiếp seed `circle`/`grid_circular_voids` của project), hierarchical honeycomb, 3D/origami/kirigami, ν âm through-thickness cho composite laminate chống va đập. Một khung phân loại tổng hợp 4 cơ chế chính (origami-kirigami, chiral, rotating-unit, re-entrant) - 11 seed hiện tại chỉ thuộc 2/4 nhóm đã biết.
- **Nhóm 2 - Thiết kế ngược vật liệu nói chung (12).** Cùng khung SIMP/homogenization đã mở rộng sang CTE cực trị, photonics, âm học, compliant mechanism, TPMS đa tính chất bằng property-VAE. Thông điệp: khung SIMP + generative model là paradigm cross-domain, auxetic chỉ một nhánh nhỏ.
- **Nhóm 3 - Nền tảng thuật toán TO (11).** Dòng lịch sử trực tiếp của code: code giáo dục tối giản → vector hóa ~100× → energy-based homogenization + PBC (anchor `simp/homogenization/compute.py`) → MMA xử lý ràng buộc phi tuyến đồng thời (anchor `mma_runner.py`) → Heaviside projection + continuation (anchor `beta_proj_t` trong `runner.py`). Đối chứng: ESO/BESO, level-set, Moving Morphable Components, 3-pha vật liệu, density field bằng NN; một review xác nhận tổng quát hóa ngoài phân phối là thách thức chung của cả lĩnh vực.
- **Nhóm 4 - Kiến trúc sinh mẫu (12).** Toán VAE → conditional VAE (kiến trúc gốc project kế thừa) → biến thể cho metamaterial (cVAE + kiểm chứng FE độc lập, graph model cho lattice truss, video diffusion cho đáp ứng phi tuyến, GAN theo nhóm đối xứng tinh thể). Đáng chú ý: 1 nghiên cứu mô tả đúng "surrogate exploitation" mà project phát hiện qua thực nghiệm; 1 thư viện FE khả vi GPU là ví dụ hiện đại của giải pháp project chọn (differentiable FE thay surrogate).
- **Nhóm 5 - Homogenization/PBC/differentiable-physics (9).** Paper khai sinh khái niệm homogenization trong TO → tài liệu hệ thống hóa energy-based homogenization → PBC cho unit cell bất kỳ → NN-surrogate thay FE và physics-informed loss → differentiable-physics hiện đại (FE khả vi GPU, learned simulator qua GNN) → cảnh báo hệ thống về surrogate bị "đánh lừa" trong vòng lặp offline → 1 CNN surrogate dự đoán Poisson trực tiếp từ ảnh mật độ, song song với kiến trúc CNN surrogate ban đầu của project trước khi chuyển sang real-physics.
- **Nhóm 6 - Xếp hạng/tối ưu đa mục tiêu (12).** Có khái niệm "composite score" (target + khả thi + đa dạng) gần nhất với composite score hiện tại (0,6 accuracy + 0,3 manuf + 0,1 aesthetic). 4 nghiên cứu dùng thuật toán tiến hóa đa mục tiêu (Pareto + MCDM) trực tiếp cho auxetic (độ bền/Poisson/khối lượng, hoặc trade-off Poisson âm-CTE âm, hoặc dị hướng đa mục tiêu). 1 nghiên cứu cân bằng manufacturability/compactness khớp gần 1-1 với `manuf_score`/`aesthetic_score`. Đáng chú ý nhất: 2 nghiên cứu dùng mô hình sinh có điều kiện (diffusion, cVAE) giải đa mục tiêu TRỰC TIẾP lúc sinh, không ranking hậu kỳ - kiến trúc gần nhất với hướng project.
- **Nhóm 7 - Vật liệu nền (9).** 1 nghiên cứu gần nhất với ablation project cần: cùng hình học, nhiều vật liệu nền kim loại → hình học quyết định DẤU, nhưng ĐỘ LỚN phụ thuộc vật liệu nền. 1 nghiên cứu mạnh hơn: giữ hình học, chỉ đổi tỉ lệ mô đun 2 vật liệu nền in đa vật liệu → Poisson hiệu dụng đổi từ âm lớn về gần 0 - chứng minh vật liệu nền là biến thiết kế độc lập với hình học. 2 nghiên cứu tối ưu hình học + phân bố vật liệu nền bằng level-set (chỉ đơn mục tiêu, không xếp hạng đa tiêu chí). 1 nghiên cứu: vật liệu nền phi tuyến (hyperelastic) đòi hỏi kiểm soát đồng thời Poisson-biến dạng lẫn ứng suất-biến dạng - cho thấy giả định "vật liệu nền tuyến tính đẳng hướng cố định" của project là đơn giản hóa mạnh. 1 review về functionally graded materials (đúng chuyên ngành GS hướng dẫn) + 2 công trình micromechanics kinh điển làm cận trên/dưới lý thuyết.
- **Nhóm 8 - Manufacturability/in 3D/sai số as-built (9).** Bộ lọc "in được" trong vòng lặp tối ưu (không cần support); kiểm soát min/max feature size - cơ sở định lượng cho tiêu chí min-feature-size hiện dùng theo kinh nghiệm. 1 nghiên cứu dùng graph + NN sửa liên thông; 1 nghiên cứu đưa ràng buộc liên thông vào loss khả vi - hướng nâng cấp so với `force_periodic()` hậu xử lý hiện tại. 1 pipeline đầy đủ density-field → STL. 2 nghiên cứu định lượng sai số as-built (khuyết tật đường nối in từng lớp; lệch hình học tích lũy dưới tải lặp) - cho thấy manufacturability check trên ảnh lý tưởng là điều kiện cần nhưng chưa đủ; 1 nghiên cứu đối trọng: sai lệch chủ yếu do đặc trưng vật liệu in, không hẳn hình học. 1 pipeline thực nghiệm đầy đủ thiết kế→in→đo cho auxetic.
- **Nhóm 9 - Động lực học/nổ/va đập (11, bổ sung theo yêu cầu mở rộng của GS).** Nhiều nghiên cứu tối ưu hình học auxetic cho specific energy absorption dưới nén/va đập - đại lượng khác hẳn Q-tensor tĩnh hiện dùng. Va đập tự nhiên là đa mục tiêu (nhiều vận tốc, không 1 điểm vận hành) - cầu nối trực tiếp với chủ đề ranking đa tính chất. 1 review toàn diện thiết kế/chế tạo/tối ưu auxetic cho energy absorption. Phân tích độ nhạy: tham số hình học ảnh hưởng khác nhau giữa đáp ứng tĩnh/động - không suy luận được trực tiếp. 5 nghiên cứu sandwich panel lõi auxetic chống nổ (độ cứng điều chỉnh, hình học cong 3D, tải xung dưới nước, foam đẳng hướng Poisson âm, khung 3-giai-đoạn biến dạng) - xác nhận "hệ số nổ" là đại lượng động lực học không suy ra được từ homogenization tuyến tính, phân nhánh theo môi trường tải. 1 trong số này nối trực tiếp gap "vật liệu nền" với miền động lực học.

**Thông điệp tổng thể:** project đứng ở giao điểm Nhóm 3+4+5 (TO/homogenization + cVAE + differentiable-physics); Nhóm 2/6/7 là 3 điểm GS yêu cầu mở rộng; Nhóm 9 là điểm mở rộng thứ 4 (miền động lực học, chưa triển khai trong code).

## 1. Phân tích vùng tối (gap analysis)

Khung 3 vòng tròn: **A = thiết kế ngược auxetic (generative/geometry)**, **B = xếp hạng đa mục tiêu**, **C = tính chất phụ thuộc vật liệu nền**.

- **A ∩ B - không trống, nhưng khác cách làm.** 4 nghiên cứu kết hợp auxetic + đa mục tiêu, nhưng đều dùng **TO/evolutionary + Pareto-search hậu kỳ**, không dùng generative sinh trực tiếp theo điều kiện. Kết luận: đóng góp của project là "cách tiếp cận khác" (generative conditioning + composite score ngay lúc sinh), không phải "chưa ai làm".
- **A ∩ C - không trống, khá dày.** 6 nghiên cứu kết hợp auxetic + vật liệu nền (so sánh đa vật liệu, đổi tỉ lệ mô đun, multi-material level-set, composite đa pha, vật liệu nền phi tuyến) - đều dùng **TO/level-set/thực nghiệm trực tiếp**, không có nghiên cứu nào dùng generative model điều kiện hóa theo vật liệu nền.
- **B ∩ C - rất mỏng.** Không tìm được nghiên cứu nào coi vật liệu nền là MỘT biến được xếp hạng/tối ưu đa mục tiêu đồng thời với tính chất hình học (2 nghiên cứu multi-material trong A∩C chỉ đơn mục tiêu).
- **A ∩ B ∩ C - trống thật.** Không có nghiên cứu nào trong 97 bài kết hợp đồng thời cả 3: (1) sinh vi cấu trúc auxetic bằng generative model có điều kiện, (2) điều kiện hóa/tối ưu đồng thời NHIỀU tiêu chí tính chất, (3) coi vật liệu nền là biến thiết kế/điều kiện chứ không phải hằng số. Nghiên cứu gần nhất về kiến trúc: 1 cVAE giải multi-objective nhưng cho phân tử (molecular graph), không phải microstructure cơ học. Nghiên cứu gần nhất về nội dung vật lý: các công trình đo độ nhạy Poisson theo vật liệu nền, nhưng dùng TO/thực nghiệm, không phải generative.

**Phát biểu vùng tối (dùng cho luận văn):**

> Chưa có công trình nào dùng mô hình sinh có điều kiện (conditional generative model) để thiết kế ngược vi cấu trúc auxetic vừa điều kiện hóa theo tính chất vật liệu nền, vừa xếp hạng đồng thời nhiều tiêu chí tính chất.

### Liên kết với roadmap kỹ thuật

Gap này khớp trực tiếp với roadmap kỹ thuật trong `docs/archive/PROJECT_PLAN.md`. **Cập nhật 2026-08-15:** phần B (xếp hạng đa tiêu chí) đã merge vào `main` (PR #13, 2026-07-25) - việc còn thiếu thật sự chỉ còn ở phần C (vật liệu nền):

1. **Vary ν0/E0 trong `pipeline/params.py::PARAM_SPACE`** (hiện cố định 0.3/199.0 trong `simp/materials/isotropic.py::Material.__init__`) - lấp A∩C bằng generative model, đóng góp trực tiếp cho luận điểm "tính chất phụ thuộc vật liệu nền" của GS. Đây là Nhóm 1/Giai đoạn A trong `docs/archive/PROJECT_PLAN.md`, effort lớn nhất, ưu tiên cao nhất.
2. **Composite ranking accuracy/manuf/aesthetic** (`best_of_n_eval.py`) + presence-mask/condition-dropout (`dataset.py`, `train.py`) đã là code production trên `main` - lấp A∩B theo hướng generative conditioning thay Pareto-search hậu kỳ, chỉ còn cần validate độc lập (Nhóm 3 trong `docs/archive/PROJECT_PLAN.md`).
3. Kết hợp cả 2 việc trên (điều kiện hóa theo cả hình học lẫn vật liệu nền, xếp hạng đa tiêu chí) chính là bước lấp A∩B∩C - khoảng trống thật sự duy nhất trong 97 bài.

### Ghi chú về Nhóm 9 (nổ/va đập)

Nằm ngoài khung A/B/C vì đòi hỏi solver vật lý khác hoàn toàn (FE động, phi tuyến hình học, có thể rate-dependency) - không tính được từ homogenization tuyến tính hiện tại. Là quyết định phạm vi (đề tài con riêng), không phải gap có thể lấp bằng sửa code hiện có. Nếu GS yêu cầu nghiêm túc, tách thành lộ trình riêng sau khi lấp xong A∩B∩C.

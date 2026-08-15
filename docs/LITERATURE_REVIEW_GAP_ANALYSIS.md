# Báo cáo: Khái quát tài liệu & Phân tích vùng tối

Nguồn: 97 bài báo, 9 nhóm chủ đề, tổng hợp qua literature review (Google Docs, 2026-08-14).
Bối cảnh: Giáo sư hướng dẫn yêu cầu mở rộng review sang (1) thiết kế ngược vật liệu nói chung,
(2) xếp hạng đa tính chất, (3) tính chất phụ thuộc vật liệu nền, và sau đó thêm (4) các tính chất
động lực học (nổ/va đập) ngoài miền tĩnh học hiện tại.

## 0. Khái quát 97 bài báo (9 nhóm)

### Bảng tổng quan

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

### Chi tiết từng nhóm

**Nhóm 1 - Cơ chế hình học auxetic (12 paper).**
Xuất phát từ bằng chứng thực nghiệm đầu tiên về foam có ν âm, mở rộng dần sang lý thuyết cellular
solids, sang thang phân tử, rồi đến các cơ chế hình học cụ thể: chiral honeycomb đạt ν ổn định gần
−1 trên dải biến dạng rộng, mô hình rotating-square, cơ chế buckling-induced auxeticity từ tấm lỗ
tròn đơn giản (liên quan trực tiếp seed "circle"/"grid_circular_voids" của project), cấu trúc
hierarchical honeycomb, mở rộng sang 3D/origami/kirigami khai thác buckling lớn, ν âm through-thickness
cho chống va đập composite laminate, và một khung phân loại tổng hợp 4 nhóm cơ chế chính (origami-
kirigami, chiral, rotating-unit, re-entrant) - cho thấy 11 seed hiện tại của project chủ yếu thuộc
2/4 nhóm cơ chế đã biết trong tài liệu.

**Nhóm 2 - Thiết kế ngược vật liệu nói chung (12 paper).**
Cùng khung SIMP/homogenization mà project dùng đã được mở rộng sang thiết kế CTE cực trị, sang
photonics (metasurface, waveguide), sang âm học (acoustic cloaking, metamaterial hấp thụ âm), sang
compliant mechanism, sang thiết kế đa tính chất/TPMS bằng property-VAE, và có các bài tổng quan định
vị auxetic chỉ là một lớp con trong họ "flexible mechanical metamaterials" rộng lớn hơn, hoặc theo
đuổi mục tiêu độ cứng cực trị thay vì Poisson ratio. Thông điệp chung: khung SIMP/homogenization +
generative model mà project dùng là một paradigm cross-domain, auxetic chỉ là một nhánh nhỏ.

**Nhóm 3 - Nền tảng thuật toán topology optimization (11 paper).**
Dòng lịch sử trực tiếp của code: từ code giáo dục tối giản, qua bản vector hóa tăng tốc ~100 lần, đến
phương pháp energy-based homogenization dùng periodic boundary conditions (anchor chính của
`simp/homogenization/compute.py`), phương pháp tối ưu "đường tiệm cận di động" xử lý nhiều ràng buộc
phi tuyến đồng thời (anchor của `mma_runner.py`), và kỹ thuật phép chiếu Heaviside có continuation để
đảm bảo hội tụ lưới cục bộ (anchor của `beta_proj_t` trong `runner.py`). Bổ sung đối chứng phương
pháp luận: các biến thể ESO/BESO, level-set, Moving Morphable Components, mở rộng ba-pha-vật-liệu,
tham số hóa trường mật độ bằng neural network, và một bài tổng quan định vị chính xác vị trí project
trong bản đồ ML-for-TO, đồng thời xác nhận thách thức tổng quát hóa ngoài phân phối huấn luyện là vấn
đề chung của cả lĩnh vực chứ không riêng project.

**Nhóm 4 - Kiến trúc học sâu sinh mẫu (12 paper).**
Từ nền tảng toán học của VAE, đến kiến trúc conditional VAE (kiến trúc gốc mà project kế thừa), rồi
các biến thể áp cho metamaterial: cVAE kết hợp kiểm chứng FE/thực nghiệm độc lập, tham số hóa hình
thái phi tuần hoàn kết hợp property-VAE, mô hình sinh dạng đồ thị cho lattice truss, video diffusion
model cho đáp ứng phi tuyến, VAE + property predictor cho metamaterial đa tỉ lệ, và GAN huấn luyện
theo nhóm đối xứng tinh thể học. Đặc biệt quan trọng: một nghiên cứu mô tả chính xác hiện tượng
"surrogate exploitation" mà project phát hiện qua thực nghiệm (optimizer trôi vào vùng surrogate
không còn tin cậy), và một thư viện FE khả vi trên GPU là ví dụ hiện đại của chính giải pháp project
chọn (differentiable FE solver thay vì surrogate). Có thêm các baseline generative khác (diffusion,
conditional GAN dựa hàm mặt ẩn) cho inverse design cellular structure không nhất thiết auxetic.

**Nhóm 5 - Homogenization, PBC, differentiable-physics (9 paper).**
Từ paper khai sinh khái niệm homogenization trong topology optimization, đến tài liệu "giáo khoa" hệ
thống hóa energy-based homogenization, kỹ thuật áp periodic boundary conditions cho unit cell hình
dạng bất kỳ, các khảo sát về NN-surrogate thay thế FE và về đưa phương trình vật lý vào hàm loss
(physics-informed neural network), các ví dụ differentiable-physics hiện đại (FE khả vi trên GPU,
learned simulator khả vi qua graph neural network), cảnh báo hệ thống về rủi ro surrogate bị "đánh
lừa" trong vòng lặp tối ưu offline, và một ví dụ CNN surrogate dự đoán trực tiếp hệ số Poisson từ
ảnh mật độ - song song trực tiếp với kiến trúc CNN surrogate ban đầu của project trước khi chuyển
sang differentiable real-physics.

**Nhóm 6 - Xếp hạng/tối ưu đa mục tiêu (12 paper).**
Có một khái niệm "điểm số tổng hợp" (aggregate performance score) kết hợp mức đạt target, tính khả
thi và tính đa dạng - gần nhất về ý tưởng với composite score hiện tại (0.6 accuracy + 0.3 manuf +
0.1 aesthetic) của project. Bốn nghiên cứu áp dụng thuật toán tiến hóa đa mục tiêu (sinh Pareto front
rồi chọn bằng phương pháp ra quyết định đa tiêu chí) trực tiếp lên thiết kế auxetic, đạt đồng thời độ
bền, Poisson âm và khối lượng thấp, hoặc phân tích rõ vùng trade-off giữa Poisson âm và giãn nở nhiệt
âm (hai tính chất xung đột nhau), hoặc kiểm soát tính dị hướng đồng thời với nhiều mục tiêu cơ học.
Có một nghiên cứu cân bằng manufacturability/compactness trong cùng khung tối ưu, khớp gần 1-1 với
`manuf_score`/`aesthetic_score` hiện tại. Có nghiên cứu kết hợp surrogate với thuật toán tiến hóa cho
thiết kế chịu va đập đa vận tốc, và nghiên cứu kết hợp deep learning với tối ưu đa mục tiêu cho cấu
trúc rỗng. Đáng chú ý nhất: hai nghiên cứu dùng mô hình sinh có điều kiện (diffusion, conditional VAE)
giải bài toán đa mục tiêu TRỰC TIẾP trong quá trình sinh - không cần ranking hậu kỳ - đây là kiến
trúc gần nhất với hướng đi của project. Có thêm một nghiên cứu dùng GAN đối chứng, và một tài liệu
nền tảng lý thuyết cho các phương pháp ra quyết định đa tiêu chí (MCDM/TOPSIS).

**Nhóm 7 - Vật liệu nền quyết định tính chất hiệu dụng (9 paper).**
Có một nghiên cứu gần nhất với thiết kế ablation project cần: giữ cùng hình học, thử trên nhiều vật
liệu nền kim loại khác nhau (mô đun đàn hồi chênh nhau khá lớn) - kết luận hình học là yếu tố chính
quyết định DẤU và dạng đáp ứng auxetic, nhưng ĐỘ LỚN của đáp ứng vẫn phụ thuộc rõ rệt vào vật liệu
nền. Có một nghiên cứu thực nghiệm mạnh hơn nữa: GIỮ NGUYÊN hình học, chỉ đổi tỉ lệ mô đun đàn hồi
giữa hai vật liệu nền trong cùng cấu trúc in đa vật liệu - hệ số Poisson hiệu dụng thay đổi từ giá
trị âm khá lớn lên gần bằng không, chứng minh vật liệu nền có thể là biến thiết kế độc lập với hình
học. Có nghiên cứu về composite đa pha trong đó tính chất từng pha thành phần quyết định trực tiếp
tính chất hiệu dụng composite. Có hai nghiên cứu tối ưu hóa hình học đồng thời với phân bố nhiều vật
liệu nền khác nhau trong cùng unit cell (dùng phương pháp level-set), nhưng chỉ hướng tới một mục
tiêu duy nhất (đạt hệ số Poisson âm), không phải xếp hạng đa tiêu chí. Có một nghiên cứu chỉ ra rằng
khi vật liệu nền không tuyến tính (hyperelastic - sát với polymer in 3D thực tế), phải kiểm soát đồng
thời cả đường cong Poisson-biến dạng lẫn ứng xử ứng suất-biến dạng phi tuyến, cho thấy giả định "vật
liệu nền tuyến tính đẳng hướng cố định" của project là một đơn giản hóa mạnh. Có một bài tổng quan hệ
thống về vật liệu phân bố chức năng (functionally graded materials) - đúng chuyên ngành của GS hướng
dẫn - cung cấp công thức chuẩn ước lượng tính chất hiệu dụng từ tính chất pha thành phần. Cuối cùng,
hai công trình kinh điển về cơ học vi mô (micromechanics/effective-medium) cung cấp cơ sở lý thuyết
và cận trên/cận dưới cho tính chất hiệu dụng của vật liệu đa pha - có thể dùng làm sanity-check lý
thuyết độc lập khi vật liệu nền thay đổi.

**Nhóm 8 - Manufacturability, in 3D, sai số as-built (9 paper).**
Có các nghiên cứu về bộ lọc đảm bảo thiết kế "in được" trực tiếp trong vòng lặp tối ưu (không cần
support structure), và về kiểm soát đồng thời kích thước tối thiểu/tối đa vật liệu đặc, kích thước
lỗ rỗng tối thiểu, khoảng cách tối thiểu giữa các thành phần - cơ sở định lượng cho tiêu chí
min-feature-size hiện dùng theo kinh nghiệm trong project. Có nghiên cứu dùng đồ thị liên thông kết
hợp mạng neural để sửa/ràng buộc tính liên thông của thiết kế sinh ra bởi mô hình generative, và một
nghiên cứu khác đưa ràng buộc liên thông vào ngay trong hàm loss huấn luyện khả vi - hướng nâng cấp
so với cách project hiện dùng `force_periodic()` như bước hậu xử lý. Có một pipeline đầy đủ từ trường
mật độ sang file STL sẵn sàng in 3D, mô tả chính xác bước còn thiếu nếu project triển khai xuất
CAD/in 3D. Hai nghiên cứu định lượng sai số as-built vs as-designed: đường nối khi in từng lớp tạo
khuyết tật làm giảm tiết diện cục bộ và độ bền đáng kể, và lệch hình học tích lũy tương tác phi tuyến
với vật liệu dưới tải lặp - cho thấy manufacturability check hiện tại trên ảnh lý tưởng là điều kiện
cần nhưng chưa đủ. Một nghiên cứu khác đưa góc nhìn đối trọng: sai lệch dự đoán-thực tế chủ yếu do
đặc trưng vật liệu in không được mô hình hóa chính xác, không hẳn do lỗi hình học. Có một pipeline
thực nghiệm hoàn chỉnh thiết kế số → in 3D → đo thực cho auxetic metamaterial.

**Nhóm 9 - Đáp ứng động lực học / nổ / va đập (11 paper, bổ sung theo yêu cầu mở rộng của GS).**
Nhiều nghiên cứu tối ưu hình học auxetic để tối đa hóa khả năng hấp thụ năng lượng riêng (specific
energy absorption) dưới tải nén/va đập - một đại lượng khác hẳn về bản chất so với Q-tensor tĩnh mà
project hiện dùng. Có nghiên cứu chỉ rõ bài toán va đập tự nhiên là đa mục tiêu (hấp thụ năng lượng
tại nhiều vận tốc khác nhau, không chỉ một điểm vận hành) - cầu nối trực tiếp với chủ đề ranking đa
tính chất. Có một bài tổng quan toàn diện thiết kế/chế tạo/tối ưu auxetic cho ứng dụng hấp thụ năng
lượng, hữu ích làm điểm neo nếu mở rộng sang miền này. Có nghiên cứu phân tích độ nhạy cho thấy tham
số hình học ảnh hưởng khác nhau đáng kể giữa đáp ứng tĩnh và đáp ứng động - không thể suy luận trực
tiếp từ kết quả tĩnh sang động. Năm nghiên cứu về khả năng chống nổ của sandwich panel lõi auxetic
(độ cứng điều chỉnh được, hình học cong 3D, tải xung kích dưới nước, lõi foam đẳng hướng có Poisson
hiệu dụng âm, khung phân tích ba giai đoạn biến dạng) - tất cả xác nhận "hệ số nổ" là một đại lượng
động lực học (chuyển vị đỉnh, năng lượng hấp thụ hệ thống, biến dạng theo pha), không suy ra được từ
homogenization tuyến tính hiện tại, và còn phân nhánh theo môi trường/kịch bản tải (nổ trong không
khí khác nổ dưới nước). Đáng chú ý: một trong các nghiên cứu về lõi foam nối trực tiếp gap "vật liệu
nền" với miền động lực học trong cùng một bài toán - vật liệu nền cụ thể ảnh hưởng đến hiệu suất
chống nổ của cấu trúc có Poisson hiệu dụng âm.

**Thông điệp tổng thể:** project đứng ở giao điểm của Nhóm 3+4+5 (công cụ TO/homogenization + kiến
trúc cVAE + differentiable-physics), trong khi Nhóm 2, 6, 7 chính là 3 điểm thầy yêu cầu mở rộng
review, và Nhóm 9 là điểm mở rộng thứ 4 (miền vật lý động lực học, chưa triển khai trong code).

## 1. Phân tích vùng tối (gap analysis)

Dùng khung 3 vòng tròn: **A = thiết kế ngược auxetic (generative/geometry)**, **B = xếp hạng đa mục
tiêu**, **C = tính chất phụ thuộc vật liệu nền**.

### A ∩ B - không trống, nhưng khác cách làm

Bốn nghiên cứu đã kết hợp thiết kế auxetic với tối ưu đa mục tiêu, đạt đồng thời độ bền/khối lượng/
Poisson âm, hoặc phân tích trade-off Poisson âm - giãn nở nhiệt âm, hoặc kiểm soát tính dị hướng đa
mục tiêu, hoặc tối ưu toàn bộ tensor độ cứng có kiểm chứng thực nghiệm. Điểm chung: tất cả đều dùng
**topology optimization cổ điển hoặc thuật toán tiến hóa, sinh tập nghiệm Pareto rồi chọn hậu kỳ**,
không dùng mô hình generative sinh trực tiếp theo điều kiện như project.

Kết luận: "auxetic + đa mục tiêu" đã có người làm, nhưng đi theo hướng **TO/evolutionary +
Pareto-search hậu kỳ**. Project đi hướng **generative conditioning + composite score ngay trong lúc
sinh** - khác biệt về phương pháp luận, không phải khoảng trống tuyệt đối. Cần diễn đạt đúng trong
luận văn: đóng góp là "cách tiếp cận khác" chứ không phải "chưa ai làm".

### A ∩ C - không trống, khá dày

Sáu nghiên cứu kết hợp thiết kế auxetic với vật liệu nền: so sánh cùng hình học trên nhiều vật liệu
nền để đo độ nhạy, đổi tỉ lệ mô đun giữa hai vật liệu nền trong cùng kiến trúc, tối ưu multi-material
bằng level-set (hai biến thể, một cho phép lỗ rỗng, một không), khảo sát composite đa pha, và mô hình
hóa khi vật liệu nền phi tuyến (hyperelastic). Tất cả đều dùng **topology optimization/level-set/thực
nghiệm trực tiếp**, không có nghiên cứu nào dùng mô hình generative điều kiện hóa theo vật liệu nền.

Kết luận: 6 nghiên cứu làm auxetic + vật liệu nền, nhưng **không có nghiên cứu nào dùng generative
model điều kiện hóa theo vật liệu nền** cho auxetic.

### B ∩ C - rất mỏng

Rà toàn bộ 97 bài, không tìm được nghiên cứu nào coi vật liệu nền (mô đun đàn hồi, hệ số Poisson của
vật liệu gốc) là MỘT trong các biến được xếp hạng/tối ưu đa mục tiêu đồng thời với tính chất hình
học. Hai nghiên cứu multi-material trong nhóm A∩C chỉ tối ưu một mục tiêu duy nhất (đạt Poisson âm
mục tiêu), không phải xếp hạng đa tiêu chí.

### A ∩ B ∩ C - trống thật

Không có nghiên cứu nào trong 97 bài kết hợp cả 3 điều kiện đồng thời:
1. Sinh vi cấu trúc auxetic bằng mô hình generative có điều kiện (cVAE/GAN/diffusion)
2. Điều kiện hóa/tối ưu đồng thời theo NHIỀU tiêu chí tính chất (không chỉ hệ số Poisson)
3. Coi vật liệu nền là một biến thiết kế/điều kiện, không phải hằng số cố định

Nghiên cứu gần nhất về kiến trúc là một conditional VAE giải multi-objective inverse design - nhưng
áp dụng cho phân tử (molecular graph), không phải microstructure cơ học. Nghiên cứu gần nhất về nội
dung vật lý là các công trình đo độ nhạy Poisson theo vật liệu nền - nhưng dùng TO/thực nghiệm, không
phải generative model.

**Phát biểu vùng tối (dùng cho luận văn):**

> Chưa có công trình nào dùng mô hình sinh có điều kiện (conditional generative model) để thiết kế
> ngược vi cấu trúc auxetic vừa điều kiện hóa theo tính chất vật liệu nền, vừa xếp hạng đồng thời
> nhiều tiêu chí tính chất.

### Liên kết với roadmap kỹ thuật

Gap này khớp trực tiếp với roadmap kỹ thuật trong `docs/PROJECT_PLAN.md` - nghĩa là gap analysis và
roadmap kỹ thuật đang chỉ về cùng một chỗ, không phải hai việc tách rời. **Cập nhật 2026-08-15:**
phần B (xếp hạng đa tiêu chí) đã merge vào `main` (PR #13, 2026-07-25) - việc còn thiếu thật sự chỉ
còn nằm ở phần C (vật liệu nền):

1. **Vary ν0/E0 trong `pipeline/params.py::PARAM_SPACE`** (hiện cố định 0.3/199.0 trong
   `simp/materials/isotropic.py::Material.__init__`) - lấp A∩C bằng generative model, đóng góp trực
   tiếp cho luận điểm "tính chất phụ thuộc vật liệu nền" của GS. Đây là Nhóm 1/Giai đoạn A trong
   `docs/PROJECT_PLAN.md`, effort lớn nhất, ưu tiên cao nhất.
2. **Composite ranking accuracy/manuf/aesthetic** (`best_of_n_eval.py`) và cơ chế presence-mask/
   condition-dropout cho điều kiện tùy chọn (`dataset.py`, `train.py`) đã là code production trên
   `main`, không còn là việc tồn đọng - lấp A∩B theo hướng generative conditioning thay vì
   Pareto-search hậu kỳ, chỉ còn cần validate độc lập (Nhóm 3 trong `docs/PROJECT_PLAN.md`).
3. Kết hợp cả 2 việc trên cùng lúc (điều kiện hóa theo cả tính chất hình học lẫn vật liệu nền, xếp
   hạng đa tiêu chí) chính là bước lấp đầy A∩B∩C - khoảng trống thật sự duy nhất trong toàn bộ 97 bài.

### Ghi chú về mở rộng sang Nhóm 9 (nổ/va đập)

Nhóm 9 nằm ngoài khung 3 vòng tròn A/B/C vì đòi hỏi solver vật lý khác hoàn toàn (FE động, phi tuyến
hình học, có thể rate-dependency) - không tính được từ homogenization tuyến tính hiện tại. Đây là
quyết định phạm vi (một đề tài con riêng) chứ không phải một gap có thể lấp bằng cách sửa code hiện
có. Nếu GS yêu cầu nghiêm túc, nên tách thành lộ trình riêng sau khi đã lấp xong A∩B∩C.

# Kịch bản báo cáo với GS.TS Nguyễn Đình Đức

> Thời lượng nói: khoảng 10 phút, sau đó hỏi đáp. Số liệu lấy từ `docs/plan.md` (cập nhật 2026-10-08),
> dùng pipeline không `force_periodic`.
> Mục tiêu buổi gặp: báo cáo tiến độ kỹ thuật và **xin thầy định hướng**. Chưa phải buổi trình bản
> thảo để nộp.

---

## 1. Mở đầu (30 giây)

> "Em chào thầy. Hôm nay em xin báo cáo tiến độ đề tài thiết kế ngược vật liệu auxetic. Em sẽ trình
> bày ba phần: em đã làm được gì, kết quả đang đứng ở đâu so với phương pháp SIMP truyền thống (kể cả
> những chỗ phương pháp của em còn thua), và mấy điểm em cần thầy góp ý về hướng đi tiếp."

---

## 2. Bài toán (1 phút)

> "Bài toán là: cho trước hệ số Poisson mong muốn, ví dụ ν12 = −0,8, ν21 = −0,3, tìm hình dạng ô
> cơ sở tuần hoàn 2D đạt đúng các giá trị đó. Cách truyền thống là chạy SIMP tối ưu hóa topo cho
> từng mục tiêu. Mỗi mục tiêu tốn vài trăm đến gần một nghìn lần giải FE, và kết quả phụ thuộc
> nhiều vào điểm khởi tạo.
>
> Hướng của em là học một mô hình sinh, cụ thể là conditional VAE, từ khoảng 13 nghìn thiết kế
> SIMP. Cho mục tiêu ν vào, mô hình sinh ra ngay một hình dạng, sau đó em tinh chỉnh bằng gradient
> từ bài toán đồng nhất hóa (homogenization) FE khả vi. Mọi kết quả đều được kiểm lại bằng FE độc
> lập, em không dùng số đo qua mô hình thay thế (surrogate)."

---

## 3. Phát hiện chính (2 phút)

> "Phát hiện quan trọng nhất là: **hàm mục tiêu phải tối ưu đúng thiết kế sẽ được kiểm chứng.**
>
> Lúc đầu em tinh chỉnh trên ảnh mật độ liên tục, tức là có vật liệu xám. Hàm mất mát gần như về
> 0, nhưng khi nhị phân hóa ảnh ra thiết kế thật rồi giải FE lại thì sai số lớn. Bộ tối ưu đã lợi
> dụng vật liệu xám, tức là thiết kế đẹp trên giấy nhưng không tồn tại thật.
>
> Em sửa bằng cách đưa phép chiếu Heaviside, với β tăng dần đến 64, vào ngay trong hàm mục tiêu,
> để bộ tối ưu nhìn thấy đúng thiết kế nhị phân sẽ được kiểm. Kết quả trên 100 mục tiêu:
> - R² của ν12: **0,874 → 0,998**
> - Sai số tuyệt đối trung bình giảm **khoảng 92%**, có khoảng tin cậy bootstrap 95%
> - Kết quả lặp lại được trên hai tập mục tiêu mới với seed khác."

---

## 4. So với SIMP: nói thẳng cả thắng lẫn thua (3 phút)

> "Em chạy SIMP từ đầu làm đối chứng (MMA, Heaviside, 4 lần khởi tạo), dùng cùng đường kiểm chứng
> FE với mô hình của em."

**Phần mô hình của em làm được:**
> "- Độ chính xác **ngang SIMP đã hội tụ**, nhưng chỉ tốn khoảng **107 lần giải FE mỗi mục tiêu,
>   so với gần 900 của SIMP**, tức ít hơn khoảng 8 lần.
> - Với mục tiêu dị hướng vừa phải (tỉ số ν21/ν12 dưới 10), sai số cặp (ν12, ν21) thấp hơn SIMP
>   khoảng **27%**. Tiêu chí này em ghi trước khi chạy, trên một tập mục tiêu mới."

**Phần SIMP vẫn thắng (cần nói rõ):**
> "- **Khả năng chế tạo:** SIMP cho khoảng 72–84% thiết kế liên thông và đủ bề dày nét, mô hình
>   của em chỉ khoảng 31%. Nhiều thiết kế của em có chỗ nối qua một điểm góc pixel, tức là bản lề
>   một nút, không in được.
> - **Ngoài dải dữ liệu:** với mục tiêu đổi dấu (ν dương) hoặc ν12 âm hơn mức nhỏ nhất trong dữ
>   liệu huấn luyện (khoảng −1,95), mô hình bão hòa, còn SIMP từ đầu tốt hơn rõ.
> - **Dị hướng cực đoan** (tỉ số ≥ 10): mô hình sinh ra ô bị đứt, vì dữ liệu huấn luyện chỉ có
>   khoảng 3% mẫu loại này.
> - **Chi phí tổng:** sinh dữ liệu tốn khoảng 3 triệu lần giải FE, nên chỉ có lợi khi cần thiết kế
>   từ khoảng 3.500 mục tiêu trở lên. Em không claim lợi về chi phí cho một mục tiêu đơn lẻ."

**Thử khắc phục khả năng chế tạo:**
> "Em đã thêm phạt điểm góc và nét mảnh vào hàm mục tiêu. Tỉ lệ chế tạo được tăng từ 31% lên
> 72–76%, ngang SIMP, nhưng sai số ν12 tăng khoảng 40% và lợi thế độ chính xác so với SIMP mất ý
> nghĩa thống kê. Đây là một đánh đổi thật. Hiện em đang để nó làm điểm vận hành thứ hai, pipeline
> chính vẫn ưu tiên độ chính xác."

---

## 5. Giới hạn em tự thấy (1 phút)

> "Em cũng xin báo các giới hạn kỹ thuật em thấy bài chưa đủ chặt:
> - Lưới FE là 50×50. Lên 200×200 thì ν lệch khoảng 0,015, lớn hơn sai số em báo. Thứ hạng giữa
>   các phương pháp vẫn giữ, nhưng con số tuyệt đối chỉ đúng tương đối với mô hình 50×50.
> - Mới chỉ đàn hồi tuyến tính, biến dạng nhỏ, một vật liệu nền. Chưa kiểm bằng FE thương mại
>   (Abaqus/ANSYS) và chưa có thực nghiệm in 3D.
> - Em thấy về mặt kỹ thuật bài chưa sẵn sàng nộp, nên em muốn xin ý kiến thầy trước khi đi tiếp."

---

## 6. Xin thầy định hướng (2 phút): phần quan trọng nhất

Hỏi theo thứ tự ưu tiên. Ghi lại câu trả lời.

1. **Góc composite/FGM.**
   > "Theo thầy, nên nối kết quả này với hướng composite hay FGM thế nào cho có ý nghĩa cơ học? Ví
   > dụ: dùng ô auxetic làm lõi tấm sandwich, hay thiết kế ô có cơ tính biến thiên theo chiều dày?"

2. **Kiểm chứng.**
   > "Theo thầy, bước kiểm chứng nào cần nhất để bài đứng được: FE thương mại, biến dạng lớn, hay in
   > 3D đo ν thật? Bên mình có phòng thí nghiệm hoặc thiết bị nào em có thể dùng không ạ?"

3. **Đánh đổi chính xác và chế tạo.**
   > "Với người làm cơ học vật liệu, thầy thấy nên đặt pipeline chính ưu tiên độ chính xác ν, hay
   > ưu tiên chế tạo được?"

4. **Hướng tiếp theo** (chọn 1, không làm song song):
   > "Sau bài này em có mấy hướng: (a) thêm môđun trượt G_xy làm mục tiêu thứ hai, gần như có sẵn
   > gradient; (b) vật liệu composite nhiều pha; (c) bài toán phi tuyến, biến dạng lớn. Thầy thấy
   > hướng nào phù hợp với nhóm nhất?"

5. **(Chỉ hỏi nếu còn thời gian)** Tạp chí và đồng tác giả.
   > "Em đang nhắm Structural and Multidisciplinary Optimization. Thầy có gợi ý tạp chí nào khác
   > không ạ?"

---

## 7. Kết (15 giây)

> "Em cảm ơn thầy. Em sẽ tổng hợp góp ý hôm nay thành kế hoạch cụ thể và gửi lại thầy trong tuần."

---

## Phụ lục: câu hỏi thầy có thể hỏi và gợi ý trả lời

| Câu hỏi có thể gặp | Gợi ý trả lời |
|---|---|
| "Sao không dùng SIMP luôn cho xong?" | Với một mục tiêu thì SIMP tốt hơn hoặc ngang. Mô hình có lợi khi cần thiết kế rất nhiều mục tiêu, ví dụ ô biến thiên dọc một tấm FGM (hàng nghìn ô khác nhau), vì mỗi ô chỉ tốn ~107 FE. |
| "ν12 = −1,95 có thật không, có vi phạm ràng buộc vật lý?" | Vật liệu trực hướng 2D chỉ cần ν12·ν21 < 1, nên ν12 rất âm vẫn hợp lệ khi đi với ν21 nhỏ. Mục tiêu đối xứng ν12 = ν21 ≤ −1 không tồn tại; em đã loại khỏi tập thử. |
| "Vật liệu nền là gì, E và ν0 bao nhiêu?" | Một vật liệu đẳng hướng, ν0 = 0,3, mật độ nhị phân (đặc/rỗng). Chưa xét nhiều pha. |
| "Kết quả có phụ thuộc lưới không?" | Có: 50² và 200² lệch ~0,015 về ν. Thứ hạng cVAE và SIMP giữ nguyên trên 200² (R²(ν12) 0,990 so với 0,985). |
| "Đã có so sánh với nghiên cứu khác chưa?" | So với SIMP từ đầu, với tra cứu dữ liệu (retrieval) đã kiểm bằng FE, và với các biến thể kiến trúc (KAN, WIRE). KAN không hơn Linear, WIRE thất bại; em báo cả kết quả âm. |
| "Làm sao biết không chọn số đẹp?" | Các tiêu chí đạt/không đạt được ghi trước khi chạy, có khoảng tin cậy bootstrap 95%, kiểm lại trên tập seed mới. Có thí nghiệm không đạt (lai cVAE→SIMP) và em vẫn báo. |
| "Bao giờ xong bài?" | Phụ thuộc bước kiểm chứng thầy chọn ở mục 6. Phần viết đã có bản nháp EN + VI. |

**Lưu ý khi nói:**
- Nói con số làm tròn, không đọc dải CI trừ khi thầy hỏi.
- Thầy là chuyên gia cơ học composite, không phải ML: giải thích cVAE bằng một câu ("học từ 13
  nghìn thiết kế SIMP để sinh hình dạng tức thì"), dành thời gian cho phần cơ học.
- Không nói "tốt hơn SIMP" chung chung. Luôn kèm điều kiện: "ngang độ chính xác, ít FE hơn 8 lần".
- Mang theo: hình `fig_designs` (cVAE vs SIMP), `fig_tiling`, `fig_deformation` trong
  `docs/paper1/figures/`.

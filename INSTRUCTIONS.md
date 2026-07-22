# Hướng dẫn chạy gamma sweep (Phase 5 — Conditional VAE)

## Bối cảnh ngắn gọn

Phase 5 (cVAE inverse design) dùng một hệ số `gamma` để quyết định trọng số của
property-consistency loss (dự đoán ν₁₂/ν₂₁ bằng surrogate model đông cứng —
frozen — từ Phase 4) trong tổng loss:

```
total_loss = recon_loss + beta * kl_loss + gamma * PROP_LOSS_SCALE * prop_loss
```

Đã chạy 3 điểm (`gamma = 1, 5, 20`), kết quả trong `outputs/phase5/eval_gamma{1,5,20}.json`:

| gamma | R² (ν₁₂) | MAE (ν₁₂) | pixel_std (đa dạng ảnh sinh) |
|---|---|---|---|
| 1  | −0.418 | 0.174 | 0.326 |
| 5  |  0.450 | 0.106 | 0.274 |
| 20 |  0.633 | 0.086 | 0.314 |

R² vẫn đang tăng theo gamma (**chưa plateau**), trong khi độ đa dạng ảnh sinh ra
(`pixel_std`) biến thiên không đơn điệu (giảm ở gamma=5 rồi tăng lại ở gamma=20).
Cần thêm điểm dữ liệu ở `gamma = 30, 50, 80, 100` để:
1. Xác nhận R² có thực sự tiếp tục tăng hay bắt đầu bão hòa/giảm ở vùng gamma lớn hơn nhiều.
2. Hiểu rõ hơn đường cong đánh đổi giữa độ chính xác property và độ đa dạng hình học sinh ra.
3. **Phát hiện sớm nếu gamma quá lớn khiến property loss lấn át reconstruction**
   (ảnh sinh ra có thể mất cấu trúc vật lý hợp lý, hoặc mode collapse — đa dạng
   giảm mạnh) — cần theo dõi không chỉ R² mà cả `recon` loss và ảnh diagnostics.

## Việc cần làm

Chạy 4 lần training (`gamma=30`, `gamma=50`, `gamma=80`, `gamma=100`), mỗi lần
**50 epoch**, rồi evaluate và gửi lại kết quả.

## Yêu cầu môi trường

- Python ≥ 3.10
- PyTorch (bản CUDA nếu máy ảo có GPU — khuyến khích; script tự nhận CPU/GPU,
  không cần chỉnh gì)
- Các thư viện khác: `numpy`, `scipy`, `pandas`, `scikit-learn`, `pillow`

```bash
pip install torch numpy scipy pandas scikit-learn pillow
```

(Nếu máy ảo có GPU, cài đúng bản `torch` tương ứng CUDA của máy — xem
https://pytorch.org/get-started/locally/ để lấy đúng lệnh `pip install`.)

## Các bước chạy

1. **Giải nén** file `simp-micro.tar.gz` gửi kèm:
   ```bash
   tar -xzvf simp-micro.tar.gz
   cd simp-micro
   ```

2. **Kiểm tra dữ liệu đã có sẵn** (đã đóng gói kèm trong tar.gz, không cần tải gì thêm):
   ```bash
   ls outputs/phase3/*.npz
   ls outputs/phase4/surrogate_for_phase5.pt
   ```
   Nếu 2 lệnh trên báo lỗi "No such file" — dữ liệu bị thiếu khi giải nén, xin
   liên hệ lại em, đừng chạy tiếp.

3. **Chạy sweep** (1 lệnh duy nhất, tự động cả 3 gamma):
   ```bash
   chmod +x run_gamma_sweep.sh
   ./run_gamma_sweep.sh
   ```

   Script sẽ tự động:
   - Kiểm tra dữ liệu/checkpoint cần thiết trước khi chạy
   - Train tuần tự `gamma=30 → 50 → 80 → 100`, mỗi lần 50 epoch
   - Evaluate ngay sau mỗi lần train
   - **Tự backup** kết quả từng gamma vào
     `outputs/phase5/gamma_sweep_results/` (vì code gốc ghi đè
     `cvae_best.pt`/`evaluation_report.json` mỗi lần chạy — script đã xử lý
     việc này, thầy không cần làm gì thêm)
   - In ra bảng tóm tắt R² theo gamma ở cuối

## Thời gian dự kiến

Dựa trên log epoch của lần chạy `gamma=20` (train: 33,120 mẫu, batch=64):
mỗi epoch trên GPU tầm trung thường vài giây đến ~1 phút; 50 epoch × 4 gamma
ước tính **tổng cộng khoảng 1.5–4 giờ trên GPU** (chênh lệch nhiều tùy loại
GPU của máy ảo). Trên CPU sẽ chậm hơn đáng kể (có thể nhiều giờ/gamma) — nếu
máy ảo không có GPU, xin báo lại em trước khi chạy full, có thể cần giảm bớt
phạm vi (ví dụ chỉ chạy gamma=50 và gamma=100 trước để có 2 điểm nhanh).

## Kết quả cần gửi lại

Chỉ cần gửi lại thư mục:
```
outputs/phase5/gamma_sweep_results/
```

Gồm 12 file (4 gamma × 3 loại file):
- `eval_gamma30.json`, `eval_gamma50.json`, `eval_gamma80.json`, `eval_gamma100.json`
- `train_history_gamma30.json`, `train_history_gamma50.json`, `train_history_gamma80.json`, `train_history_gamma100.json`
- `cvae_best_gamma30.pt`, `cvae_best_gamma50.pt`, `cvae_best_gamma80.pt`, `cvae_best_gamma100.pt`
  *(4 file .pt là checkpoint model, khá nặng — nếu dung lượng gửi bị giới hạn,
  chỉ cần 8 file .json là đủ để em phân tích, không bắt buộc gửi .pt)*

Ngoài ra, nếu gamma=80 hoặc gamma=100 cho kết quả bất thường (R² giảm đột
ngột, hoặc ảnh sinh ra trông bất hợp lý), xin gửi thêm luôn thư mục
`outputs/phase5/diagnostics/` của lần chạy đó để em kiểm tra trực quan.

Có thể nén lại:
```bash
tar -czvf gamma_sweep_results.tar.gz outputs/phase5/gamma_sweep_results/
```

## Nếu gặp lỗi

- **`CUDA out of memory`**: giảm batch size, ví dụ sửa script gọi
  `--batch-size 32` thay vì mặc định 64 (sửa trực tiếp trong
  `run_gamma_sweep.sh`, dòng gọi `train.py`).
- **`ModuleNotFoundError`**: thiếu thư viện, cài theo mục "Yêu cầu môi trường" ở trên.
- **Script dừng giữa chừng do mất kết nối máy ảo**: kết quả của các gamma đã
  chạy xong vẫn được lưu an toàn trong `gamma_sweep_results/` (script backup
  ngay sau mỗi gamma, không đợi chạy hết cả 4 mới lưu) — có thể chạy lại chỉ
  phần còn thiếu bằng cách sửa dòng `GAMMAS=(30 50 80 100)` trong script thành
  các gamma chưa chạy.

# `pipeline/phase3_dataset/` — lưu ý quan trọng trước khi dùng

**Dataset production hiện tại (`outputs/phase3/{train,val,test}.npz`, dùng để train
`surrogate_v2.pt`/`cvae_v2_finetuned.pt`) KHÔNG được tạo bởi 4 script trong thư mục
này.** Nó được tạo bởi 2 script nằm ở `analysis/scripts/`:

1. `analysis/scripts/generate_production_batch.py` — sinh mẫu thô song song
   (độc lập với `pipeline/phase2_multi_batch/`), lọc `is_connected & osc_score<0.15
   & v12<0 & not degenerate` ngay tại thời điểm sinh.
2. `analysis/scripts/assemble_phase3_v2.py` — gộp mẫu thô mới với pool sạch cũ,
   chia train/val/test 70/15/15 phân tầng theo seed, tăng cường đối xứng, ghi
   `outputs/phase3/{train,val,test,dataset_64}.npz` + `outputs/phase3_v2/split_report.json`.

Chi tiết đầy đủ: `EXPERIMENT_LOG.md` (mục "2026-07-25 - Rebuild dataset v2").

## 4 script trong thư mục này dùng để làm gì?

`scan_dataset.py` → `build_npz.py` → `finalize_dataset.py` (+ `augment_symmetry.py`
được `finalize_dataset.py` gọi) là pipeline **rebuild từ đầu** trực tiếp từ kết quả
thô của `pipeline/phase2_multi_batch/` (thư mục `outputs/multi_batch/batch_*/`).
Dùng khi cần dựng lại dataset từ batch DOE gốc (không đi qua bước sinh song song
riêng của `generate_production_batch.py`) — ví dụ khi muốn kiểm chứng độc lập,
hoặc khi `analysis/scripts/` chưa tồn tại/chưa phù hợp.

**Khác biệt lọc chất lượng quan trọng:** `finalize_dataset.py` hiện chỉ lọc theo
`converged & 0.05≤volfrac_achieved≤0.95` (KHÔNG lọc theo liên thông hình học hay
dao động nhãn `osc_score` như `assemble_phase3_v2.py::is_clean()`). Đây KHÔNG
phải bug ở dạng "tính sai" — 2 pipeline này dùng 2 tiêu chí lọc khác nhau có
chủ đích ở giai đoạn phát triển khác nhau — nhưng **không thể hợp nhất trực
tiếp** vì `dataset_64.npz` (do `build_npz.py` tạo) không lưu `sample_id`, chỉ
lưu `batch`/`seed_names`/`params` (float liên tục) — không có khóa join an
toàn để nối với `outputs/phase3/manifest_quality.csv` (nơi chứa `is_connected`/
`osc_score`, khóa theo `batch+seed+sample_id`). Muốn hợp nhất thật sự cần sửa
`build_npz.py` để giữ lại `sample_id` xuyên suốt - chưa làm trong lần audit này
(xem AUDIT_REPORT_INDEPENDENT_2026-07-29.md mục D1/A5) để tránh rủi ro thay đổi
silent hành vi pipeline hiện có.

**Nếu chạy lại pipeline này hôm nay:** `scan_dataset.py` giờ tự động quét mọi
thư mục `batch_*` trong `outputs/multi_batch/` (kể cả hậu tố như
`batch_11_dqfix_full_rebuild`) thay vì hardcode 8 batch đầu - đã sửa
2026-07-29, xem `discover_batches()`.

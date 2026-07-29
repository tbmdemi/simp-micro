# Nhật ký Thử nghiệm & Lỗi đã Sửa

Tài liệu này là **nhật ký các sai sót phát hiện được trong quá trình làm việc** (triệu chứng → nguyên nhân → cách sửa → tác động đo được) và **các đột phá**. Không lặp lại mô tả "hệ thống hoạt động thế nào ở trạng thái hiện tại" - việc đó thuộc về [`README.md`](README.md). Số liệu chi tiết/CI đầy đủ nằm trong các file JSON dưới `outputs/` được trỏ tới ở mỗi mục; tài liệu này chỉ giữ con số đủ để hiểu kết luận.

---

## Bảng tổng hợp lỗi đã sửa

| Lỗi | Triệu chứng / ảnh hưởng | Cách sửa |
|---|---|---|
| Chuyển vị FE trong đồng nhất hóa | Trường dao động χ dùng trực tiếp làm chuyển vị tổng thay vì `U0 + χ` - âm thầm sai tensor `Q` ở **mọi** lần chạy | `compute_homogenized_tensor()`: `U_total = U0 + U` |
| Công thức tắt trực hướng ν₁₂ khi xoay | `ν₁₂ = Q₁₂/Q₂₂` chỉ đúng khi `Q₁₃=Q₂₃=0`; sai ở các góc xoay như 32,4° → **0 mẫu auxetic** ở lần screening đầu tiên | `compute_nu12`/`compute_nu21` dùng nghịch đảo đầy đủ ma trận 3×3 (`S=Q⁻¹`) |
| Phạt `mu` trong mục tiêu auxetic | Sai sót khái niệm, có thể sụp vùng rỗng mà không cải thiện auxeticity | Tắt mặc định (`mu=0.0`) |
| Dấu mục tiêu `first`/`second` | Hướng cập nhật OC sai | Loại bỏ, chỉ còn mục tiêu `auxetic` |
| `max` thay vì `min` trong `aggregate_correlations.py` | Chọn mẫu "tốt nhất" ra mẫu tệ nhất | Đổi `max→min` |
| Nhiễu đo lường xuyên vòng self-play | `verify_round()` seed đổi mỗi vòng (`123+k`) + `model.generate()` không seed → "cải thiện" R² qua các vòng chỉ là nhiễu | Seed cố định 123 + `torch.manual_seed()` trước khi sinh |
| Dữ liệu adversarial vô hình khi fine-tune | Chỉ ~0,1% mỗi batch → gradient không đủ tín hiệu | `--adversarial-oversample N` (self-play v2 dùng N=40) |
| **`dQ` (sensitivity đồng nhất hóa) bị hoán vị pixel** | `reshape()` mặc định `order='C'` thay vì `'F'` → với lưới vuông tương đương TRANSPOSE map sensitivity trước `oc_update()`. Vô hại với 8/11 seed đối xứng, nhưng sai hướng gradient cho 3 seed bất đối xứng (`hourglass`,`hexagonal`,`reentrant_bowtie`) - 2/3 từng hội tụ **sai dấu hoàn toàn** | Thêm `order='F'`; kiểm chứng bằng finite-difference + SIMP thật ở 4 quy mô (xem mục dưới) |
| `converged=True` không đảm bảo hội tụ thật | Cờ này bật cả khi chỉ chạm `max_iter=150`, không cần thỏa tolerance - 76,3% dataset thuộc diện này | Thêm cột `n_iters < max_iter` (`manifest_quality.csv`) làm tiêu chí hội tụ thật |
| Notebook hardcode `n_iter=150` | `notebooks/utils.py::load_manifest_samples()` không đọc `n_iters` thật | Đọc từ `batch_{i}_results.csv` |
| Nhãn "dao động" (limit-cycle) không bị lọc | OC rơi vào chu kỳ dao động cuối vòng lặp, ảnh lưu là 1 lát cắt ngẫu nhiên chưa ổn định - một số mẫu **sai cả dấu** auxetic | Metric `osc_score` đo trực tiếp trên lịch sử `Poisson_v12` (không suy diễn qua ảnh), ngưỡng 0,15 |
| `val_loss` không an toàn để chọn checkpoint khi `gamma>0` | `beta` (KL) ramp tuyến tính trong `kl-warmup` khiến `val_loss` tăng cơ học, thiên vị chọn checkpoint SỚM/chưa train đủ - tái hiện độc lập 2 lần (weighted-sampling và Phase 5 retrain lần 1) | `--select-by fe_r2 --fe-eval-every N` (chọn theo R² FE thật đo định kỳ) |

---

## Nhật ký theo thời gian

### 2026-07-22 → 23 - Nền tảng Phase 1/2 và các thử nghiệm Phase 5 đầu tiên

- **gamma-sweep**: R² qua surrogate đóng băng tăng đơn điệu theo gamma (0,63 → 0,86), nhưng kiểm chứng bằng FE thật (`verify_fe.py`) cho R² **âm sâu ở mọi mức** (−1,2 đến −2,4) và khoảng cách nới rộng khi gamma tăng → bằng chứng **surrogate exploitation** (decoder học đánh lừa CNN đóng băng thay vì sinh hình học đúng). Từ đây `gamma=20` được giữ làm baseline (không phải vì tốt nhất qua surrogate, mà vì là điểm duy nhất có kiểm chứng FE đầy đủ trước khi chuyển hướng sang inference-time).
- **self-play** (`self_play.py`, fine-tune surrogate định kỳ trên ảnh cVAE sinh + chọn checkpoint theo R² FE thật): sau khi sửa lỗi đo lường ở bảng trên, kiểm chứng lại trên tập cố định 96 mẫu cho R² gần như phẳng, nhiễu (−2,27 đến −3,30, không xu hướng) qua 8 vòng. **Kết luận: self-play không khắc phục được single-shot exploitation** trong ngân sách đã thử.
- **ensemble surrogate** (3 surrogate độc lập + phạt bất đồng): hạ tầng đã có test, nhưng 1 lần chạy đầy đủ không hội tụ trong ngân sách phiên làm việc → để lại như tùy chọn có sẵn, **chưa phải biện pháp đã chứng minh**.
- **`force_periodic()`**: phát hiện (từ 1 nhánh nghiên cứu kiến trúc khác, `research/auxetic-breakthrough`) rằng periodicity không cần học - ép được bằng 1 phép gán pixel (`img[:,-1]=img[:,0]=mean(...)`, tương tự hàng trên/dưới), không cần gradient. Đo trên `cvae_gamma20.pt`: `passes_all` 1,7%→19,5% (>10×), chi phí property nhỏ (mean |Δv12|=0,019). Tích hợp làm hậu xử lý **mặc định BẬT** trong `best_of_n_eval.py`/`sample.py` (cờ `--no-force-periodic` để tắt).
- **Manufacturability của chính dữ liệu train** (phân tích ngược trên 7.920 mẫu thật): tương quan Spearman giữa manufacturability và các tham số DOE liên tục (`volfrac,penal,rmin,move,void_size_frac`) đều `|r|<0,12` - không đáng kể. **SEED mới là đòn bẩy chi phối** (8× chênh lệch: `reentrant_bowtie` 62,8% passes_all vs `hexagonal` 7,9%). Thêm `compute_seed_sample_allocation()` (Phase 2) để phân bổ mẫu theo `joint_rate` (auxetic∧manufacturable) thay vì chia đều - validated bằng 99 mẫu thật: joint rate 17,6%→25,3% (×1,44 so với batch công bằng nhất, ×1,76 so với trung bình lịch sử).
- **Bootstrap CI** thêm cho cả Phase 4 (`pipeline/phase4_surrogate/bootstrap_ci.py`) và Phase 5 (`pipeline/phase5_cvae/bootstrap_ci.py`) sau khi nhận ra các con số R² Phase 5 đo trên chỉ 19-24 điều kiện có CI rất rộng (vd R²=+0,60 nhưng CI cắt ngang gần 0) - từ đây mọi số Phase 5 trong tài liệu đọc kèm CI, không chỉ điểm ước lượng.
- Biện pháp huấn luyện lại cho manufacturability (regularize posterior vs regularize prior-samples): regularize trên `recon` (posterior) **không có tác dụng** (ảnh train vốn đã gần tuần hoàn sẵn, decoder lúc inference lại lấy mẫu từ prior, loss không chạm tới). Regularize trực tiếp bản giải mã từ prior (`--regularize-prior-samples`) cho cải thiện nhỏ nhưng nhất quán (~1,5-2×) - không ấn tượng, `force_periodic()` (trên) hiệu quả hơn nhiều và không cần train lại.

### 2026-07-24 - Advisor audit, dọn dữ liệu tận gốc, rebuild dataset

- **Lỗi hoán vị `dQ`** (xem bảng trên): kiểm chứng ở 4 quy mô tăng dần trước khi rebuild - 1 điểm/seed → 4 combo LHS/seed (12/12 đều delta âm, không phải nhiễu) → pilot 90 mẫu thật → **rebuild đầy đủ 2.160 mẫu** (720×3 seed, khớp quy mô lịch sử). Kết quả rebuild so với dữ liệu lỗi cũ: `hourglass` v12 trung bình −0,625 (cũ −0,115, ×5,4), `hexagonal` −0,349 (cũ −0,084, ×4,1), `reentrant_bowtie` −0,172 (cũ −0,015, ×11,4). Đã tích hợp vào `outputs/phase3/manifest.csv`/`train,val,test.npz` (backup manifest cũ: `manifest_pre_dqfix_backup.csv`). **Tác động toàn dataset (7.920 mẫu): auxetic rate 82,1%→91,9%.**
- **Audit hội tụ + hình học** (yêu cầu advisor, script `analysis/scripts/audit_convergence_and_geometry.py` trên toàn bộ 7.920 mẫu): xác nhận bug `converged=True` ở bảng trên (76,3% dataset chỉ chạm `max_iter`, không đảm bảo tolerance) và 23,9% dataset hình học rời rạc (`check_connectivity()`, `n_components>1`) - tập trung ở `grid_circular_voids`/`nine_circle`/`four_circle` (42-47%). Dữ liệu: `outputs/phase3/manifest_quality.csv`, `convergence_geometry_audit*.{json,csv}`.
- **Dọn nhãn dao động** (yêu cầu advisor, khởi phát từ 3 ảnh mờ/xám advisor gửi): heuristic đo trực tiếp trên ảnh (blur/gray-plateau) cho false positive/negative rõ ràng khi kiểm tra bằng mắt. Gốc rễ thật: mở `iteration_data.csv` của 1 mẫu lỗi thấy `Poisson_v12` **dao động qua lại giữa 2 giá trị mỗi iteration** suốt >15 vòng cuối (limit-cycle của thuật toán OC) - ảnh lưu là 1 lát cắt ngẫu nhiên của chu kỳ đó, có mẫu còn **sai cả dấu** auxetic. Metric đúng: `osc_score` (biên độ dao động `Poisson_v12` trên 20 iteration cuối, ngưỡng 0,15) đo trực tiếp trên lịch sử tối ưu - không suy diễn qua ảnh. Phát hiện phụ: 212 mẫu có `v12`/`v21` ngoài khoảng vật lý hợp lý `[-1,0, 0,5]` (nghi `Q22` gần suy biến, tập trung 188/212 ở `reentrant_bowtie`). **Tổng loại: 2.662/7.920 mẫu (33,6%)**, backup trước khi ghi đè (`manifest_pre_qualityfilter_backup.csv`), rebuild qua đúng pipeline gốc → train 33.246→22.080 mẫu (sau augment), val/test 1.188→789.
- **Retrain Phase 4/5 trên dataset sạch**: surrogate `surrogate_clean.pt` R²(v12) **0,48→0,92**, R²(v21) 0,38→0,89 (CI 95% [0,907,0,935]/[0,860,0,913], n=789). Lần retrain Phase 5 đầu tiên chọn checkpoint theo `val_loss` mặc định → tái hiện đúng bug ở bảng trên **lần thứ 2 độc lập** (chọn epoch 13, giữa KL-warmup, R² thấp hơn cả baseline). Chạy lại với `--select-by fe_r2` chọn đúng epoch 28: oracle R²=0,68 [0,23,0,93].
- **+ weighted-sampling trên dataset sạch** (`cvae_clean_weighted.pt`, dùng `v12_bin_weights.json` ưu tiên vùng auxetic thưa mẫu): checkpoint tốt nhất trước khi có differentiable-physics - oracle R²=**0,83** [0,59,0,95], K=10 R²=**0,65** [0,26,0,83], lần đầu tiên CI 95% của cả 2 biến thể hoàn toàn dương.
- **Confound `force_periodic()` phát hiện giữa chừng**: số baseline "0,5955" đã lưu từ 2026-07-23 được đo TRƯỚC KHI `force_periodic()` thành mặc định - so trực tiếp với checkpoint mới (đo sau khi có `force_periodic`) là so sánh lẫn 2 biến, không chỉ 1. Đo lại `cvae_gamma20.pt` bằng code hiện tại: R²=0,314 (không phải 0,5955). Từ đây mọi so sánh checkpoint theo thời gian đều đo lại CẢ HAI bằng code hiện tại, không tin số liệu JSON cũ.
- **Ablation nhãn-sạch (osc_score) - có kiểm soát, 3 seed độc lập**: câu hỏi là R² Phase 4 tăng 0,48→0,92 do lọc `osc_score` hay do bản vá `dQ` (làm trước, ảnh hưởng cả dataset)? Lần thử đầu bị confound (subsample theo ảnh, lẫn cả trạng thái vá-dQ và đa dạng mẫu-gốc) → sửa lại bằng subsample cấp mẫu-gốc, khớp chính xác số lượng + logic augment/split gốc. Kết quả 3 seed (42,7,123): **`osc_score` KHÔNG cải thiện R² Phase 4** (trung bình 0,952±0,004 không lọc vs 0,922 đã lọc, CI v12 không chồng lấn) - phần lớn gain đến từ bản vá `dQ`, không phải bộ lọc này. Ở Phase 5 best-of-N thì điểm ước lượng nghiêng có lợi cho việc lọc (oracle 0,68 vs 0,52) nhưng CI chồng lấn ở n=24 - chưa đủ mạnh để kết luận. Khuyến nghị thực dụng: vẫn giữ bộ lọc mặc định (mẫu bị lọc có nhãn sai rõ ràng, giá trị vệ sinh dữ liệu độc lập với R²).
- **Đánh giá lại ở quy mô lớn (n=24→300→789, toàn bộ test set)**: oracle mode giữ nguyên kết luận nhưng khiêm tốn hơn nhiều khi n tăng (checkpoint tốt nhất trước physics: 0,83→0,44→0,32). **Chế độ "thực dụng" K=10 sụp đổ về CI âm/cắt-ngang-0 ở quy mô đầy đủ cho MỌI checkpoint trước differentiable-physics** (vd checkpoint cũ: R²=−0,212, CI hoàn toàn âm ở n=789) - khuyến nghị trước đây trong README ("K=10 giữ được gần hết mức tăng R²", dựa trên n=24) **SAI**, đã sửa lại trong README. Chẩn đoán riêng (K=10-ngẫu-nhiên vs K=10-xếp-hạng-bởi-surrogate, n=300): surrogate ranking CÓ giá trị thật (CI tách biệt hoàn toàn khỏi ngẫu nhiên) nhưng còn cách xa chất lượng oracle.

### 2026-07-24 (cùng ngày) - đột phá differentiable-physics

`real_physics_prior_loss()` (FE-solve thật + gradient giải tích trong training loop, không qua surrogate có thể bị đánh lừa) đã được viết + kiểm thử đầu-cuối từ trước nhưng **chưa từng chạy huấn luyện thật**. Chạy lần đầu: fine-tune 35 epoch từ `cvae_clean_weighted.pt` (`--lambda-real-physics 20.0 --real-physics-subsample 8 --real-physics-every 2 --real-physics-workers 8 --select-by fe_r2`, lệnh đầy đủ và tham số xem `pipeline/phase5_cvae/train.py --help`).

R²(FE, single-shot validation đo mỗi 2 epoch) tăng dựng đứng: epoch 2 = 0,11 → epoch 10 = 0,41 → epoch 20 = 0,92 - **lần đầu tiên chỉ số này dương và ổn định** trong cả dự án (mọi checkpoint trước luôn âm sâu, -5 đến -13). Kiểm chứng ở quy mô đầy đủ (n=789, toàn bộ test set, không subsample - rút kinh nghiệm từ phát hiện K=10-sụp-đổ ở trên):

| Chế độ | R²(FE) [95% CI] | hit rate single-shot | frac manufacturable |
|---|---|---|---|
| oracle (N=30) | **0,9988** [0,9985, 0,9990] | 98,7% | 0,350 |
| thực dụng (K=10) | **0,9920** [0,9899, 0,9937] | 98,7% | 0,350 |

`hit_rate_single_shot` nhảy từ ~35-40% (mọi checkpoint trước) lên 98,7% - generate() **1 lần duy nhất** giờ gần như luôn đúng, không còn cần best-of-N để "né" surrogate exploitation.

**Kiểm tra chống ngộ nhận:** (1) không mode-collapse - 4 ảnh sinh ở target khác nhau (v12=−0,70/−0,35/−0,10/+0,05) khác nhau rõ rệt về hình học; (2) không rò rỉ test set - loss chỉ dùng điều kiện từ `train.npz`, đánh giá luôn đọc từ `test.npz`; (3) không phải artefact n nhỏ - đo trên toàn bộ 789 mẫu test, CI cực hẹp.

**Kết luận:** đây là lần đầu tiên vấn đề surrogate-exploitation (theo đuổi suốt dự án - self-play và ensemble surrogate ở trên đều thất bại) được sửa **tận gốc ở decoder**, không phải né ở inference-time. `cvae_realphysics.pt` là checkpoint khuyến nghị mặc định cho mọi việc dùng Phase 5 tiếp theo.

**Việc CHƯA làm:** train from-scratch với differentiable-physics (thay vì chỉ fine-tune) để xem có cần 2 giai đoạn hay không; đo trên seed hình học/điều kiện ngoài khoảng train; CI riêng cho `frac_manufacturable=0,350`; quét `--lambda-real-physics` khác 20,0; multi-seed ablation osc_score ở Phase 5 (chỉ Phase 4 đã lặp 3 seed).

---

### 2026-07-25 - Rebuild dataset v2 (mở rộng quy mô + 2 bug trong sinh dữ liệu mới), retrain Phase 4/5

**Động lực:** dataset production cần lớn hơn cho việc huấn luyện Phase 5 tiếp theo (mục tiêu 40-50k ảnh train sau augmentation). Viết script sinh dữ liệu song song mới `analysis/scripts/generate_production_batch.py` (dùng `multiprocessing.Pool`, độc lập với `pipeline/phase2_multi_batch/` gốc) để sinh thêm mẫu thô nhanh hơn.

**Bug #1 - range tham số quá rộng cho seed nhạy cảm:** dùng nguyên `ACTIVE_PARAMETERS` (volfrac 0,3-0,7 / void_size_frac 0,1-0,4) cho MỌI seed khiến `reentrant_bowtie` chỉ đạt yield 1,4% (lịch sử 77,2%). `hourglass` không bị ảnh hưởng (96,3%, vốn robust trên toàn dải). Sửa: thêm `SEED_PARAM_RANGES`, `hexagonal`/`reentrant_bowtie` dùng range hẹp đã kiểm chứng (volfrac 0,50-0,58 / void_size_frac 0,28-0,38), giữ nguyên range rộng cho `hourglass`.

**Bug #2 - `beta=0,8` thay vì mặc định production `1,0` (nghiêm trọng hơn, phát hiện SAU khi bug #1 vẫn không giải quyết hết vấn đề):** sau khi sửa range, `reentrant_bowtie` vẫn 0% sạch - `v12` kẹt ở đúng ~2e-11, 100% rời rạc, dừng ở 13-14/150 iteration. Root-cause qua `git diff simp/runner.py` (loại trừ các thay đổi `move_min`/`penal_init` của phiên trước vì no-op khi không set) và trace `pipeline/phase2_multi_batch/runner.py::build_params_dict()`: hàm này **không bao giờ set `beta`**, nghĩa là production luôn chạy `beta=1.0` (mặc định nội bộ của `run_simp`) - không phải `beta=0.8` mà mọi script sinh/pilot tuỳ chỉnh trong phiên này đã hardcode từ đầu. Xác nhận qua tái hiện trực tiếp bằng `run_simp()`: `beta=0.8` → kẹt 13 iter, `v12≈0`; `beta=1.0` → chạy đủ 150 iter, `v12=-0,063`. Sửa `generate_production_batch.py`: `beta=0.8`→`1.0`.

**Kết quả sau khi sửa cả 2 bug** (batch đầy đủ, sau khi xoá và chạy lại batch bị lỗi beta=0,8): yield sạch `hourglass`=95,5%, `hexagonal`=96,4%, `reentrant_bowtie`=42,4%. `reentrant_bowtie` cải thiện rất nhiều so với 1,4% ban đầu nhưng **vẫn thấp hơn lịch sử 77,2%** - chênh lệch còn lại chưa điều tra thêm (không chặn tiến độ vì tổng thể dataset đã đạt quy mô mục tiêu).

**Lưu ý về `/tmp`:** batch raw đầu tiên (~10.700 mẫu, ~3,5h compute) bị MẤT vì ghi vào `/tmp/phase3_v2_raw` (tmpfs, xoá qua reboot máy giữa 2 phiên làm việc). Từ đó mọi output raw/scratch ghi vào `outputs/` (đĩa thật), không dùng `/tmp` cho dữ liệu cần giữ.

**Assemble + retrain:** `analysis/scripts/assemble_phase3_v2.py` lọc mẫu thô mới bằng `is_connected & osc_score<0,15 & v12<0 & not degenerate` (**không** loại theo `hit_cap`/`converged` - phiên này xác nhận lại rằng cờ `converged` bị gán sai và hit-cap không phải tiêu chí chất lượng đáng tin, xem mục 2026-07-24 ở trên), gộp với pool sạch cũ, chia 70/15/15 theo phân tầng seed, tăng cường đối xứng ×6 (train). Kết quả: train=**57.216** (đạt mục tiêu), val=test=**2.044**. Đã thay `outputs/phase3/{train,val,test,dataset_64}.npz` (backup bộ cũ nguyên vẹn: `outputs/phase3_backup/`).

Retrain Phase 4 (`surrogate_v2.pt`) trên dataset v2: R²(v12/v21/volfrac) = **0,974/0,964/0,983** (test set v2, 2.044 mẫu) - cải thiện so với `surrogate_clean.pt` (0,922/0,889), chủ yếu nhờ quy mô dữ liệu (~2,6×).

Retrain Phase 5: thử train from-scratch với real-physics loss active từ epoch 1 → **thất bại** (R²(FE) âm sâu -2 đến -17,4 suốt 60 epoch) - xác nhận quy trình 2-stage (`cvae_clean_weighted.pt`→`cvae_realphysics.pt` ở mục trên) là **bắt buộc**, không thể gộp 1 bước dù đã ngờ vực điều này từ trước. Chạy lại đúng 2-stage: Stage 1 base training (`cvae_v2_base.pt`, không real-physics, R²(FE) tại checkpoint tốt nhất = -1,34) → Stage 2 fine-tune 35 epoch (`--lambda-real-physics 20.0`, giống hệt tham số của `cvae_realphysics.pt`) cho `cvae_v2_finetuned.pt`, R²(FE) hội tụ ổn định qua các lần đo (0,44→0,81→0,84→0,86→0,91→0,90, chọn best=0,9054 theo `--select-by fe_r2`).

**So sánh chính thức** (`best_of_n_eval.py --n-conditions 300 --n-samples 30`, CÙNG test set v2 cho cả 2 checkpoint để công bằng):

| Checkpoint | R²(FE) oracle | hit rate single-shot | hit rate best-of-N | frac manufacturable |
|---|---|---|---|---|
| `cvae_v2_finetuned.pt` | 0,9953 | **0,997** | 1,000 | 0,247 |
| `cvae_realphysics.pt` (checkpoint cũ, train trên dataset trước rebuild) | **0,9984** | 0,983 | 1,000 | **0,280** |

**Quyết định:** giữ dataset v2 làm production (rollback path còn nguyên: `outputs/phase3_backup/`, `cvae_realphysics.pt`, `surrogate_best.pt` không bị đụng). Hai checkpoint Phase 5 ngang ngửa - khuyến nghị `cvae_v2_finetuned.pt` cho nhất quán với dataset hiện tại, nhưng `cvae_realphysics.pt` vẫn dùng tốt nếu ưu tiên R²/manufacturability nhỉnh hơn chút.

**Việc CHƯA làm:** điều tra tiếp vì sao `reentrant_bowtie` chưa về lại mức yield lịch sử (77,2%) dù đã sửa cả 2 bug đã biết; cập nhật default checkpoint path trong code (hiện `surrogate_best.pt`/`surrogate_for_phase5.pt`/`cvae_realphysics.pt` vẫn là default, bản `_v2` chỉ dùng qua flag tường minh - quyết định có đổi default hay không chưa được đưa ra).

---

*Xem [`CHANGELOG.md`](CHANGELOG.md) cho lịch sử thay đổi theo phiên bản, và [`README.md`](README.md) cho trạng thái/cách hoạt động hiện tại của dự án.*

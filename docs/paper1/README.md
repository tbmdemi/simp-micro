# Bài báo #1 - bản thảo LaTeX (EN + VI)

**Tên bài (C2, chốt 2026-10-07):** *Optimizing What Is Verified: Binarization-Aware Latent Refinement
for Generative Inverse Design of Auxetic Metamaterials* - tác giả Trịnh Bình Minh. Tạp chí đích:
*Structural and Multidisciplinary Optimization* (C4).

> **[2026-10-08] P1.10:** `main_en.tex` / `main_vi.tex` đã viết lại toàn bộ theo Khung A (C1), cấu
> trúc SMO (Intro → Methods → Results → Discussion → Conclusions → Appendix fp → Statements), mọi số
> trên pipeline không `force_periodic` (C5). Bản 30/09 và các nháp P1.10 ở `archive/`.
> **Chưa làm (cần cắm sạc):** (1) vẽ lại hình - `make_figures.py` đã trỏ sang file không fp nhưng
> chưa chạy, `fig_signflip` viết lại chưa chạy thử; (2) chạy lại `final_design_eval.py` trên thiết
> kế không fp (số 200² và `fig_stiffness` vẫn là bản có fp, đã ghi chú trong bài); (3) biên dịch
> PDF bằng tectonic (PDF hiện tại là bản 30/09, đã cũ).

Bản thảo viết theo `docs/plan.md` v3, mục P1.5. Cấu trúc theo kiểu npj Computational
Materials: Abstract → Introduction → Results → Discussion → Methods, trích dẫn đánh số.

| File | Nội dung |
|---|---|
| `main_en.tex` / `main_en.pdf` | Bản tiếng Anh |
| `main_vi.tex` / `main_vi.pdf` | Bản tiếng Việt (cùng số liệu, cùng hình, cùng trích dẫn) |
| `refs.bib` | 41 tài liệu tham khảo, dùng chung. DOI/arXiv đã đối chiếu ngày 2026-09-30 |
| `figures/fig_{parity,gap,spearman,signflip,ood}_{en,vi}.pdf` | Hình 3-7 (Hình 1-2 vẽ bằng TikZ ngay trong file .tex) |
| `scripts/make_figures.py` | Sinh lại toàn bộ hình từ JSON kết quả, không chạy lại FE/GPU |

## Biên dịch

Máy không có TeX Live; dùng `tectonic` (engine XeTeX, tự chạy BibTeX, tự tải package):

```bash
cd docs/paper1
~/miniconda3/envs/simp/bin/python scripts/make_figures.py   # chỉ khi cần vẽ lại hình
tectonic main_en.tex
tectonic main_vi.tex
```

Font chữ TeX Gyre Termes được gọi theo tên file `.otf` có sẵn trong bundle tectonic.
Nếu dùng TeX Live: `latexmk -xelatex main_en.tex`.

## Nguồn gốc số liệu (mọi số trong bài đều truy được về file JSON)

| Trong bài | File trong `outputs/phase5/plan_v3/` |
|---|---|
| Bảng 1, Hình 3-4 - mẫu đơn lẻ | `p1_1a_v2_finetuned_in100.json` (liên tục), `p1_1e_a_v2_finetuned_in100.json` (nhận biết nhị phân hóa) |
| Bảng 1 - best-of-30 | `p1_1b_v2_finetuned_in100_bo30.json`, `p1_1e_b_v2_finetuned_in100_bo30.json` |
| Bảng 5, Hình 7 - quét biên độ OOD | `p1_1d_v2_finetuned_ood.json`, `p1_1e_d_v2_finetuned_ood.json` |
| Bảng 2, Hình 5 - Spearman điểm tổng hợp vs Pareto | `outputs/phase5/reports/composite_score_pareto_validation.json` |
| Bảng 4, Hình 6 - OOD đổi dấu vs retrieval | `outputs/phase5/reports/ood_baseline_comparison_cvae_realphysics.json` (chỉ nhóm `non_auxetic`; nhóm `extreme_auxetic` bị loại vì cả 6 mục tiêu vi phạm ν12·ν21<1) |
| Bảng 6 - KAN vs Linear | `p12_eval_{linear,kan}_s{123,7}_{ss,bo30}.json` (số lấy theo `docs/plan.md` mục P1.2) |
| WIRE | `p13_wire_bestofn_in24.json`, `p13_wire_refine_in24.json` |
| Bảng 2 khối trên (R² oracle, hit-rate, tỉ lệ chế tạo n=300) | `docs/PIPELINE.md` mục 2026-07-25 |
| Số liệu dataset/surrogate | `docs/PIPELINE.md`, `README.md` |

### Kết quả mới 2026-10-06 (P1.6, CHƯA đưa vào bản thảo - chờ chốt khung claim)

Mọi file trong `outputs/phase5/plan_v3/`. Script trong `scripts/`, chạy từ gốc repo.

| Kết quả | File JSON/npz | Sinh bởi |
|---|---|---|
| SIMP từ đầu vs cVAE (IN100), CI ghép cặp | `p1_6a_simp_baseline_in100.json`, `p1_6a_comparison.json` | `pipeline/phase5_cvae/benchmark_simp_baseline.py`, `scripts/p1_6a_compare.py` |
| OOD đổi dấu, checkpoint production (28 mục tiêu) + SIMP | `p1_6b_signflip_{sym,v21_01,v21_03}.json`, `p1_6b_simp_*.json` | harness `--targets`, `benchmark_simp_baseline.py --targets-json` |
| Chế tạo được sau refine + ảnh thiết kế | `p1_6c_in100_bo30_images.json` | harness `--save-images` |
| Ví dụ refine liên tục vs nhận biết nhị phân hóa (ảnh xám) | `p1_6c_refine_examples.{npz,json}` | `scripts/refine_examples.py` (z kiểm khớp JSON) |
| Trường biến dạng kéo đơn trục | `p1_6c_deformation_examples.{npz,json}` | `scripts/deformation_examples.py` |
| Xác nhận trên tập mới IN100-B (seed 456) | `p1_6x_in100b_e_{a,b}_*.json` | harness `--seed 456` |
| Hội tụ lưới 50²/100²/200² | `p1_6x_mesh_convergence.json` | `scripts/mesh_convergence.py` |
| Thiết kế cuối trên lưới 200² + E_x/E₀ | `p1_6x_final_design_eval.json` | `scripts/final_design_eval.py` |
| Retrieval verified (có/không force_periodic) | `p1_6x_retrieval_in100{,_nofp}.json` | `scripts/retrieval_in100.py` |
| Lai cVAE → SIMP (P1.7, tiêu chí ghi trước, KHÔNG ĐẠT) | `p1_7_hybrid_in100.json` | `pipeline/phase5_cvae/benchmark_hybrid.py` |
| SIMP từ đầu + cVAE có ảnh trên IN100-B (seed 456) | `p1_7_simp_in100b_full.json`, `p1_7_in100b_bo30_images.json` | `benchmark_simp_baseline.py --budgets full`, harness `--seed 456 --save-images` |
| Ablation `force_periodic` (C5) | `p1_7_c5_in100_bo30_nofp.json`, `p1_7_c5_comparison.json` | harness không `--force-periodic`, `scripts/c5_force_periodic.py` |
| Xác nhận phân tầng dị hướng trên IN100-C (seed 789, P1.8, tiêu chí ghi trước) | `p1_8_in100c_targets.json`, `p1_8_in100c_bo30_images.json`, `p1_8_simp_in100c_full.json`, `p1_8_anisotropy_strata.json` | `scripts/p1_8_anisotropy_strata.py` (tính cả IN100, IN100-B hậu kiểm) |
| Chế tạo được không fp: single-shot IN100 + best-of-30 n=300 (P1.9, cho C5) | `p1_9_in100_single_nofp_images.json`, `p1_9_bon300_{nofp,fp,fp_acconly}.json` | harness `--save-images`; `best_of_n_eval.py --periodicity-tol 1.0` (± `--no-force-periodic`; `_acconly` = `--w-accuracy 1 --w-manuf 0 --w-aesthetic 0`, tái lập R² 0,995 cũ) |
| **Mọi số của bài trên pipeline không fp (C5, P1.9)** - Bảng 1, OOD biên độ, bảng SIMP 3 tập, OOD đổi dấu, Bảng 2, Spearman Pareto | `p1_9_r{1,3,4,5,6,7,8}_*.json`, `p1_9_pareto_nofp.json` → `p1_9_nofp_summary.json` | `scripts/p1_9_nofp_summary.py`, `scripts/c5_pareto_nofp.py` |

Hình mới (EN + VI, `make_figures.py`): `fig_refine_examples`, `fig_designs`, `fig_tiling`,
`fig_stiffness`, `fig_deformation`; `fig_ood` thêm panel mật độ dữ liệu train. Tóm tắt số + hệ quả
cho claim: `docs/plan.md` mục P1.6-P1.8, `docs/LIMITATIONS.md` #32-37.

**Mới 2026-10-08 (P1.10):** `p1_10_degenerate_check.json` - kiểm suy biến (min(E_x,E_y)/E0 < 1e-3)
cho thiết kế cuối cVAE và SIMP trên IN100/B/C + đổi dấu: SIMP trả về **ô rỗng** (ν = ν0 = 0,3) ở 3/28
mục tiêu đổi dấu, cVAE suy biến 3/28; SIMP vẫn thắng OOD khi chỉ so mục tiêu cả hai không suy biến.
Sinh bởi `scripts/p1_10_degenerate_check.py`.

## Việc tác giả cần bổ sung trước khi nộp

- Tên tác giả, đơn vị, tác giả liên hệ, mục đóng góp tác giả/xung đột lợi ích, URL kho mã.
- Chọn tạp chí đích và chỉnh theo hướng dẫn tác giả (giới hạn số từ tóm tắt, định dạng tài liệu tham khảo).
- Nên cân nhắc: hình minh họa ô cơ sở trước/sau tinh chỉnh (cần chạy lại decoder để xuất ảnh, JSON hiện chưa lưu ảnh).

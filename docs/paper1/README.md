# Bài báo #1 - bản thảo LaTeX (EN + VI)

**Tên bài (C2, chốt 2026-10-07):** *Optimizing What Is Verified: Binarization-Aware Latent Refinement
for Generative Inverse Design of Auxetic Metamaterials* - tác giả Trịnh Bình Minh. Tạp chí đích:
*Structural and Multidisciplinary Optimization* (C4).

> **[2026-10-08] P1.10:** `main_en.tex` / `main_vi.tex` đã viết lại toàn bộ theo Khung A (C1), cấu
> trúc SMO (Intro → Methods → Results → Discussion → Conclusions → Appendix fp → Statements), mọi số
> trên pipeline không `force_periodic` (C5). Bản 30/09 và các nháp P1.10 ở `archive/`.
> **Xong 2026-10-08 19:35:** hình vẽ lại toàn bộ (không fp), số 200² từ
> `p1_10_final_design_eval_nofp.json`, PDF EN + VI biên dịch lại bằng tectonic.

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
| Thiết kế cuối trên lưới 200² + E_x/E₀ | `p1_10_final_design_eval_nofp.json` (bản có fp: `p1_6x_…`) | `scripts/final_design_eval.py` |
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

### K1 (2026-10-08, ĐÃ có trong bản thảo: Methods, Results đoạn K1 + Hình 8 `fig_k1_tradeoff`)

Mọi file trong `outputs/phase5/plan_v3/k1/`. Tiêu chí ghi trước + bảng số: `docs/plan.md` mục K1.

| Kết quả | File | Sinh bởi |
|---|---|---|
| Pilot chỉnh λ (IN100-B, 20 condition) | `pilot_B20_l{0,0.03,0.1,1}.json` | harness `--corner-weight λ --thin-weight λ --seed 456` |
| Kết quả chính λ = 0,1 (IN100) + xác nhận (IN100-C) | `k1_in100_bo30_l0.1.json`, `k1_in100c_bo30_l0.1.json` → `k1_comparison.json` | harness; `scripts/k1_compare.py` |
| Đường đánh đổi λ (IN100-B, 100 target) - Hình 8 | `sweep_in100b_l{0.03,0.1,0.3,1}.json` (λ = 0: `../p1_9_r5_in100b_bo30_nofp.json`) | harness; `scripts/make_figures.py` |

### N1, N2 (2026-10-10, CHƯA đưa vào bản thảo - chờ tác giả chốt N3-D1/D2, `docs/plan.md` mục N3)

Tiêu chí ghi trước + bảng số: `docs/plan.md` mục N1, N2, N2-E1′, N2-E3-F, N2-C6, N2-S, N2-E3-S15.
Phương pháp, câu gợi ý, trích dẫn cần thêm: `ghi_chu_viet_bao.md`. Thước đo: e_verify (50²), e_mesh
(200² kron), e_real (hiện thực hóa làm mượt 200², σ 0,5 / 1,0), e_shift, e_ed, chế tạo 4 hướng.

| Kết quả | File (trong `outputs/phase5/plan_v3/`) | Sinh bởi |
|---|---|---|
| N1 hội tụ lưới 6 thiết kế (50² → 400², biên bậc thang vs bo) | `n1/mesh_convergence.json` | `n1/mesh_convergence.py` |
| N1 pilot objective (A, K1, N1, lưới 100², N1 + 100²; IN100-B 20) | `n1/pilot_B20_*.json`, `n1/pilot_B20_scores*.json` | harness `--realization-shifts`, `--fe-upsample`; `n1/eval_pilot.py`, `n1/sigma_check.py` |
| N2 pilot E1 robust, E2 lưới 100² mức cuối | `n2/pilot_B20_E1_robust.json`, `n2/pilot_B20_E2_last100.json`, `n2/score_pilot.log` | harness `--robust-etas`, `--fe-upsample-last-only` |
| N2 pilot F (lọc σ = 1) và E1′ | `n2/pilot_B20_F_filter.json`, `n2/pilot_B20_E1p_robust.json`, `n2/score_e1p_pilot.{json,log}` | harness `--design-filter-sigma` |
| **E3-F xác nhận n = 100 (IN100 + IN100-C) - ĐẠT** | `n2/e3f_in100{,c}_filter.json` → `n2/e3f_scores_in100{,c}{,_vs_simp}.json`, `n2/final_e3f.log` | `n2/run_e3f.sh`, `n2/eval_general.py` |
| Xóa đảo rời (A, F, SIMP) | `n2/island_cleanup.{json,log}` | `n2/island_cleanup.py` |
| Độ nhạy σ ∈ {0,5; 0,75; 1,5} (pilot) | `n2/pilot_B20_F_s*.json`, `n2/sigma_scores.json`, `n2/run_sigma.log` | `n2/run_sigma.sh` |
| Xác nhận σ = 1,5 (n = 100 × 2) | `n2/e3s15_in100{,c}.json` → `n2/e3s15_scores_*.json`, `n2/e3s15_vs_simp_*.json` | `n2/run_e3s15.sh` |
| Hình so sánh A / F / SIMP (5 mục tiêu, lát 2×2) | `n2/fig_designs_A_F_SIMP.png` | script tạm, **chưa lưu trong repo** - cần viết lại khi đưa vào bài (N3-3) |

Nếu tác giả chọn F + xóa đảo làm pipeline chính (N3-D1), mọi số của bài phải chạy lại theo N3-2
(giống P1.9) trước khi sửa bản thảo (N3-3).

## Việc tác giả cần bổ sung trước khi nộp

- Tên tác giả, đơn vị, tác giả liên hệ, mục đóng góp tác giả/xung đột lợi ích, URL kho mã.
- Chọn tạp chí đích và chỉnh theo hướng dẫn tác giả (giới hạn số từ tóm tắt, định dạng tài liệu tham khảo).
- Nên cân nhắc: hình minh họa ô cơ sở trước/sau tinh chỉnh (cần chạy lại decoder để xuất ảnh, JSON hiện chưa lưu ảnh).

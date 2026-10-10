# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 2026-10-10 (khuya) - Rà soát code: lỗi train/resume, solver dự phòng hỏng, xóa code chết (nhánh `test_algo_1`)

#### Fixed
- `pipeline/phase5_cvae/train.py`: `--resume-from` dựng kiến trúc từ cờ CLI (mặc định KAN + ép đối xứng) thay vì từ checkpoint → resume mọi checkpoint Linear (production, và bước train lại của `self_play.py`) crash khi load; nếu chỉ truyền `--use-mlp-head` thì load im lặng nhưng ép ảnh đối xứng (v12 = v21) suốt fine-tune (đo trên `cvae_v2_finetuned.pt`: lệch 0,126/pixel, bất đối xứng 0,253 → 0). Nay `resolve_architecture()` lấy kiến trúc từ checkpoint qua `cvae_kwargs_from_checkpoint()`; cờ head/decoder mâu thuẫn → báo lỗi. Đã kiểm: không checkpoint nào đang có bị ảnh hưởng (đều lưu hoặc suy ra `enforce_symmetry=False`).
- `train.py::resize_condition_dim_weights`: chỉ hiểu head KAN → `--extended-condition --resume-from cvae_v2_finetuned.pt` (lệnh trong `PIPELINE.md`) crash `KeyError 'encoder.fc_mu.base_weight'`; nay hỗ trợ cả `nn.Linear`. Đã chạy thật cả 2 lệnh resume (thường + mở rộng condition) trên tập con.
- `train.py`: in "Đã lưu checkpoint (R2=-inf)" khi không có lần FE-eval hữu hạn nào → nay cảnh báo rõ là KHÔNG lưu checkpoint.
- `simp/core/solver.py`: nhánh dự phòng CG gọi `cg(tol=)` mà SciPy 1.15 không còn tham số này → luôn `TypeError` (kích hoạt 0/700 thiết kế thật đã đo). Thay bằng `LinAlgError` rõ ràng - hành vi thực tế không đổi; ν của 100 thiết kế IN100 tính lại khớp JSON đã lưu tới 1e-13.
- `analysis/scripts/generate_production_batch.py --help` crash (`%` chưa escape trong help argparse).
- `analysis/scripts/backfill_f1_f2_npz.py`: sanity check chỉ so v12 dù docstring hứa cả v21; thêm v21 (kiểm trên kết quả đã lưu: |Δv21| TB 0,021 ≈ |Δv12| 0,019, không có tráo v12↔v21).

#### Changed
- Mặc định `CVAE` / `Encoder` / `Decoder` / `WireContinuousDecoder` và `train.py`: head Linear, không ép đối xứng (quyết định P1.2; trước đây phải nhớ `--use-mlp-head --disable-symmetry`). Cờ mới `--use-kan-head`, `--enforce-symmetry`; cờ cũ vẫn nhận.
- Ví dụ lệnh trong docstring trỏ checkpoint đã xóa (`cvae_gamma20.pt`, `gamma_sweep_results/…`) → `cvae_v2_finetuned.pt`; nhánh chọn checkpoint theo `val_loss` (không truy cập được, `--select-by` chỉ có `fe_r2`) đã xóa.

#### Removed
- Code chết (không entry point, không nơi gọi): `analysis/sensitivity/` (cả package), `analysis/pareto/runner.py`, `analysis/pareto/visualize.py`, `compute_pareto_front`/`compute_hypervolume`, `analysis/utils.py`, `analysis/conftest.py`, `simp/io/logger.py::SimpLogger` (test thay bằng test cho `save_csv` - bộ ghi CSV thật), `sampling.append_samples_to_csv`.
- 34 import thừa, 7 biến gán không dùng, 12 f-string không có biến (pyflakes nay sạch, trừ 9 chú thích kiểu dạng chuỗi).

#### Tests
- 732 → 735: +8 (`TestResolveArchitecture`, resize head Linear, load sau resize cho cả 2 head), +3 `save_csv`, −8 `SimpLogger`; 5 test KAN/đối xứng truyền tường minh `use_kan=True`/`enforce_symmetry=True`.

### 2026-10-10 (tối) - Dọn `outputs/`, sửa test ghi vào `outputs/`, đồng bộ tài liệu (nhánh `test_algo_1`)

#### Removed
- `outputs/` (~286 MB): artifact của các nhánh thí nghiệm đã đóng/bị thay thế, kết quả đã ghi trong docs - checkpoint gamma sweep (`cvae_gamma*`, `gamma_sweep_results/`), dòng clean/ablation cũ (`cvae_clean`, `cvae_clean_v2`, `cvae_ablation_v2`), `cvae_best`, `cvae_manuf_prior`, active learning (`cvae_al_round1`, `active_learning/`), KAN cũ (`cvae_kan_*` + latents), WIRE (`cvae_wire_*`), `exp3_mlp_physics_150`, smoke test (`task1_kan_smoke20`, `*_smoke*.pt`), `cvae_a4_full8d_finetuned` v1 (đã có v2); surrogate ablation / ConvKAN / `al_round1`; history, bootstrap report và kết quả `self_play` của các checkpoint đó (chỉ file không được tài liệu nào trích dẫn); `design_library/`, `samples/`, `diagnostics/`, `pilot_nu_convergence/`, `logs_a4/`, `simp_results_hexagonal/`, `manifest_pre_qualityfilter_backup.csv`, `phase2_manuf_analysis.csv`. File git theo dõi: `git rm` (commit `d327e5214`, khôi phục được từ lịch sử); file gitignored: chuyển vào Thùng rác.
- Giữ nguyên: `phase5/plan_v3/`, `phase5/reports/`, dataset `phase3`, `phase3_a4`, `phase3_a4_nu_raw`, `multi_batch`, `figures/`, `_backup/`, mọi checkpoint Bài #1 dùng cùng base/surrogate của chúng (vd `cvae_clean_weighted.pt` + `surrogate_clean.pt` là nguồn fine-tune của `cvae_realphysics.pt`).
- Nhánh local `substrate-material` (đã merge PR #15).

#### Fixed
- `tests/test_core_smoke.py`: 7 smoke test gọi `run_simp()` không truyền `output_dir` nên ghi vào `outputs/simp_results_<seed>/`, và `run_simp()` `rmtree` thư mục đó trước khi ghi → mỗi lần `pytest` xóa kết quả SIMP chạy tay cho `circle`/`hourglass`/`reentrant_bowtie`. Nay ghi vào `tmp_path`.
- `benchmark_physics_guided_refinement.py`, `build_latent_dataset.py`: mặc định `--cvae-ckpt` trỏ `cvae_kan_realphysics_v2.pt` (dòng KAN đã đóng ở P1.2, checkpoint đã xóa) → `cvae_v2_finetuned.pt` (checkpoint production; mọi run Bài #1 đều truyền rõ checkpoint này).

#### Changed (tài liệu)
- `EXPERIMENT_LOG.md`: mục 2026-10-10 (N1, N2); chân trang chuyển về cuối file.
- `docs/LIMITATIONS.md`: mục mới #39-42 (refine khớp lưới 50², ν pixel chỉ xác định tới ~0,02, giới hạn của F và robust formulation, thước đo chế tạo cũ đếm đảo rời); cập nhật #2, #32, #33, phạm vi claim (claim 7) và bản tiếng Anh.
- `README.md`: trạng thái 2026-10-08 (K1) và 2026-10-10 (N1/N2), giới hạn, danh sách tài liệu (`docs/plan.md`, `docs/paper1/`), bảng test Phase 5.
- `docs/PIPELINE.md` § 5: K1, N1/N2 + lệnh E3-F; ghi chú checkpoint đã xóa. `docs/CLI_GUIDE.md`: 8.3b bỏ `--force-periodic` khỏi ví dụ pipeline bài (C5), mục mới 8.3e cờ K1/N1/N2. `docs/ARCHITECT.md`: cây module Phase 5. `docs/paper1/README.md`: bảng nguồn gốc số liệu K1, N1, N2.

### 2026-10-10 - N1/N2: hiện thực hóa lệch lưới, robust formulation, thiết kế qua bộ lọc (nhánh `substrate-material`)

#### Added
- `pipeline/phase5_cvae/realization.py`: `realize_shifted` (hiện thực hóa làm mượt + dịch lệch lưới, ngưỡng `eta` cho bản co/giãn) và `filtered_design` (thiết kế vật lý = Heaviside(lọc Gauss → lấy mẫu lưới FE)), khả vi.
- `tandem_lbfgs` + `benchmark_physics_guided_refinement.py`: cờ mặc định tắt `--realization-shifts/--realization-sigma` (N1), `--fe-upsample [--fe-upsample-last-only]` (đối chứng lưới mịn, E2), `--robust-etas/--robust-sigma` (robust formulation, E1/E1′), `--design-filter-sigma` (thiết kế qua bộ lọc F: chọn best-of-N, refine, verify, chế tạo, ảnh lưu trên cùng ảnh; guard theo loss robust khi có `--robust-etas`), `--n-workers`.
- Script đánh giá `outputs/phase5/plan_v3/n1/` (`eval_pilot.py`, `sigma_check.py`, `mesh_convergence.py`) và `n2/` (`eval_general.py` - bảng điểm e_verify / e_mesh / e_real / e_shift / e_ed / chế tạo, có cache; `island_cleanup.py`).
- Test mới: `tests/test_phase5_realization.py`, thêm vào `test_tandem_lbfgs.py`, `test_phase5_refinement_benchmark.py` (706 → 732).
- Tài liệu: `docs/de_xuat_cai_thien_ket_qua.md` (tra cứu + đề xuất), `docs/paper1/ghi_chu_viet_bao.md` (ghi chú viết báo), `docs/plan.md` mục N1, N2, N2-E1′, N2-E3-F, N2-C6, N2-S, N2-E3-S15.

#### Kết quả chính (chi tiết `docs/plan.md`)
- Refine giảm MAE ν12 68% trên lưới verify 50² nhưng chỉ 26% trên lưới mịn; ν của thiết kế pixel chỉ xác định tới ~0,02 tùy biểu diễn biên.
- Thiết kế qua bộ lọc σ = 1 (F) ĐẠT xác nhận đặt trước n = 100 × 2 tập: sai số biên thực tế −43…−47% so với pipeline cũ, −53% so với SIMP, cùng chi phí.
- Thước đo chế tạo cũ chủ yếu đếm đảo rời: xóa đảo (không đổi ν) cho A 0,84 / F 0,93 / SIMP 1,00.

### 2026-10-08 - Dọn dẹp repo: `outputs/` theo dõi bởi git, docs lỗi thời, notebooks

#### Changed
- `notebooks/`: 10 notebook → 3. `01_doe_dataset_analysis.ipynb` gộp 01/02/03 cũ (34 ô gốc giữ nguyên văn); đặt lại tên cho đúng nguồn dữ liệu thật - 3 notebook cũ tên "Phase 1" nhưng luôn đọc manifest DOE Phase 3 (7.920 mẫu), không phải cây LHS Phase 1. `07_geometric_feature_influence` → `02_…`, `09_cheap_physical_properties` → `03_…` (nội dung không đổi).
- `docs/PROJECT_PLAN.md`, `task_progress.md`, `auxforge_architecture_comparison-v2.md`, `paper1_{methods,results,discussion}_draft.md` → `docs/archive/` (đều tự ghi là đã bị thay thế; `docs/plan.md` v3 là nguồn kế hoạch duy nhất).
- `.gitignore`: bỏ theo dõi ảnh snapshot `outputs/**/iteration_*.png`, `*_backup.csv`, `*_partial_log.txt` (artifact trung gian, tạo lại được; file trên đĩa giữ nguyên). Giữ nguyên `outputs/phase5/plan_v3/` (script Bài #1 đọc trực tiếp), `self_play/`, `multi_batch/` (bằng chứng cho `EXPERIMENT_LOG.md`).

#### Removed
- Notebook 04/05/06 (hiển thị chỉ số lỗi thời: surrogate R² 0,91 đời đầu, `property_accuracy` qua surrogate), 08 (thay bằng `docs/paper1/scripts/c5_pareto_nofp.py`), `gamma_sweep_analysis`; file trung gian `cleaned_dataset.parquet`, `cleaned_dataset_preview.csv`, `gamma_sweep_chart.png`, `next_sweep_config.json` (đề xuất `mu=0.2` trái mặc định hiện hành). Tất cả còn trong git history.
- `notebooks/utils.py`: `classify_void`, `compute_coupling_ratio`, `PHASE4_DIR`, `PHASE5_DIR`, `SEEDS` (không còn nơi dùng; `SEEDS` thiếu `reentrant_bowtie`).

#### Fixed
- Notebook 01, tổng kết "tham số ảnh hưởng mạnh nhất lên ν12" xếp hạng cả metric đầu ra (`final_obj`, `n_iter`) → giờ chỉ xếp trong `param_cols` (kết quả: `rmin`, `move`, `void_size_frac`).

### 2026-10-07 - P1.7-P1.9: lai cVAE → SIMP, xác nhận phân tầng, bỏ `force_periodic` (nhánh `substrate-material`)

#### Added
- `pipeline/phase5_cvae/benchmark_hybrid.py`: thí nghiệm lai cVAE → SIMP với 4 tiêu chí đăng ký trước (P1.7, KHÔNG ĐẠT).
- `benchmark_simp_baseline.run_simp_target(x0=, betas=)`: khởi tạo SIMP từ thiết kế tùy ý, β continuation tùy chỉnh (mặc định giữ hành vi cũ).
- `docs/paper1/scripts/`: `c5_force_periodic.py` (ablation C5), `p1_8_anisotropy_strata.py` (phân tầng dị hướng, P1.8), `c5_pareto_nofp.py` (Spearman Pareto không fp), `p1_9_nofp_summary.py` (tổng hợp mọi số không fp).
- Nháp bản thảo Khung A: `docs/paper1/drafts/p1_10_sections_{en,vi}.tex`.
- 5 test mới (701 → 706).

#### Changed
- Quyết định tác giả C1-C5 (`docs/plan.md` mục 4): Khung A, tên bài mới, nộp SMO, không ν0, **bỏ `force_periodic`** + bỏ kiểm tra cạnh khỏi định nghĩa chế tạo được. Mọi số của bài chạy lại không fp (`p1_9_*`).

#### Fixed (tài liệu)
- `LIMITATIONS.md` #35: "refine giảm chế tạo được 30%→24%" không vững (5 lần chạy, p=0,12). Thêm #36 (lai không đạt), #37 (lỗi tập trung ở dị hướng r ≥ 10), #38 (R² 0,995 của Bảng 2 là chọn thuần độ chính xác; composite 0,983).
- Phạm vi claim (`LIMITATIONS.md`, `README.md`): bỏ claim "cVAE chính xác hơn SIMP về ν₁₂" → "ngang SIMP hội tụ với ~8× ít FE".

### 2026-10-06 - P1.6: baseline SIMP + kiểm tra bổ sung Bài #1 (nhánh `substrate-material`)

#### Added
- `pipeline/phase5_cvae/benchmark_simp_baseline.py`: baseline SIMP chạy từ đầu (inverse homogenization ½‖ν−ν*‖², MMA, Heaviside β 1→64, đa khởi tạo, verify giống cVAE).
- `benchmark_physics_guided_refinement.py --save-images`; cờ manufacturable baseline/refined + `frac_manufacturable_*` trong summary.
- `docs/paper1/scripts/`: `p1_6a_compare.py`, `mesh_convergence.py`, `final_design_eval.py`, `retrieval_in100.py`, `refine_examples.py`, `deformation_examples.py`.
- `make_figures.py`: `fig_refine_examples`, `fig_designs`, `fig_tiling`, `fig_stiffness`, `fig_deformation`; `fig_ood` thêm panel mật độ dữ liệu train.
- 8 test mới (693 → 701).

#### Fixed (tài liệu)
- `LIMITATIONS.md` #18: "retrieval R²=1,000" đo trên nhãn, không verify - đính chính (verify: 0,941-0,977). Thêm #32-35 (so SIMP, sai số rời rạc hóa lưới, `force_periodic`, refine giảm chế tạo được).

### 2026-09-30 - Tính chất vật lý suy từ Q (nhánh `substrate-material`)

#### Added
- `real_physics.solve_elastic_with_grad()`: v12/v21/E_x/E_y/G_xy/B_eff kèm gradient giải tích theo pixel, chỉ 1 lần FE-solve; hằng `ELASTIC_KEYS`.
- `auxetic.compute_wave_speeds()`: tốc độ sóng quasi-static (bài toán Christoffel trên Q), chuẩn hóa theo sqrt(E0/ρs).
- `analysis/scripts/backfill_elastic_props_npz.py`: backfill Q + 13 tính chất cho `outputs/phase3_a4/{train,val,test}_props.npz` (73.164 mẫu, 0 lỗi FE).
- `notebooks/03_cheap_physical_properties.ipynb`: kiểm tra cận Voigt/Hashin–Shtrikman, đo thông tin mới so với 5 điều kiện, proxy ấn lõm, biên Pareto ν12 ↔ E_x/ρ.
- 14 test mới (679 → 693).

#### Changed
- `real_physics.solve_nu_with_grad()` giờ là lát cắt ν của `solve_elastic_with_grad()` (1 nguồn công thức đạo hàm; giá trị và gradient không đổi, có test regression).

### 2026-09-25 - plan v3 (nhánh `substrate-material`)

#### Added
- `pipeline/phase5_cvae/heaviside.py`: Heaviside projection, force_periodic và resize nearest (khớp PIL tuyệt đối) bản torch khả vi + `project_for_fe()` dùng chung cho refine và loss train.
- `tandem_lbfgs(projection_betas=, periodic=)`: refine latent nhận thức nhị phân hóa (β continuation, L-BFGS khởi động lại mỗi mức β).
- `benchmark_physics_guided_refinement.py`: `--targets/--target-v21/--n-samples/--force-periodic/--projection-betas/--n-boot`, chỉ số guarded, CI paired; `bootstrap_ci.bootstrap_paired_mae_reduction()`.
- `train.py --rp-projection-beta-max/--rp-periodic`: real-physics loss trên ảnh đã chiếu Heaviside.
- `model.cvae_kwargs_from_checkpoint()`: nguồn suy luận kiến trúc duy nhất cho mọi loader.
- 33 test mới (646 → 679).

#### Fixed
- `adversarial_dataset.load_cvae()` (và `verify_fe`) bỏ qua `use_kan`/`enforce_symmetry` của checkpoint - checkpoint KAN cũ bị ép đối xứng lúc đánh giá, checkpoint Linear crash (`docs/LIMITATIONS.md` mục 28).

#### Changed
- `docs/plan.md` viết lại thành v3 - nguồn kế hoạch + trạng thái duy nhất; Bài báo #1 định vị lại quanh physics-guided refinement. `PROJECT_PLAN.md`, `task_progress.md`, bản nháp paper #1 thêm banner trỏ về plan v3.

> Ghi chú: khối lượng công việc dưới đây (Phase 3-5 đầy đủ) đã hoàn thành và có mặt trong `main`/`FixLoss` từ lâu, nhưng chưa từng được ghi vào CHANGELOG - mục này bù lại khoảng trống đó. Chưa gắn số phiên bản mới vì đó là quyết định phát hành, không tự ý bump.

### Added
- **Phase 3 (`pipeline/phase3_dataset/`)**: pipeline build dataset (`scan_dataset.py`, `build_npz.py`, `augment_symmetry.py`, `finalize_dataset.py`) - 7.920 lần chạy SIMP → trường mật độ 64×64 + target (`v12`, `v21`, `volfrac_achieved`), chia 70/15/15 phân tầng theo seed, tăng cường đối xứng vật lý (train ×6 → 33.120 mẫu).
- **Phase 4 (`pipeline/phase4_surrogate/`)**: CNN surrogate (`SurrogateCNN`) dự đoán (v12, v21, volfrac) từ trường mật độ. R² trên test set = 0,910 / 0,911 / 0,982.
- **Phase 5 (`pipeline/phase5_cvae/`)**: conditional VAE cho thiết kế ngược (`dataset.py`, `model.py`, `losses.py`, `train.py`, `evaluate.py`, `sample.py`, `verify_fe.py`), cùng:
  - gamma-sweep cho trọng số property-loss, kiểm chứng độc lập bằng FE thực (`verify_fe.py`) - phát hiện hiện tượng khai thác surrogate.
  - hai biện pháp khắc phục ở giai đoạn huấn luyện (`self_play.py` self-play adversarial retraining, ensemble surrogate trong `losses.py`) - đã thử và không khắc phục được vấn đề single-shot trong ngân sách thời gian đã thử (xem EXPERIMENT_LOG.md).
  - `best_of_n_eval.py`: sinh N + chọn bằng FE thực, nay là đường suy luận (inference) chính thức (R²=+0,44 đến +0,60, so với single-shot âm sâu).
  - `manufacturability.py` + `coverage_eval.py`: kiểm tra liên thông/tuần hoàn (roadmap 6.2/6.3) và bản đồ độ phủ không gian thuộc tính (7.4).
  - `bootstrap_ci.py`: khoảng tin cậy bootstrap/Wilson cho các con số R²/hit-rate best-of-N đo trên cỡ mẫu nhỏ.
- Bộ test Phase 4/5 (`tests/test_phase4_*.py`, `tests/test_phase5_*.py`) - tổng 208 test (từ baseline Phase 0-3).
- Mục "Giới hạn Đã biết / Known Limitations" trong README.md (song ngữ) gộp các khoảng trống đã biết: cỡ mẫu nhỏ, khả năng chế tạo thấp, `mu` tắt, `f1/f2` chưa có, v.v.

### Changed
- Đổi tên dự án từ "SIMP Analyst" thành **AuxForge** trên toàn bộ tài liệu và codebase.
- README viết lại để phản ánh Phase 4/5 hoàn thành và quy trình thiết kế ngược best-of-N.

### Fixed
- Một số lỗi tính đúng ở Phase 4/5 phát hiện qua kiểm chứng FE (chi tiết nguyên nhân gốc: xem EXPERIMENT_LOG.md).
- `html/inverse_auxetic_report.html` (báo cáo sơ bộ Phase 1, sinh 2026-07-16) có bảng xếp hạng seed trái ngược dữ liệu Phase 2 đã xác thực - thêm banner cảnh báo.
- Metadata packaging lỗi thời trong `pyproject.toml` (tên project, author, URL repository) còn sót lại từ trước khi đổi tên sang AuxForge.

### Project structure
- Di chuyển `docs/workflow.html` (trước đó chưa từng được commit - `docs/` bị gitignore như "personal docs") vào `html/dashboards/workflow.html` để được version-control và khớp với chính mô tả của nó trong README.
- Di chuyển `pipeline/REVIEW_ALGORITHMS_VI.md` (báo cáo review độc lập, đã có ngày tháng cố định) ra khỏi thư mục code `pipeline/`, đưa về root cùng các tài liệu khác.

### 2026-07-24 - Dọn dữ liệu tận gốc + differentiable-physics (chưa gắn số phiên bản)

> Chi tiết đầy đủ (triệu chứng/nguyên nhân/sửa/số liệu): [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Mục này chỉ tóm tắt theo định dạng changelog.

#### Added
- `pipeline/phase5_cvae/real_physics.py`: `real_physics_prior_loss()` - loss differentiable từ FE-solve + đồng nhất hóa thật (gradient giải tích, không qua surrogate), wiring vào `train.py` qua `--lambda-real-physics/--real-physics-subsample/--real-physics-every/--real-physics-workers`.
- `manufacturability.py::force_periodic()` - hậu xử lý ép cứng periodicity bằng 1 phép gán pixel, mặc định BẬT trong `best_of_n_eval.py`/`sample.py` (`--no-force-periodic` để tắt).
- `pipeline/phase2_multi_batch/adaptive.py::compute_seed_sample_allocation()` - phân bổ mẫu theo seed dựa trên tỷ lệ auxetic∧manufacturable, thay vì chia đều.
- `analysis/scripts/audit_label_stability.py` (metric `osc_score`) - phát hiện nhãn dao động/limit-cycle trực tiếp trên lịch sử tối ưu, không suy diễn qua pixel ảnh.
- `train.py --select-by fe_r2 --fe-eval-every N` (chọn checkpoint theo R² FE thật thay vì `val_loss`), `train.py --weighted-sampling` (lấy mẫu theo trọng số nghịch mật độ phổ auxetic).
- `pipeline/phase4_surrogate/bootstrap_ci.py`, `pipeline/phase5_cvae/bootstrap_ci.py` - khoảng tin cậy bootstrap/Wilson cho R²/hit-rate.
- `tests/test_phase5_real_physics.py` + ~58 test khác cho các mục trên (tổng 371/371 pass).

#### Fixed
- **`dQ` bị hoán vị pixel** trong `compute_homogenized_tensor()` (`reshape()` thiếu `order='F'`) - sai hướng gradient OC cho 3 seed bất đối xứng suốt cả quá trình tối ưu; rebuild đầy đủ 2.160 mẫu bị ảnh hưởng.
- **`converged=True` không đảm bảo hội tụ thật** - chỉ chạm `max_iter=150` cũng được gán cờ này; thêm cột hội tụ-theo-tolerance thật (`manifest_quality.csv`).
- **Chọn checkpoint theo `val_loss` khi `gamma>0`** bị KL-warmup đánh lừa, thiên vị chọn checkpoint chưa train đủ - tái hiện độc lập 2 lần trước khi được ghi nhận là bug hệ thống.

#### Changed
- Dataset train lọc bỏ 33,6% mẫu nhãn dao động/rời rạc/ngoài khoảng vật lý hợp lý (7.920→5.258 mẫu gốc); Phase 4 surrogate và Phase 5 cVAE đã train lại trên bản sạch (checkpoint cũ giữ nguyên làm baseline lịch sử, không ghi đè).
- **Checkpoint Phase 5 khuyến nghị mặc định đổi thành `cvae_realphysics.pt`**: R²(FE, n=789 toàn bộ test set) = 0,999 (oracle)/0,992 (K=10), hit rate single-shot 98,7% - so với 0,20-0,32 (oracle)/CI âm (K=10) ở checkpoint tốt nhất trước đó.
- README.md/EXPERIMENT_LOG.md rút gọn đáng kể (EXPERIMENT_LOG: 708→70 dòng), chuẩn hóa dùng `-` thay cho em-dash `-` trên toàn bộ `.md`/`.py`/`.sh`.
- Khuyến nghị "K=10 giữ gần hết mức tăng R²" (dựa trên n=24) bị rút lại sau khi đánh giá ở n=789 cho thấy nó sụp đổ về CI âm/cắt-0 với mọi checkpoint trước differentiable-physics.

### 2026-07-25 - Rebuild dataset v2 + retrain Phase 4/5 (chưa gắn số phiên bản)

> Chi tiết đầy đủ (triệu chứng/nguyên nhân/sửa/số liệu): [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Mục này chỉ tóm tắt theo định dạng changelog.

#### Added
- `analysis/scripts/generate_production_batch.py` - sinh raw sample song song (`multiprocessing.Pool`), độc lập với `pipeline/phase2_multi_batch/`, dùng range tham số theo seed (`SEED_PARAM_RANGES`).
- `analysis/scripts/assemble_phase3_v2.py` - gộp pool raw mới + pool sạch cũ, build/split/augment thành dataset v2.
- `outputs/phase4/surrogate_v2.pt`, `outputs/phase5/cvae_v2_base.pt`/`cvae_v2_finetuned.pt` - checkpoint retrain trên dataset v2.
- `--ckpt` (`evaluate.py`), `--src/--dst/--eval-report` (`export_for_phase5.py`) - tham số CLI, trước đó hardcode hoàn toàn.

#### Fixed
- **Range tham số quá rộng cho `reentrant_bowtie`/`hexagonal`** trong script sinh dữ liệu mới - dùng nguyên `ACTIVE_PARAMETERS` (0,3-0,7/0,1-0,4) thay vì dải hẹp lịch sử (0,50-0,58/0,28-0,38) khiến yield sập xuống 1,4%.
- **`beta=0,8` thay vì mặc định production `1,0`** trong script sinh dữ liệu mới (nghiêm trọng hơn) - khiến `reentrant_bowtie` suy biến hoàn toàn (kẹt ~13/150 iteration). Production pipeline gốc (`pipeline/phase2_multi_batch/`) không bị ảnh hưởng - không bao giờ set `beta` nên luôn dùng đúng mặc định `1.0`.
- Mất ~10.700 mẫu raw (~3,5h compute) do ghi tạm vào `/tmp` (tmpfs, xoá qua reboot) - từ nay mọi output raw ghi vào `outputs/` (đĩa thật).

#### Changed
- **Dataset production thay hoàn toàn**: `outputs/phase3/{train,val,test,dataset_64}.npz` - train 57.216 mẫu (từ 22.080), val/test 2.044 mẫu (từ 789). Bộ cũ backup nguyên vẹn tại `outputs/phase3_backup/`.
- Phase 4 surrogate retrain trên dataset v2: R²(v12/v21/volfrac) 0,922/0,889/- → **0,974/0,964/0,983**.
- Phase 5 cVAE retrain trên dataset v2 (`cvae_v2_finetuned.pt`, recipe 2-stage bắt buộc - train from-scratch với real-physics loss ngay từ đầu thất bại): R²(FE, n=300)=0,9953, hit rate single-shot=99,7% - ngang ngửa `cvae_realphysics.pt` (checkpoint cũ, R²=0,9984/98,3%, đo lại trên cùng test set v2). Cả 2 checkpoint được giữ, khuyến nghị dùng bản v2 cho nhất quán với dataset hiện tại.

### 2026-07-29 - Audit độc lập toàn diện + fix (chưa gắn số phiên bản)

> Chi tiết đầy đủ (bằng chứng số, severity, đề xuất): [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Mục này chỉ tóm tắt theo định dạng changelog.

#### Fixed
- **Bug scaling `Q` trong `compute_homogenized_tensor()`** - `k_e` thiếu chia `E0` (double-counting, `KE` đã chứa sẵn `E0`) + chia dư `/(nelx*nely)` khi `U0` đã dùng tọa độ chuẩn hóa `[0,1]` (diện tích miền=1). `Q_cũ = Q_đúng * E0/nele` - đã kiểm chứng bằng bài test chuẩn "ô cơ sở đặc → Q=D" (2 mesh size, rtol<1e-6). KHÔNG ảnh hưởng `ν₁₂/ν₂₁` (bất biến với hệ số nhân đồng đều), nhưng ngưỡng phạt cứng `delta` trong `objectives/auxetic.py` từng lệch lên **~46%** độ cứng thay vì 10% thiết kế - giờ đúng lại.
- Checkpoint/surrogate mặc định lỗi thời ở 5 script Phase 5 (`sample.py`, `best_of_n_eval.py`, `coverage_eval.py`, `losses.py`, `self_play.py`) - đổi sang `cvae_v2_finetuned.pt`/`surrogate_for_phase5_v2.pt`; thêm banner cảnh báo runtime vào `train.py` cho tổ hợp `val_loss`+`lambda_real_physics=0` (đã 2 lần xác nhận chọn nhầm checkpoint).
- `torch`/`scikit-learn`/`pandas`/`Pillow` thiếu trong `requirements.txt`/`pyproject.toml` dù Phase 3-5 phụ thuộc hoàn toàn - `pip install -r requirements.txt` trước đây không đủ để chạy Phase 3-5.
- `beta` không được truyền từ Phase 1 sang Phase 2 (`DEFAULT_FIXED` thiếu key) - giờ tường minh `beta=1.0` (khớp dữ liệu production thật, không đổi hành vi). `fixed_parameters` (quyết định active/fixed của Phase 1) bị bỏ quên khi gọi `run_batch_from_design()` trong `main.py` - đã nối lại.
- `scan_dataset.py` hardcode `N_BATCHES=8`, bỏ sót batch 9-11 (kể cả bản rebuild sửa lỗi `dQ`) - giờ tự động quét mọi thư mục `batch_*`.
- `outputs/phase3/split_report.json` mô tả dataset CŨ (5.258 mẫu) trong khi `train.npz` thật đã là dataset v2 (57.216 mẫu, seed lệch nặng về `hourglass`) - script một lần tính lại khớp dữ liệu thật (đã xóa 2026-08-15 sau khi hoàn thành việc regenerate).

#### Added (tính năng thử nghiệm, TẮT mặc định)
- `simp/core/filter.py::apply_heaviside_projection()` - robust/minimum-length-scale projection (Wang-Lazarov-Sigmund 2011), qua `run_simp(params={'projection': 'heaviside', ...})`. Pilot 3 seed manufacturability thấp: kết quả lẫn lộn (cải thiện rõ `hexagonal`, không cải thiện `four_circle`/`grid_circular_voids`) - có xấp xỉ đã biết (ràng buộc thể tích trong OC nhắm x̃ trước-projection).
- `simp/runner.py::nudge_disconnected_islands()` - heuristic hạ mật độ đảo vật liệu rời rạc mỗi N vòng lặp, qua `params={'enforce_connectivity': True, ...}`. Pilot: cải thiện mạnh `grid_circular_voids`/`nine_circle` (0/3→3/3 manufacturable) nhưng gây regression 1 case ở `four_circle`.
- `simp/objectives/auxetic.py::compute_auxetic_normalized_objective()` - biến thể `c = Q12/sqrt(Q11*Q22)` (bị chặn tự nhiên [-1,1] bởi Cauchy-Schwarz/PSD của Q), qua `params={'objective_variant': 'normalized'}`. Pilot 3 seed × 2 điều kiện: thắng rõ 4/6, đặc biệt ngăn sụp cấu trúc mà `q12` gốc vẫn gặp phải (`hexagonal`).
- Test hồi quy/gradient mới cho cả 4 mục trên trong `tests/test_core_smoke.py` (8 test).

#### Known issue (không sửa trong lần này, đã ghi chú lý do)
- `pipeline/phase3_dataset/finalize_dataset.py` chưa dùng chung logic lọc `is_connected`/`osc_score` với `analysis/scripts/assemble_phase3_v2.py::is_clean()` - `dataset_64.npz` không lưu `sample_id` nên thiếu khóa join an toàn. Xem `pipeline/phase3_dataset/README.md`.

### 2026-07-30 - Bug scaling `Q` (A1) + MMA optimizer cho `hexagonal`/`hourglass` (chưa gắn số phiên bản)

> Chi tiết đầy đủ (bằng chứng số, pilot N=100→N=400): [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Mục này chỉ tóm tắt.

#### Added
- `simp/mma_runner.py::run_simp_mma()` - `nlopt.LD_MMA` (Svanberg 1987) thay `oc_update()`, dùng cho `hexagonal`/`hourglass` (yield 76,5%→99,8% và 76,8%→88,0%, N=400, CI không chồng lấn). `reentrant_bowtie` giữ nguyên `gate`/OC - MMA thất bại nặng trên seed này (90,0%→0,0%, rơi vào local optimum sai dấu).
- `analysis/scripts/generate_production_batch.py::SEED_OPTIMIZER` - dispatch optimizer theo seed, `--optimizer {auto,gate,mma}` (CHƯA chạy để regenerate dataset 57k production).
- `nlopt>=2.7` vào `requirements.txt`/`pyproject.toml`.
- `tests/test_mma_runner.py` (5 test).

#### Fixed
- Bug scaling `Q` (A1) trong `compute_homogenized_tensor()` - `Q_cũ = Q_đúng × E0/nele`, khiến ngưỡng phạt stiffness hiệu lực ~46% thay vì 10% thiết kế suốt lịch sử dự án (vô tình làm regularizer cho `hexagonal`). Sau khi sửa đúng, lộ ra bug thứ 2 (`X_MIN=0,0` thay vì `0,001`, trạng thái hấp thụ toán học) - sửa cả 2, tỷ lệ sụp cấu trúc 16,7%→0% (N=400).

#### Known issue → [ĐÃ GIẢI QUYẾT, xem "Added" ở trên] (sửa 2026-08-05, dòng này trước đó mâu thuẫn với chính mục Added)
- `hexagonal` yield thấp hơn lịch sử trong khung OC/gate sau khi sửa 2 bug trên (~80,5% so với 93,9%) - đã thử 3 hướng khắc phục THAM SỐ (không đổi optimizer), cả 3 đều phản tác dụng. Nhưng đổi hẳn optimizer (MMA, xem "Added" ngay trên) giải quyết triệt để (76,5%→99,8%, N=400) - dòng "known issue" gốc ở đây đã lỗi thời ngay từ khi CHANGELOG được viết. Khoảng trống còn lại thật: dataset 57k production hiện dùng sinh bằng code TRƯỚC MMA dispatch, chưa rebuild. Xem [docs/LIMITATIONS.md](docs/LIMITATIONS.md) mục #16 (đã sửa 2026-08-05).

### 2026-07-31 - Phase 7 (active-learning) kết luận + Phase 8.1/8.3 (chưa gắn số phiên bản)

> Chi tiết đầy đủ: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), [`outputs/phase5/reports/final_status_report_2026-07-31.md`](outputs/phase5/reports/final_status_report_2026-07-31.md). Mục này chỉ tóm tắt.

#### Added
- `pipeline/phase5_cvae/active_learning.py` (roadmap 7.1-7.3+7.5) - vòng lặp active-learning thật: tìm vùng yếu qua `coverage_eval()` → sinh+FE-verify ứng viên → lọc manufacturable → fine-tune surrogate rồi cVAE → đo lại coverage → quyết định dừng.
- `analysis/scripts/skfem_homogenization.py` (roadmap 8.1) - implementation FE/homogenization độc lập trên `scikit-fem` (không tái dùng `simp/core/pbc.py`/`compute.py`), cùng `analysis/scripts/run_independent_fe_check.py` (script so sánh).
- `analysis/scripts/build_design_library.py` (roadmap 8.3) - thư viện thiết kế tuyển chọn qua `best_of_n_eval.py`, xuất `outputs/phase5/design_library/`.
- `scikit-fem>=10.0` vào `requirements.txt`/`pyproject.toml` (optional, nhóm `analysis`).
- `tests/test_phase5_active_learning.py` (12 test), `tests/test_skfem_homogenization.py` (7 test).

#### Changed
- README §Giới hạn Đã biết #15 (đối chiếu FE độc lập): GIẢI QUYẾT PHẦN LỚN - `scikit-fem` khớp engine nội bộ tới ~1e-9 (R²=1,000000, n=24 mẫu thật), xác nhận đúng vật lý, không chỉ nhất quán nội bộ.
- README bảng roadmap Phase 6-8 viết lại theo đúng đánh số roadmap chi tiết (trước đó "Phase 6" trong bảng lẫn với nâng cấp cGAN/diffusion tuỳ chọn của Phase 5).

#### Known result (âm tính, có giá trị phương pháp luận)
- Active-learning loop chạy production trên `cvae_realphysics.pt` (đã gần mức trần: `frac_manufacturable`=0,35, hit-rate 98,7%) - round 1 làm `mean_abs_error` TỆ ĐI (0,0246→0,0686), không cải thiện. Kết luận: giữ nguyên checkpoint production, không promote checkpoint mới.

### 2026-08-02 đến 2026-08-09 - Optional multi-condition (Pha A) + OOD baseline + tái cấu trúc tài liệu (chưa gắn số phiên bản)

> Chi tiết đầy đủ: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) các mục 2026-08-02 đến 2026-08-05. Mục này chỉ tóm tắt theo định dạng changelog.

#### Added
- **Optional multi-condition (Pha A, PR #13, nhánh `feature/optional-multi-condition`)**: mở rộng cVAE để nhận thêm điều kiện tuỳ chọn (`volfrac`, `void_size_frac`) bên cạnh `v12`/`v21`, cùng novelty/diversity metric và điểm số tổng hợp accuracy/manufacturability/aesthetic cho việc chọn ứng viên - đã fix 2 bug thật phát hiện qua audit kiểu phản biện sau khi rebase lên `main` (một bug nằm ngay trên lệnh khuyến nghị trong README).
- `analysis/scripts/` - script so sánh cVAE với retrieval baseline ngoài phân phối train (OOD, Thử nghiệm E): cVAE thắng rõ trên sign-flip Poisson (R²=0,42 so với 0,06) nhưng cả hai đều không ngoại suy được về biên độ vượt khoảng train - kết quả có tính hai mặt, không phải chiến thắng toàn diện, tránh diễn giải quá mức.
- `docs/slides/` - slide deck trình bày dự án AuxForge.

#### Changed
- Phát hiện `hit_rate` (best-of-N) là metric yếu khi so với retrieval baseline trên test set hiện tại - bổ sung novelty/diversity metric để đánh giá đầy đủ hơn thay vì chỉ dựa vào hit_rate.
- Tách `docs/PIPELINE.md` ra thành file riêng khỏi tài liệu gộp trước đó; xoá script gamma-sweep không còn dùng.
- Đồng bộ đường dẫn trong notebooks sang dùng `utils.REPO_ROOT` thay vì đường dẫn tương đối cứng.
- README §Giới hạn Đã biết cập nhật phản ánh các phát hiện audit/OOD ở trên.

### 2026-09-11 - Exp 3 MLP/KAN ablation và hiệu chỉnh metric

#### Added
- Thêm nhánh `--use-mlp-head` để đối chứng `nn.Linear` với
  `EfficientKANLinear` trên cùng CVAE.
- Bổ sung `lambda_volfrac` dùng nhãn `volfrac_achieved` thực từ dataset,
  đối xứng transpose ở output decoder, seed reproducibility và metadata
  `use_kan`/`enforce_symmetry` trong checkpoint.

#### Results
- Exp 3 MLP + Physics 150 epoch: `R²(v12)=0,432`, `R²(v21)=0,087`,
  `R²(volfrac)=-2,483`, `R²(FE,v12)=-9,834` trên cùng test protocol.
- KAN + Physics v2 đo lại cùng protocol: `0,836`, `0,476`, `-0,221`,
  `0,581` tương ứng.
- KAN v2 chưa retrain lại với code mới; không xem chênh lệch này là claim
  nhân quả cuối cùng cho tới khi chạy cặp retrain đồng nhất.

### 2026-08-23 - KAN regression head + WIRE INR decoder (Task 1/2, nhánh `substrate-material`)

> Chi tiết đầy đủ + số liệu: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) mục 2026-08-23, [docs/archive/task_progress.md](docs/archive/task_progress.md).

#### Added
- `pipeline/phase5_cvae/model.py`: `WireContinuousDecoder` + `ComplexGaborActivation` (Gabor Wavelet phức) - WIRE INR decoder sinh mật độ ρ ∈ [0,1] tại tọa độ liên tục (x,y) ∈ [-1,1]², resolution-agnostic. `CVAE` thêm flag `decoder_type="conv"|"wire"` (mặc định `conv`, tương thích ngược) và `generate(resolution=...)` (128²/256²/512²) giữ API đầu ra `(B,1,H,W)`.
- `train.py`: `--decoder-type --wire-hidden-dim --wire-omega0 --wire-s0` (lưu vào checkpoint); `sample.py`: `--resolution`.
- Loader cập nhật đọc `decoder_type` từ checkpoint: `sample.py::load_model`, `adversarial_dataset.py::load_cvae` (self-play/best_of_n), `verify_fe.py::load_cvae_checkpoint`.
- Tests: `TestWireDecoder` (forward/shape/unit range/resolution-agnostic/gradient flow) + cập nhật mock `tracking_generate` (tổng **625/625 pass**).

#### Changed
- **KAN-hóa bộ hồi quy cVAE (Task 1):** `Decoder.fc` chuyển từ `nn.Linear` sang `EfficientKANLinear` (khớp `Encoder.fc_mu`/`fc_logvar` đã KAN) - mọi `fc` của cVAE giờ là KAN. `resize_condition_dim_weights()` đã xử lý KAN (`base_weight`/`spline_weight`/grid).
- Training KAN chạy 5 lần: base val_loss R²(surrogate)=0,19; **fine-tune real-physics 2 vòng → `cvae_kan_realphysics_v2.pt` R²=0,64 (v12 0,82, v21 0,45), R²(FE thật)=0,8889** - vượt baseline Linear đo lại (0,29/0,35) gần 2×. DoN R² ≥ 0,990 chưa đạt (xem docs/archive/task_progress.md).

#### Fixed
- Phát hiện `outputs/phase5/evaluation_report.json` (2026-07-29, R²=0,85) **không tái lập** với code hiện tại (cùng checkpoint đo lại = 0,35 với surrogate v2, −1,81 với v1) - chưa tìm ra gốc rễ, khả năng report cũ sinh bằng code path khác; mọi so sánh R² cũ cần đo lại cùng code hiện tại (xem EXPERIMENT_LOG.md 2026-08-23).

## [1.4.0] - 2026-07-10

### Fixed
- **OC sqrt tranh cãi (Bug #1)**: Thêm tham số `use_sqrt=False` vào `oc_update()`. Mặc định `False` để khớp MATLAB (không sqrt). Khi cần Sigmund 2001 heuristic, có thể set `True`. Xem docs/summary.md và bug_reports.md.
- **Symmetrize K (Bug #6)**: Thêm `K_global = (K_global + K_global.T) * 0.5` trong `solve_fe()` - giống MATLAB, tăng ổn định số học.
- **xPhys unfiltered cho First_Obj (Bug #4)**: Runner.py nay dùng unfiltered `xPhys` từ OC update cho First_Obj, đúng với MATLAB behavior. Code cũ dùng filtered cho mọi objective.
- **rho0 scaling (Bug #8)**: Thêm tham số `rho0` (mặc định 1.0) vào `solve_fe()`, `compute_homogenized_tensor()`, `compute_second_objective()`. MATLAB Second_Obj dùng `rho0=7850`, `E0=1` - Python giờ hỗ trợ đồng bộ.
- **Error handling (R3)**: Thêm `try/except` trong vòng lặp chính của `run_simp()`. Khi lỗi xảy ra, gán objective lớn + gradient mạnh để OC tránh điểm đó. Tự động dừng nếu có 5 lỗi liên tiếp.
- **Metadata reproducibility (R6)**: Thêm `metadata.json` vào mỗi output directory - lưu git hash, timestamp, version, params.

### Changed
- **`simp/core/oc.py`**: `oc_update()` signature thay đổi - thêm `use_sqrt=False`. Tham số mới, backward compatible.
- **`simp/core/solver.py`**: `solve_fe()` signature thay đổi - thêm `rho0=1.0`. Backward compatible.
- **`simp/homogenization/compute.py`**: `compute_homogenized_tensor()` signature thay đổi - thêm `rho0=1.0`. Backward compatible.
- **`simp/runner.py`**: Thêm `rho0` vào params extraction.

## [1.3.0] - 2026-07-10

### Added
- **`pipeline/phase2_multi_batch/`**: Multi-batch adaptive sampling pipeline - a complete module for intelligent, sequential design space exploration.

  - **`params.py`**: Configuration layer with `BatchConfig` and `PipelineConfig` dataclasses, parameter management (fixed vs active), and JSON serialization.
    - `SamplingStrategy` enum: `SOBOL`, `LHS`, `OPTIMIZED_LHS`, `RANDOM`
    - `BatchMode` enum: `EXPLORE`, `REFINE`, `TARGETED`, `VALIDATE`
    - `load_phase1_params()` - extract parameter ranges from Phase 1 summary JSON
    - `prepare_output()` - create output directory structure with metadata

  - **`sampling.py`**: Design generation engine supporting three strategies:
    - **Sobol sequence** (`sobol`) - deterministic low-discrepancy sequence via SciPy, preferred for initial exploration
    - **Latin Hypercube Sampling** (`lhs`) - stratified random sampling via pyDOE
    - **Optimized LHS** (`optimized_lhs`) - LHS with SPSA-like pairwise correlation minimization
    - Outputs a clean `pandas.DataFrame` with `param_ranges` columns + `seed`, `objective` columns for SIMP dispatch

  - **`runner.py`**: Batch execution harness wrapping `run_single_simp`:
    - Parallel execution via `concurrent.futures.ProcessPoolExecutor` (up to 4 workers default)
    - Per-sample JSON result aggregation into batch-level summary
    - Error tolerance: individual sample failures don't crash the batch
    - Batch results saved as `batch_{id}_results.json` (list of `{sample_id, seed, objective, params, v12, v21, obj_value, success}`)

  - **`coverage.py`**: N-dimensional coverage analysis engine:
    - `coverage_report()` - discretise property space into ND bins, compute per-bin statistics
    - `find_sparse_regions()` - identify bins with low sample density for targeted follow-up
    - `compare_batches()` - measure coverage improvement between consecutive batches
    - Tracks `v12`, `v21`, `obj_value` as property dimensions by default

  - **`adaptive.py`**: Decision-making orchestrator that closes the loop:
    - `decide_next_action()` - analyzes accumulated batch summaries against configurable thresholds
    - **Stop**: if objective hasn't improved for N batches AND coverage is adequate
    - **Expand**: if sparsity > 30% of property space → sample new seeds/objectives
    - **Refine**: if sparsity < 10% but best objective still far from theoretical → narrow param bounds
    - Returns structured `{'action', 'reason', 'next_config', 'coverage'}` dict

  - **`visualize.py`**: HTML report generators for human-readable monitoring:
    - `generate_coverage_html()` - interactive coverage grid with sparse region highlights
    - `generate_batch_progression_html()` - side-by-side coverage comparison across batches

  - **`main.py`**: CLI entry point orchestrating the full loop:
    ```
    python -m pipeline.phase2_multi_batch.main --phase1-summary <path> [options]
    ```
    - `--phase1-summary` - path to Phase 1 summary JSON or directory
    - `--max-batches` - iteration limit (default: 5)
    - `--n-batch1` - sample count for first batch (default: 120)
    - `--strategy` - sampling strategy: `sobol`, `lhs`, `optimized_lhs`
    - `--skip-run` - dry-run mode (mock results) for testing
    - `--resume` - continue from a previous `decision_log.json`
    - `--only-report` - regenerate HTML reports from existing results
    - `--seeds` / `--objectives` - filter which seed shapes and objectives to include
    - Generates `decision_log.json`, per-batch summaries, and HTML coverage reports

### Changed
- **`pipeline/params.py`**: Added `multi_batch` key to `REFINED_PARAMETERS_TEMPLATE` for cross-phase data handoff (active + fixed param metadata).

### Technical highlights
- **Zero new external dependencies**: uses only `scipy.stats.qmc`, `scipy.optimize`, `numpy`, `pandas` - all already in `requirements.txt`
- **Backward compatible**: existing Phase 1, Phase 2 pipelines unaffected - `multi_batch/` is a standalone optional addition
- **Easy extension**: new sampling strategies can be added by extending `generate_design()`; new decision heuristics by extending `decide_next_action()` - both accept pluggable callables
- **All HTML reports self-contained**: single-file, interactive, no server needed

## [1.2.2] - 2026-06-15

### Fixed
- **Review-driven cleanup**: Toàn bộ các vấn đề từ review_output.md đã được khắc phục:
  - **README/simp/README.md**: Sửa công thức auxetic từ `c = ν₁₂ = −Q₁₂/Q₂₂` → `c = Q₁₂` (có penalty stiffness).
  - **README tree**: Sửa version metadata (1.1.0→1.2.1), sửa mô tả auxetic objective, thêm phase2_tuning.py vào tree, xóa tham chiếu notebooks/ không tồn tại.
  - **simp/README.md**: Sửa default beta (0.85→0.8), beta_second (1.0→100.0), sửa công thức Poisson ratio trong CSV description, xóa tham chiếu notebooks/.
  - **hourglass_seed rotation**: Implement xoay tọa độ (giống các seed khác) thay vì bỏ qua tham số rotation_deg.
  - **NaN guard trong runner.py**: Khởi tạo sẵn c, Q, v12, v21 trước vòng lặp để tránh lỗi nếu break sớm do NaN.
  - **Duplicate line**: Xóa dòng "# Print progress" trùng trong phase1_screening_parallel.py.
  - **Version đồng bộ**: Cập nhật simp/__init__.py (1.2.1), pyproject.toml (1.2.1), analysis/_version.py (1.2.1).
  - **requirements.txt**: Clean up - xóa CUDA/torch/smolagent/HuggingFace dependencies, chỉ giữ numpy/scipy/matplotlib.
  - **second_obj iteration < 20 → <= 20**: Sửa để khớp MATLAB dùng `loop <= 20`.
  - **analysis CLI**: Thêm 'auxetic' vào choices của --objective (trước đây chỉ hỗ trợ 'first', 'second').
  - **main.py**: Thêm --version, --list-seeds, --verbose flags (đã ghi trong CHANGELOG 1.1.0 nhưng chưa implement).
  - **convergence.py**: Thêm docstring giải thích heuristic `obj_converged and change <= tol_change * 2`.
  - **oc.py**: Thêm docstring giải thích approximation (Q cũ trong stiffness constraint).
  - **config.py**: Thêm docstring ghi nhận hai interface params song song.
  - **utils.py**: Thêm docstring ghi chú resolve_phase1_dir không được dùng.
  - **auxetic_first_second_representative.json**: Thêm _note về placeholder.
  - **CHANGELOG.md**: Sửa ngày [1.0.0] từ "2026-04-xx" → "2026-04-15".
  - **Magic numbers**: Thay eps=1e-12 bằng local constant self-documenting.
  - **NaN check optimization**: Dùng `np.isnan(np.sum(xPhys))` thay vì `np.isnan(np.mean(xPhys))`.
  - **Core smoke tests**: Thêm 31 tests cho FEM, Material, Filter, Solver, OC, Objectives, PBC, Homogenization, Runner.

## [1.2.0] - 2026-06-04

### Added
- **`pipeline/phase2_tuning.py`**: Phase 2 - Parameter tuning với các thuật toán tối ưu hóa toàn cục.
  - `differential_evolution` (DE): Global search robust, sử dụng `scipy.optimize.differential_evolution`
  - `shgo` (Simplicial Homology Global Optimization): Phương pháp global thay thế
  - `basinhopping`: Stochastic global search + local refinement (L-BFGS-B)
  - `refine`: Local refinement (L-BFGS-B) từ Phase 1 best points
  - Tự động ghi log JSON + CSV cho mỗi eval, lưu lịch sử hội tụ
  - Hỗ trợ chạy đơn lẻ hoặc quét toàn bộ combo (seed × objective)
- `pipeline/__init__.py`: Export Phase 2 symbols

## [1.2.1] - 2026-06-07

### Changed
- **`simp/seeds/hourglass.py`**: Đã viết lại hoàn toàn để tạo hình đồng hồ cát theo đúng MATLAB (`topK_Hourglass.m`)
  - Góc nghiêng 50° từ eo ra đỉnh/đáy
  - Bề rộng eo = nelx/14
  - Mật độ vùng lõm = volfrac/2 ("soft void", không phải 0)
  - Hàm seed nhận (nelx, nely, volfrac, rotation_deg) để khớp MATLAB
- **`simp/objectives/second_obj.py`**: Thêm first-20-iter scaling và sửa hệ số phạt để khớp MATLAB (`topK_Hourglass_New_obj.m`)
  - Thêm bộ scale `(1 - 0.02*iteration)` cho 20 iteration đầu (giảm dần ảnh hưởng của Q₁₁+Q₂₂)
  - Sửa `beta_second` mặc định từ 1.0 → 100.0 để khớp MATLAB penalty=100
- **`simp/config.py`**: Đã chuẩn hoá các tham số mặc định cho MATLAB compatibility
  - `beta` thay đổi từ 0.85 → 0.8 (khớp MATLAB First_Obj hardcoded 0.8)
  - `beta_second` thay đổi từ 1.0 → 100.0 (khớp MATLAB Second_Obj penalty=100)
- **`simp/runner.py`**: Cập nhật để hỗ trợ các thay đổi trên
  - Truyền `volfrac` (thay vì `void_size_frac`) cho seed hourglass
  - Thêm hỗ trợ truyền `Q` và `delta` vào `oc_update` để ràng buộc stiffness trong First_Obj
  - Cập nhật giá trị mặc định cho `beta` và `beta_second`
- **`simp/core/oc.py`**: Đã mở rộng để hỗ trợ ràng buộc stiffness bổ sung (tùy chọn)
  - Thêm tham số `Q` và `delta` để kiểm tra `Q[0,0] >= delta && Q[1,1] >= delta` trong vòng bisection
  - Khi được cung cấp, mô phỏng đúng MATLAB `mean(xPhys)>volfrac && Q11>=delta && Q22>=delta`
  - Giữ nguyên công thức OC chuẩn có sqrt (Sigmund 2001), tuỳ chọn có thể bỏ qua nếu cần

### Fixed
- Đã sửa chữ ký hàm `hourglass_seed` để nhận `volfrac` thay vì `void_size_frac` để khớp MATLAB reference
- Đã sửa giá trị mặc định `beta` và `beta_second` trong config để khớp MATLAB hardcoded values
- Đã sửa lỗi trong second objective (thiếu first-20-iter scaling và hệ số phạt sai lệch 100 lần)

### Removed
- Không có.

## [1.1.0] - 2026-05-21

### Added
- `analysis/` module: new structured analysis pipeline replacing `src/`.
  - `analysis/dataset.py`: Dataset overview, convergence metrics, auxetic classification.
  - `analysis/image.py`: Image quality metrics (binary rate, edge density, noise, symmetry).
  - `analysis/report.py`: Self-contained HTML report generation.
  - `analysis/cli.py`: CLI interface for analysis commands.
- `simp/core/convergence.py`: Dedicated `ConvergenceChecker` class with design change and objective stability criteria.
- `pyproject.toml`: Standard Python packaging with setuptools.
- `.gitignore`: Comprehensive Python project ignore rules.
- `requirements-core.txt`, `requirements-analysis.txt`, `requirements-dev.txt`: Split dependency files.
- `Makefile`: Common commands (install, test, lint, format, run).
- `CHANGELOG.md`: This file.

### Changed
- **`simp/io/logger.py`**: Buffered CSV writing for performance (configurable buffer size).
- **`simp/io/visualizer.py`**: Full docstrings, type hints, cleaner API.
- **`simp/core/solver.py`**: Reduced sparse↔dense conversions; all submatrix operations stay in sparse format.
- **`simp/config.py`**: Full `SimpConfig` dataclass with `__post_init__` validation.
- **`simp/runner.py`**: Integrated `ConvergenceChecker`; cleaner loop structure; returns results dict.
- **`simp/main.py`**: Added `--version`, `--list-seeds`, `--verbose` flags; seed registry; error handling.
- **`simp/__init__.py`**: Added `__version__`, `__author__`, `__license__`.
- **`simp/core/__init__.py`**: Exports `ConvergenceChecker`.
- **`simp/io/__init__.py`**: Clean `__all__` exports.

### Removed
- `src/` directory (replaced by `analysis/` module).
- `requirements.txt` (split into `requirements-*.txt` files).

### Fixed
- Logger no longer opens file on every `log()` call (buffered I/O).
- Solver no longer converts sparse submatrices to dense unnecessarily.
- Config validation catches invalid parameters early.

## [1.0.0] - 2026-04-15

### Added
- Initial SIMP topology optimization engine.
- Core modules: fem, filter, pbc, solver, oc.
- Material properties, homogenization, objective functions.
- Seed patterns: circle, square, hourglass, four_circle, hexagonal, nine_circle, cross, grid_voids, small_cross, half_circle.
- CSV logging and PNG visualization.
- HTML workflow documentation (workflow.html, workflow_vi.html).
- MATLAB reference implementations in `data/`.
- Analysis notebooks in `notebooks/`.
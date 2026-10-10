# Ghi chú phục vụ viết báo (N1, N2 - từ 2026-10-10)

> Sổ ghi chép cho bản thảo: phương pháp, số liệu (kèm đường dẫn file), tài liệu cần trích dẫn,
> câu chữ gợi ý và giới hạn. Cập nhật sau mỗi thí nghiệm. Tiêu chí đặt trước nằm ở `docs/plan.md`
> mục N1, N2; tra cứu chi tiết ở `docs/de_xuat_cai_thien_ket_qua.md`.

---

## A. Tài liệu cần trích dẫn trong bản thảo (đa số chưa có trong `refs.bib` - kiểm DOI trước khi thêm; danh sách đầy đủ mọi tài liệu đã tham khảo ở mục D)

| Khóa gợi ý | Tài liệu | Dùng ở đâu trong bài |
|---|---|---|
| `hammond2025ssp` | A. M. Hammond, A. Oskooi, I. M. Hammond, M. Chen, S. E. Ralph, S. G. Johnson, "Unifying and accelerating level-set and density-based topology optimization by subpixel-smoothed projection", Opt. Express 33(16):33620-33642, 2025. arXiv:2503.20189 | Related work: chiếu nhị phân khả vi (gần nhất với P1.1e); Discussion: hướng thay tanh-Heaviside |
| `arrieta2025lengthscale` | R. Arrieta, G. Romano, S. G. Johnson, "Hyperparameter-free minimum-lengthscale constraints for topology optimization", arXiv:2507.16108 (2025); SMO 2026 (s00158-026-04388-6) | Manufacturability: ràng buộc nét tối thiểu |
| `wang2011projection` (ĐÃ CÓ trong refs.bib) | F. Wang, B. S. Lazarov, O. Sigmund, "On projection methods, convergence and robust formulations in topology optimization", SMO 43:767-784, 2011 | Robust formulation co/gốc/giãn (E1) |
| `sigmund2009manufacturing` | O. Sigmund, "Manufacturing tolerant topology optimization", Acta Mech. Sin. 25:227-239, 2009 | Robust formulation gốc |
| `andreassen2014extremal` | E. Andreassen, B. S. Lazarov, O. Sigmund, "Design of manufacturable 3D extremal elastic microstructure", Mech. Mater. 69:1-10, 2014 | Auxetic + robust → in được; đối chiếu E1 |
| `sigmund2000auxetic` | O. Sigmund, "A new class of extremal composites", J. Mech. Phys. Solids 48:397-428, 2000 | Bản lề mảnh ở auxetic tối ưu topo |
| `poulsen2002hinge` | T. A. Poulsen, "A simple scheme to prevent checkerboard patterns and one-node connected hinges in topology optimization", SMO 24:396-399, 2002 | Bản lề 1 nút là artifact rời rạc hóa |
| `kabel2015composite` | M. Kabel, D. Merkert, M. Schneider, "Use of composite voxels in FFT-based homogenization", CMAME 294:168-188, 2015 | Sai số biên bậc thang trong đồng nhất hóa pixel |
| `zhou2015minlength` | M. Zhou, B. S. Lazarov, F. Wang, O. Sigmund, "Minimum length scale in topology optimization by geometric constraints", CMAME 293:266-282, 2015 | Ràng buộc nét tối thiểu |
| `cool2025connectivity` | Cool, Aage, Sigmund, "A practical review on promoting connectivity in topology optimization", arXiv:2501.09402 | Ràng buộc liên thông (r ≥ 10) |
| `chen2020glonet` | M. Chen, J. Jiang, J. A. Fan, "Design space reparameterization enforces hard geometric constraints in inverse-designed nanophotonic devices", arXiv:2007.12991 | Robust formulation trong tối ưu qua mạng nơ-ron (trọng số 0,5/0,25/0,25) |
| `bourdin2001filters` (ĐÃ CÓ trong refs.bib) | B. Bourdin, "Filters in topology optimization", IJNME 50:2143-2158, 2001 | Density filter = một phần tham số hóa thiết kế (F) |
| `bruns2001filter` | T. E. Bruns, D. A. Tortorelli, "Topology optimization of non-linear elastic structures and compliant mechanisms", CMAME 190:3443-3459, 2001 | Density filter (F) |

---

## B. Phát hiện N1 (2026-10-10) - "khe giữa lưới verify và hiện thực hóa"

### B1. Phương pháp chẩn đoán
- Hiện thực hóa R(n, s) của ảnh nhị phân: làm mượt Gauss tuần hoàn σ (đơn vị phần tử lưới 50, cố
  định vật lý) → lấy mẫu song tuyến tại tâm phần tử lưới n×n dịch s → ngưỡng 0,5 → FE. R(N, 0) tái
  tạo đúng 100% ảnh gốc (kiểm cho ảnh 64² và 50²).
- Độ nhạy dịch S = std ν12 qua 8 phép dịch s ∈ {0; 0,5}² ∪ {0,25; 0,75}².
- Thiết kế được chấm = ảnh 50² mà FE verify thật sự giải (với cVAE: ảnh 64² đã resize nearest).
- Script: `/tmp/.../n1_diag.py` (bản chính thức nằm trong `outputs/phase5/plan_v3/n1/eval_pilot.py`,
  hàm `realize`). Bản torch khả vi: `pipeline/phase5_cvae/realization.py`.

### B2. Số liệu (IN100, n = 100 mỗi phương pháp, thiết kế cuối guarded)
| | MAE ν12 lưới 50² (verify) | MAE ν12 làm mượt R(200), σ = 0,5 | S trung vị | Chế tạo 4 hướng |
|---|---|---|---|---|
| cVAE λ = 0 | 0,0046 | 0,0164 | 0,031 | 0,31 |
| K1 λ = 0,1 | 0,0064 | 0,0195 | 0,029 | 0,72 |
| SIMP full | 0,0071 | 0,0187 | 0,038 | 0,72 |

- Spearman(S, |ν_verify − ν_R200|) = 0,52 (p ≈ 1e-22), trong từng phương pháp 0,49-0,55 → độ nhạy
  dịch (8 FE ở 50²) dự báo được khe lưới thô ↔ hiện thực hóa mịn.
- S không khác giữa chế tạo được / không (p = 0,25); SIMP nhạy dịch hơn cVAE (H1 đặt trước trượt).
- Hiệu ghép cặp so với cVAE λ = 0 (bootstrap 5000): K1 +37% [CI +0,0003; +0,0033] trên 50² nhưng chỉ
  +19% [+0,0010; +0,0054] trên R(200) σ = 0,5; SIMP +53% → +14% [−0,0005; +0,0052] (hết ý nghĩa).
  → **Lợi thế độ chính xác của cVAE so với SIMP co lại và mất ý nghĩa thống kê trên hiện thực hóa
  mịn.** Đây là điểm reviewer sẽ hỏi; cần báo trong bài.

### B3. Refine phần lớn khớp lưới 50² (ảnh 64² gốc, FE lưới 64² và kron 128²)
| | MAE ν12 50² (verify) | MAE ν12 64² gốc | MAE ν12 kron 128² |
|---|---|---|---|
| best-of-30 chưa refine | 0,0142 | 0,0242 | 0,0202 |
| sau refine (guarded) | 0,0046 (−68%) | 0,0189 (−22%) | 0,0149 (−26%) |
| K1 sau refine | 0,0064 | 0,0192 | 0,0159 |
Nguồn: `/tmp/.../n1_native64.json` (tạo lại bằng script cùng thư mục n1 nếu cần).
**Câu gợi ý:** "On a finer discretization the refinement gain shrinks from 68 % to 26 %; most of the
reported improvement is specific to the 50×50 verification mesh, the same objective–verification
gap the paper addresses, now at the level of discretization."

### B4. Hội tụ lưới (6 thiết kế IN100, `outputs/phase5/plan_v3/n1/mesh_convergence.json`)
- Biên bậc thang (kron) hội tụ đều 50 → 100 → 200 → 400 (vd #0: −0,7088, −0,7227, −0,7282,
  −0,7303); lệch verify ~0,022 cùng chiều ở cả 6 thiết kế.
- Biên bo nhẹ (σ = 0,25): trùng kron ở 50², 100²; ở 400² quay về sát giá trị 50² (±0,002), chưa hội
  tụ (800² vượt RAM).
- Khe kron vẫn có khi không có điểm chạm góc (0,0134 vs 0,0194, p = 0,13) → khe 0,015 trong
  `LIMITATIONS.md` là hiệu ứng biên bậc thang, không phải do bản lề.
- **Kết luận cho bài:** ν của một thiết kế pixel chỉ xác định tới ~0,02 tùy cách biểu diễn biên,
  ~4× MAE đang báo. Nên báo độ chính xác "tương đối với mô hình FE 50²" (đã có) + thêm cột lưới mịn.

### B5. Pilot objective N1 (IN100-B 20 condition, `outputs/phase5/plan_v3/n1/`)
| Nhánh (refined) | e_verify | e_mesh | e_real σ=0,5 | e_real σ=1,0 | Chế tạo |
|---|---|---|---|---|---|
| A λ = 0 | 0,0053 | 0,0136 | 0,0173 | 0,090 | 0,45 |
| D K1 | 0,0034 | 0,0144 | 0,0174 | 0,080 | 0,80 |
| N1 (verify + 4 dịch) | 0,0074 | 0,0158 | **0,0101** | **0,058** | 0,25 |
| C (FE 100²) | 0,0093 | **0,0076** | 0,0224 | 0,096 | 0,35 |
| N1 + C | 0,0096 | **0,0080** | 0,0197 | 0,069 | 0,30 |
- Quy luật: mỗi objective cải thiện đúng hiện thực hóa nó tối ưu; không nhánh nào trội mọi thước đo.
- N1 tổng quát hóa sang σ lớn hơn σ tối ưu (σ = 0,75/1,0: −37%/−36%, CI không chứa 0).

---

## C. N2 - cải thiện kết quả (xong 2026-10-10)

### C1. E1 robust formulation (phương pháp, để viết Methods)
- Loss = 0,5·MSE(ν; chuỗi verify) + 0,25·MSE(ν; bản giãn) + 0,25·MSE(ν; bản co). Bản co/giãn: ảnh
  decoder → Heaviside(β, 0,5) → làm mượt Gauss tuần hoàn σ = 1,0 phần tử → lấy mẫu lưới 50² →
  Heaviside(β, η), η = 0,25 (giãn) / 0,75 (co). Với biên thẳng, biên dịch σ·Φ⁻¹(η) ≈ ±0,67 phần tử.
- Khác robust formulation gốc: bản gốc là chính chuỗi verify (không làm mượt) để vẫn tối ưu đúng thứ
  được kiểm; trọng số trung bình (GLOnet) thay vì max để L-BFGS ổn định.
- Code: `realization.realize_shifted(eta=...)`, `tandem_lbfgs(robust_etas=, robust_sigma=)`, cờ
  `--robust-etas --robust-sigma`. Test: `tests/test_phase5_realization.py`,
  `tests/test_tandem_lbfgs.py::TestRealizationObjective`.

### C2. E2 refine đa độ phân giải
- 3 mức β đầu (1, 4, 16) giải FE lưới 50²; mức β = 64 cuối giải FE lưới 100² (cùng hình học, mỗi
  phần tử lặp 2×2). Cờ `--fe-upsample 2 --fe-upsample-last-only`.

### C3. Kết quả pilot (IN100-B 20 condition; `outputs/phase5/plan_v3/n2/score_pilot.log`)
Thiết kế guarded (luật pipeline), MAE ν12; Δ = hiệu ghép cặp so với A, CI bootstrap 95%.

| Nhánh | e_verify 50² | e_mesh 200² | e_real σ=0,5 | e_real σ=1,0 | Chế tạo 4 hướng | Nhận refine |
|---|---|---|---|---|---|---|
| A λ = 0 | 0,0053 | 0,0143 | 0,0165 | 0,090 | 0,45 | 95% |
| D K1 | 0,0034 | 0,0144 | 0,0174 | 0,080 | 0,80 | 100% |
| C FE 100² mọi mức | 0,0093 | 0,0082 | 0,0220 | 0,097 | 0,40 | – |
| E1 robust | 0,0081 | 0,0136 | 0,0163 | 0,071 (Δ −0,019 [−0,038; −0,001]) | 0,45 | 55% |
| E1 robust, refined chưa guard | 0,0098 | 0,0120 | 0,0163 | 0,033 (Δ −0,057 [−0,074; −0,040]) | 0,30 | – |
| E2 mức cuối 100² | 0,0104 | 0,0093 (Δ −0,005 [−0,008; −0,002]) | 0,0226 (Δ +0,006 [+0,002; +0,010]) | 0,094 | 0,50 | 80% |

Đánh giá theo tiêu chí đặt trước (`docs/plan.md` N2):
- **E1 không đạt**: chế tạo 0,45 < 0,60. Độ bền σ = 1,0 đạt, e_mesh không tệ hơn.
- **E2 đạt 2/3**: e_mesh tốt hơn, giữ 81% mức cải thiện của C. Tiêu chí chi phí chưa đo công bằng
  (E1 và E2 chạy song song, thời gian thực ~362 s và ~366 s; A chạy riêng ~200 s).

Chẩn đoán nguyên nhân (để viết Discussion):
1. **Đốm vật liệu "miễn phí"** ở E1: ảnh refined có trung bình 4,2 mảnh rời (A: 2,0; K1: 1,2), 14/20
  có nét mảnh. Bản gốc trong loss là chuỗi verify (resize nearest 64→50 bỏ 14 hàng/cột) và bản co/giãn
  đi qua làm mượt → đốm nhỏ không ảnh hưởng tới số hạng nào của loss, nên không bị phạt. Lại là khe
  giữa thứ được tối ưu và thứ được kiểm (ở đây: thứ kiểm chế tạo là ảnh 64², thứ kiểm FE là ảnh 50²).
2. **Luật guarded dùng thước đo dễ bị khai thác**: guard chấm bằng sai số lưới 50², nên loại 45%
  refine của E1 dù các refine đó bền hơn nhiều (σ = 1,0: 0,033 nếu không guard vs 0,071 có guard).
3. **Đánh đổi lưới mịn ↔ độ bền** (thấy ở C, E2, N1, E1): tối ưu trên lưới mịn của biên bậc thang
  khiến thiết kế dựa vào góc sắc pixel → mất khi biên bị bo; tối ưu độ bền thì ngược lại.
  **Câu gợi ý:** "Mesh convergence of the pixel staircase is not the right target, because nobody
  manufactures the staircase; robustness to plausible boundary realizations is."

### C4. Vòng 2: thiết kế qua bộ lọc (F) và robust formulation chuẩn (E1′) - pilot
Tác giả chốt thước đo chính = **e_real** (độ bền với biên thực tế, σ = 0,5 và 1,0, lưới 200²).

**Phương pháp F (để viết Methods):** thiết kế vật lý x_phys = Heaviside_η(lấy mẫu song tuyến tuần hoàn
lên lưới 50² của (ảnh decoder lọc Gauss tuần hoàn σ = 1 phần tử)), η = 0,5. Khi verify dùng ngưỡng cứng
(β = ∞); khi refine dùng chuỗi β {1, 4, 16, 64}. Bộ lọc là một phần của tham số hóa thiết kế, giống
density filter trong tối ưu topo (Bourdin 2001; Bruns & Tortorelli 2001; Wang et al. 2011). Chọn
best-of-30, refine, guard, kiểm chế tạo và ảnh lưu đều trên CÙNG x_phys - bỏ khe 64² (chế tạo) ↔ 50²
(FE) của pipeline cũ. Code: `realization.filtered_design`, cờ `--design-filter-sigma`.
**E1′** = F + loss robust 0,5·gốc + 0,25·giãn (η = 0,25) + 0,25·co (η = 0,75) từ cùng trường đã lọc,
guard theo loss robust.

Kết quả (IN100-B 20 condition, `outputs/phase5/plan_v3/n2/score_e1p_pilot.log`, guarded, Δ so với A):

| Nhánh | e_verify | e_mesh | e_real σ=0,5 | e_real σ=1,0 | sai số cặp σ=0,5 | Chế tạo 50² | Thời gian |
|---|---|---|---|---|---|---|---|
| A | 0,0053 | 0,0143 | 0,0165 | 0,090 | 0,031 | 0,50 | ~200 s |
| F | 0,0050 | 0,0147 | **0,0078** (−52% [CI −0,014; −0,004]) | **0,034** (−62%) | **0,015** (−52%) | **0,80** | 200 s |
| E1′ | 0,0228 | 0,0235 | 0,0253 (+53%) | 0,042 (−54%) | 0,052 | 0,80 | (song song) |

- E1′ trượt tiêu chí đặt trước (e_real σ=0,5 tệ hơn): co/giãn ±0,67 phần tử quá mạnh, phải khớp ν cho
  3 hình học cùng lúc nên bản gốc lệch.
- F (ablation không có tiêu chí đặt trước) không tệ hơn A ở thước đo nào, tốt hơn rõ ở độ bền, sai số
  cặp và chế tạo (ngang K1, nhưng **không mất độ chính xác** như K1), cùng chi phí. Đang xác nhận n = 100
  (E3-F, tiêu chí trong `docs/plan.md`).
- **Câu gợi ý (nếu E3-F xác nhận):** "Defining the design through a filter-and-project map, and
  verifying, checking manufacturability and plotting that same map, halves the error under realistic
  boundary realizations and raises manufacturability from 0.50 to 0.80 at no cost in nominal accuracy
  or compute - the same principle (optimize what is verified) applied to the definition of the design
  itself."
- Lưu ý vòng lặp: bộ lọc thiết kế và e_real cùng là làm mượt Gauss → E3-F báo thêm 2 thước đo độ bền
  độc lập (dịch lệch lưới; co/giãn η 0,35/0,65).

### C5. Xác nhận E3-F (n = 100, IN100 + IN100-C) - ĐẠT CẢ 5 TIÊU CHÍ ĐẶT TRƯỚC
Nguồn: `outputs/phase5/plan_v3/n2/final_e3f.log`, `e3f_scores_in100*.json`; run:
`e3f_in100_filter.json` (seed 123), `e3f_in100c_filter.json` (seed 789). A = `p1_7_c5_in100_bo30_nofp.json`,
`p1_9_r6_in100c_bo30_nofp.json`; SIMP = `p1_6a_simp_baseline_in100.json`, `p1_8_simp_in100c_full.json`.

**Bảng chính cho bài (IN100 / IN100-C, guarded, MAE ν12):**

| | verify 50² | lưới 200² | biên thực tế σ=0,5 | σ=1,0 | sai số cặp σ=0,5 | dịch lệch lưới | co/giãn đều | chế tạo | FE/target |
|---|---|---|---|---|---|---|---|---|---|
| cVAE + refine (cũ, A) | 0,0046 / 0,0043 | 0,0141 / 0,0148 | 0,0164 / 0,0163 | 0,086 / 0,087 | 0,038 / 0,031 | 0,038 / 0,038 | 0,046 / 0,045 | 0,39 / 0,37 | 107 |
| **cVAE + refine qua bộ lọc (F)** | 0,0061 / 0,0062 | 0,0127 / 0,0126 | **0,0087 / 0,0093** | **0,035 / 0,036** | **0,026 / 0,020** | **0,017 / 0,018** | 0,052 / 0,050 | **0,78 / 0,81** | 107 |
| SIMP từ đầu (4 khởi tạo) | 0,0071 / 0,0064 | 0,0162 / 0,0174 | 0,0187 / 0,0198 | 0,103 / 0,111 | 0,035 / 0,038 | 0,046 / 0,047 | 0,050 / 0,051 | 0,72 / 0,80 | ~900 |

Hiệu ghép cặp (CI bootstrap 95%):
- F − A: biên thực tế σ=0,5 −47% [−0,0100; −0,0049] / −43% [−0,0098; −0,0040]; σ=1,0 −59% / −59%; chế
  tạo +0,47 [+0,34; +0,59] / +0,50; verify +32% (CI chứa 0) / +46% (CI chứa 0); lưới 200² −10% / −15%
  (CI chứa 0); co/giãn đều +11% / +12% (CI > 0).
- F − SIMP: biên thực tế σ=0,5 −53% / −53%; σ=1,0 −66% / −68%; lưới 200² −21% [−0,0065; −0,0001] /
  −28% [−0,0076; −0,0018]; verify −14% / −3% (CI chứa 0); chế tạo +0,06 / +0,01 (CI chứa 0); co/giãn đều
  +4% / −0% (CI chứa 0).

**Claim đề xuất cho bài (nếu tác giả chọn F làm pipeline chính):**
- "Defining the design through a filter-and-project map and verifying that same design halves the
  error under realistic boundary realizations (−43 % to −47 %, two independent target sets) and raises
  manufacturability from 0.38 to 0.80, with unchanged nominal accuracy and cost."
- "Against topology optimization from scratch, the generative pipeline is now at least as
  manufacturable (0.78–0.81 vs 0.72–0.80) and more accurate under realistic boundary realizations
  (−53 %) and on a 4× finer mesh (−21 % to −28 %), with about 8× fewer FE solves."
- Phải nói kèm: không bền hơn với co/giãn đều (over/under-etch); thước đo biên thực tế cùng họ làm mượt
  với bộ lọc (thước đo không làm mượt - verify, lưới mịn - cho thấy F không tệ hơn); dị hướng r ≥ 10 vẫn
  thất bại (1 mục tiêu/tập).
- Điều này đảo một kết luận cũ của bài ("SIMP wins manufacturability") → cần sửa Abstract, Bảng
  "which method when", Discussion.
- **Hình:** `outputs/phase5/plan_v3/n2/fig_designs_A_F_SIMP.png` - 5 mục tiêu IN100 (#0, #9, #27, #45,
  #72), lát 2×2, mỗi ô ghi sai số biên thực tế σ = 0,5 và chế tạo được/không. F mượt hơn, không bản lề
  mảnh; ca F trượt chế tạo (#27) chỉ do đốm 1-2 px rời → kiểm hậu xử lý xóa đảo (mục C6).

### C6. Xóa đảo rời - thước đo chế tạo cũ chủ yếu đếm đảo (2026-10-10)
- Phương pháp: giữ thành phần rắn chính theo liên thông 4 hướng tuần hoàn (lát 3×3, lấy thành phần chiếm
  nhiều pixel nhất trong ô giữa), xóa phần còn lại. Script `outputs/phase5/plan_v3/n2/island_cleanup.py`.
- Đảo rời không chịu lực: ν không đổi (|Δν12| tối đa 0,00000, 29/200 thiết kế F có đảo, trung vị 7 px).
- Chế tạo được trước → sau xóa đảo (IN100 / IN100-C): A 0,39 → 0,84 / 0,37 → 0,85; F 0,78 → 0,92 /
  0,81 → 0,94; SIMP 0,72 → 1,00 / 0,80 → 1,00. Hiệu ghép cặp sau xóa đảo: F − A +0,08 / +0,09; F − SIMP
  −0,08 / −0,06 (CI < 0).
- **Viết vào bài:** định nghĩa "thiết kế cuối" = khung liên thông chính (đảo rời bị xóa, không đổi tính
  chất). Mọi số chế tạo trong bài nên báo sau xóa đảo cho mọi phương pháp. Bỏ claim cũ "SIMP thắng chế tạo
  0,72-0,84 vs ~0,31" (khoảng cách thật sau xóa đảo: 1,00 vs 0,84 cho pipeline cũ). K1 nên được trình bày
  lại: phần lớn lợi ích chế tạo của K1 đạt được miễn phí bằng xóa đảo.

### C7. Độ nhạy σ của bộ lọc
- Pilot IN100-B (n = 20, `outputs/phase5/plan_v3/n2/run_sigma.log`): σ ∈ {0,5; 0,75; 1,0; 1,5} → sai số
  biên thực tế σ=0,5: 0,0150 / 0,0130 / 0,0078 / 0,0061 (A 0,0165); chế tạo thô 0,50 / 0,65 / 0,80 / 1,00;
  co/giãn đều tệ dần 0,052 → 0,061. Đơn điệu.
- Xác nhận σ = 1,5 (n = 100, IN100 / IN100-C): độ chính xác ngang σ = 1,0 (CI chứa 0 mọi thước đo),
  chế tạo sau xóa đảo 0,96 / 0,99 (σ = 1,0: 0,92 / 0,94; SIMP 1,00 / 1,00), co/giãn đều tệ hơn 8-9%.
- **Viết vào bài:** "Performance is monotone in the filter radius up to σ = 1.5 elements; σ = 1 is used
  throughout; σ = 1.5 trades a slightly higher manufacturability for a higher sensitivity to uniform
  erosion/dilation."


---

## D. Danh sách đầy đủ tài liệu đã tham khảo (tra cứu 2026-10-10)

> **Mức đọc:** "TT" = đọc tóm tắt/trích đoạn qua tìm kiếm web; "CT" = đọc chi tiết phần phương pháp
> (công thức) từ bản arXiv. Chưa đọc toàn văn bài nào - **kiểm lại số liệu, DOI, danh sách tác giả từ bản
> gốc trước khi trích dẫn**. Cột "bib": ✅ đã có trong `docs/paper1/refs.bib`; ➕ đề xuất thêm; – chỉ để
> tham khảo nền.

### D1. Tối ưu topo: robust formulation, kích thước nét, chiếu nhị phân

| # | Tài liệu | Link | Mức đọc | bib | Dùng cho |
|---|---|---|---|---|---|
| 1 | O. Sigmund (2009), Manufacturing tolerant topology optimization, Acta Mech. Sin. 25:227-239 | (qua trích dẫn trong #3, #4) | TT | ➕ | Robust formulation gốc |
| 2 | F. Wang, B. S. Lazarov, O. Sigmund (2011), On projection methods, convergence and robust formulations in topology optimization, SMO 43:767-784 | (qua #4, #5) | TT | ✅ `wang2011projection` | Robust co/gốc/giãn (E1, E1′); Heaviside |
| 3 | E. Andreassen, B. S. Lazarov, O. Sigmund (2014), Design of manufacturable 3D extremal elastic microstructure, Mech. Mater. 69:1-10 | [DTU Orbit](https://backend.orbit.dtu.dk/ws/files/119929621/Design_of_manufacturable_3D_extremal_elasticc_microstructure.pdf) | TT | ➕ | Auxetic + robust → in được, ν = −0,5 kiểm thực nghiệm |
| 4 | (Fernández et al.), Analytical relationships for imposing minimum length scale in the robust topology optimization formulation (2021) | [arXiv 2101.08605](https://arxiv.org/pdf/2101.08605) | TT | – | Liên hệ co/giãn ↔ kích thước nét |
| 5 | (Fernández et al.), Imposing minimum and maximum member size, minimum cavity size, and minimum separation distance between solid members in topology optimization (2020) | [arXiv 2003.00263](https://arxiv.org/pdf/2003.00263) | TT | – | Mở rộng ràng buộc kích thước |
| 6 | M. Zhou, B. S. Lazarov, F. Wang, O. Sigmund (2015), Minimum length scale in topology optimization by geometric constraints, CMAME 293:266-282 | [DTU Orbit](https://backend.orbit.dtu.dk/ws/files/128050147/Minimum_length_scale_in_topology_optimization_by_geometric_constraints.pdf) | TT | ➕ | Ràng buộc nét tối thiểu |
| 7 | A. M. Hammond, A. Oskooi, I. M. Hammond, M. Chen, S. E. Ralph, S. G. Johnson (2025), Unifying and accelerating level-set and density-based topology optimization by subpixel-smoothed projection, Opt. Express 33(16):33620-33642 | [arXiv 2503.20189](https://arxiv.org/abs/2503.20189), [code](https://github.com/NanoComp/SSP) | CT (công thức SSP) | ➕ | **Gần nhất với P1.1e** (chiếu nhị phân khả vi, β = ∞); hướng E6 |
| 8 | (tác giả chưa kiểm), Differentiating through binarized topology changes: Second-order subpixel-smoothed projection (2026) | [arXiv 2601.10737](https://arxiv.org/html/2601.10737) | TT | – | SSP bậc 2 |
| 9 | R. Arrieta, G. Romano, S. G. Johnson (2025), Hyperparameter-free minimum-lengthscale constraints for topology optimization, arXiv; SMO 2026 | [arXiv 2507.16108](https://arxiv.org/pdf/2507.16108) | CT (siêu tham số giải tích) | ➕ | Ràng buộc nét tối thiểu dựa trên SSP; chi phí ≤ 3-20% |
| 10 | J. K. Guest, J. H. Prévost, T. Belytschko (2004), Achieving minimum length scale in topology optimization using nodal design variables and projection functions, IJNME | – | – | ✅ `guest2004achieving` | Heaviside projection |
| 11 | B. Bourdin (2001), Filters in topology optimization, IJNME 50:2143-2158 | – | – | ✅ `bourdin2001filters` | Density filter = tham số hóa thiết kế (F) |
| 12 | T. E. Bruns, D. A. Tortorelli (2001), Topology optimization of non-linear elastic structures and compliant mechanisms, CMAME 190:3443-3459 | – | – | ➕ | Density filter (F) |
| 13 | T. A. Poulsen (2002), A simple scheme to prevent checkerboard patterns and one-node connected hinges in topology optimization, SMO 24:396-399 | (qua #14) | TT | ➕ | Bản lề 1 nút = artifact rời rạc hóa |
| 14 | G. H. Yoon et al. (2004), Hinge-free topology optimization with embedded translation-invariant differentiable wavelet shrinkage, SMO | [Springer](https://link.springer.com/article/10.1007/s00158-004-0378-z) | TT | – | Chống bản lề bằng lọc wavelet |
| 15 | (nhóm Paulino), Auxetic structure design using compliant mechanisms: a topology optimization approach with polygonal finite elements | [Princeton](https://collaborate.princeton.edu/en/publications/auxetic-structure-design-using-compliant-mechanisms-a-topology-op/) | TT | – | Auxetic không bản lề bằng lưới đa giác |
| 16 | O. Sigmund (2000), A new class of extremal composites, JMPS 48:397-428 | (qua #3) | TT | ➕ | Auxetic tối ưu topo hay ra bản lề mảnh |
| 17 | (tác giả chưa kiểm), Stress-constrained topology optimization for metamaterial microstructure design (2026) | [arXiv 2602.19662](https://arxiv.org/pdf/2602.19662) | TT | – | Thiếu ràng buộc ứng suất → khung mảnh + bản lề |

### D2. Liên thông và tô-pô

| # | Tài liệu | Link | Mức đọc | bib | Dùng cho |
|---|---|---|---|---|---|
| 18 | Cool, Aage, Sigmund, A practical review on promoting connectivity in topology optimization (2025) | [arXiv 2501.09402](https://ar5iv.labs.arxiv.org/html/2501.09402) | TT | ➕ | Tổng quan ràng buộc liên thông (E4) |
| 19 | S. Liu et al. (2015), virtual temperature method, Front. Mech. Eng. 10:126 | [HEP](https://academic.hep.com.cn/fme/EN/10.1007/s11465-015-0340-3) | TT | – | Liên thông = ràng buộc nhiệt độ max |
| 20 | Li, Gao, Yin, Lin (2026), Persistent homology-based explicit topological control for 2D topology optimization with MMA | [arXiv 2602.13856](https://arxiv.org/html/2602.13856v1) | TT | – | Ràng buộc số lỗ/liên thông tường minh |
| 21 | M. Carrière et al. (2021), Optimizing persistent homology based functions, ICML | [arXiv 2010.08356](https://arxiv.org/pdf/2010.08356) | TT | – | Gradient cho loss persistent homology |
| 22 | (tổng quan), Persistence-based topological optimization: a survey (2026) | [arXiv 2603.24613](https://arxiv.org/pdf/2603.24613) | TT | – | Nền |

### D3. Đồng nhất hóa trên lưới pixel/voxel - sai số biên bậc thang

| # | Tài liệu | Link | Mức đọc | bib | Dùng cho |
|---|---|---|---|---|---|
| 23 | M. Kabel, D. Merkert, M. Schneider (2015), Use of composite voxels in FFT-based homogenization, CMAME 294:168-188 | (qua #24) | TT | ➕ | Biên bậc thang + sửa bằng độ cứng laminate (B4, Discussion) |
| 24 | S. Keshav, F. Fritzen, M. Kabel (2022/23), FFT-based homogenization at finite strains using composite boxels (ComBo), Comput. Mech. | [arXiv 2204.13624](https://arxiv.org/pdf/2204.13624) | TT | – | Mở rộng composite voxel |
| 25 | (tác giả chưa kiểm), A stable and accurate X-FFT solver for linear elastic homogenization problems in 3D (2026) | [arXiv 2601.02172](https://arxiv.org/pdf/2601.02172) | TT | – | Tốc độ hội tụ voxel FE O(h) (dẫn Schneider) |
| 26 | Merkert et al. (2014) interface voxels; Lendvai & Schneider - composite voxel trên level set | (qua #23-#25) | TT | – | Nền |

### D4. Mô hình sinh, tối ưu latent, khai thác surrogate

| # | Tài liệu | Link | Mức đọc | bib | Dùng cho |
|---|---|---|---|---|---|
| 27 | S. Hoyer, J. Sohl-Dickstein, S. Greydanus (2019), Neural reparameterization improves structural optimization | [arXiv 1909.04240](https://arxiv.org/abs/1909.04240v1) | TT | ✅ `hoyer2019neural` | Tối ưu qua mạng nơ-ron |
| 28 | (tác giả chưa kiểm), Robust topology optimization using variational autoencoders (2021) | [arXiv 2107.10661](https://ar5iv.arxiv.org/html/2107.10661) | TT | – | Robust TO trong latent VAE (qua surrogate) |
| 29 | M. Chen, J. Jiang, J. A. Fan (2020), Design space reparameterization enforces hard geometric constraints in inverse-designed nanophotonic devices | [arXiv 2007.12991](https://arxiv.org/pdf/2007.12991) | TT | ➕ | Robust co/giãn trong tối ưu qua mạng (trọng số 0,5/0,25/0,25) |
| 30 | J.-H. Bastek, D. M. Kochmann (2023), Inverse design of nonlinear mechanical metamaterials via video denoising diffusion models, Nat. Mach. Intell. | [Nature](https://www.nature.com/articles/s42256-023-00762-x) | TT | ✅ `bastek2023diffusion` | Mô hình sinh cho metamaterial |
| 31 | J.-H. Bastek, W. Sun, D. M. Kochmann (2025), Physics-informed diffusion models, ICLR | [arXiv 2403.14404](https://ui.adsabs.harvard.edu/abs/2024arXiv240314404B/abstract) | TT | – | Loss vật lý lúc train |
| 32 | (tác giả chưa kiểm), Guided diffusion for fast inverse design of density-based mechanical metamaterials (2024) | [arXiv 2401.13570](https://arxiv.org/pdf/2401.13570) | TT | ➕ | Active learning phủ dữ liệu; diffusion làm khởi tạo TO (E4, E5) |
| 33 | (tác giả chưa kiểm), Inverse design of curved mechanical metamaterials with geometric AI: a generative diffusion operates in compact latent space of cellular structures (2025) | [Nature npj](https://www.nature.com/articles/s44455-025-00005-6) | TT | ➕ | Tối ưu latent + diffusion; R² một số thành phần ~0,6 |
| 34 | (tác giả chưa kiểm), Unifying the design space and optimizing linear and nonlinear truss metamaterials by generative modeling (2023) | [arXiv 2306.14773](https://arxiv.org/pdf/2306.14773) | TT | – | Encode-decode giữ thiết kế hợp lệ |
| 35 | (tác giả chưa kiểm), DiffuMeta: Algebraic language models for inverse design of metamaterials via diffusion transformers (2025) | [arXiv 2507.15753](https://arxiv.org/pdf/2507.15753) | TT | – | Claim ngoại suy tính chất |
| 36 | (tác giả chưa kiểm), Nonlinear inverse design of mechanical multi-material metamaterials enabled by video denoising diffusion and structure identifier (2024) | [arXiv 2409.13908](https://arxiv.org/pdf/2409.13908) | TT | – | Nền |
| 37 | (tác giả chưa kiểm), Steering topology distributions for unified generative design of architected metamaterials (GenTO) | [arXiv 2607.24777](https://arxiv.org/pdf/2607.24777) | TT | – | Claim OOD |
| 38 | Kong et al. (2025), Diffusion models as constrained samplers for optimization (DiffOPT), AISTATS | [PDF](https://www.cs.cornell.edu/gomes/pdf/2025_kong_aistats_diffusion.pdf) | TT | – | Tối ưu theo surrogate ngoài phân phối bị khai thác (giống P1.1) |
| 39 | (nhóm W. Chen), IH-GAN: a conditional generative model for implicit surface-based inverse design of cellular structures | [Boise State](https://experts.boisestate.edu/en/publications/ih-gan-a-conditional-generative-model-for-implicit-surface-based-/) | TT | – | Inverse homogenization bằng GAN |
| 40 | (tác giả chưa kiểm), MetaGen: a DSL, database, and benchmark for VLM-assisted metamaterial generation (2025) | [arXiv 2508.17568](https://arxiv.org/pdf/2508.17568) | TT | – | Đo độ phủ dữ liệu theo tính chất |
| 41 | B. Trabucco et al. (2021), Conservative objective models (COMs) | – | – | ✅ `trabucco2021coms` | Khai thác surrogate |
| 42 | H. Pahlavani et al. (2024), Deep-DRAM | – | – | ✅ `pahlavani2024deep` | Best-of-N chọn bằng FE |
| 43 | R. V. Woldseth et al. (2022), On the use of artificial neural networks in topology optimisation | – | – | ✅ `woldseth2022use` | Phê phán: so với TO cổ điển |

### D5. Giới hạn Poisson / dị hướng

| # | Tài liệu | Link | Mức đọc | bib | Dùng cho |
|---|---|---|---|---|---|
| 44 | (Nature Commun. 2023), freeform self-bridging metamaterials vượt giới hạn Poisson nhiệt động | [Nat. Commun.](https://link.springer.com/10.1038/s41467-023-39792-9) | TT | – | Giới hạn 0 ≤ ν_ij·ν_ji < 1 (trực hướng) |
| 45 | A. N. Norris (2006), Poisson's ratio in cubic materials / Conditions for a maximum or minimum of Poisson's ratio | [JoMMS](https://msp.org/jomms/2006/1-4/p10.xhtml), [arXiv cond-mat/0603820](https://ar5iv.labs.arxiv.org/html/cond-mat/0603820) | TT | – | |ν| không bị chặn ở vật liệu dị hướng |
| 46 | (tác giả chưa kiểm), Topology optimization of micro-structured materials featured with the specific mechanical properties (2018) | [arXiv 1808.08647](https://arxiv.org/pdf/1808.08647) | TT | – | 2 định nghĩa ν trực hướng không tương đương; ν < −1 khả thi |
| 47 | A. Clausen, F. Wang, J. S. Jensen, O. Sigmund, J. A. Lewis (2015), Topology optimized architectures with programmable Poisson's ratio over large deformations, Adv. Mater. | (qua #3) | TT | – | Auxetic biến dạng lớn (P1.6f) |

### Ghi chú dùng danh sách
- Ưu tiên thêm vào `refs.bib` khi sửa bản thảo (N3-3): #1, #3, #6, #7, #9, #12, #13, #16, #18, #23, #29, #32, #33.
- Các mục "(tác giả chưa kiểm)" phải tra lại tác giả và nơi đăng trước khi trích dẫn.
- Chưa tra cứu có hệ thống (Scopus/Web of Science); trước khi viết "first"/"novel" cần một lượt tra đầy đủ.

## E. Dự định ảnh hưởng tới bản thảo
Danh sách đầy đủ và ETA ở `docs/plan.md` mục **N3**. Tóm tắt phần chạm tới bài báo:
1. Chốt N3-D1 (pipeline F + xóa đảo), N3-D2 (σ = 1,0 khuyến nghị), N3-D3 (E4).
2. N3-1 → N3-2: đưa xóa đảo vào code, chạy lại mọi số của bài với pipeline mới (kể cả OOD, n = 300).
3. N3-3: sửa Methods / Results / Discussion / Abstract / bảng "which method when", thêm trích dẫn mục D,
   vẽ lại hình (dùng mẫu `outputs/phase5/plan_v3/n2/fig_designs_A_F_SIMP.png`).
4. N3-4: cập nhật `LIMITATIONS.md` (khe lưới, thước đo chế tạo, co/giãn đều, vòng lặp họ làm mượt).
5. N3-5 (E4, r ≥ 10), N3-6 (robust nhẹ cho co/giãn đều), N3-7 (kiểm chứng ν bằng FE biên trơn) - kết quả
   nào xong trước khi nộp thì đưa vào bài, không thì ghi ở Limitations / Future work.

# Audit toàn diện AuxForge - 2026-08-09

> Audit theo 3 khía cạnh do user yêu cầu: (1) kiến trúc/thuật toán/logic, (2) thống nhất tên gọi & phiên bản, (3) độ sẵn sàng "sản phẩm cuối". Thực hiện bằng 3 agent khảo sát song song (đọc-only) trên toàn repo tại HEAD `81df0e6e7` (branch `main`), sau đó áp dụng các sửa an toàn. Không bump version, không tag release, không sửa lịch sử git, không đổi tên GitHub repo - các quyết định này thuộc về user, chỉ ghi nhận bên dưới.

## 1. Đã sửa trong lần audit này

### Thống nhất tên gọi/phiên bản
- **Xoá tên cũ "SIMP Analyst"** còn sót trong 10 file HTML tĩnh (title/meta/h1/footer/nav) - dự án đã đổi tên sang **AuxForge** từ trước (ghi trong CHANGELOG) nhưng các file HTML này bị bỏ sót: `html/index.html`, `html/dashboards/{workflow,phase1_screening_dashboard}.html`, `html/inverse_auxetic_report.html`, `html/reports/{auxetic_phase1_analysis,auxetic_report,ml_pipeline_phase3to5}.html`, `html/guides/{optimization_pipeline,simp_guide_and_roadmap,multi_batch_adaptive_sampling}.html`.
- **Sửa số phiên bản lệch khỏi 1.4.0** (phiên bản hiện tại theo `pyproject.toml`/`simp/__init__.py`/README badge):
  - `analysis/report.py:256` - footer template "v1.1.0" → "v1.4.0".
  - `simp/README.md:94` - "version 1.1.0" → "version 1.4.0".
  - `html/guides/multi_batch_adaptive_sampling.html:630,1351` - badge "v1.3.0" → "v1.4.0".
  - `html/dashboards/workflow.html:915` - badge "v2.0.0" đổi nhãn thành "Dashboard v2.0.0" để rõ đây là version riêng của diagram, không phải version package (giữ nguyên số vì nó không thật sự sai - chỉ mập mờ).
- **Đồng bộ `requirements.txt`** với `pyproject.toml`: thêm `isort>=5.12` vào khối dev-deps đã comment cho khớp `[project.optional-dependencies.dev]`.
- **Ghi chú mismatch repo GitHub `simp-micro` vs tên project `auxforge`**: thêm 1 dòng trong README giải thích đây là slug lịch sử, trỏ sang CHANGELOG - không tự ý đổi tên repo GitHub thật vì ảnh hưởng remote chung.

### Sản phẩm cuối / release readiness
- **Thêm file `LICENSE` (MIT)** - trước đó hoàn toàn không tồn tại dù `pyproject.toml`, `simp/__init__.py`, và badge README đều tuyên bố MIT. Đây là lỗ hổng dễ thấy nhất với người clone repo lần đầu, đã vá.
- **Cập nhật CHANGELOG.md**: `[Unreleased]` trước đó dừng ở mục "2026-07-31" trong khi có 8+ commit từ 2026-08-02 đến 2026-08-09 chưa được ghi (PR #13 optional multi-condition, script so sánh baseline OOD, tách `docs/PIPELINE.md`, slide deck, refactor path notebook sang `REPO_ROOT`). Đã bổ sung mục "2026-08-02 đến 2026-08-09" theo đúng văn phong/tiếng Việt hiện có của file, không bump version số (theo đúng ghi chú tự đặt ra trong CHANGELOG: "không tự ý bump").
- **Dọn `.gitignore`**: thêm `.pi-subagents/` (thư mục scratch của agent tooling, trước đó untracked nhưng không có gitignore entry, nguy cơ bị `git add -A` commit nhầm), thêm `outputs/pilot_normalized_*/` cho nhất quán với các `pilot_mma_*/`/`pilot_damping_decay/`/`pilot_oc_dual_multiplier/` đã ignore.
- **Untrack `outputs/pilot_normalized_{hexagonal,hourglass,reentrant_bowtie}/`** (~9MB) khỏi git - đây là dữ liệu scratch của A/B test `objective_variant='normalized'` (đã REJECTED, xem EXPERIMENT_LOG.md 2026-08-05), bị track nhầm không nhất quán với các thư mục pilot khác. File vẫn còn nguyên trên đĩa, chỉ ngừng theo dõi bởi git (dùng `git rm -r --cached`, không xoá vật lý).

## 2. Ghi nhận - KHÔNG sửa trong lần này (cần scope riêng hoặc quyết định của user)

### Rủi ro cao / cần quyết định user
- **`.git` nặng 267MB** trong khi file lớn nhất đang track chỉ ~2.1MB - binary lịch sử (checkpoint/npz) từng được commit rồi gỡ vẫn còn vĩnh viễn trong lịch sử, làm mọi lần clone tương lai nặng hơn cần thiết. Sửa cần `git filter-repo`/BFG + force-push, phá vỡ clone hiện có của bất kỳ collaborator nào - **cần quyết định riêng của user**, không tự làm.
- **Chưa có git tag nào** (kể cả cho 1.4.0) - CHANGELOG tự ghi rõ việc bump version là "quyết định phát hành, không tự ý bump", nên không tag trong lần audit này.
- **Repo GitHub tên `simp-micro`** khác tên project `auxforge` - chỉ thêm ghi chú giải thích trong README, không đổi tên repo thật.

### Kiến trúc/logic - cần refactor có phạm vi riêng
- **Hai interface tham số song song**: `simp/config.py::SimpConfig` (dataclass) và interface dict-param dùng bởi `pipeline/params.py`/`simp/run.py` - docstring trong `simp/config.py` tự cảnh báo có thể lệch nhau. Hợp nhất là refactor lớn, chạm nhiều file, nên làm task riêng.
- **`REPO_ROOT` dùng không nhất quán**: `pipeline/phase3_dataset`, `phase4_surrogate`, `phase5_cvae`, `analysis/scripts`, và giờ cả notebooks (commit gần nhất) đều dùng `REPO_ROOT`, nhưng `pipeline/phase1_screening` và `phase2_multi_batch` vẫn dùng đường dẫn tương đối giả định CWD = repo root (`screening_parallel.py:265,380,538`, `refine_params.py:109-110,147,152`, `phase2_multi_batch/params.py`, `main.py:65`). Nên đồng bộ hoá về sau.
- **`phase2_multi_batch/runner.py`** có `DEFAULT_FIXED` (`nelx=nely=50`) hardcode độc lập, lệch với `pipeline/params.py::FIXED_PARAMS` (`nelx=80`) - cần xác nhận đây có phải cố ý (vd. test nhanh) trước khi đổi, không tự sửa vì có thể phá kỳ vọng hiện có.
- **Thiếu validate rõ ràng trước khi load output phase trước** (`train.npz`, `manifest.csv`, checkpoint) ở phase3/4/5 - lỗi hiện ra dạng `FileNotFoundError`/npz-load error thô thay vì thông báo hướng dẫn. Cải thiện UX lỗi, không khẩn cấp về mặt đúng-sai.
- **Thiếu test cho `pipeline/phase5_cvae/evaluate.py`** - mọi module khác trong `phase5_cvae/` đều có file test riêng (`tests/test_phase5_*.py`), trừ module này.
- **`pipeline/phase5_cvae/losses.py:108`** - TODO còn tồn: `property_consistency_loss` dùng one-hot vector của seed gốc (xấp xỉ) thay vì seed dự đoán thật cho ảnh sinh ra, vì ảnh sinh không có nhãn seed. Đã biết, chưa fix.
- **`simp/core/oc.py::oc_update` chế độ `'gate'`** vẫn là default dù có limit-cycle failure mode tự ghi nhận trong code, và MMA thất bại trên `reentrant_bowtie` (xem memory dự án `hexagonal_yield_oc_dual_multiplier`) - dispatch theo seed trong `generate_production_batch.py` đã xử lý việc này, không cần sửa thêm ở đây.

### Không phải bug, chỉ là quan sát
- Một số khối `except Exception` khá rộng trong `pipeline/phase1_screening/analyst.py`, `phase2_multi_batch/sampling.py`, `simp/core/solver.py:128`, và nhiều nơi khác - đã kiểm tra 2 chỗ nghi ngờ "silent swallow" cụ thể (`analyst.py:136-137` đọc field optional trong metadata.json với fallback hợp lý; `sampling.py:337-338` bắt `FileNotFoundError`/`EmptyDataError` khi CSV chưa tồn tại, đúng ý đồ) - đây là pattern chấp nhận được, không phải bug, không sửa.

## 3. Xác minh đã chạy
- `grep -rn "SIMP Analyst" html/` → rỗng.
- `grep -rn "v1\.1\.0\|v1\.3\.0" html/ analysis/ simp/README.md` → rỗng.
- Tất cả thay đổi ở trên là sửa văn bản/docs/gitignore, không đụng logic Python nào trong `simp/`/`pipeline/` - không cần chạy lại test suite để xác nhận đúng-sai thuật toán.

## 4. Trạng thái git
Toàn bộ thay đổi ở trên **chưa được commit** - nằm trong working tree, do user tự quyết định thời điểm commit. `pyproject.toml` cũng đang có diff chưa commit từ trước (bỏ `pre-commit`/`scikit-image`/`seaborn`, thêm `isort`/Python 3.13 classifier) - đã xác minh an toàn (không còn import nào dùng các gói bị bỏ) nhưng vẫn để nguyên trạng thái uncommitted, không tự gộp commit.

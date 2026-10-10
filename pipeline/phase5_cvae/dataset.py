"""
Phase 5 - dataset.py
=====================
Đọc outputs/phase3/{train,val,test}.npz (cùng file Phase 4 dùng). Khác Phase 4:
v12/v21 ở đây là CONDITION đầu vào cVAE (không phải target regress), giữ
nguyên đơn vị vật lý (không chuẩn hoá). seed_onehot vẫn trả về nhưng chỉ dùng
phụ ở evaluate.py, không đưa vào condition vector (xem model.py).

Mỗi mẫu: image (1,RES,RES) [0,1], condition (2,)=[v12,v21] (mặc định), mở
rộng theo 2 cờ ĐỘC LẬP cộng dồn (xem `CVAEDataset.__init__`):
  - `extended_condition=True`: +4 chiều [volfrac,volfrac_mask,
    void_size_frac,void_size_frac_mask]
  - `include_nu0=True` (Giai đoạn A, docs/archive/PROJECT_PLAN.md A6): +2 chiều
    [nu0,nu0_mask] - LUÔN ở 2 cột CUỐI CÙNG của condition, sau nhóm
    extended_condition nếu cả 2 cùng bật.
Tổ hợp 2 cờ cho condition_dim ∈ {2,4,6,8}. seed_vec (n_seeds,) one-hot,
volfrac scalar (luôn trả riêng, kể cả khi đã có trong condition - dùng cho
volfrac_consistency_loss).

extended_condition/include_nu0 thêm tham số OPTIONAL: mask ở đây LUÔN=1
(dataset chỉ mô tả dữ liệu thật, có sẵn); train.py mới là nơi áp
condition-dropout (zero value + mask=0 ngẫu nhiên) để model học bỏ qua các
chiều optional lúc suy diễn không được chỉ định - xem losses.py
`volfrac_consistency_loss`, `real_physics_loss` (dùng đúng nu0 per-sample
khi mask=1, xem real_physics.py A5) và train.py `run_epoch`.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")
V12_WEIGHTS_PATH = os.path.join(PHASE3_DIR, "v12_bin_weights.json")


class CVAEDataset(Dataset):
    def __init__(self, npz_path: str, extended_condition: bool = False,
                 include_nu0: bool = False):
        """include_nu0: đọc thêm ν0 (hệ số Poisson vật liệu nền) từ field
        "nu" trong npz, thêm 2 cột [nu0, nu0_mask] vào CUỐI condition vector
        (Giai đoạn A, docs/archive/PROJECT_PLAN.md A6 - tái dùng đúng pattern
        extended_condition/dropout). Độc lập với `extended_condition` (có
        thể bật riêng hoặc cùng lúc - xem module docstring cho thứ tự cột).

        Field "nu" do `pipeline/phase3_dataset/build_npz.py` ghi (thêm cho
        A4) - CHỈ npz build lại sau đó mới có (vd `outputs/phase3_a4/`).
        Các file `outputs/phase3/*.npz` sinh trước A4 KHÔNG có field này -
        include_nu0=True trên các file đó raise lỗi rõ ràng ngay lúc load,
        thay vì âm thầm coi mọi mẫu là ν0=0.3 (validate ở biên, CLAUDE.md).
        """
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"]                       # (N, RES, RES) [0,1]
        self.v12 = data["v12"].astype(np.float32)
        self.v21 = data["v21"].astype(np.float32)
        self.volfrac_achieved = data["volfrac_achieved"].astype(np.float32)
        self.seed_onehot = data["seed_onehot"].astype(np.float32)  # (N, n_seeds)
        self.seed_classes = data["seed_classes"]

        self.extended_condition = extended_condition
        self.void_size_frac = None
        if extended_condition:
            param_names = list(data["param_names"])
            void_idx = param_names.index("void_size_frac")
            self.void_size_frac = data["params"][:, void_idx].astype(np.float32)

        self.include_nu0 = include_nu0
        self.nu0 = None
        if include_nu0:
            if "nu" not in data.files:
                raise ValueError(
                    f"include_nu0=True nhưng '{npz_path}' không có field 'nu'. "
                    "Cần dataset build sau A4 (pipeline/phase3_dataset/build_npz.py "
                    "đã thêm field này) - vd outputs/phase3_a4/*.npz, KHÔNG phải "
                    "outputs/phase3/*.npz cũ "
                    "(xem docs/archive/PROJECT_PLAN.md A6)."
                )
            self.nu0 = data["nu"].astype(np.float32)

    def __len__(self):
        return len(self.images)

    @property
    def n_seeds(self) -> int:
        return self.seed_onehot.shape[1]

    @property
    def resolution(self) -> int:
        return self.images.shape[-1]

    @property
    def condition_dim(self) -> int:
        dim = 2
        if self.extended_condition:
            dim += 4
        if self.include_nu0:
            dim += 2
        return dim

    @property
    def nu0_col(self):
        """Cột giá trị nu0 trong condition vector (cột mask = nu0_col+1),
        hoặc None nếu include_nu0=False - dùng bởi losses.real_physics_loss
        để trích ν0 THẬT per-sample thay vì fe_params['nu'] cố định (A5)."""
        if not self.include_nu0:
            return None
        return 6 if self.extended_condition else 2

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, RES, RES)
        cond_values = [self.v12[idx], self.v21[idx]]
        if self.extended_condition:
            cond_values += [
                self.volfrac_achieved[idx], 1.0,
                self.void_size_frac[idx], 1.0,
            ]
        if self.include_nu0:
            cond_values += [self.nu0[idx], 1.0]
        condition = torch.tensor(cond_values, dtype=torch.float32)
        seed_vec = torch.from_numpy(self.seed_onehot[idx])
        volfrac = torch.tensor(self.volfrac_achieved[idx], dtype=torch.float32)
        return image, condition, seed_vec, volfrac


def condition_flags_from_dim(condition_dim: int):
    """Suy ngược (extended_condition, include_nu0) từ condition_dim đã lưu
    trong checkpoint (vd sample.py/best_of_n_eval.py chỉ có condition_dim,
    không lưu riêng 2 cờ gốc) - NGUỒN DUY NHẤT cho phép ánh xạ {2,4,6,8} ->
    (extended_condition, include_nu0), tránh suy luận rời rạc kiểu
    `condition_dim == 6` lặp lại ở nhiều nơi rồi lệch nhau khi thêm A6
    (condition_dim=4/8 cần include_nu0=True mà check cũ không biết tới)."""
    if condition_dim not in (2, 4, 6, 8):
        raise ValueError(
            f"condition_dim={condition_dim} không được hỗ trợ (chỉ 2, 4, 6 hoặc 8)."
        )
    extended_condition = condition_dim in (6, 8)
    include_nu0 = condition_dim in (4, 8)
    return extended_condition, include_nu0


def build_condition_vector(v12: float, v21: float, condition_dim: int,
                            volfrac: float = None, void_size_frac: float = None,
                            nu0: float = None) -> np.ndarray:
    """Dựng condition vector cho suy diễn (sample.py/best_of_n_eval.py) -
    dùng chung 1 chỗ để các script không lệch quy ước. condition_dim==2:
    hành vi cũ, bỏ qua volfrac/void_size_frac/nu0 nếu lỡ truyền (in cảnh báo
    ở caller). condition_dim ∈ {4,6,8}: giá trị optional=None -> (0.0,
    mask=0.0) ĐÚNG quy ước condition-dropout lúc train (xem train.py
    apply_condition_dropout) - giá trị có -> (value, mask=1.0). Thứ tự cột
    khớp CVAEDataset: nhóm volfrac/void_size_frac (nếu condition_dim ∈
    {6,8}) đứng TRƯỚC nhóm nu0 (nếu condition_dim ∈ {4,8})."""
    if condition_dim == 2:
        return np.array([v12, v21], dtype=np.float32)
    extended_condition, include_nu0 = condition_flags_from_dim(condition_dim)
    parts = [v12, v21]
    if extended_condition:
        vol_val, vol_mask = (volfrac, 1.0) if volfrac is not None else (0.0, 0.0)
        void_val, void_mask = (void_size_frac, 1.0) if void_size_frac is not None else (0.0, 0.0)
        parts += [vol_val, vol_mask, void_val, void_mask]
    if include_nu0:
        nu0_val, nu0_mask = (nu0, 1.0) if nu0 is not None else (0.0, 0.0)
        parts += [nu0_val, nu0_mask]
    return np.array(parts, dtype=np.float32)


def compute_v12_bin_weights(v12: np.ndarray, bin_edges: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """weight(bin) = (1/count(bin))^alpha, chuẩn hoá mean=1 - xem
    analysis/scripts/analyze_auxetic_distribution.py (nguồn phân tích gốc,
    yêu cầu advisor 2026-07-24: gán trọng số cao hơn cho vùng v12 thưa mẫu
    để dataloader lấy mẫu đều hơn qua toàn phổ auxetic thay vì chỉ học tốt
    vùng mode [-0.45,-0.30) chiếm ~38% dữ liệu). alpha=0 tắt (mọi bin weight
    =1), alpha=1 nghịch đảo tần suất hoàn toàn, alpha=0.5 (mặc định) làm
    mượt bằng sqrt để tránh few-sample bin nhận trọng số cực đoan."""
    counts, _ = np.histogram(v12, bins=bin_edges)
    counts_safe = np.maximum(counts, 1)
    raw_weight = counts_safe.astype(np.float64) ** (-alpha)
    return raw_weight / raw_weight.mean()


def compute_v12_sample_weights(v12: np.ndarray, weights_path: str = V12_WEIGHTS_PATH,
                                alpha: float = 0.5) -> np.ndarray:
    """Trọng số per-sample cho WeightedRandomSampler (train.py --weighted-sampling),
    dựa trên bin của v12 mỗi mẫu. Ưu tiên đọc bin_edges/bin_weight đã lưu sẵn
    ở `weights_path` (từ analyze_auxetic_distribution.py, chạy 1 lần trên
    train.npz) - nếu file không tồn tại, tính lại tại chỗ với bin rộng 0.05
    trên đúng range của `v12` truyền vào (fallback, vd khi dùng tập dữ liệu
    khác chưa chạy script phân tích)."""
    if weights_path and os.path.exists(weights_path):
        with open(weights_path) as f:
            saved = json.load(f)
        bin_edges = np.array(saved["bin_edges"])
        bin_weight = np.array(saved["bin_weight"])
    else:
        bin_edges = np.arange(v12.min() - 0.05, v12.max() + 0.05, 0.05)
        bin_weight = compute_v12_bin_weights(v12, bin_edges, alpha=alpha)

    bin_idx = np.clip(np.digitize(v12, bin_edges) - 1, 0, len(bin_weight) - 1)
    return bin_weight[bin_idx].astype(np.float32)


if __name__ == "__main__":
    # Self-test: `python3 pipeline/phase5_cvae/dataset.py`
    path = os.path.join(PHASE3_DIR, "val.npz")
    ds = CVAEDataset(path)
    print(f"Số mẫu: {len(ds)}, resolution: {ds.resolution}, n_seeds: {ds.n_seeds}")
    img, cond, seed_vec, vf = ds[0]
    print(f"image: {img.shape}, condition (v12,v21): {cond.tolist()}, "
          f"seed_vec: {seed_vec.shape}, volfrac: {vf.item():.3f}")
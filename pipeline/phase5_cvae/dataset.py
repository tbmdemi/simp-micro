"""
Phase 5 - dataset.py
=====================
Đọc outputs/phase3/{train,val,test}.npz (cùng file Phase 4 dùng). Khác Phase 4:
v12/v21 ở đây là CONDITION đầu vào cVAE (không phải target regress), giữ
nguyên đơn vị vật lý (không chuẩn hoá). seed_onehot vẫn trả về nhưng chỉ dùng
phụ ở evaluate.py, không đưa vào condition vector (xem model.py).

Mỗi mẫu: image (1,RES,RES) [0,1], condition (2,)=[v12,v21], seed_vec
(n_seeds,) one-hot, volfrac scalar.
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
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"]                       # (N, RES, RES) [0,1]
        self.v12 = data["v12"].astype(np.float32)
        self.v21 = data["v21"].astype(np.float32)
        self.volfrac_achieved = data["volfrac_achieved"].astype(np.float32)
        self.seed_onehot = data["seed_onehot"].astype(np.float32)  # (N, n_seeds)
        self.seed_classes = data["seed_classes"]

    def __len__(self):
        return len(self.images)

    @property
    def n_seeds(self) -> int:
        return self.seed_onehot.shape[1]

    @property
    def resolution(self) -> int:
        return self.images.shape[-1]

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, RES, RES)
        condition = torch.tensor(
            [self.v12[idx], self.v21[idx]], dtype=torch.float32
        )
        seed_vec = torch.from_numpy(self.seed_onehot[idx])
        volfrac = torch.tensor(self.volfrac_achieved[idx], dtype=torch.float32)
        return image, condition, seed_vec, volfrac


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
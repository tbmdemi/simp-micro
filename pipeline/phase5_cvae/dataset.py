"""
Phase 5 - dataset.py
=====================
PyTorch Dataset đọc trực tiếp outputs/phase3/{train,val,test}.npz - CÙNG file
mà Phase 4 dùng, không cần build lại dataset.

Khác với Phase 4 (dataset.py trả targets = [v12, v21, volfrac] để REGRESS),
ở đây v12/v21 đóng vai trò CONDITION (đầu vào cVAE), không phải target dự
đoán. seed_onehot vẫn được trả về nhưng CHỈ dùng phụ (debug / phân tích theo
seed sau này ở evaluate.py) - mặc định KHÔNG đưa vào condition vector, xem
lý do ở docstring của model.py.

Mỗi mẫu trả về:
    image      : Tensor (1, RES, RES) float32, [0,1]
    condition  : Tensor (2,) float32 = [v12, v21]   (giữ nguyên đơn vị vật lý)
    seed_vec   : Tensor (n_seeds,) float32, one-hot - dùng phụ, không bắt buộc
    volfrac    : Tensor scalar float32 - dùng phụ cho evaluate.py nếu cần
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "outputs", "phase3")


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


if __name__ == "__main__":
    # Self-test: `python3 pipeline/phase5_cvae/dataset.py`
    path = os.path.join(PHASE3_DIR, "val.npz")
    ds = CVAEDataset(path)
    print(f"Số mẫu: {len(ds)}, resolution: {ds.resolution}, n_seeds: {ds.n_seeds}")
    img, cond, seed_vec, vf = ds[0]
    print(f"image: {img.shape}, condition (v12,v21): {cond.tolist()}, "
          f"seed_vec: {seed_vec.shape}, volfrac: {vf.item():.3f}")
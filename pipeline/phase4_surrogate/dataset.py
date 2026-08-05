"""
Phase 4 - dataset.py
=====================
Đọc outputs/phase3/{train,val,test}.npz.

Mỗi mẫu: image (1,RES,RES) [0,1], seed_vec (n_seeds,) one-hot, targets
(3,)=[v12,v21,volfrac_achieved] (không chuẩn hoá, giữ đơn vị vật lý gốc).
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class AuxeticDataset(Dataset):
    def __init__(self, npz_path: str, include_f1f2: bool = False):
        """include_f1f2: doc them f1=E11/E0, f2=E22/E0 tu {split}_ext.npz
        (backfill 2026-08-05, xem analysis/scripts/backfill_f1_f2_npz.py) -
        targets thanh 5 chieu [v12,v21,volfrac_achieved,f1,f2] thay vi 3.
        Mac dinh False (hanh vi cu, tuong thich nguoc voi checkpoint san co)."""
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"]                 # (N, RES, RES) float32 [0,1]
        self.v12 = data["v12"].astype(np.float32)
        self.v21 = data["v21"].astype(np.float32)
        self.volfrac_achieved = data["volfrac_achieved"].astype(np.float32)
        self.seed_onehot = data["seed_onehot"].astype(np.float32)  # (N, n_seeds)
        self.seed_classes = data["seed_classes"]      # tên seed theo thứ tự cột onehot
        self.include_f1f2 = include_f1f2
        if include_f1f2:
            self.f1 = data["f1"].astype(np.float32)
            self.f2 = data["f2"].astype(np.float32)

    def __len__(self):
        return len(self.images)

    @property
    def n_seeds(self) -> int:
        return self.seed_onehot.shape[1]

    @property
    def n_targets(self) -> int:
        return 5 if self.include_f1f2 else 3

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, RES, RES)
        seed_vec = torch.from_numpy(self.seed_onehot[idx])
        vals = [self.v12[idx], self.v21[idx], self.volfrac_achieved[idx]]
        if self.include_f1f2:
            vals += [self.f1[idx], self.f2[idx]]
        targets = torch.tensor(vals, dtype=torch.float32)
        return image, seed_vec, targets


if __name__ == "__main__":
    # Self-test nhanh: chạy `python3 pipeline/phase4_surrogate/dataset.py`
    # để kiểm tra file .npz đọc đúng trước khi viết model/train.
    import os
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "outputs", "phase3", "val.npz"
    )
    ds = AuxeticDataset(path)
    print(f"Số mẫu: {len(ds)}, số seed classes: {ds.n_seeds}")
    img, seed_vec, targets = ds[0]
    print(f"image shape: {img.shape}, seed_vec shape: {seed_vec.shape}, "
          f"targets: {targets.tolist()}")
"""
Phase 4 - dataset.py
=====================
PyTorch Dataset đọc trực tiếp file .npz sinh ra từ Phase 3
(outputs/phase3/{train,val,test}.npz).

Mỗi mẫu trả về:
    image      : Tensor (1, RES, RES) float32, giá trị [0,1]
    seed_vec   : Tensor (n_seeds,) float32, one-hot - dùng làm input phụ
    targets    : Tensor (3,) float32 = [v12, v21, volfrac_achieved]

Không cần sửa file này để chạy baseline. Chỉ sửa nếu bạn:
  - Thêm target mới (VD f1, f2 sau khi làm bước 4.0 mở rộng homogenization)
    -> sửa trong hàm __getitem__, phần "targets = ..."
  - Đổi cách chuẩn hoá target (hiện KHÔNG scale v12/v21, giữ đơn vị vật lý)
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class AuxeticDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"]                 # (N, RES, RES) float32 [0,1]
        self.v12 = data["v12"].astype(np.float32)
        self.v21 = data["v21"].astype(np.float32)
        self.volfrac_achieved = data["volfrac_achieved"].astype(np.float32)
        self.seed_onehot = data["seed_onehot"].astype(np.float32)  # (N, n_seeds)
        self.seed_classes = data["seed_classes"]      # tên seed theo thứ tự cột onehot

    def __len__(self):
        return len(self.images)

    @property
    def n_seeds(self) -> int:
        return self.seed_onehot.shape[1]

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, RES, RES)
        seed_vec = torch.from_numpy(self.seed_onehot[idx])
        targets = torch.tensor(
            [self.v12[idx], self.v21[idx], self.volfrac_achieved[idx]],
            dtype=torch.float32,
        )
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
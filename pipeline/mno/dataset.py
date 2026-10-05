"""Dataset adapter for density and displacement-field NPZ files."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class HomogenizationDataset(Dataset):
    """Read ``images`` and ``displacements`` arrays from an NPZ file.

    Args:
        path: NPZ path. ``images`` is ``(N,H,W)`` or ``(N,1,H,W)`` and
            ``displacements`` is ``(N,18,H,W)``.
    """

    def __init__(self, path: str) -> None:
        data = np.load(path)
        if "images" not in data or "displacements" not in data:
            raise ValueError(
                "NPZ must contain images and displacements arrays"
            )
        self.images = torch.as_tensor(data["images"], dtype=torch.float32)
        self.targets = torch.as_tensor(
            data["displacements"], dtype=torch.float32
        )
        if self.images.ndim == 3:
            self.images = self.images.unsqueeze(1)
        if self.images.ndim != 4 or self.images.size(1) != 1:
            raise ValueError("images must have shape (N,1,H,W) or (N,H,W)")
        if self.targets.ndim != 4 or self.targets.size(1) != 18:
            raise ValueError("displacements must have shape (N,18,H,W)")

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.images.size(0)

    def __getitem__(self, index: int):
        """Return one density field and its displacement target."""
        return self.images[index], self.targets[index]

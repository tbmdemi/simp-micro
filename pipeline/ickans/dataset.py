"""Synthetic composite strain-energy data for ICKAN experiments."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CompositeEnergyDataset(Dataset):
    """Generate deterministic isotropic two-phase elastic energy samples."""

    def __init__(self, size: int = 4096, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.strain = torch.rand(size, 3, generator=generator) * 0.4
        self.strain[:, 0] -= 0.2
        self.strain[:, 1] -= 0.2
        self.volume_fraction = torch.rand(size, 1, generator=generator)
        self.features = torch.cat([self.strain, self.volume_fraction], dim=1)
        bulk = 2.0 + 8.0 * self.volume_fraction
        shear = 1.0 + 4.0 * self.volume_fraction
        trace = self.strain[:, :2].sum(dim=1, keepdim=True)
        self.energy = (
            0.5 * bulk * trace.square()
            + shear * self.strain.square().sum(dim=1, keepdim=True)
        )

    def __len__(self) -> int:
        """Return the generated sample count."""
        return self.features.size(0)

    def __getitem__(self, index: int):
        """Return composite features and reference energy."""
        return self.features[index], self.energy[index]

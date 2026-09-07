"""Evaluate ICKAN energy fit and tangent positive definiteness."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import CompositeEnergyDataset
from .model import ICKAN


def main() -> None:
    """Evaluate R2 and the minimum tangent eigenvalue."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    dataset = CompositeEnergyDataset(size=256, seed=1)
    model = ICKAN(input_dim=4)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
    )
    model.eval()
    features, target = next(iter(DataLoader(dataset, batch_size=64)))
    with torch.no_grad():
        prediction = model(features)
    ss_res = (target - prediction).square().sum()
    ss_tot = (target - target.mean()).square().sum()
    hessian = model.tangent_hessian(features[:8])
    minimum_eigenvalue = torch.linalg.eigvalsh(hessian).min()
    print(
        {
            "r2": float(1.0 - ss_res / ss_tot),
            "min_tangent_eigenvalue": float(minimum_eigenvalue),
        }
    )


if __name__ == "__main__":
    main()

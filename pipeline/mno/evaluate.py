"""Evaluate MNO relative L2 error and inference throughput."""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from .dataset import HomogenizationDataset
from .model import MambaNeuralOperator


def main() -> None:
    """Evaluate a trained MNO checkpoint on an NPZ dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HomogenizationDataset(args.data)
    model = MambaNeuralOperator().to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(dataset, batch_size=1)
    errors = []
    start = time.perf_counter()
    with torch.no_grad():
        for density, target in loader:
            prediction = model(density.to(device)).cpu()
            errors.append(
                (prediction - target).norm() / target.norm().clamp_min(1e-12)
            )
    elapsed = time.perf_counter() - start
    print(
        {
            "relative_l2": float(torch.stack(errors).mean()),
            "samples_per_second": len(dataset) / elapsed,
        }
    )


if __name__ == "__main__":
    main()

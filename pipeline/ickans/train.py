"""Train the convex constitutive ICKAN model."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import CompositeEnergyDataset
from .model import ICKAN


def main() -> None:
    """Fit ICKAN to synthetic two-phase elastic energies."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="outputs/ickan.pt")
    args = parser.parse_args()
    dataset = CompositeEnergyDataset()
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = ICKAN(input_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        total = 0.0
        for features, target in loader:
            loss = torch.nn.functional.mse_loss(model(features), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} mse={total / len(loader):.6g}")
    torch.save({"model_state_dict": model.state_dict()}, args.output)


if __name__ == "__main__":
    main()

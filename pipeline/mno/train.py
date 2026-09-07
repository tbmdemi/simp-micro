"""Train the MNO against precomputed FE displacement fields."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import HomogenizationDataset
from .model import MambaNeuralOperator


def main() -> None:
    """Run a compact supervised MNO training job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="outputs/mno.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HomogenizationDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = MambaNeuralOperator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for density, target in loader:
            density, target = density.to(device), target.to(device)
            loss = torch.nn.functional.mse_loss(model(density), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch={epoch + 1} loss={total / len(loader):.6g}")
    torch.save(
        {"model_state_dict": model.state_dict(), "backend": model.backend},
        args.output,
    )


if __name__ == "__main__":
    main()

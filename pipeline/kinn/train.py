"""Train a KINN with the differentiable deep-energy objective."""

from __future__ import annotations

import argparse

import torch

from .model import KINN
from .physics import deep_energy_loss


def main() -> None:
    """Run a small physics-only KINN optimization experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--output", default="outputs/kinn.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        coordinates = torch.rand(
            1, args.points, 2, device=device, requires_grad=True
        )
        material = torch.ones(1, args.points, 1, device=device)
        loss = deep_energy_loss(model, coordinates, material)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} energy={loss.item():.6g}")
    torch.save({"model_state_dict": model.state_dict()}, args.output)


if __name__ == "__main__":
    main()

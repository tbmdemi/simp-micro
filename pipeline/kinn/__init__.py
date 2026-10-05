"""KAN-informed differentiable mechanics components."""

from .model import KINN
from .physics import deep_energy_loss, solve_linear_amg

__all__ = ["KINN", "deep_energy_loss", "solve_linear_amg"]

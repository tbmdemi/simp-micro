"""GPU surrogate for density-to-displacement homogenization fields."""

from __future__ import annotations

import torch
from torch import nn


class MambaNeuralOperator(nn.Module):
    """Predict 18 displacement channels from a density field.

    The implementation uses ``mamba_ssm`` when it is installed. A bidirectional
    GRU is the deterministic PyTorch fallback, so the model remains runnable in
    environments where Mamba CUDA kernels are unavailable.

    Args:
        d_model: Token embedding width.
        n_outputs: Number of displacement channels, 18 by default.
        layers: Number of recurrent/ Mamba layers per direction.
        use_mamba: Whether to try the optional Mamba backend.

    Returns:
        A module whose forward pass maps ``(B, 1, H, W)`` to
        ``(B, n_outputs, H, W)``.
    """

    def __init__(
        self,
        d_model: int = 96,
        n_outputs: int = 18,
        layers: int = 2,
        use_mamba: bool = True,
    ) -> None:
        super().__init__()
        if d_model < 8 or n_outputs < 1 or layers < 1:
            raise ValueError("d_model, n_outputs and layers must be positive")
        self.d_model = d_model
        self.n_outputs = n_outputs
        self.input_projection = nn.Conv2d(1, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.backend = "gru"
        self.forward_scan = None
        self.backward_scan = None
        if use_mamba:
            try:
                from mamba_ssm import Mamba

                self.forward_scan = nn.ModuleList(
                    [
                        Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                        for _ in range(layers)
                    ]
                )
                self.backward_scan = nn.ModuleList(
                    [
                        Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                        for _ in range(layers)
                    ]
                )
                self.backend = "mamba"
            except (ImportError, RuntimeError):
                self.forward_scan = None
                self.backward_scan = None
        if self.backend == "gru":
            self.forward_scan = nn.GRU(
                d_model, d_model, num_layers=layers, batch_first=True
            )
            self.backward_scan = nn.GRU(
                d_model, d_model, num_layers=layers, batch_first=True
            )
        self.output_projection = nn.Linear(2 * d_model, n_outputs)

    def _scan(self, tokens: torch.Tensor, module: nn.Module) -> torch.Tensor:
        """Apply one directional sequence model."""
        if self.backend == "gru":
            return module(tokens)[0]
        for block in module:
            tokens = tokens + block(self.norm(tokens))
        return tokens

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """Predict displacement fields while preserving the input grid."""
        if density.ndim != 4 or density.size(1) != 1:
            raise ValueError("density must have shape (B, 1, H, W)")
        batch, _, height, width = density.shape
        tokens = self.input_projection(density).flatten(2).transpose(1, 2)
        forward = self._scan(tokens, self.forward_scan)
        reverse = torch.flip(tokens, dims=[1])
        backward = torch.flip(
            self._scan(reverse, self.backward_scan), dims=[1]
        )
        fields = self.output_projection(torch.cat([forward, backward], dim=-1))
        return fields.transpose(1, 2).reshape(
            batch, self.n_outputs, height, width
        )

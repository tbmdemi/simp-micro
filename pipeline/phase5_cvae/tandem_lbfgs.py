"""Differentiable tandem inverse design for the Phase 5 generator.

The generator and surrogate remain frozen.  Only the latent vector is
optimized, so the routine can be used with both the convolutional decoder and
the continuous WIRE decoder without changing either model's public API.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def tandem_inverse_design_lbfgs(
    target_poisson: torch.Tensor,
    generator_model: torch.nn.Module,
    surrogate_model: torch.nn.Module,
    steps: int = 50,
    condition: Optional[torch.Tensor] = None,
    seed_vec: Optional[torch.Tensor] = None,
    initial_z: Optional[torch.Tensor] = None,
    latent_dim: Optional[int] = None,
    learning_rate: float = 1.0,
    history: Optional[list[float]] = None,
) -> Dict[str, Any]:
    """Optimize a latent vector against a surrogate Poisson target.

    Args:
        target_poisson: Target ``[v12, v21]`` tensor, shape ``(2,)`` or
            ``(1, 2)``.
        generator_model: A frozen CVAE-like model exposing ``decoder(z, c)``.
        surrogate_model: A frozen surrogate exposing
            ``surrogate(image, seed_vec)``.
        steps: Maximum number of L-BFGS closure evaluations.
        condition: Full generator condition. Defaults to target_poisson.
        seed_vec: One-hot seed vector for the surrogate. Required when the
            surrogate has no default seed contract.
        initial_z: Optional starting latent vector. Defaults to zeros.
        latent_dim: Latent width when ``initial_z`` is omitted. Defaults to
            ``generator_model.latent_dim``.
        learning_rate: L-BFGS step size.
        history: Optional list to extend with objective values.

    Returns:
        A dictionary containing ``z``, ``image``, ``prediction``, ``loss`` and
        ``history``. All returned tensors retain the generator's device.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if target_poisson.numel() != 2:
        raise ValueError("target_poisson must contain exactly [v12, v21]")

    device = next(generator_model.parameters(), target_poisson).device
    target = target_poisson.reshape(1, 2).to(device=device, dtype=torch.float32)
    if condition is None:
        condition = target
    condition = condition.reshape(1, -1).to(device=device, dtype=torch.float32)

    if initial_z is None:
        width = latent_dim or getattr(generator_model, "latent_dim", None)
        if width is None:
            raise ValueError("latent_dim is required when initial_z is omitted")
        z = torch.zeros(1, width, device=device, requires_grad=True)
    else:
        z = initial_z.reshape(1, -1).to(device=device, dtype=torch.float32)
        z = z.detach().requires_grad_(True)

    if seed_vec is None:
        n_seeds = getattr(surrogate_model, "n_seeds", None)
        if n_seeds is None:
            raise ValueError("seed_vec is required for this surrogate")
        seed_vec = torch.zeros(1, n_seeds, device=device)
        seed_vec[:, 0] = 1.0
    else:
        seed_vec = seed_vec.reshape(1, -1).to(device=device, dtype=torch.float32)

    model_states = (generator_model.training, surrogate_model.training)
    generator_model.eval()
    surrogate_model.eval()
    values = history if history is not None else []
    optimizer = torch.optim.LBFGS(
        [z], lr=learning_rate, max_iter=1, line_search_fn="strong_wolfe"
    )

    def objective() -> torch.Tensor:
        image = generator_model.decoder(z, condition)
        prediction = surrogate_model(image, seed_vec)
        predicted_poisson = prediction[:, :2]
        return F.mse_loss(predicted_poisson, target)

    for _ in range(steps):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            values.append(float(loss.detach().cpu()))
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        image = generator_model.decoder(z, condition)
        prediction = surrogate_model(image, seed_vec)
        final_loss = F.mse_loss(prediction[:, :2], target)

    generator_model.train(model_states[0])
    surrogate_model.train(model_states[1])
    return {
        "z": z.detach(),
        "image": image.detach(),
        "prediction": prediction.detach(),
        "loss": float(final_loss.cpu()),
        "history": values,
    }
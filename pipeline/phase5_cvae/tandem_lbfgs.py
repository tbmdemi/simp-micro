"""Differentiable tandem inverse design for the Phase 5 generator.

The generator remains frozen. Only the latent vector is optimized, so the
routine can be used with both the convolutional decoder and the continuous
WIRE decoder without changing either model's public API.

Two sources of gradient guidance are supported via `guidance_source`:

- "surrogate" (default, backward-compatible): gradient comes from a frozen
  Phase 4 `SurrogateCNN`-like model. Cheap, but the surrogate is known to be
  exploitable by the decoder (see `losses.py` module docstring warning) -
  optimizing z against it can converge to a latent that fools the surrogate
  without being physically auxetic.
- "real_physics": gradient comes from `losses.real_physics_loss`, which
  calls the REAL differentiable FE solver (`real_physics.RealPhysicsNu`,
  exact analytic gradient, no surrogate approximation involved). Slower
  (one FE-solve per step) but not exploitable the same way - this is the
  "physics-guided latent refinement" building block for the diffusion-prior
  roadmap (see `plan.md`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

try:
    # Khi tandem_lbfgs.py được import theo đường dẫn dotted đầy đủ
    # (pipeline.phase5_cvae.tandem_lbfgs, vd tests/test_tandem_lbfgs.py).
    from .losses import real_physics_loss
    from .real_physics import RealPhysicsNu
    from .verify_fe import FE_PARAMS as _DEFAULT_FE_PARAMS
except ImportError:
    # Khi tandem_lbfgs.py được import bằng "from tandem_lbfgs import ..."
    # sau sys.path.insert(dirname(__file__)) - xem README "bare-import
    # landmine" note trong CLAUDE.md/memory (phase4/phase5 module trùng tên
    # sibling collide trong sys.modules nếu import lẫn lộn 2 kiểu).
    from losses import real_physics_loss
    from real_physics import RealPhysicsNu
    from verify_fe import FE_PARAMS as _DEFAULT_FE_PARAMS

_GUIDANCE_SOURCES = ("surrogate", "real_physics")


def tandem_inverse_design_lbfgs(
    target_poisson: torch.Tensor,
    generator_model: torch.nn.Module,
    surrogate_model: Optional[torch.nn.Module] = None,
    steps: int = 50,
    condition: Optional[torch.Tensor] = None,
    seed_vec: Optional[torch.Tensor] = None,
    initial_z: Optional[torch.Tensor] = None,
    latent_dim: Optional[int] = None,
    learning_rate: float = 1.0,
    history: Optional[list[float]] = None,
    guidance_source: str = "surrogate",
    fe_params: Optional[dict] = None,
    subsample: Optional[int] = None,
    n_workers: int = 0,
) -> Dict[str, Any]:
    """Optimize a latent vector against a Poisson-ratio target.

    Args:
        target_poisson: Target ``[v12, v21]`` tensor, shape ``(2,)`` or
            ``(1, 2)``.
        generator_model: A frozen CVAE-like model exposing ``decoder(z, c)``.
        surrogate_model: A frozen surrogate exposing
            ``surrogate(image, seed_vec)``. Required when
            ``guidance_source="surrogate"``; ignored (may be ``None``) when
            ``guidance_source="real_physics"``.
        steps: Maximum number of L-BFGS closure evaluations.
        condition: Full generator condition. Defaults to target_poisson.
        seed_vec: One-hot seed vector for the surrogate. Required when the
            surrogate has no default seed contract and
            ``guidance_source="surrogate"``. Unused for ``"real_physics"``.
        initial_z: Optional starting latent vector. Defaults to zeros.
        latent_dim: Latent width when ``initial_z`` is omitted. Defaults to
            ``generator_model.latent_dim``.
        learning_rate: L-BFGS step size.
        history: Optional list to extend with objective values.
        guidance_source: ``"surrogate"`` (default, backward-compatible) or
            ``"real_physics"`` - xem docstring module này để biết đánh đổi.
        fe_params: dict FE params (``nelx, nely, penal, E0, Emin, nu,
            rho0``) dùng khi ``guidance_source="real_physics"``. Mặc định
            (``None``) dùng ``verify_fe.FE_PARAMS`` (nelx=nely=50, khớp
            cách sinh dataset production). Bỏ qua khi ``guidance_source=
            "surrogate"``.
        subsample: chỉ dùng khi ``guidance_source="real_physics"`` - xem
            ``losses.real_physics_loss`` (mỗi bước tối ưu ở đây chỉ có 1
            mẫu nên tham số này hầu như không cần, giữ lại để đồng nhất
            chữ ký với ``real_physics_loss``).
        n_workers: chỉ dùng khi ``guidance_source="real_physics"`` - số
            worker `multiprocessing.Pool` cho FE-solve (0 = tuần tự).

    Returns:
        A dictionary containing ``z``, ``image``, ``prediction``, ``loss``
        and ``history``. All returned tensors retain the generator's
        device. ``prediction`` is ``[v12, v21]`` regardless of
        ``guidance_source``.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if target_poisson.numel() != 2:
        raise ValueError("target_poisson must contain exactly [v12, v21]")
    if guidance_source not in _GUIDANCE_SOURCES:
        raise ValueError(
            f"guidance_source={guidance_source!r} không hợp lệ - "
            f"chỉ hỗ trợ {_GUIDANCE_SOURCES}"
        )
    if guidance_source == "surrogate" and surrogate_model is None:
        raise ValueError(
            "surrogate_model is required when guidance_source='surrogate'"
        )
    if guidance_source == "real_physics" and fe_params is None:
        fe_params = _DEFAULT_FE_PARAMS

    device = next(generator_model.parameters(), target_poisson).device
    target = target_poisson.reshape(1, 2).to(
        device=device, dtype=torch.float32
    )
    if condition is None:
        condition = target
    condition = condition.reshape(1, -1).to(device=device, dtype=torch.float32)

    if initial_z is None:
        width = latent_dim or getattr(generator_model, "latent_dim", None)
        if width is None:
            raise ValueError(
                "latent_dim is required when initial_z is omitted"
            )
        z = torch.zeros(1, width, device=device, requires_grad=True)
    else:
        z = initial_z.reshape(1, -1).to(device=device, dtype=torch.float32)
        z = z.detach().requires_grad_(True)

    if guidance_source == "surrogate":
        if seed_vec is None:
            n_seeds = getattr(surrogate_model, "n_seeds", None)
            if n_seeds is None:
                raise ValueError("seed_vec is required for this surrogate")
            seed_vec = torch.zeros(1, n_seeds, device=device)
            seed_vec[:, 0] = 1.0
        else:
            seed_vec = seed_vec.reshape(1, -1).to(
                device=device, dtype=torch.float32
            )

    generator_was_training = generator_model.training
    generator_model.eval()
    if surrogate_model is not None:
        surrogate_was_training = surrogate_model.training
        surrogate_model.eval()

    values = history if history is not None else []
    optimizer = torch.optim.LBFGS(
        [z], lr=learning_rate, max_iter=1, line_search_fn="strong_wolfe"
    )

    def objective() -> torch.Tensor:
        image = generator_model.decoder(z, condition)
        if guidance_source == "real_physics":
            return real_physics_loss(
                image,
                target,
                fe_params,
                subsample=subsample,
                n_workers=n_workers,
            )
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
        if guidance_source == "real_physics":
            density = image.squeeze(1) if image.dim() == 4 else image
            density_fe = F.interpolate(
                density.unsqueeze(1),
                size=(fe_params["nely"], fe_params["nelx"]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            prediction = RealPhysicsNu.apply(
                density_fe,
                fe_params.get("penal", 3.0),
                fe_params.get("E0", 199.0),
                fe_params.get("Emin", 1e-9),
                fe_params.get("nu", 0.3),
                fe_params.get("rho0", 1.0),
                n_workers,
            )
        else:
            prediction = surrogate_model(image, seed_vec)[:, :2]
        final_loss = F.mse_loss(prediction, target)

    generator_model.train(generator_was_training)
    if surrogate_model is not None:
        surrogate_model.train(surrogate_was_training)
    return {
        "z": z.detach(),
        "image": image.detach(),
        "prediction": prediction.detach(),
        "loss": float(final_loss.cpu()),
        "history": values,
    }

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

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

try:
    # Khi tandem_lbfgs.py được import theo đường dẫn dotted đầy đủ
    # (pipeline.phase5_cvae.tandem_lbfgs, vd tests/test_tandem_lbfgs.py).
    from .heaviside import heaviside_projection_torch, project_for_fe
    from .losses import real_physics_loss
    from .manuf_penalty import corner_contact_penalty, thin_feature_penalty
    from .real_physics import RealPhysicsNu
    from .realization import filtered_design, realize_shifted
    from .verify_fe import FE_PARAMS as _DEFAULT_FE_PARAMS
except ImportError:
    # Khi tandem_lbfgs.py được import bằng "from tandem_lbfgs import ..."
    # sau sys.path.insert(dirname(__file__)) - xem README "bare-import
    # landmine" note trong CLAUDE.md/memory (phase4/phase5 module trùng tên
    # sibling collide trong sys.modules nếu import lẫn lộn 2 kiểu).
    from heaviside import heaviside_projection_torch, project_for_fe
    from losses import real_physics_loss
    from manuf_penalty import corner_contact_penalty, thin_feature_penalty
    from real_physics import RealPhysicsNu
    from realization import filtered_design, realize_shifted
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
    projection_betas: Optional[Sequence[float]] = None,
    periodic: bool = False,
    corner_weight: float = 0.0,
    thin_weight: float = 0.0,
    realization_shifts: Optional[Sequence[Sequence[float]]] = None,
    realization_sigma: float = 0.5,
    fe_upsample: int = 1,
    fe_upsample_last_only: bool = False,
    robust_etas: Optional[Sequence[float]] = None,
    robust_sigma: float = 1.0,
    design_filter_sigma: Optional[float] = None,
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
        projection_betas: chỉ dùng khi ``guidance_source="real_physics"``.
            None (mặc định) = hành vi cũ: FE trên ảnh liên tục, resize
            bilinear. Có giá trị (vd ``(1, 4, 16, 64)``) = refine nhận thức
            nhị phân hóa (plan.md v3 P1.1e): ảnh -> [force_periodic] ->
            Heaviside projection (η=0,5) -> resize nearest khớp verify, β
            tăng dần theo từng giai đoạn; ``steps`` chia đều cho các giai
            đoạn và L-BFGS khởi động lại mỗi khi đổi β (bộ nhớ độ cong của
            giai đoạn trước không còn đúng với objective mới).
        periodic: áp force_periodic khả vi trước projection - chỉ có tác
            dụng khi ``projection_betas`` được đặt; bật khi verify dùng
            ``--force-periodic``.
        corner_weight: hệ số phạt chạm góc pixel (bản lề 1 nút), xem
            ``manuf_penalty.corner_contact_penalty``. 0 = tắt (hành vi cũ).
            Cần ``projection_betas`` (phạt tính trên ảnh sau Heaviside β).
        thin_weight: hệ số phạt nét mảnh < 3 px, xem
            ``manuf_penalty.thin_feature_penalty``. 0 = tắt.
        realization_shifts: N1 (plan.md) - danh sách (dy, dx) theo phần tử
            lưới FE. Có giá trị = loss là trung bình MSE trên chuỗi verify
            VÀ K hiện thực hóa dịch lệch lưới (``realization.realize_shifted``)
            để không thưởng thiết kế khai thác lỗi rời rạc hóa. None = tắt.
            Cần ``projection_betas``.
        realization_sigma: độ mượt của hiện thực hóa (đơn vị phần tử FE).
        fe_upsample: đối chứng của N1 - giải FE trên lưới mịn hơn k lần
            (lặp mỗi phần tử k×k, cùng hình học) thay vì lưới verify. 1 = tắt.
            Cần ``projection_betas``.
        fe_upsample_last_only: N2-E2 - chỉ giải FE lưới mịn ở mức β cuối
            (mức quyết định hình học nhị phân), các mức trước giữ lưới verify
            để rẻ. False = mọi mức (hành vi cũ của ``fe_upsample``).
        robust_etas: N2-E1 robust formulation (Wang, Lazarov & Sigmund 2011)
            - các ngưỡng chiếu cho bản giãn/co, vd ``(0.25, 0.75)``. Loss =
            0,5·MSE(chuỗi verify) + 0,5·trung bình MSE(các bản co/giãn) (dạng
            trọng số GLOnet: 0,5 gốc / 0,25 / 0,25). Bản gốc là chính chuỗi
            verify để vẫn tối ưu đúng thứ được kiểm. None = tắt.
        robust_sigma: độ mượt (phần tử FE) của trường dùng để co/giãn - quyết
            định khoảng co/giãn σ·Φ⁻¹(eta), tức kích thước nét tối thiểu.
        design_filter_sigma: N2-E1′ - thiết kế vật lý = Heaviside(lọc Gauss σ
            (ảnh decoder) lấy mẫu lưới FE) (``realization.filtered_design``)
            thay cho chuỗi resize nearest. Khi đặt, bản co/giãn của
            ``robust_etas`` cũng sinh từ cùng trường đã lọc (``robust_sigma``
            bị bỏ qua) - đúng robust formulation chuẩn. None = tắt.

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
    if (realization_shifts or fe_upsample != 1 or robust_etas) and not (
        guidance_source == "real_physics" and projection_betas
    ):
        raise ValueError(
            "realization_shifts/fe_upsample/robust_etas cần guidance_source="
            "'real_physics' và projection_betas"
        )
    if fe_upsample < 1:
        raise ValueError("fe_upsample phải >= 1")
    if design_filter_sigma is not None:
        if not (guidance_source == "real_physics" and projection_betas):
            raise ValueError(
                "design_filter_sigma cần guidance_source='real_physics' và "
                "projection_betas"
            )
        if periodic or realization_shifts:
            # Hai biểu diễn thiết kế khác nhau - không trộn trong 1 loss.
            raise ValueError(
                "design_filter_sigma không dùng chung periodic/"
                "realization_shifts"
            )
    use_manuf = corner_weight > 0 or thin_weight > 0
    if use_manuf and not (
        guidance_source == "real_physics" and projection_betas
    ):
        # Phạt chỉ có nghĩa trên ảnh đã chiếu gần nhị phân (K1) - trên ảnh
        # xám, đa thức chạm góc/opening không khớp thứ verify đo.
        raise ValueError(
            "corner_weight/thin_weight cần guidance_source='real_physics' "
            "và projection_betas"
        )

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

    def density_for_fe(image: torch.Tensor, beta) -> torch.Tensor:
        """Ảnh decoder -> mật độ đưa vào FE. beta=None: hành vi cũ (ảnh
        liên tục, real_physics_loss tự resize bilinear)."""
        if beta is None:
            return image
        if design_filter_sigma is not None:
            return filtered_design(
                image,
                beta,
                fe_params["nely"],
                fe_params["nelx"],
                design_filter_sigma,
            )
        # Đã ở đúng lưới FE -> bilinear cùng kích thước bên trong
        # real_physics_loss là identity.
        return project_for_fe(
            image, beta, fe_params["nely"], fe_params["nelx"], periodic
        )

    def objective(beta) -> torch.Tensor:
        image = generator_model.decoder(z, condition)
        if guidance_source == "real_physics":
            density = density_for_fe(image, beta)
            if beta is not None and realization_shifts:
                # Gộp chuỗi verify + K hiện thực hóa thành 1 batch FE: MSE
                # trung bình = kỳ vọng sai số trên các cách đặt lưới.
                density = torch.cat(
                    [
                        density,
                        realize_shifted(
                            image,
                            beta,
                            fe_params["nely"],
                            fe_params["nelx"],
                            realization_shifts,
                            realization_sigma,
                        ),
                    ]
                )
            fine = (
                beta is not None
                and fe_upsample > 1
                and (not fe_upsample_last_only or beta == last_beta)
            )

            def fe_loss(dens: torch.Tensor) -> torch.Tensor:
                """MSE FE trên 1 batch mật độ, trên lưới mịn nếu ``fine``."""
                p = fe_params
                if fine:
                    # Cùng hình học, lưới mịn hơn: lặp phần tử k×k.
                    k = fe_upsample
                    dens = dens.repeat_interleave(k, -2)
                    dens = dens.repeat_interleave(k, -1)
                    p = dict(
                        fe_params,
                        nely=fe_params["nely"] * k,
                        nelx=fe_params["nelx"] * k,
                    )
                return real_physics_loss(
                    dens,
                    target.expand(dens.shape[0], -1),
                    p,
                    subsample=subsample,
                    n_workers=n_workers,
                )

            loss = fe_loss(density)
            if beta is not None and robust_etas:
                if design_filter_sigma is not None:
                    # E1′: co/giãn từ CÙNG trường đã lọc với bản gốc.
                    pert = torch.cat(
                        [
                            filtered_design(
                                image,
                                beta,
                                fe_params["nely"],
                                fe_params["nelx"],
                                design_filter_sigma,
                                eta,
                            )
                            for eta in robust_etas
                        ]
                    )
                else:
                    pert = torch.cat(
                        [
                            realize_shifted(
                                image,
                                beta,
                                fe_params["nely"],
                                fe_params["nelx"],
                                [(0.0, 0.0)],
                                robust_sigma,
                                eta,
                            )
                            for eta in robust_etas
                        ]
                    )
                loss = 0.5 * loss + 0.5 * fe_loss(pert)
            if use_manuf:
                # Phạt trên ảnh 64² sau Heaviside cùng β - đúng lưới mà
                # check_connectivity đo chế tạo (không phải lưới FE 50²).
                x = heaviside_projection_torch(image, beta)
                loss = loss + corner_weight * corner_contact_penalty(x)
                loss = loss + thin_weight * thin_feature_penalty(x)
            return loss
        prediction = surrogate_model(image, seed_vec)
        predicted_poisson = prediction[:, :2]
        return F.mse_loss(predicted_poisson, target)

    if guidance_source == "real_physics" and projection_betas:
        betas = [float(b) for b in projection_betas]
        # Chia đều steps cho các mức β, phần dư dồn vào mức cuối (β lớn
        # nhất - gần verify nhất nên đáng nhận thêm bước).
        per = [steps // len(betas)] * len(betas)
        per[-1] += steps - sum(per)
        stages = [(b, n) for b, n in zip(betas, per) if n > 0]
    else:
        stages = [(None, steps)]
    last_beta = stages[-1][0]

    for beta, n_steps in stages:
        optimizer = torch.optim.LBFGS(
            [z], lr=learning_rate, max_iter=1, line_search_fn="strong_wolfe"
        )
        for _ in range(n_steps):

            def closure(beta=beta) -> torch.Tensor:
                optimizer.zero_grad()
                loss = objective(beta)
                loss.backward()
                values.append(float(loss.detach().cpu()))
                return loss

            optimizer.step(closure)

    final_beta = stages[-1][0]
    with torch.no_grad():
        image = generator_model.decoder(z, condition)
        if guidance_source == "real_physics":
            density = density_for_fe(image, final_beta)
            density = density.squeeze(1) if density.dim() == 4 else density
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

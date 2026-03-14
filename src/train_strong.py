from __future__ import annotations

import torch

from .geometry import generate_training_points
from .model import PINN, apply_hard_bc
from .physics import equilibrium_residual, lame_parameters, strain_from_displacement, stress_from_strain, traction


def _boundary_loss(model: PINN, pts: torch.Tensor, normal: torch.Tensor, target_t: torch.Tensor, L: float, lam: float, mu: float) -> torch.Tensor:
    xy = pts.clone().requires_grad_(True)
    u = apply_hard_bc(xy, model(xy), L)
    exx, eyy, exy = strain_from_displacement(xy, u)
    sxx, syy, sxy = stress_from_strain(exx, eyy, exy, lam, mu)
    pred_t = traction(sxx, syy, sxy, normal)
    return torch.mean((pred_t - target_t) ** 2)


def strong_form_loss(model: PINN, points: dict, L: float, P_side: float, P_top: float, lam: float, mu: float) -> torch.Tensor:
    xy = points["domain"].clone().requires_grad_(True)
    u = apply_hard_bc(xy, model(xy), L)

    exx, eyy, exy = strain_from_displacement(xy, u)
    sxx, syy, sxy = stress_from_strain(exx, eyy, exy, lam, mu)
    res = equilibrium_residual(xy, sxx, syy, sxy)
    loss_eq = torch.mean(res ** 2)

    left = points["left"]
    right = points["right"]
    top = points["top"]
    hole = points["hole"]

    n_left = torch.tensor([[-1.0, 0.0]], device=left.device).repeat(left.shape[0], 1)
    n_right = torch.tensor([[1.0, 0.0]], device=right.device).repeat(right.shape[0], 1)
    n_top = torch.tensor([[0.0, 1.0]], device=top.device).repeat(top.shape[0], 1)

    n_hole = -hole / torch.linalg.norm(hole, dim=1, keepdim=True)

    t_left = torch.tensor([[-P_side, 0.0]], device=left.device).repeat(left.shape[0], 1)
    t_right = torch.tensor([[P_side, 0.0]], device=right.device).repeat(right.shape[0], 1)
    t_top = torch.tensor([[0.0, P_top]], device=top.device).repeat(top.shape[0], 1)
    t_hole = torch.zeros((hole.shape[0], 2), device=hole.device)

    loss_left = _boundary_loss(model, left, n_left, t_left, L, lam, mu)
    loss_right = _boundary_loss(model, right, n_right, t_right, L, lam, mu)
    loss_top = _boundary_loss(model, top, n_top, t_top, L, lam, mu)
    loss_hole = _boundary_loss(model, hole, n_hole, t_hole, L, lam, mu)

    return loss_eq + loss_left + loss_right + loss_top + loss_hole


def train_strong_form(cfg, device: torch.device) -> tuple[PINN, list[float]]:
    L = cfg.geometry.L
    R = cfg.geometry.R
    P_side = cfg.load.P_side
    P_top = cfg.load.P_top
    lam, mu = lame_parameters(cfg.material.E, cfg.material.nu)

    model = PINN(width=cfg.train.hidden_width, depth=cfg.train.hidden_depth).to(device)
    history: list[float] = []

    adam = torch.optim.Adam(model.parameters(), lr=cfg.train.adam_lr)
    for step in range(1, cfg.train.adam_steps + 1):
        points = generate_training_points(
            cfg.train.n_domain,
            cfg.train.n_boundary,
            cfg.train.n_hole,
            L,
            R,
            device,
        )
        adam.zero_grad()
        loss = strong_form_loss(model, points, L, P_side, P_top, lam, mu)
        loss.backward()
        adam.step()
        history.append(float(loss.item()))

        if step % cfg.train.print_every == 0:
            print(f"[Strong-Adam] step={step}, loss={loss.item():.6e}")

    fixed_points = generate_training_points(
        cfg.train.n_domain,
        cfg.train.n_boundary,
        cfg.train.n_hole,
        L,
        R,
        device,
    )

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.train.lbfgs_lr,
        max_iter=1,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    for step in range(1, cfg.train.strong_lbfgs_steps + 1):
        def closure():
            lbfgs.zero_grad()
            lbfgs_loss = strong_form_loss(model, fixed_points, L, P_side, P_top, lam, mu)
            lbfgs_loss.backward()
            return lbfgs_loss

        loss = lbfgs.step(closure)
        history.append(float(loss.item()))

        if step % cfg.train.print_every == 0:
            print(f"[Strong-LBFGS] step={step}, loss={loss.item():.6e}")

    return model, history

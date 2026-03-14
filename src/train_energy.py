from __future__ import annotations

import torch

from .geometry import generate_training_points
from .model import PINN, apply_hard_bc
from .physics import domain_area, edge_length, lame_parameters, strain_energy_density, strain_from_displacement, stress_from_strain


def potential_energy_loss(model: PINN, points: dict, L: float, R: float, P_side: float, P_top: float, lam: float, mu: float) -> torch.Tensor:
    area = domain_area(L, R)
    edge = edge_length(L)

    xy = points["domain"].clone().requires_grad_(True)
    u = apply_hard_bc(xy, model(xy), L)
    exx, eyy, exy = strain_from_displacement(xy, u)
    sxx, syy, sxy = stress_from_strain(exx, eyy, exy, lam, mu)

    w = strain_energy_density(exx, eyy, exy, sxx, syy, sxy)
    internal_energy = area * torch.mean(w)

    u_left = apply_hard_bc(points["left"], model(points["left"]), L)
    u_right = apply_hard_bc(points["right"], model(points["right"]), L)
    u_top = apply_hard_bc(points["top"], model(points["top"]), L)

    t_left = torch.tensor([-P_side, 0.0], device=u_left.device)
    t_right = torch.tensor([P_side, 0.0], device=u_right.device)
    t_top = torch.tensor([0.0, P_top], device=u_top.device)

    ext_left = edge * torch.mean(torch.sum(u_left * t_left, dim=1, keepdim=True))
    ext_right = edge * torch.mean(torch.sum(u_right * t_right, dim=1, keepdim=True))
    ext_top = edge * torch.mean(torch.sum(u_top * t_top, dim=1, keepdim=True))

    return internal_energy - (ext_left + ext_right + ext_top)


def train_energy_form(cfg, device: torch.device) -> tuple[PINN, list[float]]:
    L = cfg.geometry.L
    R = cfg.geometry.R
    P_side = cfg.load.P_side
    P_top = cfg.load.P_top
    lam, mu = lame_parameters(cfg.material.E, cfg.material.nu)

    model = PINN(width=cfg.train.hidden_width, depth=cfg.train.hidden_depth).to(device)

    points = generate_training_points(
        cfg.train.n_domain,
        cfg.train.n_boundary,
        cfg.train.n_hole,
        L,
        R,
        device,
    )

    history: list[float] = []
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.train.lbfgs_lr,
        max_iter=1,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    for step in range(1, cfg.train.energy_lbfgs_steps + 1):
        def closure():
            lbfgs.zero_grad()
            loss = potential_energy_loss(model, points, L, R, P_side, P_top, lam, mu)
            loss.backward()
            return loss

        loss = lbfgs.step(closure)
        history.append(float(loss.item()))

        if step % cfg.train.print_every == 0:
            print(f"[Energy-LBFGS] step={step}, potential={loss.item():.6e}")

    return model, history

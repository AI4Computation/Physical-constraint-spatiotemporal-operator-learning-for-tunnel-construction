from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import apply_hard_bc


def save_loss_curve(history: list[float], save_path: Path, title: str, ylabel: str = "loss") -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, len(history) + 1), history, lw=1.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def save_displacement_contour(
    model,
    L: float,
    R: float,
    grid_n: int,
    device: torch.device,
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-L, L, grid_n)
    y = np.linspace(-L, L, grid_n)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    rr = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    mask = rr > R

    xy_t = torch.tensor(xy[mask], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(xy_t)
        disp = apply_hard_bc(xy_t, pred, L).cpu().numpy()

    ux = np.full((xy.shape[0],), np.nan)
    uy = np.full((xy.shape[0],), np.nan)
    ux[mask] = disp[:, 0]
    uy[mask] = disp[:, 1]

    ux = ux.reshape(grid_n, grid_n)
    uy = uy.reshape(grid_n, grid_n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].contourf(xx, yy, ux, levels=60, cmap="RdBu_r")
    axes[0].set_title("u_x")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], shrink=0.88)

    im1 = axes[1].contourf(xx, yy, uy, levels=60, cmap="RdBu_r")
    axes[1].set_title("u_y")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], shrink=0.88)

    fig.suptitle(title)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

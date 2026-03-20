import math

import torch


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def strain_from_displacement(xy: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ux = u[:, 0:1]
    uy = u[:, 1:2]

    dux = grad(ux, xy)
    duy = grad(uy, xy)

    exx = dux[:, 0:1]
    eyy = duy[:, 1:2]
    exy = 0.5 * (dux[:, 1:2] + duy[:, 0:1])
    return exx, eyy, exy


def stress_from_strain(exx: torch.Tensor, eyy: torch.Tensor, exy: torch.Tensor, lam: float, mu: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tr = exx + eyy
    sxx = lam * tr + 2.0 * mu * exx
    syy = lam * tr + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return sxx, syy, sxy


def traction(sxx: torch.Tensor, syy: torch.Tensor, sxy: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    nx = n[:, 0:1]
    ny = n[:, 1:2]
    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny
    return torch.cat([tx, ty], dim=1)


def equilibrium_residual(xy: torch.Tensor, sxx: torch.Tensor, syy: torch.Tensor, sxy: torch.Tensor) -> torch.Tensor:
    dsxx = grad(sxx, xy)
    dsyy = grad(syy, xy)
    dsxy = grad(sxy, xy)

    rx = dsxx[:, 0:1] + dsxy[:, 1:2]
    ry = dsxy[:, 0:1] + dsyy[:, 1:2]
    return torch.cat([rx, ry], dim=1)


def strain_energy_density(exx: torch.Tensor, eyy: torch.Tensor, exy: torch.Tensor, sxx: torch.Tensor, syy: torch.Tensor, sxy: torch.Tensor) -> torch.Tensor:
    return 0.5 * (sxx * exx + syy * eyy + 2.0 * sxy * exy)


def domain_area(L: float, R: float) -> float:
    return (2.0 * L) ** 2 - math.pi * R ** 2


def edge_length(L: float) -> float:
    return 2.0 * L

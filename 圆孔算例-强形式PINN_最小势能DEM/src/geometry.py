import math

import torch


def sample_domain(n: int, L: float, R: float, device: torch.device) -> torch.Tensor:
    points = []
    batch = max(2 * n, 2048)
    while len(points) < n:
        pts = (torch.rand(batch, 2, device=device) * 2.0 - 1.0) * L
        valid = torch.linalg.norm(pts, dim=1) > R
        points.extend(pts[valid].unbind(0))
    return torch.stack(points[:n], dim=0)


def sample_vertical_boundary(x_const: float, n: int, L: float, device: torch.device) -> torch.Tensor:
    y = (torch.rand(n, 1, device=device) * 2.0 - 1.0) * L
    x = torch.full((n, 1), float(x_const), device=device)
    return torch.cat([x, y], dim=1)


def sample_horizontal_boundary(y_const: float, n: int, L: float, device: torch.device) -> torch.Tensor:
    x = (torch.rand(n, 1, device=device) * 2.0 - 1.0) * L
    y = torch.full((n, 1), float(y_const), device=device)
    return torch.cat([x, y], dim=1)


def sample_hole_boundary(n: int, R: float, device: torch.device) -> torch.Tensor:
    theta = torch.linspace(0.0, 2.0 * math.pi, n + 1, device=device)[:-1]
    x = R * torch.cos(theta)
    y = R * torch.sin(theta)
    return torch.stack([x, y], dim=1)


def generate_training_points(n_domain: int, n_boundary: int, n_hole: int, L: float, R: float, device: torch.device) -> dict:
    return {
        "domain": sample_domain(n_domain, L, R, device),
        "left": sample_vertical_boundary(-L, n_boundary, L, device),
        "right": sample_vertical_boundary(L, n_boundary, L, device),
        "top": sample_horizontal_boundary(L, n_boundary, L, device),
        "bottom": sample_horizontal_boundary(-L, n_boundary, L, device),
        "hole": sample_hole_boundary(n_hole, R, device),
    }

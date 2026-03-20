import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 2, width: int = 200, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def apply_hard_bc(xy: torch.Tensor, u_theta: torch.Tensor, L: float) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    u = (x.pow(2) + (y + L).pow(2)) * u_theta[:, 0:1]
    v = (y + L) * u_theta[:, 1:2]
    return torch.cat([u, v], dim=1)

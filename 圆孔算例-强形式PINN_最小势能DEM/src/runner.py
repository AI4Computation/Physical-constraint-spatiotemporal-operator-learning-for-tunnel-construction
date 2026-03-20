from pathlib import Path

import numpy as np
import torch

from .plotting import save_displacement_contour, save_loss_curve
from .train_energy import train_energy_form
from .train_strong import train_strong_form
from .utils import get_device, set_seed


def run_all(cfg, output_root: Path | str = "outputs") -> None:
    output_root = Path(output_root)
    strong_dir = output_root / "strong_form"
    energy_dir = output_root / "energy_form"
    strong_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = get_device()
    print(f"使用设备: {device}")

    strong_model, strong_history = train_strong_form(cfg, device)
    save_loss_curve(strong_history, strong_dir / "training_curve.png", "强形式 PINN（Adam + L-BFGS）")
    save_displacement_contour(
        strong_model,
        cfg.geometry.L,
        cfg.geometry.R,
        cfg.grid_n,
        device,
        strong_dir / "final_displacement.png",
        "强形式 PINN 最终位移场",
    )
    np.savetxt(strong_dir / "loss_history.csv", np.array(strong_history), delimiter=",", header="value", comments="")
    torch.save(strong_model.state_dict(), strong_dir / "model.pth")

    energy_model, energy_history = train_energy_form(cfg, device)
    save_loss_curve(energy_history, energy_dir / "training_curve.png", "能量 PINN（L-BFGS）", ylabel="势能")
    save_displacement_contour(
        energy_model,
        cfg.geometry.L,
        cfg.geometry.R,
        cfg.grid_n,
        device,
        energy_dir / "final_displacement.png",
        "能量 PINN 最终位移场",
    )
    np.savetxt(energy_dir / "loss_history.csv", np.array(energy_history), delimiter=",", header="value", comments="")
    torch.save(energy_model.state_dict(), energy_dir / "model.pth")

    print("运行完成，结果已保存到:")
    print(f"  - {strong_dir}")
    print(f"  - {energy_dir}")

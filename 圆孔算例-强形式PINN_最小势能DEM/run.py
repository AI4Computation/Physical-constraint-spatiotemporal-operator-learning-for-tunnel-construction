import os
import argparse

from src.config import ExperimentConfig
from src.runner import run_all

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="圆孔薄板二维弹性：强形式 PINN 与能量 PINN 对比")
    parser.add_argument("--adam_steps", type=int, default=3000)
    parser.add_argument("--strong_lbfgs_steps", type=int, default=600)
    parser.add_argument("--energy_lbfgs_steps", type=int, default=1000)
    parser.add_argument("--n_domain", type=int, default=5000)
    parser.add_argument("--n_boundary", type=int, default=1000)
    parser.add_argument("--n_hole", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()
    cfg.train.adam_steps = args.adam_steps
    cfg.train.strong_lbfgs_steps = args.strong_lbfgs_steps
    cfg.train.energy_lbfgs_steps = args.energy_lbfgs_steps
    cfg.train.n_domain = args.n_domain
    cfg.train.n_boundary = args.n_boundary
    cfg.train.n_hole = args.n_hole

    run_all(cfg, output_root="outputs")


if __name__ == "__main__":
    main()

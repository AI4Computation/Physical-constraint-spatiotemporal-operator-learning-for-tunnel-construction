from dataclasses import dataclass, field


@dataclass
class MaterialConfig:
    E: float = 1.333
    nu: float = 0.3333


@dataclass
class GeometryConfig:
    L: float = 0.5
    R: float = 0.1


@dataclass
class LoadConfig:
    P_side: float = -4.0
    P_top: float = -2.0


@dataclass
class TrainConfig:
    n_domain: int = 5000
    n_boundary: int = 1000
    n_hole: int = 720
    hidden_width: int = 200
    hidden_depth: int = 3
    adam_steps: int = 3000
    adam_lr: float = 1e-3
    strong_lbfgs_steps: int = 600
    energy_lbfgs_steps: int = 1000
    lbfgs_lr: float = 1.0
    print_every: int = 100


@dataclass
class ExperimentConfig:
    seed: int = 42
    grid_n: int = 181
    material: MaterialConfig = field(default_factory=MaterialConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

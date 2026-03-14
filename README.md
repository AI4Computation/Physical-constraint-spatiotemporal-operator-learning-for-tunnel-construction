# 圆孔薄板 PINN 对比项目

本项目用于研究二维圆孔薄板线弹性问题，比较两类 PINN 方法：
- 强形式 PINN：平衡方程残差 + 力边界残差，优化器采用 Adam + L-BFGS。
- 能量 PINN：基于最小势能原理，优化器直接采用 L-BFGS。

## 1. 问题设置
- 几何域：`[-L, L] x [-L, L]` 去掉圆孔 `r < R`。
- 参数（参考既有强形式代码）：
  - `L = 0.5`
  - `R = 0.1`
  - `E = 1.333`
  - `nu = 0.3333`
  - 左右侧压力 `P_side = -4.0`
  - 顶部压力 `P_top = -2.0`
- 位移边界（硬约束）：
  - 底边竖向约束：`v(x, -L) = 0`
  - 底边中点水平约束：`u(0, -L) = 0`

网络输出统一为二维位移 `(u_x, u_y)`。

## 2. 项目结构

```text
circular_hole_pinn_compare/
  src/
    config.py
    geometry.py
    model.py
    physics.py
    plotting.py
    train_strong.py
    train_energy.py
    runner.py
    utils.py
  scripts/
    check_large_files.py
  .githooks/
    pre-commit
  outputs/
    .gitkeep
  run.py
  requirements.txt
  README.md
  .gitignore
```

## 3. 安装与运行

```bash
pip install -r requirements.txt
python run.py
```

可选参数：

```bash
python run.py --adam_steps 3000 --strong_lbfgs_steps 600 --energy_lbfgs_steps 1000 --n_domain 5000 --n_boundary 1000 --n_hole 720
```

## 4. 输出结果
运行后会自动生成：
- `outputs/strong_form/training_curve.png`
- `outputs/strong_form/final_displacement.png`
- `outputs/energy_form/training_curve.png`
- `outputs/energy_form/final_displacement.png`
- 两个模型各自的 `loss_history.csv`

## 5. 大文件控制
- `.gitignore` 仅忽略常见大二进制类型（如 `*.pth`、`*.npz`、压缩包）。
- 图片、Excel、CSV 可以正常提交。
- 提交前阈值检查由 `pre-commit` 完成，默认单文件上限为 `30MB`。

启用 hook：

```bash
git config core.hooksPath .githooks
```

临时将上限改为 50MB：

```powershell
$env:MAX_FILE_SIZE_MB=50; git commit -m "msg"
```

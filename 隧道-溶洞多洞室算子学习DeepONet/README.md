# 隧道应力预测项目（DeepONet）

本项目用于隧道场景下的应力预测，对比了 4 种核心方法。

## 目录说明

- `methods/`：核心方法代码（4 个）
- `figures/`：每个方法的输出图
- `train_data/`：训练数据
- `src/deeponet_tunnels/`：可复用的数据读取、模型与训练模块
- `requirements.txt`：依赖列表

## 方法与图片对应关系

1. `method_01_baseline_deeponet.py`
   - 图片目录：`figures/method_01_baseline_deeponet/`
2. `method_02_weighted_loss_deeponet.py`
   - 图片目录：`figures/method_02_weighted_loss_deeponet/`
3. `method_03_dualcoord_mlp.py`
   - 图片目录：`figures/method_03_dualcoord_mlp/`
4. `method_04_dualcoord_attention.py`
   - 图片目录：`figures/method_04_dualcoord_attention/`

## 依赖安装

```bash
pip install -r requirements.txt
```

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

# 设置matplotlib字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 25

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检测设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 问题参数
L = 0.5  # 板的半边长 (mm)
R = 0.1   # 圆孔半径 (mm)
E = 1.333 # 杨氏模量 (MPa)
nu = 0.3333   # 泊松比
P = -4.0    # 左右侧载荷 (MPa)
P_top = -2.0

# Lame常数
lam = E * nu / ((1 + nu) * (1 - 2*nu))
mu = E / (2 * (1 + nu))

def load_reference_data():
    """读取参考解数据"""
    try:
        data = np.loadtxt('圆孔.csv', delimiter=',', skiprows=1)
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]  # X, Y, U1, U2
    except:
        print("警告：无法读取参考数据文件")
        return None, None, None, None

def compute_error_at_iteration(model, iteration, ref_x, ref_y, ref_u1, ref_u2, phase=""):
    """计算指定iteration的误差并输出结果"""
    if ref_x is None:
        print(f"{phase}Iteration {iteration}: 无参考数据，跳过误差计算")
        return
    
    # 用参考数据的坐标进行预测
    xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_theta = model(xy_tensor)
        u_pred = apply_hard_bc(xy_tensor, u_theta).cpu().numpy()
    
    # 计算误差
    error_u_vals = u_pred[:, 0] - ref_u1  # 预测值 - 真值
    error_v_vals = u_pred[:, 1] - ref_u2
    
    mean_error_u = np.mean(np.abs(error_u_vals))
    mean_error_v = np.mean(np.abs(error_v_vals))
    max_error_u = np.max(np.abs(error_u_vals))
    max_error_v = np.max(np.abs(error_v_vals))
    
    print(f"{phase}Iteration {iteration}:")
    print(f"  平均误差 - U: {mean_error_u:.6f}, V: {mean_error_v:.6f}")
    print(f"  最大误差 - U: {max_error_u:.6f}, V: {max_error_v:.6f}")

# 神经网络定义
class PINN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 2)
        )
    
    def forward(self, x):
        return self.net(x)

# 硬边界条件
def apply_hard_bc(xy, u_theta):
    """
    应用硬边界条件：
    - u分量：只在底部中点(0, -L)处约束为0
    - v分量：在整个底部(y = -L)约束为0
    """
    x = xy[:, 0:1]  # x坐标
    y = xy[:, 1:2]  # y坐标
    
    # u分量：使用点约束，只在(0, -L)处为0
    u = (x**2 + (y + L)**2) * u_theta[:, 0:1]
    
    # v分量：使用线约束，在整个底部y = -L处为0
    v = (y + L) * u_theta[:, 1:2]
    
    return torch.cat([u, v], dim=1)

# 生成采样点
def generate_points(n_domain, n_boundary_right, n_boundary_other):
    # 域内点（排除圆孔）
    domain_pts = []
    pts = (torch.rand(n_domain*2, 2, device=DEVICE) * 2 - 1) * L
    r = torch.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    valid = (r > R)
    domain_pts.extend(pts[valid])
    domain_pts = domain_pts[:n_domain]
    domain_pts = torch.stack(domain_pts[:])
    
    # 左边界点（受力边界）
    y_right = (torch.rand(n_boundary_right, 1, device=DEVICE) * 2 - 1) * L
    x_right = torch.ones(n_boundary_right, 1, device=DEVICE) * (-L)
    boundary_left = torch.cat([x_right, y_right], dim=1)
    
    # 右边界点（受力边界）
    y_right = (torch.rand(n_boundary_right, 1, device=DEVICE) * 2 - 1) * L
    x_right = torch.ones(n_boundary_right, 1, device=DEVICE) * L
    boundary_right = torch.cat([x_right, y_right], dim=1)
    
    # 上边界点
    x_top = (torch.rand(n_boundary_right, 1, device=DEVICE) * 2 - 1) * L
    y_top = torch.ones(n_boundary_right, 1, device=DEVICE) * L
    boundary_top = torch.cat([x_top, y_top], dim=1)
    
    # 下边界点
    x_bottom = (torch.rand(n_boundary_right, 1, device=DEVICE) * 2 - 1) * L
    y_bottom = torch.ones(n_boundary_right, 1, device=DEVICE) * (-L)
    boundary_bottom = torch.cat([x_bottom, y_bottom], dim=1)

    # 圆孔边界点
    theta = torch.linspace(0, 2*np.pi, n_boundary_other, device=DEVICE)
    x_hole = R * torch.cos(theta).unsqueeze(1)
    y_hole = R * torch.sin(theta).unsqueeze(1)
    boundary_hole = torch.cat([x_hole, y_hole], dim=1)
    
    return domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole

# 方案2 - Log Variance形式的自适应权重损失函数
def compute_loss_scheme2_log_variance(model, domain_points, boundary_left, boundary_right, 
                                    boundary_top, boundary_bottom, boundary_hole,
                                    s_pde, s_bc):
    """
    方案2：Log Variance形式的自适应权重损失函数
    
    参数：
    s_pde: log(ε_pde²) - PDE损失的log variance
    s_bc: log(ε_bc²) - 边界条件损失的log variance
    
    损失函数（按论文方程11）：
    L = 0.5 * exp(-s_pde) * L_PDE + 0.5 * exp(-s_bc) * L_BC + s_pde + s_bc
    """
    
    # 1. 域内平衡方程残差: ∇·σ + f = 0
    xy_domain = domain_points.clone().requires_grad_(True)
    u_theta = model(xy_domain)
    u = apply_hard_bc(xy_domain, u_theta)
    
    # 计算一阶导数
    u_x = u[:, 0:1]
    u_y = u[:, 1:2]
    
    # 计算应变分量（需要一阶导数）
    du_dx = torch.autograd.grad(u_x.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    du_dy = torch.autograd.grad(u_x.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    dv_dx = torch.autograd.grad(u_y.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dv_dy = torch.autograd.grad(u_y.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    
    # 应变张量
    eps_xx = du_dx
    eps_yy = dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)
    
    # 应力张量（本构方程）
    tr_eps = eps_xx + eps_yy
    sigma_xx = lam * tr_eps + 2 * mu * eps_xx
    sigma_yy = lam * tr_eps + 2 * mu * eps_yy
    sigma_xy = 2 * mu * eps_xy
    
    # 计算平衡方程残差（需要二阶导数）
    dsigma_xx_dx = torch.autograd.grad(sigma_xx.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dsigma_xy_dy = torch.autograd.grad(sigma_xy.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    dsigma_xy_dx = torch.autograd.grad(sigma_xy.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dsigma_yy_dy = torch.autograd.grad(sigma_yy.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    
    # 平衡方程残差（体力为0）
    residual_x = dsigma_xx_dx + dsigma_xy_dy
    residual_y = dsigma_xy_dx + dsigma_yy_dy
    
    loss_pde = torch.mean(residual_x**2) + torch.mean(residual_y**2)
    
    # 2. 右边界应力条件
    xy_right = boundary_right.clone().requires_grad_(True)
    u_theta_right = model(xy_right)
    u_right = apply_hard_bc(xy_right, u_theta_right)
    
    du_dx_r = torch.autograd.grad(u_right[:, 0].sum(), xy_right, create_graph=True)[0][:, 0]
    du_dy_r = torch.autograd.grad(u_right[:, 0].sum(), xy_right, create_graph=True)[0][:, 1]
    dv_dx_r = torch.autograd.grad(u_right[:, 1].sum(), xy_right, create_graph=True)[0][:, 0]
    dv_dy_r = torch.autograd.grad(u_right[:, 1].sum(), xy_right, create_graph=True)[0][:, 1]
    
    eps_xx_r = du_dx_r
    eps_yy_r = dv_dy_r
    eps_xy_r = 0.5 * (du_dy_r + dv_dx_r)
    
    tr_eps_r = eps_xx_r + eps_yy_r
    sigma_xx_r = lam * tr_eps_r + 2 * mu * eps_xx_r
    sigma_xy_r = 2 * mu * eps_xy_r
    
    traction_x_error = sigma_xx_r - P
    traction_y_error = sigma_xy_r - 0
    
    loss_bc_right = torch.mean(traction_x_error**2) + torch.mean(traction_y_error**2)
    
    # 3. 左边界应力条件
    xy_left = boundary_left.clone().requires_grad_(True)
    u_theta_left = model(xy_left)
    u_left = apply_hard_bc(xy_left, u_theta_left)
    
    du_dx_l = torch.autograd.grad(u_left[:, 0].sum(), xy_left, create_graph=True)[0][:, 0]
    du_dy_l = torch.autograd.grad(u_left[:, 0].sum(), xy_left, create_graph=True)[0][:, 1]
    dv_dx_l = torch.autograd.grad(u_left[:, 1].sum(), xy_left, create_graph=True)[0][:, 0]
    dv_dy_l = torch.autograd.grad(u_left[:, 1].sum(), xy_left, create_graph=True)[0][:, 1]
    
    eps_xx_l = du_dx_l
    eps_yy_l = dv_dy_l
    eps_xy_l = 0.5 * (du_dy_l + dv_dx_l)
    
    tr_eps_l = eps_xx_l + eps_yy_l
    sigma_xx_l = lam * tr_eps_l + 2 * mu * eps_xx_l
    sigma_xy_l = 2 * mu * eps_xy_l
    
    traction_x_error = sigma_xx_l - P
    traction_y_error = sigma_xy_l - 0
    
    loss_bc_left = torch.mean(traction_x_error**2) + torch.mean(traction_y_error**2)

    # 4. 上边界
    xy_top = boundary_top.clone().requires_grad_(True)
    u_theta_top = model(xy_top)
    u_top = apply_hard_bc(xy_top, u_theta_top)
    
    du_dx_t = torch.autograd.grad(u_top[:, 0].sum(), xy_top, create_graph=True)[0][:, 0]
    du_dy_t = torch.autograd.grad(u_top[:, 0].sum(), xy_top, create_graph=True)[0][:, 1]
    dv_dx_t = torch.autograd.grad(u_top[:, 1].sum(), xy_top, create_graph=True)[0][:, 0]
    dv_dy_t = torch.autograd.grad(u_top[:, 1].sum(), xy_top, create_graph=True)[0][:, 1]
    
    eps_xx_t = du_dx_t
    eps_yy_t = dv_dy_t
    eps_xy_t = 0.5 * (du_dy_t + dv_dx_t)
    
    tr_eps_t = eps_xx_t + eps_yy_t
    sigma_yy_t = lam * tr_eps_t + 2 * mu * eps_yy_t
    sigma_xy_t = 2 * mu * eps_xy_t
    
    traction_x_error = sigma_xy_t - 0
    traction_y_error = sigma_yy_t - P_top
    
    loss_bc_top = torch.mean(traction_x_error**2) + torch.mean(traction_y_error**2)
    
    # 5. 下边界
    xy_bottom = boundary_bottom.clone().requires_grad_(True)
    u_theta_bottom = model(xy_bottom)
    u_bottom = apply_hard_bc(xy_bottom, u_theta_bottom)
    
    du_dx_b = torch.autograd.grad(u_bottom[:, 0].sum(), xy_bottom, create_graph=True)[0][:, 0]
    du_dy_b = torch.autograd.grad(u_bottom[:, 0].sum(), xy_bottom, create_graph=True)[0][:, 1]
    dv_dx_b = torch.autograd.grad(u_bottom[:, 1].sum(), xy_bottom, create_graph=True)[0][:, 0]
    dv_dy_b = torch.autograd.grad(u_bottom[:, 1].sum(), xy_bottom, create_graph=True)[0][:, 1]
    
    eps_xx_b = du_dx_b
    eps_yy_b = dv_dy_b
    eps_xy_b = 0.5 * (du_dy_b + dv_dx_b)
    
    tr_eps_b = eps_xx_b + eps_yy_b
    sigma_yy_b = lam * tr_eps_b + 2 * mu * eps_yy_b
    sigma_xy_b = 2 * mu * eps_xy_b
    
    traction_x_error = sigma_xy_b - 0
    
    loss_bc_bottom = torch.mean(traction_x_error**2) 
    
    # 6. 圆孔边界: σ·n = 0
    xy_hole = boundary_hole.clone().requires_grad_(True)
    u_theta_hole = model(xy_hole)
    u_hole = apply_hard_bc(xy_hole, u_theta_hole)
    
    du_dx_h = torch.autograd.grad(u_hole[:, 0].sum(), xy_hole, create_graph=True)[0][:, 0]
    du_dy_h = torch.autograd.grad(u_hole[:, 0].sum(), xy_hole, create_graph=True)[0][:, 1]
    dv_dx_h = torch.autograd.grad(u_hole[:, 1].sum(), xy_hole, create_graph=True)[0][:, 0]
    dv_dy_h = torch.autograd.grad(u_hole[:, 1].sum(), xy_hole, create_graph=True)[0][:, 1]
    
    eps_xx_h = du_dx_h
    eps_yy_h = dv_dy_h
    eps_xy_h = 0.5 * (du_dy_h + dv_dx_h)
    
    tr_eps_h = eps_xx_h + eps_yy_h
    sigma_xx_h = lam * tr_eps_h + 2 * mu * eps_xx_h
    sigma_yy_h = lam * tr_eps_h + 2 * mu * eps_yy_h
    sigma_xy_h = 2 * mu * eps_xy_h
    
    # 圆孔边界法向量
    n_x = -xy_hole[:, 0] / R
    n_y = -xy_hole[:, 1] / R
    
    # fx=fy = 0
    traction_x_hole = sigma_xx_h * n_x + sigma_xy_h * n_y
    traction_y_hole = sigma_xy_h * n_x + sigma_yy_h * n_y
    
    loss_bc_hole = torch.mean(traction_x_hole**2 + traction_y_hole**2)
    
    # 合并所有边界条件损失 - 方案2的关键
    loss_bc_combined = loss_bc_left + loss_bc_right + loss_bc_top + loss_bc_bottom + loss_bc_hole
    
    # ===== 关键：Log Variance形式的自适应损失平衡（按论文方程11） =====
    # L = 0.5 * exp(-s_pde) * L_PDE + 0.5 * exp(-s_bc) * L_BC + s_pde + s_bc
    # 其中：s_pde = log(ε_pde²), s_bc = log(ε_bc²)
    adaptive_loss = (0.5 * torch.exp(-s_pde) * loss_pde + 
                    0.5 * torch.exp(-s_bc) * loss_bc_combined +
                    s_pde + s_bc)
    
    return adaptive_loss, loss_pde, loss_bc_combined, loss_bc_left, loss_bc_right, loss_bc_top, loss_bc_bottom, loss_bc_hole

# Adam优化训练函数 - 方案2 + Log Variance + 单一优化器
def train_scheme2_log_variance(n_domain=6000, n_boundary=500, 
                             adam_epochs=10000, adam_lr=1e-3):
    """
    方案2：Log Variance + 单一Adam优化器
    """
    
    # 读取参考数据
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    
    # 初始化模型
    model = PINN_Network().to(DEVICE)
    domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole = generate_points(
        n_domain, n_boundary, n_boundary*2
    )
    
    # ===== 关键：Log Variance参数初始化 =====
    # s = log(ε²)，如果我们想要初始ε ≈ 2，那么 s = log(4) ≈ 1.386
    s_pde = torch.tensor(1.386, dtype=torch.float32, device=DEVICE, requires_grad=True)  # log(ε_pde²)
    s_bc = torch.tensor(1.386, dtype=torch.float32, device=DEVICE, requires_grad=True)   # log(ε_bc²)
    
    # ===== 关键：单一优化器优化所有参数 =====
    all_params = list(model.parameters()) + [s_pde, s_bc]
    optimizer = torch.optim.Adam(all_params, lr=adam_lr)
    
    # 训练历史记录
    loss_history = []
    s_history = {'pde': [], 'bc': []}
    epsilon_history = {'pde': [], 'bc': []}  # 从s转换的等效ε值
    weight_history = {'pde': [], 'bc': []}   # 实际权重值
    individual_bc_history = {
        'left': [], 'right': [], 'top': [], 'bottom': [], 'hole': []
    }
    
    # 检查点设置
    check_points = [1000, 2500, 5000, 7500, 10000]
    
    # 训练计时
    start_time = time.time()
    
    print("=" * 60)
    print("方案2：Log Variance形式 + 单一Adam优化器")
    print("=" * 60)
    print("关键特点：")
    print("- 使用log variance: s = log(ε²)")
    print("- 损失函数: L = 0.5*exp(-s_pde)*L_PDE + 0.5*exp(-s_bc)*L_BC + s_pde + s_bc")
    print("- 单一Adam优化器优化网络参数和ε参数")
    print("- 无缩放因子（严格按论文）")
    print(f"- 参数数量: 2个log variance参数")
    print(f"- 初始log variance: s_pde={s_pde.item():.3f}, s_bc={s_bc.item():.3f}")
    print(f"- 对应初始ε: ε_pde={torch.sqrt(torch.exp(s_pde)).item():.3f}, ε_bc={torch.sqrt(torch.exp(s_bc)).item():.3f}")
    print("=" * 60)
    
    for epoch in range(adam_epochs):
        # 清零梯度
        optimizer.zero_grad()
        
        # 计算损失
        loss, loss_pde, loss_bc_combined, loss_bc_l, loss_bc_r, loss_bc_t, loss_bc_b, loss_bc_h = compute_loss_scheme2_log_variance(
            model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole,
            s_pde, s_bc
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录训练历史
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 记录log variance值
        s_history['pde'].append(s_pde.item())
        s_history['bc'].append(s_bc.item())
        
        # 转换为等效ε值 (ε = sqrt(exp(s)))
        epsilon_pde = torch.sqrt(torch.exp(s_pde)).item()
        epsilon_bc = torch.sqrt(torch.exp(s_bc)).item()
        epsilon_history['pde'].append(epsilon_pde)
        epsilon_history['bc'].append(epsilon_bc)
        
        # 计算实际权重 (0.5 * exp(-s))
        weight_pde = (0.5 * torch.exp(-s_pde)).item()
        weight_bc = (0.5 * torch.exp(-s_bc)).item()
        weight_history['pde'].append(weight_pde)
        weight_history['bc'].append(weight_bc)
        
        # 记录各个边界条件的损失值（用于分析）
        individual_bc_history['left'].append(loss_bc_l.item())
        individual_bc_history['right'].append(loss_bc_r.item())
        individual_bc_history['top'].append(loss_bc_t.item())
        individual_bc_history['bottom'].append(loss_bc_b.item())
        individual_bc_history['hole'].append(loss_bc_h.item())
        
        # 检查是否需要输出结果
        if epoch + 1 in check_points:
            current_time = time.time() - start_time
            print(f"\n--- 检查点 {epoch + 1} ---")
            print(f"训练时长: {current_time:.2f} 秒")
            print(f"当前总损失: {loss_value:.6f}")
            print(f"PDE损失: {loss_pde.item():.6f}")
            print(f"BC总损失: {loss_bc_combined.item():.6f}")
            print(f"Log variance值:")
            print(f"  s_PDE: {s_pde.item():.4f}, s_BC: {s_bc.item():.4f}")
            print(f"等效ε值:")
            print(f"  ε_PDE: {epsilon_pde:.4f}, ε_BC: {epsilon_bc:.4f}")
            print(f"实际权重:")
            print(f"  w_PDE: {weight_pde:.4f}, w_BC: {weight_bc:.4f}")
            print(f"权重比例:")
            total_weight = weight_pde + weight_bc
            print(f"  PDE: {(weight_pde/total_weight)*100:.1f}%, BC: {(weight_bc/total_weight)*100:.1f}%")
            
            # 计算误差
            compute_error_at_iteration(model, epoch + 1, ref_x, ref_y, ref_u1, ref_u2, "Log-Var ")
        
        # 定期输出训练进度
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d}: Loss={loss_value:.6f}, "
                  f"PDE={loss_pde.item():.6f}, BC={loss_bc_combined.item():.6f}, "
                  f"ε_PDE={epsilon_pde:.3f}, ε_BC={epsilon_bc:.3f}")
    
    training_time = time.time() - start_time
    print(f"\n方案2 Log Variance训练完成！")
    print(f"总训练时间: {training_time:.2f} 秒")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"最终Log variance值:")
    print(f"  s_PDE: {s_pde.item():.4f}, s_BC: {s_bc.item():.4f}")
    print(f"最终等效ε值:")
    print(f"  ε_PDE: {epsilon_history['pde'][-1]:.4f}, ε_BC: {epsilon_history['bc'][-1]:.4f}")
    print(f"最终权重值:")
    print(f"  w_PDE: {weight_history['pde'][-1]:.4f}, w_BC: {weight_history['bc'][-1]:.4f}")
    
    return model, loss_history, s_history, epsilon_history, weight_history, individual_bc_history

# 可视化Log Variance训练过程
def plot_log_variance_evolution(s_history, epsilon_history, weight_history, individual_bc_history):
    """绘制Log Variance形式的训练演化曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    iterations = range(len(s_history['pde']))
    
    # 第一行左：Log variance演化
    axes[0, 0].plot(iterations, s_history['pde'], 'blue', linewidth=2, label='s_PDE = log(ε²_PDE)')
    axes[0, 0].plot(iterations, s_history['bc'], 'red', linewidth=2, label='s_BC = log(ε²_BC)')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Log Variance (s)')
    axes[0, 0].set_title('Log Variance Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 第一行中：等效ε演化
    axes[0, 1].plot(iterations, epsilon_history['pde'], 'blue', linewidth=2, label='ε_PDE')
    axes[0, 1].plot(iterations, epsilon_history['bc'], 'red', linewidth=2, label='ε_BC')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Epsilon Value')
    axes[0, 1].set_title('Equivalent ε Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 第一行右：实际权重演化
    axes[0, 2].semilogy(iterations, weight_history['pde'], 'blue', linewidth=2, label='w_PDE = 0.5×exp(-s_PDE)')
    axes[0, 2].semilogy(iterations, weight_history['bc'], 'red', linewidth=2, label='w_BC = 0.5×exp(-s_BC)')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Actual Weight Value')
    axes[0, 2].set_title('Actual Weights Evolution')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # 第二行左：权重比例
    weight_ratio = [w_pde/(w_pde + w_bc) for w_pde, w_bc in zip(weight_history['pde'], weight_history['bc'])]
    axes[1, 0].plot(iterations, weight_ratio, 'green', linewidth=2, label='PDE Weight Ratio')
    axes[1, 0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Equal Weight (50%)')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('PDE/(PDE+BC) Ratio')
    axes[1, 0].set_title('PDE Weight Proportion')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    
    # 第二行中：各边界条件损失演化（前三个）
    colors = ['red', 'green', 'orange']
    labels = ['Left BC', 'Right BC', 'Top BC']
    for i, (key, label, color) in enumerate(zip(['left', 'right', 'top'], labels, colors)):
        axes[1, 1].semilogy(iterations, individual_bc_history[key], color=color, linewidth=2, label=label)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss Value')
    axes[1, 1].set_title('Boundary Conditions (Left, Right, Top)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 第二行右：各边界条件损失演化（后两个）
    colors = ['purple', 'brown']
    labels = ['Bottom BC', 'Hole BC']
    for i, (key, label, color) in enumerate(zip(['bottom', 'hole'], labels, colors)):
        axes[1, 2].semilogy(iterations, individual_bc_history[key], color=color, linewidth=2, label=label)
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('Loss Value')
    axes[1, 2].set_title('Boundary Conditions (Bottom, Hole)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('scheme2_log_variance_evolution.svg', format='svg', bbox_inches='tight')
    plt.show()

# 可视化最终结果
def visualize_results(model):
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    if ref_x is None:
        print("无法可视化结果：缺少参考数据")
        return
        
    # 用参考数据的坐标进行预测
    xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_theta = model(xy_tensor)
        u_pred = apply_hard_bc(xy_tensor, u_theta).cpu().numpy()
    
    U = u_pred[:, 0] 
    V = u_pred[:, 1] 
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # u位移
    im1 = axes[0].tricontourf(ref_x, ref_y, U, levels=70, cmap='RdBu_r')
    axes[0].set_title('Final U Displacement (Log Variance)')
    axes[0].set_aspect('equal')
    axes[0].set_xticks([-L, 0, L])
    axes[0].set_yticks([-L, 0, L])
    axes[0].tick_params(axis='both', pad=12)
    circle1 = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[0].add_patch(circle1)
    cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.062, pad=0.08, aspect = 14)
    cb1.locator = ticker.MaxNLocator(nbins=3)
    cb1.update_ticks()
    
    # v位移
    im2 = axes[1].tricontourf(ref_x, ref_y, V, levels=70, cmap='RdBu_r')
    axes[1].set_title('Final V Displacement (Log Variance)')
    axes[1].set_aspect('equal')
    axes[1].set_xticks([-L, 0, L])
    axes[1].set_yticks([-L, 0, L])
    axes[1].tick_params(axis='both', pad=12)
    circle2 = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[1].add_patch(circle2)
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.062, pad=0.08, aspect = 14)
    cb2.locator = ticker.MaxNLocator(nbins=3)
    cb2.update_ticks()
    
    plt.tight_layout()
    plt.savefig('final_displacement_scheme2_log_variance.svg', format='svg', bbox_inches='tight')
    plt.show()

# 绘制损失历史
def plot_loss_history(loss_history):
    """绘制损失演化曲线"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Evolution (Scheme 2: Log Variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_history_scheme2_log_variance.svg', format='svg', bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    print("方案2：Log Variance + 单一优化器 - 带圆孔板拉伸问题")
    print("严格按照论文方程(11)实现")
    print("=" * 50)
    print(f"问题参数：")
    print(f"板尺寸: {2*L} × {2*L} mm")
    print(f"圆孔半径: {R} mm")
    print(f"杨氏模量: {E} MPa")
    print(f"泊松比: {nu}")
    print(f"载荷: 左右{P} MPa, 上{P_top} MPa")
    print("=" * 50)
    
    # 训练参数
    params = {
        'n_domain': 6000,
        'n_boundary': 500,
        'adam_epochs': 20000,
        'adam_lr': 1e-3
    }
    
    print(f"\n训练参数：")
    print(f"域内采样点: {params['n_domain']}")
    print(f"边界采样点: {params['n_boundary']}")
    print(f"Adam迭代数: {params['adam_epochs']}")
    print(f"学习率: {params['adam_lr']}")
    print(f"自适应参数: 2个log variance (s_pde, s_bc)")
    
    # 开始训练
    print(f"\n开始方案2 Log Variance训练...")
    model, loss_history, s_history, epsilon_history, weight_history, bc_history = train_scheme2_log_variance(**params)
    
    # 绘制损失曲线
    plot_loss_history(loss_history)
    
    # 绘制Log Variance演化
    plot_log_variance_evolution(s_history, epsilon_history, weight_history, bc_history)
    
    # 可视化最终结果
    print("\n生成最终位移场...")
    visualize_results(model)
    
    print("\n方案2 Log Variance训练完成！")
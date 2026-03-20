import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from itertools import chain

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

def compute_error_at_iteration(models, iteration, ref_x, ref_y, ref_u1, ref_u2, phase=""):
    """计算指定iteration的误差并输出结果"""
    if ref_x is None:
        print(f"{phase}Iteration {iteration}: 无参考数据，跳过误差计算")
        return
    
    # 用参考数据的坐标进行预测
    xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_pred = predict_displacement(models, xy_tensor).cpu().numpy()
    
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
    
    # 位移云图（用参考数据的坐标）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].tricontourf(ref_x, ref_y, u_pred[:, 0], levels=70, cmap='RdBu_r')
    axes[0].set_title(f'{phase}Iter {iteration}')
    axes[0].set_xticks([-L, 0, L])  # x轴只显示：-0.5, 0, 0.5
    axes[0].set_yticks([-L, 0, L])  # y轴只显示：-0.5, 0, 0.5
    axes[0].set_aspect('equal')
    axes[0].tick_params(axis='both', pad=12)  # 刻度数字距离轴线更远
    circle = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[0].add_patch(circle)
    cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.062, pad=0.08, aspect = 14)
    cb1.locator = ticker.MaxNLocator(nbins=3)
    cb1.update_ticks()
    
    im2 = axes[1].tricontourf(ref_x, ref_y, u_pred[:, 1], levels=70, cmap='RdBu_r')
    axes[1].set_title(f'{phase}Iter {iteration}')
    axes[1].set_xticks([-L, 0, L])  # x轴只显示：-0.5, 0, 0.5
    axes[1].set_yticks([-L, 0, L])  # y轴只显示：-0.5, 0, 0.5
    axes[1].set_aspect('equal')
    axes[1].tick_params(axis='both', pad=12)  # 刻度数字距离轴线更远
    circle = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[1].add_patch(circle)
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.062, pad=0.08, aspect = 14)
    cb2.locator = ticker.MaxNLocator(nbins=3)
    cb2.update_ticks()
    
    plt.tight_layout()
    plt.savefig(f'displacement_{phase.lower().replace(" ", "_")}iter_{iteration}.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    # 误差云图（用参考数据的坐标和三角剖分）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # u分量误差云图
    im1 = axes[0].tricontourf(ref_x, ref_y, error_u_vals, levels=70, cmap='RdBu_r')
    axes[0].set_title(f'{phase}Iter {iteration}')
    axes[0].set_xticks([-L, 0, L])  # x轴只显示：-0.5, 0, 0.5
    axes[0].set_yticks([-L, 0, L])  # y轴只显示：-0.5, 0, 0.5
    axes[0].set_aspect('equal')
    axes[0].tick_params(axis='both', pad=12)  # 刻度数字距离轴线更远
    circle = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[0].add_patch(circle)
    cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.062, pad=0.08, aspect = 14)
    cb1.locator = ticker.MaxNLocator(nbins=3)
    cb1.update_ticks()
    
    # v分量误差云图
    im2 = axes[1].tricontourf(ref_x, ref_y, error_v_vals, levels=70, cmap='RdBu_r')
    axes[1].set_title(f'{phase}Iter {iteration}')
    axes[1].set_xticks([-L, 0, L])  # x轴只显示：-0.5, 0, 0.5
    axes[1].set_yticks([-L, 0, L])  # y轴只显示：-0.5, 0, 0.5
    axes[1].set_aspect('equal')
    axes[1].tick_params(axis='both', pad=12)  # 刻度数字距离轴线更远
    circle = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[1].add_patch(circle)
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.062, pad=0.08, aspect = 14)
    cb2.locator = ticker.MaxNLocator(nbins=3)
    cb2.update_ticks()
    
    plt.tight_layout()
    plt.savefig(f'error_{phase.lower().replace(" ", "_")}iter_{iteration}.svg', format='svg', bbox_inches='tight')
    plt.show()

# 神经网络定义 - 5个独立的神经网络
class SingleOutputNetwork(nn.Module):
    """单输出神经网络"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1)  # 单输出
        )
    
    def forward(self, x):
        return self.net(x)

def create_models():
    """创建5个独立的神经网络"""
    u_net = SingleOutputNetwork().to(DEVICE)        # 水平位移网络
    v_net = SingleOutputNetwork().to(DEVICE)        # 竖直位移网络
    sigma_xx_net = SingleOutputNetwork().to(DEVICE) # σ_xx应力网络
    sigma_yy_net = SingleOutputNetwork().to(DEVICE) # σ_yy应力网络
    sigma_xy_net = SingleOutputNetwork().to(DEVICE) # σ_xy应力网络
    
    return {
        'u_net': u_net,
        'v_net': v_net,
        'sigma_xx_net': sigma_xx_net,
        'sigma_yy_net': sigma_yy_net,
        'sigma_xy_net': sigma_xy_net
    }

def get_all_parameters(models):
    """获取所有网络的参数 - 使用itertools.chain正确合并"""
    return chain(models['u_net'].parameters(), 
                 models['v_net'].parameters(),
                 models['sigma_xx_net'].parameters(),
                 models['sigma_yy_net'].parameters(),
                 models['sigma_xy_net'].parameters())

# 硬边界条件
def apply_hard_bc_displacement(xy, u_theta, v_theta):
    """
    应用硬边界条件：
    - u分量：只在底部中点(0, -L)处约束为0
    - v分量：在整个底部(y = -L)约束为0
    """
    x = xy[:, 0:1]  # x坐标
    y = xy[:, 1:2]  # y坐标
    
    # u分量：使用点约束，只在(0, -L)处为0
    # 使用径向距离函数：(x-0)² + (y-(-L))² = x² + (y+L)²
    u = (x**2 + (y + L)**2) * u_theta
    
    # v分量：使用线约束，在整个底部y = -L处为0
    v = (y + L) * v_theta
    
    return u, v

def predict_displacement(models, xy):
    """预测位移分量并应用硬边界条件"""
    u_theta = models['u_net'](xy)
    v_theta = models['v_net'](xy)
    u, v = apply_hard_bc_displacement(xy, u_theta, v_theta)
    return torch.cat([u, v], dim=1)

def predict_stress(models, xy):
    """预测应力分量"""
    sigma_xx = models['sigma_xx_net'](xy)
    sigma_yy = models['sigma_yy_net'](xy)
    sigma_xy = models['sigma_xy_net'](xy)
    return sigma_xx, sigma_yy, sigma_xy

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

# 改进的损失函数（基于5个独立网络）
def compute_loss_multi_network(models, domain_points, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole):
    """
    使用5个独立网络的损失函数:
    1. 平衡方程: ∇·σ = 0 (使用应力的一阶微分)
    2. 物理方程: σ = D:ε (位移一阶微分得到应变，与应力输出构成物理方程)
    3. 边界条件: 直接使用应力输出
    """
    
    # 1. 域内损失：平衡方程 + 物理方程
    xy_domain = domain_points.clone().requires_grad_(True)
    
    # 获取位移和应力预测
    u_theta = models['u_net'](xy_domain)
    v_theta = models['v_net'](xy_domain)
    u, v = apply_hard_bc_displacement(xy_domain, u_theta, v_theta)
    
    sigma_xx_pred, sigma_yy_pred, sigma_xy_pred = predict_stress(models, xy_domain)
    
    # 1.1 平衡方程损失: ∇·σ = 0
    # ∂σ_xx/∂x + ∂σ_xy/∂y = 0
    # ∂σ_xy/∂x + ∂σ_yy/∂y = 0
    dsigma_xx_dx = torch.autograd.grad(sigma_xx_pred.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dsigma_xy_dy = torch.autograd.grad(sigma_xy_pred.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    dsigma_xy_dx = torch.autograd.grad(sigma_xy_pred.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dsigma_yy_dy = torch.autograd.grad(sigma_yy_pred.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    
    equilibrium_x = dsigma_xx_dx + dsigma_xy_dy
    equilibrium_y = dsigma_xy_dx + dsigma_yy_dy
    
    loss_equilibrium = torch.mean(equilibrium_x**2) + torch.mean(equilibrium_y**2)
    
    # 1.2 物理方程损失: σ = D:ε
    # 从位移计算应变
    du_dx = torch.autograd.grad(u.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    du_dy = torch.autograd.grad(u.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    dv_dx = torch.autograd.grad(v.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dv_dy = torch.autograd.grad(v.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    
    eps_xx = du_dx
    eps_yy = dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)
    
    # 从应变计算理论应力
    tr_eps = eps_xx + eps_yy
    sigma_xx_theory = lam * tr_eps + 2 * mu * eps_xx
    sigma_yy_theory = lam * tr_eps + 2 * mu * eps_yy
    sigma_xy_theory = 2 * mu * eps_xy
    
    # 物理方程误差
    constitutive_xx = sigma_xx_pred - sigma_xx_theory
    constitutive_yy = sigma_yy_pred - sigma_yy_theory
    constitutive_xy = sigma_xy_pred - sigma_xy_theory
    
    loss_constitutive = torch.mean(constitutive_xx**2) + torch.mean(constitutive_yy**2) + torch.mean(constitutive_xy**2)
    
    # 2. 边界条件损失（直接使用应力输出）
    
    # 2.1 右边界: σ·n = [P, 0], n = [1, 0]
    sigma_xx_right, sigma_yy_right, sigma_xy_right = predict_stress(models, boundary_right)
    traction_x_error_right = sigma_xx_right - P
    traction_y_error_right = sigma_xy_right - 0
    loss_bc_right = torch.mean(traction_x_error_right**2) + torch.mean(traction_y_error_right**2)
    
    # 2.2 左边界: σ·n = [P, 0], n = [-1, 0]
    sigma_xx_left, sigma_yy_left, sigma_xy_left = predict_stress(models, boundary_left)
    traction_x_error_left = (-sigma_xx_left) - P  # 注意法向量为[-1, 0]
    traction_y_error_left = (-sigma_xy_left) - 0
    loss_bc_left = torch.mean(traction_x_error_left**2) + torch.mean(traction_y_error_left**2)
    
    # 2.3 上边界: σ·n = [0, P_top], n = [0, 1]
    sigma_xx_top, sigma_yy_top, sigma_xy_top = predict_stress(models, boundary_top)
    traction_x_error_top = sigma_xy_top - 0
    traction_y_error_top = sigma_yy_top - P_top
    loss_bc_top = torch.mean(traction_x_error_top**2) + torch.mean(traction_y_error_top**2)
    
    # 2.4 下边界: σ·n = [0, 0], n = [0, -1]
    sigma_xx_bottom, sigma_yy_bottom, sigma_xy_bottom = predict_stress(models, boundary_bottom)
    traction_x_error_bottom = (-sigma_xy_bottom) - 0  # 注意法向量为[0, -1]
    loss_bc_bottom = torch.mean(traction_x_error_bottom**2)
    
    # 2.5 圆孔边界: σ·n = [0, 0]
    sigma_xx_hole, sigma_yy_hole, sigma_xy_hole = predict_stress(models, boundary_hole)
    
    # 圆孔边界法向量
    n_x = -boundary_hole[:, 0:1] / R
    n_y = -boundary_hole[:, 1:2] / R
    
    # 牵引力计算
    traction_x_hole = sigma_xx_hole * n_x + sigma_xy_hole * n_y
    traction_y_hole = sigma_xy_hole * n_x + sigma_yy_hole * n_y
    
    loss_bc_hole = torch.mean(traction_x_hole**2 + traction_y_hole**2)
    
    # 总损失（加权）
    w_eq = 1.0      # 平衡方程权重
    w_const = 1.0   # 物理方程权重
    w_bc = 10.0     # 边界条件权重
    
    total_loss = (w_eq * loss_equilibrium + 
                  w_const * loss_constitutive + 
                  w_bc * (loss_bc_left + loss_bc_right + loss_bc_top + loss_bc_bottom + loss_bc_hole))
    
    return (total_loss, loss_equilibrium, loss_constitutive, 
            loss_bc_left, loss_bc_right, loss_bc_top, loss_bc_bottom, loss_bc_hole)

# 混合优化训练函数（修改为支持5个网络）
def train_hybrid_optimization(n_domain=6000, n_boundary=500, 
                            adam_epochs=5000, adam_lr=1e-3,
                            lbfgs_max_iter=10000):
    """
    混合优化策略：Adam预训练 + L-BFGS精调（5个独立网络）
    """
    
    # 读取参考数据
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    
    # Adam阶段的检查点
    adam_check_points = [1000, 2500, 4000, 5000]
    # L-BFGS阶段的检查点（相对于总迭代数）
    lbfgs_check_points = [adam_epochs + 2000, adam_epochs + 5000, adam_epochs + 8000, adam_epochs + 10000]
    
    # 初始化5个神经网络
    models = create_models()
    domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole = generate_points(
        n_domain, n_boundary, n_boundary*2
    )
    
    # 训练历史
    loss_history = []
    
    # 训练计时
    start_time = time.time()
    
    print("=" * 60)
    print("Phase 1: Adam预训练阶段（5个独立网络）")
    print("=" * 60)
    
    # 第一阶段：Adam预训练
    model_parameters = get_all_parameters(models)
    adam_optimizer = torch.optim.Adam(model_parameters, lr=adam_lr)
    
    for epoch in range(adam_epochs):
        adam_optimizer.zero_grad()
        (loss, loss_eq, loss_const, loss_bc_l, loss_bc_r, 
         loss_bc_t, loss_bc_b, loss_bc_h) = compute_loss_multi_network(
            models, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole
        )
        loss.backward()
        adam_optimizer.step()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 检查是否需要输出结果
        if epoch + 1 in adam_check_points:
            current_time = time.time() - start_time
            print(f"\n训练时长: {current_time:.2f} 秒")
            compute_error_at_iteration(models, epoch + 1, ref_x, ref_y, ref_u1, ref_u2, "Adam ")
        
        if epoch % 500 == 0:
            print(f"Adam Epoch {epoch:4d}: Loss={loss_value:.6f}, "
                  f"Eq={loss_eq:.6f}, Const={loss_const:.6f}, BC_right={loss_bc_r:.6f}")
    
    adam_time = time.time() - start_time
    print(f"\nAdam预训练完成，用时: {adam_time:.2f} 秒")
    print(f"Adam最终损失: {loss_history[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("Phase 2: L-BFGS精调阶段（5个独立网络）")
    print("=" * 60)
    
    # 第二阶段：L-BFGS精调
    # 重新获取参数，因为chain迭代器在Adam阶段已被消耗
    model_parameters_lbfgs = get_all_parameters(models)
    lbfgs_optimizer = torch.optim.LBFGS(
        model_parameters_lbfgs,
        max_iter=lbfgs_max_iter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-7,
        tolerance_change=1e-12
    )
    
    lbfgs_iteration = [adam_epochs]  # 从Adam结束的迭代数开始计数
    
    def closure():
        lbfgs_optimizer.zero_grad()
        (loss, loss_eq, loss_const, loss_bc_l, loss_bc_r, 
         loss_bc_t, loss_bc_b, loss_bc_h) = compute_loss_multi_network(
            models, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole
        )
        loss.backward()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 检查是否需要输出结果
        if lbfgs_iteration[0] in lbfgs_check_points:
            current_time = time.time() - start_time
            print(f"\n训练时长: {current_time:.2f} 秒")
            compute_error_at_iteration(models, lbfgs_iteration[0], ref_x, ref_y, ref_u1, ref_u2, "L-BFGS ")
        
        if lbfgs_iteration[0] % 100 == 0:
            print(f"L-BFGS Iter {lbfgs_iteration[0]:4d}: Loss={loss_value:.6f}, "
                  f"Eq={loss_eq:.6f}, Const={loss_const:.6f}, BC_right={loss_bc_r:.6f}")
        
        lbfgs_iteration[0] += 1
        return loss
    
    # 执行L-BFGS优化
    iterations_before = lbfgs_iteration[0]
    lbfgs_optimizer.step(closure)
    total_lbfgs_iterations = lbfgs_iteration[0] - iterations_before
    
    training_time = time.time() - start_time
    print(f"\n总训练时间: {training_time:.2f} 秒")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"Adam阶段: {adam_epochs} 迭代")
    print(f"L-BFGS阶段: {total_lbfgs_iterations} 迭代")
    print(f"总迭代数: {lbfgs_iteration[0]}")
    
    if total_lbfgs_iterations < lbfgs_max_iter * 0.5:
        print(f"\n注意：L-BFGS在{total_lbfgs_iterations}次迭代后停止，")
        print(f"可能是由于梯度容忍度或函数变化容忍度达到条件而提前停止")
    
    return models, loss_history, adam_epochs

# 可视化结果（修改为支持5个网络）
def visualize_results(models):
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    # 用参考数据的坐标进行预测
    xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_pred = predict_displacement(models, xy_tensor).cpu().numpy()
    
    U = u_pred[:, 0] 
    V = u_pred[:, 1] 
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # u位移
    im1 = axes[0].tricontourf(ref_x, ref_y, U, levels=70, cmap='RdBu_r')
    axes[0].set_title('  ')
    axes[0].set_aspect('equal')
    axes[0].set_xticks([-L, 0, L])  # x轴只显示：-0.5, 0, 0.5
    axes[0].set_yticks([-L, 0, L])  # y轴只显示：-0.5, 0, 0.5
    axes[0].tick_params(axis='both', pad=12)  # 刻度数字距离轴线更远
    circle1 = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[0].add_patch(circle1)
    cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.062, pad=0.08, aspect = 14)
    cb1.locator = ticker.MaxNLocator(nbins=3)
    cb1.update_ticks()
    
    # v位移
    im2 = axes[1].tricontourf(ref_x, ref_y, V, levels=70, cmap='RdBu_r')
    axes[1].set_title('  ')
    axes[1].set_aspect('equal')
    axes[1].set_xticks([-L, 0, L])  # x轴只显示：-0.5, 0, 0.5
    axes[1].set_yticks([-L, 0, L])  # y轴只显示：-0.5, 0, 0.5
    axes[1].tick_params(axis='both', pad=12)  # 刻度数字距离轴线更远
    circle2 = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[1].add_patch(circle2)
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.062, pad=0.08, aspect = 14)
    cb2.locator = ticker.MaxNLocator(nbins=3)
    cb2.update_ticks()
    
    plt.tight_layout()
    plt.savefig('final_displacement_hybrid_multi.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    # 输出关键点位移
    test_points = torch.tensor([
        [10.0, 0.0],   # 右侧中点
        [0.0, 10.0],   # 上侧中点
        [-10.0, 0.0],  # 左侧中点
        [0.0, -10.0],  # 下侧中点
        [15.0, 0.0],   # 右侧3/4处
        [R*1.2, 0.0],  # 圆孔右侧附近
    ], dtype=torch.float32, device=DEVICE)
    
    with torch.no_grad():
        u_test = predict_displacement(models, test_points).cpu().numpy()
    
    print("\n关键点位移值：")
    print("点位置\t\t\tu (mm)\t\tv (mm)")
    print("-" * 50)
    for i, pt in enumerate(test_points.cpu().numpy()):
        print(f"({pt[0]:5.1f}, {pt[1]:5.1f})\t{u_test[i, 0]:10.6f}\t{u_test[i, 1]:10.6f}")

# 绘制混合优化损失曲线
def plot_loss_history_hybrid(loss_history, adam_epochs):
    """绘制混合优化的损失曲线，区分Adam和L-BFGS阶段"""
    plt.figure(figsize=(12, 6))
    
    iterations = range(len(loss_history))
    
    # 绘制完整的损失曲线
    plt.semilogy(iterations, loss_history, 'b-', linewidth=2, alpha=0.8)
    
    # 标记Adam和L-BFGS的分界线
    plt.axvline(x=adam_epochs, color='red', linestyle='--', linewidth=2, 
                label=f'Adam→L-BFGS (iter {adam_epochs})')
    
    # 添加区域标注
    plt.fill_betweenx([min(loss_history), max(loss_history)], 0, adam_epochs, 
                      alpha=0.2, color='green', label='Adam Phase')
    plt.fill_betweenx([min(loss_history), max(loss_history)], adam_epochs, len(loss_history), 
                      alpha=0.2, color='blue', label='L-BFGS Phase')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Hybrid Optimization: Adam + L-BFGS Training Loss (5 Networks)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('loss_history_hybrid_multi.svg', format='svg', bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    print("改进的多网络PINNs - 带圆孔板拉伸问题")
    print("使用5个独立神经网络：u, v, σ_xx, σ_yy, σ_xy")
    print("=" * 50)
    print(f"问题参数：")
    print(f"板尺寸: {L} × {L} mm")
    print(f"圆孔半径: {R} mm")
    print(f"杨氏模量: {E} MPa")
    print(f"泊松比: {nu}")
    print(f"载荷: {P} MPa")
    print("=" * 50)
    
    # 混合优化参数设置
    params = {
        'n_domain': 6000,
        'n_boundary': 500,
        'adam_epochs': 5000,
        'adam_lr': 1e-3,
        'lbfgs_max_iter': 30000
    }
    
    print(f"\n混合优化参数（5个独立网络）：")
    print(f"Adam预训练: {params['adam_epochs']} epochs, lr={params['adam_lr']}")
    print(f"L-BFGS精调: 最大{params['lbfgs_max_iter']} 迭代")
    print(f"L-BFGS设置: tolerance_grad=1e-7, tolerance_change=1e-12")
    
    # 训练模型
    print(f"\n开始混合优化训练（5个独立网络）...")
    models, loss_history, adam_epochs = train_hybrid_optimization(**params)
    
    # 绘制损失曲线
    plot_loss_history_hybrid(loss_history, adam_epochs)
    
    # 可视化结果
    print("\n生成最终位移场...")
    visualize_results(models)
    
    
    # 计算并输出最终MSE、MAPE和R
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    if ref_x is not None:
        xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            u_pred = predict_displacement(models, xy_tensor).cpu().numpy()
        
        # 计算MSE
        mse_u = np.mean((u_pred[:, 0] - ref_u1)**2)
        mse_v = np.mean((u_pred[:, 1] - ref_u2)**2)
        
        # 计算MAPE (Mean Absolute Percentage Error)
        mape_u = np.mean(np.abs((ref_u1 - u_pred[:, 0]) / (np.abs(ref_u1) + 1e-8))) * 100
        mape_v = np.mean(np.abs((ref_u2 - u_pred[:, 1]) / (np.abs(ref_u2) + 1e-8))) * 100
        
        # 计算相关系数R
        r_u = np.corrcoef(ref_u1, u_pred[:, 0])[0, 1]
        r_v = np.corrcoef(ref_u2, u_pred[:, 1])[0, 1]
        
        print(f"\n最终误差指标结果：")
        print(f"水平位移(U) - MSE: {mse_u:.8e}, MAPE: {mape_u:.4f}%, R: {r_u:.6f}")
        print(f"竖向位移(V) - MSE: {mse_v:.8e}, MAPE: {mape_v:.4f}%, R: {r_v:.6f}")
    
    print("\n多网络混合优化训练完成！")
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
        u_pred = apply_hard_bc(xy_tensor, model).cpu().numpy()
    
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

# 修改后的神经网络定义 - 双网络结构
class SingleComponentNetwork(nn.Module):
    """单个位移分量的神经网络"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1)  # 输出1个分量
        )
    
    def forward(self, x):
        return self.net(x)

class DualPINN_Network(nn.Module):
    """双网络PINN结构：包含两个独立的子网络"""
    def __init__(self):
        super().__init__()
        self.u_network = SingleComponentNetwork()  # 水平位移网络
        self.v_network = SingleComponentNetwork()  # 竖向位移网络
    
    def forward(self, x):
        # 分别计算两个位移分量
        u_theta = self.u_network(x)
        v_theta = self.v_network(x)
        return u_theta, v_theta

# 修改后的硬边界条件
def apply_hard_bc(xy, model):
    """
    应用硬边界条件：
    - u分量：只在底部中点(0, -L)处约束为0
    - v分量：在整个底部(y = -L)约束为0
    """
    x = xy[:, 0:1]  # x坐标
    y = xy[:, 1:2]  # y坐标
    
    # 分别获取两个网络的输出
    u_theta, v_theta = model(xy)
    
    # u分量：使用点约束，只在(0, -L)处为0
    # 使用径向距离函数：(x-0)² + (y-(-L))² = x² + (y+L)²
    u = (x**2 + (y + L)**2) * u_theta
    
    # v分量：使用线约束，在整个底部y = -L处为0
    v = (y + L) * v_theta
    
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

# 强形式损失函数（基于平衡方程）
def compute_loss_strong(model, domain_points, boundary_left, boundary_right, boundary_top,boundary_bottom, boundary_hole):
    # 1. 域内平衡方程残差: ∇·σ + f = 0
    xy_domain = domain_points.clone().requires_grad_(True)
    u = apply_hard_bc(xy_domain, model)
    
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
    # ∂σ_xx/∂x + ∂σ_xy/∂y = 0
    # ∂σ_xy/∂x + ∂σ_yy/∂y = 0
    dsigma_xx_dx = torch.autograd.grad(sigma_xx.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dsigma_xy_dy = torch.autograd.grad(sigma_xy.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    dsigma_xy_dx = torch.autograd.grad(sigma_xy.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dsigma_yy_dy = torch.autograd.grad(sigma_yy.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    
    # 平衡方程残差（体力为0）
    residual_x = dsigma_xx_dx + dsigma_xy_dy
    residual_y = dsigma_xy_dx + dsigma_yy_dy
    
    loss_pde = torch.mean(residual_x**2) + torch.mean(residual_y**2)
    
    # 2. 右边界应力条件: σ·n = [P, 0]
    xy_right = boundary_right.clone().requires_grad_(True)
    u_right = apply_hard_bc(xy_right, model)
    
    # 计算右边界的应力
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
    
    # 右边界: n = [1, 0], σ·n = [σ_xx, σ_xy]
    traction_x_error = sigma_xx_r - P
    traction_y_error = sigma_xy_r - 0
    
    loss_bc_right = torch.mean(traction_x_error**2) + torch.mean(traction_y_error**2)
    
    # 3. 左边界应力条件: σ·n = [P, 0]
    xy_left = boundary_left.clone().requires_grad_(True)
    u_left = apply_hard_bc(xy_left, model)
    
    # 计算左边界的应力
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
    
    # 左边界误差计算
    traction_x_error = sigma_xx_l - P
    traction_y_error = sigma_xy_l - 0
    
    loss_bc_left = torch.mean(traction_x_error**2) + torch.mean(traction_y_error**2)

    # 4. 上边界
    xy_top = boundary_top.clone().requires_grad_(True)
    u_top = apply_hard_bc(xy_top, model)
    
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
    
    # 上边界误差计算:
    traction_x_error = sigma_xy_t - 0
    traction_y_error = sigma_yy_t - P_top
    
    loss_bc_top = torch.mean(traction_x_error**2) + torch.mean(traction_y_error**2)
    
    # 5. 下边界
    xy_bottom = boundary_bottom.clone().requires_grad_(True)
    u_bottom = apply_hard_bc(xy_bottom, model)
    
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
    
    # 下边界误差计算:
    traction_x_error = sigma_xy_b - 0
    
    loss_bc_bottom = torch.mean(traction_x_error**2) 
    
    # 6. 圆孔边界: σ·n = 0
    xy_hole = boundary_hole.clone().requires_grad_(True)
    u_hole = apply_hard_bc(xy_hole, model)
    
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
    
    # 总损失（加权）
    w_pde = 1.0
    w_bc = 10.0  # 边界条件权重更高
    
    total_loss = w_pde * loss_pde + w_bc * (loss_bc_left+ loss_bc_right + loss_bc_top + loss_bc_bottom+ loss_bc_hole)
    
    return total_loss, loss_pde, loss_bc_left, loss_bc_right, loss_bc_top, loss_bc_bottom, loss_bc_hole

# 修改后的混合优化训练函数
def train_hybrid_optimization(n_domain=6000, n_boundary=500, 
                            adam_epochs=5000, adam_lr=1e-3,
                            lbfgs_max_iter=10000):
    """
    混合优化策略：Adam预训练 + L-BFGS精调
    
    Args:
        adam_epochs: Adam预训练的轮数
        adam_lr: Adam学习率
        lbfgs_max_iter: L-BFGS的最大迭代数
    """
    
    # 读取参考数据
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    
    # Adam阶段的检查点
    adam_check_points = [1000, 2500, 4000, 5000]
    # L-BFGS阶段的检查点（相对于总迭代数）
    lbfgs_check_points = [adam_epochs + 2000, adam_epochs + 5000, adam_epochs + 8000, adam_epochs + 10000]
    
    # 初始化双网络模型
    model = DualPINN_Network().to(DEVICE)
    domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole = generate_points(
        n_domain, n_boundary, n_boundary*2
    )
    
    # 训练历史
    loss_history = []
    
    # 训练计时
    start_time = time.time()
    
    print("=" * 60)
    print("Phase 1: Adam预训练阶段 (双网络)")
    print("=" * 60)
    
    # 第一阶段：Adam预训练 - 同时优化两个网络的参数
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    
    for epoch in range(adam_epochs):
        adam_optimizer.zero_grad()
        loss, loss_pde, loss_bc_l, loss_bc_r, loss_bc_t, loss_bc_b, loss_bc_h = compute_loss_strong(
            model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole
        )
        loss.backward()
        adam_optimizer.step()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 检查是否需要输出结果
        if epoch + 1 in adam_check_points:
            current_time = time.time() - start_time
            print(f"\n训练时长: {current_time:.2f} 秒")
            compute_error_at_iteration(model, epoch + 1, ref_x, ref_y, ref_u1, ref_u2, "Adam ")
        
        if epoch % 500 == 0:
            print(f"Adam Epoch {epoch:4d}: Loss={loss_value:.6f}, "
                  f"PDE={loss_pde:.6f}, BC_right={loss_bc_r:.6f}")
    
    adam_time = time.time() - start_time
    print(f"\nAdam预训练完成，用时: {adam_time:.2f} 秒")
    print(f"Adam最终损失: {loss_history[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("Phase 2: L-BFGS精调阶段 (双网络)")
    print("=" * 60)
    
    # 第二阶段：L-BFGS精调 - 同时优化两个网络的参数
    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_max_iter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-7,
        tolerance_change=1e-12
    )
    
    lbfgs_iteration = [adam_epochs]  # 从Adam结束的迭代数开始计数
    
    def closure():
        lbfgs_optimizer.zero_grad()
        loss, loss_pde, loss_bc_l, loss_bc_r, loss_bc_t, loss_bc_b, loss_bc_h = compute_loss_strong(
            model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole
        )
        loss.backward()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 检查是否需要输出结果
        if lbfgs_iteration[0] in lbfgs_check_points:
            current_time = time.time() - start_time
            print(f"\n训练时长: {current_time:.2f} 秒")
            compute_error_at_iteration(model, lbfgs_iteration[0], ref_x, ref_y, ref_u1, ref_u2, "L-BFGS ")
        
        if lbfgs_iteration[0] % 100 == 0:
            print(f"L-BFGS Iter {lbfgs_iteration[0]:4d}: Loss={loss_value:.6f}, "
                  f"PDE={loss_pde:.6f}, BC_right={loss_bc_r:.6f}")
        
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
    
    return model, loss_history, adam_epochs

# 可视化结果
def visualize_results(model):
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    # 用参考数据的坐标进行预测
    xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_pred = apply_hard_bc(xy_tensor, model).cpu().numpy()
    
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
    plt.savefig('final_displacement_dual_network.svg', format='svg', bbox_inches='tight')
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
        u_test = apply_hard_bc(test_points, model).cpu().numpy()
    
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
    plt.title('Dual Network Hybrid Optimization: Adam + L-BFGS Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('loss_history_dual_network.svg', format='svg', bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    print("双网络混合优化策略PINNs - 带圆孔板拉伸问题")
    print("=" * 50)
    print(f"问题参数：")
    print(f"板尺寸: {L} × {L} mm")
    print(f"圆孔半径: {R} mm")
    print(f"杨氏模量: {E} MPa")
    print(f"泊松比: {nu}")
    print(f"载荷: {P} MPa")
    print("=" * 50)
    print("网络架构: 双独立网络 (u网络 + v网络)")
    print("每个网络: 2→200→200→200→1")
    
    # 混合优化参数设置
    params = {
        'n_domain': 6000,
        'n_boundary': 500,
        'adam_epochs': 5000,
        'adam_lr': 1e-3,
        'lbfgs_max_iter': 30000
    }
    
    print(f"\n混合优化参数：")
    print(f"Adam预训练: {params['adam_epochs']} epochs, lr={params['adam_lr']}")
    print(f"L-BFGS精调: 最大{params['lbfgs_max_iter']} 迭代")
    print(f"L-BFGS设置: tolerance_grad=1e-7, tolerance_change=1e-12")
    
    # 训练模型
    print(f"\n开始双网络混合优化训练...")
    model, loss_history, adam_epochs = train_hybrid_optimization(**params)
    
    # 绘制损失曲线
    plot_loss_history_hybrid(loss_history, adam_epochs)
    
    # 可视化结果
    print("\n生成最终位移场...")
    visualize_results(model)
    
    # 计算并输出最终MSE、MAPE和R
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    if ref_x is not None:
        xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            u_theta = model(xy_tensor)
            u_pred = apply_hard_bc(xy_tensor, model).cpu().numpy()
        
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
    
    print("\n双网络混合优化训练完成！")
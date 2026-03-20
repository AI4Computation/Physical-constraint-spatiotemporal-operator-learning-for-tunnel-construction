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
    
    # 位移云图（用参考数据的坐标）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].tricontourf(ref_x, ref_y, u_pred[:, 0], levels=70, cmap='RdBu_r')
    axes[0].set_title(f'{phase}Iter {iteration}')
    axes[0].set_xticks([-L, 0, L])
    axes[0].set_yticks([-L, 0, L])
    axes[0].set_aspect('equal')
    axes[0].tick_params(axis='both', pad=12)
    circle = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[0].add_patch(circle)
    cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.062, pad=0.08, aspect = 14)
    cb1.locator = ticker.MaxNLocator(nbins=3)
    cb1.update_ticks()
    
    im2 = axes[1].tricontourf(ref_x, ref_y, u_pred[:, 1], levels=70, cmap='RdBu_r')
    axes[1].set_title(f'{phase}Iter {iteration}')
    axes[1].set_xticks([-L, 0, L])
    axes[1].set_yticks([-L, 0, L])
    axes[1].set_aspect('equal')
    axes[1].tick_params(axis='both', pad=12)
    circle = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[1].add_patch(circle)
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.062, pad=0.08, aspect = 14)
    cb2.locator = ticker.MaxNLocator(nbins=3)
    cb2.update_ticks()
    
    plt.tight_layout()
    plt.savefig(f'displacement_{phase.lower().replace(" ", "_")}iter_{iteration}.svg', format='svg', bbox_inches='tight')
    plt.show()

# SA-PINN 神经网络定义
class SA_PINN_Network(nn.Module):
    def __init__(self, n_domain, n_boundary_left, n_boundary_right, n_boundary_top, n_boundary_bottom, n_boundary_hole):
        super().__init__()
        # 原有网络结构
        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 2)
        )
        
        # 自适应权重参数 - 参考TensorFlow代码的初始化方式
        self.lambda_domain = nn.Parameter(torch.rand(n_domain, 1, device=DEVICE))  # 0-1
        self.lambda_boundary_left = nn.Parameter(10 * torch.rand(n_boundary_left, 1, device=DEVICE))  # 0-10
        self.lambda_boundary_right = nn.Parameter(10 * torch.rand(n_boundary_right, 1, device=DEVICE))  # 0-10
        self.lambda_boundary_top = nn.Parameter(10 * torch.rand(n_boundary_top, 1, device=DEVICE))  # 0-10
        self.lambda_boundary_bottom = nn.Parameter(10 * torch.rand(n_boundary_bottom, 1, device=DEVICE))  # 0-10
        self.lambda_boundary_hole = nn.Parameter(10 * torch.rand(n_boundary_hole, 1, device=DEVICE))  # 0-10
        
        print(f"初始化SA-PINN权重:")
        print(f"  域内点权重范围: {self.lambda_domain.min().item():.3f} - {self.lambda_domain.max().item():.3f}")
        print(f"  边界点权重范围: {self.lambda_boundary_left.min().item():.3f} - {self.lambda_boundary_left.max().item():.3f}")
    
    def forward(self, x):
        return self.net(x)

# 硬边界条件 (保持不变)
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

# 生成采样点 (保持不变)
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

# SA-PINN 损失函数
def compute_sa_loss_strong(model, domain_points, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole):
    # 1. 域内平衡方程残差
    xy_domain = domain_points.clone().requires_grad_(True)
    u_theta = model(xy_domain)
    u = apply_hard_bc(xy_domain, u_theta)
    
    # 计算应变和应力 (与原代码相同)
    u_x = u[:, 0:1]
    u_y = u[:, 1:2]
    
    du_dx = torch.autograd.grad(u_x.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    du_dy = torch.autograd.grad(u_x.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    dv_dx = torch.autograd.grad(u_y.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dv_dy = torch.autograd.grad(u_y.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    
    eps_xx = du_dx
    eps_yy = dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)
    
    tr_eps = eps_xx + eps_yy
    sigma_xx = lam * tr_eps + 2 * mu * eps_xx
    sigma_yy = lam * tr_eps + 2 * mu * eps_yy
    sigma_xy = 2 * mu * eps_xy
    
    dsigma_xx_dx = torch.autograd.grad(sigma_xx.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dsigma_xy_dy = torch.autograd.grad(sigma_xy.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    dsigma_xy_dx = torch.autograd.grad(sigma_xy.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dsigma_yy_dy = torch.autograd.grad(sigma_yy.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    
    residual_x = dsigma_xx_dx + dsigma_xy_dy
    residual_y = dsigma_xy_dx + dsigma_yy_dy
    
    # SA-PINN: 应用自适应权重到域内残差
    residual_total = residual_x**2 + residual_y**2
    loss_pde = torch.mean(model.lambda_domain * residual_total)
    
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
    
    # SA-PINN: 应用自适应权重到右边界
    bc_error_right = traction_x_error**2 + traction_y_error**2
    loss_bc_right = torch.mean(model.lambda_boundary_right * bc_error_right.unsqueeze(1))
    
    # 3. 左边界应力条件 (类似处理)
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
    
    bc_error_left = traction_x_error**2 + traction_y_error**2
    loss_bc_left = torch.mean(model.lambda_boundary_left * bc_error_left.unsqueeze(1))
    
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
    
    bc_error_top = traction_x_error**2 + traction_y_error**2
    loss_bc_top = torch.mean(model.lambda_boundary_top * bc_error_top.unsqueeze(1))
    
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
    
    bc_error_bottom = traction_x_error**2
    loss_bc_bottom = torch.mean(model.lambda_boundary_bottom * bc_error_bottom.unsqueeze(1))
    
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
    
    traction_x_hole = sigma_xx_h * n_x + sigma_xy_h * n_y
    traction_y_hole = sigma_xy_h * n_x + sigma_yy_h * n_y
    
    bc_error_hole = traction_x_hole**2 + traction_y_hole**2
    loss_bc_hole = torch.mean(model.lambda_boundary_hole * bc_error_hole.unsqueeze(1))
    
    # 总损失
    total_loss = loss_pde + loss_bc_left + loss_bc_right + loss_bc_top + loss_bc_bottom + loss_bc_hole
    
    return total_loss, loss_pde, loss_bc_left, loss_bc_right, loss_bc_top, loss_bc_bottom, loss_bc_hole

# SA-PINN 混合优化训练函数
def train_sa_hybrid_optimization(n_domain=6000, n_boundary=500, 
                               adam_epochs=5000, adam_lr=1e-3,
                               lbfgs_max_iter=10000):
    """
    SA-PINN混合优化策略：Adam预训练 + L-BFGS精调
    """
    
    # 读取参考数据
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    
    # Adam阶段的检查点
    adam_check_points = [1000, 2500, 4000, 5000]
    # L-BFGS阶段的检查点
    lbfgs_check_points = [adam_epochs + 2000, adam_epochs + 5000, adam_epochs + 8000, adam_epochs + 10000]
    
    # 生成训练点
    domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole = generate_points(
        n_domain, n_boundary, n_boundary*2
    )
    
    # 初始化SA-PINN模型
    model = SA_PINN_Network(
        n_domain=len(domain_pts),
        n_boundary_left=len(boundary_left),
        n_boundary_right=len(boundary_right),
        n_boundary_top=len(boundary_top),
        n_boundary_bottom=len(boundary_bottom),
        n_boundary_hole=len(boundary_hole)
    ).to(DEVICE)
    
    # 训练历史
    loss_history = []
    
    # 训练计时
    start_time = time.time()
    
    print("=" * 60)
    print("Phase 1: SA-PINN Adam预训练阶段")
    print("=" * 60)
    
    # 第一阶段：Adam预训练 - 双重优化器
    # 网络权重优化器
    optimizer_net = torch.optim.Adam(model.net.parameters(), lr=adam_lr)
    # 自适应权重优化器 (更大的学习率)
    optimizer_weights = torch.optim.Adam([
        model.lambda_domain,
        model.lambda_boundary_left,
        model.lambda_boundary_right,
        model.lambda_boundary_top,
        model.lambda_boundary_bottom,
        model.lambda_boundary_hole
    ], lr=adam_lr * 10)  # 自适应权重使用更大的学习率
    
    for epoch in range(adam_epochs):
        # 清零梯度
        optimizer_net.zero_grad()
        optimizer_weights.zero_grad()
        
        # 计算损失
        loss, loss_pde, loss_bc_l, loss_bc_r, loss_bc_t, loss_bc_b, loss_bc_h = compute_sa_loss_strong(
            model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole
        )
        
        # 反向传播
        loss.backward()
        
        # 网络权重：梯度下降
        optimizer_net.step()
        
        # 自适应权重：梯度上升 (参考TensorFlow代码的实现)
        for param in [model.lambda_domain, model.lambda_boundary_left, model.lambda_boundary_right,
                     model.lambda_boundary_top, model.lambda_boundary_bottom, model.lambda_boundary_hole]:
            if param.grad is not None:
                param.grad = -param.grad  # 梯度上升：取负号
        optimizer_weights.step()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 检查是否需要输出结果
        if epoch + 1 in adam_check_points:
            current_time = time.time() - start_time
            print(f"\n训练时长: {current_time:.2f} 秒")
            print(f"权重统计 - 域内: {model.lambda_domain.mean().item():.3f}±{model.lambda_domain.std().item():.3f}, "
                  f"边界: {model.lambda_boundary_right.mean().item():.3f}±{model.lambda_boundary_right.std().item():.3f}")
            compute_error_at_iteration(model, epoch + 1, ref_x, ref_y, ref_u1, ref_u2, "SA-Adam ")
        
        if epoch % 500 == 0:
            print(f"SA-Adam Epoch {epoch:4d}: Loss={loss_value:.6f}, "
                  f"PDE={loss_pde:.6f}, BC_right={loss_bc_r:.6f}")
    
    adam_time = time.time() - start_time
    print(f"\nSA-Adam预训练完成，用时: {adam_time:.2f} 秒")
    print(f"SA-Adam最终损失: {loss_history[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("Phase 2: SA-PINN L-BFGS精调阶段 (固定自适应权重)")
    print("=" * 60)
    
    # 第二阶段：L-BFGS精调 (只优化网络权重，固定自适应权重)
    lbfgs_optimizer = torch.optim.LBFGS(
        model.net.parameters(),  # 只优化网络权重
        max_iter=lbfgs_max_iter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-7,
        tolerance_change=1e-12
    )
    
    lbfgs_iteration = [adam_epochs]
    
    def closure():
        lbfgs_optimizer.zero_grad()
        loss, loss_pde, loss_bc_l, loss_bc_r, loss_bc_t, loss_bc_b, loss_bc_h = compute_sa_loss_strong(
            model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole
        )
        loss.backward()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 检查是否需要输出结果
        if lbfgs_iteration[0] in lbfgs_check_points:
            current_time = time.time() - start_time
            print(f"\n训练时长: {current_time:.2f} 秒")
            compute_error_at_iteration(model, lbfgs_iteration[0], ref_x, ref_y, ref_u1, ref_u2, "SA-L-BFGS ")
        
        if lbfgs_iteration[0] % 100 == 0:
            print(f"SA-L-BFGS Iter {lbfgs_iteration[0]:4d}: Loss={loss_value:.6f}, "
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
    print(f"SA-Adam阶段: {adam_epochs} 迭代")
    print(f"SA-L-BFGS阶段: {total_lbfgs_iterations} 迭代")
    print(f"总迭代数: {lbfgs_iteration[0]}")
    
    # 输出最终自适应权重统计
    print(f"\n最终自适应权重统计:")
    print(f"  域内点权重: 均值={model.lambda_domain.mean().item():.3f}, 标准差={model.lambda_domain.std().item():.3f}")
    print(f"  右边界权重: 均值={model.lambda_boundary_right.mean().item():.3f}, 标准差={model.lambda_boundary_right.std().item():.3f}")
    print(f"  圆孔边界权重: 均值={model.lambda_boundary_hole.mean().item():.3f}, 标准差={model.lambda_boundary_hole.std().item():.3f}")
    
    return model, loss_history, adam_epochs

# 可视化结果 (保持不变，但支持SA-PINN模型)
def visualize_sa_results(model):
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
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
    axes[0].set_title('  ')#SA-PINN U
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
    axes[1].set_title('  ')#SA-PINN V
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
    plt.savefig('final_displacement_sa_pinn.svg', format='svg', bbox_inches='tight')
    plt.show()

# 可视化自适应权重分布
def visualize_adaptive_weights(model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole):
    """可视化学习到的自适应权重分布"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 域内点权重
    domain_weights = model.lambda_domain.detach().cpu().numpy().flatten()
    domain_coords = domain_pts.cpu().numpy()
    scatter1 = axes[0, 0].scatter(domain_coords[:, 0], domain_coords[:, 1], 
                                 c=domain_weights, s=20, cmap='viridis')
    axes[0, 0].set_title('Domain points weights', pad=15)
    axes[0, 0].set_aspect('equal')
    #circle = plt.Circle((0, 0), R, fill=False, edgecolor='red', linewidth=2)
    #axes[0, 0].add_patch(circle)
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # 右边界权重
    right_weights = model.lambda_boundary_right.detach().cpu().numpy().flatten()
    right_coords = boundary_right.cpu().numpy()
    scatter2 = axes[0, 1].scatter(right_coords[:, 0], right_coords[:, 1], 
                                 c=right_weights, s=50, cmap='plasma')
    axes[0, 1].set_title('Right boundary weights', pad=15)
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # 圆孔边界权重
    hole_weights = model.lambda_boundary_hole.detach().cpu().numpy().flatten()
    hole_coords = boundary_hole.cpu().numpy()
    scatter3 = axes[0, 2].scatter(hole_coords[:, 0], hole_coords[:, 1], 
                                 c=hole_weights, s=50, cmap='coolwarm')
    axes[0, 2].set_title('Hole boundary weights', pad=15)
    axes[0, 2].set_aspect('equal')
    plt.colorbar(scatter3, ax=axes[0, 2])
    
    # 权重直方图
    axes[1, 0].hist(domain_weights, bins=30, alpha=0.7, label='Domain')
    axes[1, 0].set_title('Domain weights distribution', pad=15)
    axes[1, 0].set_xlabel('Weight value')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(right_weights, bins=30, alpha=0.7, label='Right BC', color='orange')
    axes[1, 1].set_title('Right boundary weights distribution', pad=15)
    axes[1, 1].set_xlabel('Weight value')
    axes[1, 1].set_ylabel('Frequency')
    
    axes[1, 2].hist(hole_weights, bins=30, alpha=0.7, label='Hole BC', color='green')
    axes[1, 2].set_title('Hole boundary weights distribution', pad=15)
    axes[1, 2].set_xlabel('Weight value')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('sa_pinn_adaptive_weights.svg', format='svg', bbox_inches='tight')
    plt.show()

# 绘制SA-PINN混合优化损失曲线
def plot_sa_loss_history_hybrid(loss_history, adam_epochs):
    """绘制SA-PINN混合优化的损失曲线"""
    plt.figure(figsize=(12, 6))
    
    iterations = range(len(loss_history))
    
    plt.semilogy(iterations, loss_history, 'b-', linewidth=2, alpha=0.8)
    
    plt.axvline(x=adam_epochs, color='red', linestyle='--', linewidth=2, 
                label=f'SA-Adam→SA-L-BFGS (iter {adam_epochs})')
    
    plt.fill_betweenx([min(loss_history), max(loss_history)], 0, adam_epochs, 
                      alpha=0.2, color='green', label='SA-Adam Phase')
    plt.fill_betweenx([min(loss_history), max(loss_history)], adam_epochs, len(loss_history), 
                      alpha=0.2, color='blue', label='SA-L-BFGS Phase')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('SA-PINN Hybrid Optimization: Adam + L-BFGS Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('sa_loss_history_hybrid.svg', format='svg', bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    print("SA-PINN (Self-Adaptive PINNs) - 带圆孔板拉伸问题")
    print("=" * 50)
    print(f"问题参数：")
    print(f"板尺寸: {L} × {L} mm")
    print(f"圆孔半径: {R} mm")
    print(f"杨氏模量: {E} MPa")
    print(f"泊松比: {nu}")
    print(f"载荷: {P} MPa")
    print("=" * 50)
    
    # SA-PINN混合优化参数设置
    params = {
        'n_domain': 6000,
        'n_boundary': 500,
        'adam_epochs': 10000,
        'adam_lr': 1e-3,
        'lbfgs_max_iter': 30000
    }
    
    print(f"\nSA-PINN混合优化参数：")
    print(f"SA-Adam预训练: {params['adam_epochs']} epochs, lr={params['adam_lr']}")
    print(f"自适应权重学习率: {params['adam_lr'] * 10}")
    print(f"SA-L-BFGS精调: 最大{params['lbfgs_max_iter']} 迭代")
    
    # 训练SA-PINN模型
    print(f"\n开始SA-PINN混合优化训练...")
    model, loss_history, adam_epochs = train_sa_hybrid_optimization(**params)
    
    # 绘制损失曲线
    plot_sa_loss_history_hybrid(loss_history, adam_epochs)
    
    # 可视化结果
    print("\n生成最终位移场...")
    visualize_sa_results(model)
    
    # 可视化自适应权重
    print("\n可视化学习到的自适应权重...")
    domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole = generate_points(
        params['n_domain'], params['n_boundary'], params['n_boundary']*2
    )
    visualize_adaptive_weights(model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole)
    
    # 计算并输出最终误差指标
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    if ref_x is not None:
        xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            u_theta = model(xy_tensor)
            u_pred = apply_hard_bc(xy_tensor, u_theta).cpu().numpy()
        
        # 计算MSE
        mse_u = np.mean((u_pred[:, 0] - ref_u1)**2)
        mse_v = np.mean((u_pred[:, 1] - ref_u2)**2)
        
        # 计算MAPE
        mape_u = np.mean(np.abs((ref_u1 - u_pred[:, 0]) / (np.abs(ref_u1) + 1e-8))) * 100
        mape_v = np.mean(np.abs((ref_u2 - u_pred[:, 1]) / (np.abs(ref_u2) + 1e-8))) * 100
        
        # 计算相关系数R
        r_u = np.corrcoef(ref_u1, u_pred[:, 0])[0, 1]
        r_v = np.corrcoef(ref_u2, u_pred[:, 1])[0, 1]
        
        print(f"\nSA-PINN最终误差指标结果：")
        print(f"水平位移(U) - MSE: {mse_u:.8e}, MAPE: {mape_u:.4f}%, R: {r_u:.6f}")
        print(f"竖向位移(V) - MSE: {mse_v:.8e}, MAPE: {mape_v:.4f}%, R: {r_v:.6f}")
    
    print("\nSA-PINN混合优化训练完成！")
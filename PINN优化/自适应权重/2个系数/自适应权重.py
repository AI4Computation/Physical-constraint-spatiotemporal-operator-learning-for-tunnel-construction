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

# 自适应权重损失函数 - 方案2
def compute_loss_adaptive(model, domain_points, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole,
                         sigma_pde, sigma_bc):
    """
    方案2：自适应权重损失函数（PDE vs 合并的边界条件）
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
    
    # 自适应损失平衡 - 方案2：只有两个sigma参数
    adaptive_loss = (loss_pde / (2 * sigma_pde.pow(2)) + 
                    loss_bc_combined / (2 * sigma_bc.pow(2)) +
                    torch.log(sigma_pde) + torch.log(sigma_bc)) * 1e3
    
    return adaptive_loss, loss_pde, loss_bc_combined, loss_bc_left, loss_bc_right, loss_bc_top, loss_bc_bottom, loss_bc_hole

# Adam优化训练函数 - 方案2 + 仅Adam
def train_adaptive_adam_only(n_domain=6000, n_boundary=500, 
                           adam_epochs=10000, adam_lr=1e-3):
    """
    方案2：自适应权重 + 仅Adam优化器
    """
    
    # 读取参考数据
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    
    # 初始化模型
    model = PINN_Network().to(DEVICE)
    domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole = generate_points(
        n_domain, n_boundary, n_boundary*2
    )
    
    # 初始化自适应权重参数 - 方案2：只有2个sigma
    sigma_pde = torch.tensor(2.0, dtype=torch.float32, device=DEVICE, requires_grad=True)
    sigma_bc = torch.tensor(2.0, dtype=torch.float32, device=DEVICE, requires_grad=True)
    
    # 优化器
    model_optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    sigma_optimizer = torch.optim.Adam([sigma_pde, sigma_bc], lr=adam_lr)
    
    # 训练历史记录
    loss_history = []
    Pde_loss_history = []
    sigma_history = {'pde': [], 'bc': []}
    weight_history = {'pde': [], 'bc': []}
    individual_bc_history = {
        'left': [], 'right': [], 'top': [], 'bottom': [], 'hole': []
    }
    
    # 检查点设置
    check_points = [1000, 2500, 5000, 7500, 10000]
    
    # 训练计时
    start_time = time.time()
    
    print("=" * 60)
    print("方案2：自适应权重 + Adam优化器训练")
    print("=" * 60)
    print(f"训练参数：")
    print(f"- 域内点数: {n_domain}")
    print(f"- 边界点数: {n_boundary}")
    print(f"- 总迭代数: {adam_epochs}")
    print(f"- 学习率: {adam_lr}")
    print(f"- 初始sigma值: PDE={sigma_pde.item()}, BC={sigma_bc.item()}")
    print("=" * 60)
    
    for epoch in range(adam_epochs):
        # 清零梯度
        model_optimizer.zero_grad()
        sigma_optimizer.zero_grad()
        
        # 计算损失
        loss, loss_pde, loss_bc_combined, loss_bc_l, loss_bc_r, loss_bc_t, loss_bc_b, loss_bc_h = compute_loss_adaptive(
            model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole,
            sigma_pde, sigma_bc
        )
        
        # 反向传播
        loss.backward()
        model_optimizer.step()
        sigma_optimizer.step()
        
        # 记录训练历史
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 记录训练历史
        PDE_loss_value = loss_pde.item()
        Pde_loss_history.append(PDE_loss_value)
        
        # 记录sigma值和权重
        sigma_history['pde'].append(sigma_pde.item())
        sigma_history['bc'].append(sigma_bc.item())
        
        weight_history['pde'].append((1/(2*sigma_pde.pow(2))).item())
        weight_history['bc'].append((1/(2*sigma_bc.pow(2))).item())
        
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
            print(f"当前损失: {loss_value:.6f}")
            print(f"PDE损失: {loss_pde.item():.6f}")
            print(f"BC总损失: {loss_bc_combined.item():.6f}")
            print(f"Sigma值 - PDE: {sigma_pde.item():.4f}, BC: {sigma_bc.item():.4f}")
            print(f"实际权重 - PDE: {(1/(2*sigma_pde.pow(2))).item():.4f}, BC: {(1/(2*sigma_bc.pow(2))).item():.4f}")
            print(f"权重比例 - PDE/(PDE+BC): {(1/(2*sigma_pde.pow(2)))/(1/(2*sigma_pde.pow(2)) + 1/(2*sigma_bc.pow(2))):.4f}")
            
            # 计算误差
            compute_error_at_iteration(model, epoch + 1, ref_x, ref_y, ref_u1, ref_u2, "Adam ")
        
        # 定期输出训练进度
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d}: Loss={loss_value:.6f}, "
                  f"PDE={loss_pde.item():.6f}, BC={loss_bc_combined.item():.6f}, "
                  f"σ_PDE={sigma_pde.item():.3f}, σ_BC={sigma_bc.item():.3f}")
    
    training_time = time.time() - start_time
    print(f"\n训练完成！")
    print(f"总训练时间: {training_time:.2f} 秒")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"最终Sigma值 - PDE: {sigma_pde.item():.4f}, BC: {sigma_bc.item():.4f}")
    print(f"最终权重值 - PDE: {(1/(2*sigma_pde.pow(2))).item():.4f}, BC: {(1/(2*sigma_bc.pow(2))).item():.4f}")
    
    return model, loss_history, sigma_history, weight_history, individual_bc_history, Pde_loss_history

# 可视化自适应权重演化
def plot_adaptive_weights_evolution(sigma_history, weight_history, individual_bc_history,Pde_loss_history):
    """绘制自适应权重演化曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    iterations = range(len(sigma_history['pde']))
    
    # 第一行左：Sigma值演化
    axes[0].semilogy(iterations, sigma_history['pde'], 'blue', linewidth=2, label='σ_PDE')
    axes[0].semilogy(iterations, sigma_history['bc'], 'red', linewidth=2, label='σ_BC')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Sigma Value')
    axes[0].set_title('  ')#'Adaptive Weights (σ) Evolution'
    axes[0].grid(True, alpha=0.3)
    #axes[0, 0].legend()
    
    # 第一行中：实际权重值演化
    axes[1].semilogy(iterations, weight_history['pde'], 'blue', linewidth=2, label='w_PDE = 1/(2σ²_PDE)')
    axes[1].semilogy(iterations, weight_history['bc'], 'red', linewidth=2, label='w_BC = 1/(2σ²_BC)')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Actual Weight Value')
    axes[1].set_title('  ')#'Actual Weights Evolution'
    axes[1].grid(True, alpha=0.3)
    #axes[0, 1].legend()
    
    # 第一行右：Lbc vs Lpde
    
    bc_total = [sum([individual_bc_history[key][i] for key in individual_bc_history.keys()]) 
                for i in range(len(iterations))]
    axes[2].semilogy(bc_total, 'orange', linewidth=2, label='Total BC Loss')
    axes[2].semilogy(Pde_loss_history, 'green', linewidth=2, label='Total PDE Loss')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('  ')
    axes[2].grid(True, alpha=0.3)
    #axes[0, 2].legend()
    
  
    
    plt.tight_layout()
    plt.savefig('adaptive_weights_evolution_adam_only.svg', format='svg', bbox_inches='tight')
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
    axes[0].set_title('Final U Displacement')
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
    axes[1].set_title('Final V Displacement')
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
    plt.savefig('final_displacement_adaptive_adam.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    # 输出关键点位移
    test_points = torch.tensor([
        [0.4, 0.0],   # 右侧中点
        [0.0, 0.4],   # 上侧中点
        [-0.4, 0.0],  # 左侧中点
        [0.0, -0.4],  # 下侧中点
        [0.3, 0.0],   # 右侧3/4处
        [R*1.2, 0.0],  # 圆孔右侧附近
    ], dtype=torch.float32, device=DEVICE)
    
    with torch.no_grad():
        u_theta_test = model(test_points)
        u_test = apply_hard_bc(test_points, u_theta_test).cpu().numpy()
    
    print("\n关键点位移值：")
    print("点位置\t\t\tu (mm)\t\tv (mm)")
    print("-" * 50)
    for i, pt in enumerate(test_points.cpu().numpy()):
        print(f"({pt[0]:5.1f}, {pt[1]:5.1f})\t{u_test[i, 0]:10.6f}\t{u_test[i, 1]:10.6f}")

# 绘制损失历史
def plot_loss_history(loss_history):
    """绘制损失演化曲线"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, 'b-', linewidth=2, label='Total Loss')
    

    plt.legend()
    
    
    
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('  ')#'Training Loss Evolution (Adaptive Weights + Adam)'
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_history_adaptive_adam.svg', format='svg', bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    print("自适应权重PINNs - 带圆孔板拉伸问题")
    print("方案2：简化自适应权重 + Adam优化器")
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
    
    # 开始训练
    print(f"\n开始自适应权重训练...")
    model, loss_history, sigma_history, weight_history, bc_history, PDE_history = train_adaptive_adam_only(**params)
    
    # 绘制损失曲线
    plot_loss_history(loss_history)
    
    # 绘制自适应权重演化
    plot_adaptive_weights_evolution(sigma_history, weight_history, bc_history, PDE_history)
    
    # 可视化最终结果
    print("\n生成最终位移场...")
    visualize_results(model)
    
    # 计算并输出最终MSE、MAPE和R
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    if ref_x is not None:
        xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            u_theta = model(xy_tensor)
            u_pred = apply_hard_bc(xy_tensor, u_theta).cpu().numpy()
        
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
    
    print("\n自适应权重训练完成！")
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from torch.optim import Optimizer
from torch.func import vmap
from functools import reduce

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

# ===================== NysNewtonCG 实现 =====================
def _armijo(f, x, gx, dx, t, alpha=0.1, beta=0.5):
    """Line search to find a step size that satisfies the Armijo condition."""
    f0 = f(x, 0, dx)
    f1 = f(x, t, dx)
    while f1 > f0 + alpha * t * gx.dot(dx):
        t *= beta
        f1 = f(x, t, dx)
    return t

def _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, x):
    """Applies the inverse of the Nystrom approximation of the Hessian to a vector."""
    z = U.T @ x
    z = (lambd_r + mu) * (U @ (S_mu_inv * z)) + (x - U @ z)
    return z

def _nystrom_pcg(hess, b, x, mu, U, S, r, tol, max_iters):
    """Solves a positive-definite linear system using NyströmPCG."""
    lambd_r = S[r - 1]
    S_mu_inv = (S + mu) ** (-1)

    resid = b - (hess(x) + mu * x)
    with torch.no_grad():
        z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
        p = z.clone()

    i = 0

    while torch.norm(resid) > tol and i < max_iters:
        v = hess(p) + mu * p
        with torch.no_grad():
            alpha = torch.dot(resid, z) / torch.dot(p, v)
            x += alpha * p

            rTz = torch.dot(resid, z)
            resid -= alpha * v
            z = _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, resid)
            beta = torch.dot(resid, z) / rTz

            p = z + beta * p

        i += 1

    if torch.norm(resid) > tol:
        print(f"Warning: PCG did not converge to tolerance. Tolerance was {tol} but norm of residual is {torch.norm(resid)}")

    return x

class NysNewtonCG(Optimizer):
    """Implementation of NysNewtonCG, a damped Newton-CG method that uses Nyström preconditioning."""
    
    def __init__(self, params, lr=1.0, rank=10, mu=1e-4, chunk_size=1,
                 cg_tol=1e-16, cg_max_iters=1000, line_search_fn=None, verbose=False):
        defaults = dict(lr=lr, rank=rank, chunk_size=chunk_size, mu=mu, cg_tol=cg_tol,
                        cg_max_iters=cg_max_iters, line_search_fn=line_search_fn)
        self.rank = rank
        self.mu = mu
        self.chunk_size = chunk_size
        self.cg_tol = cg_tol
        self.cg_max_iters = cg_max_iters
        self.line_search_fn = line_search_fn
        self.verbose = verbose
        self.U = None
        self.S = None
        self.n_iters = 0
        super(NysNewtonCG, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError(
                "NysNewtonCG doesn't currently support per-parameter options (parameter groups)")

        if self.line_search_fn is not None and self.line_search_fn != 'armijo':
            raise ValueError("NysNewtonCG only supports Armijo line search")

        self._params = self.param_groups[0]['params']
        self._params_list = list(self._params)
        self._numel_cache = None

    def step(self, closure=None):
        """Perform a single optimization step."""
        if self.n_iters == 0:
            # Store the previous direction for warm starting PCG
            self.old_dir = torch.zeros(
                self._numel(), device=self._params[0].device)

        # NOTE: The closure must return both the loss and the gradient
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss, grad_tuple = closure()

        g = torch.cat([grad.view(-1) for grad in grad_tuple if grad is not None])

        # One step update
        for group_idx, group in enumerate(self.param_groups):
            def hvp_temp(x):
                return self._hvp(g, self._params_list, x)

            # Calculate the Newton direction
            d = _nystrom_pcg(hvp_temp, g, self.old_dir,
                             self.mu, self.U, self.S, self.rank, self.cg_tol, self.cg_max_iters)

            # Store the previous direction for warm starting PCG
            self.old_dir = d

            # Check if d is a descent direction
            if torch.dot(d, g) <= 0:
                print("Warning: d is not a descent direction")

            if self.line_search_fn == 'armijo':
                x_init = self._clone_param()

                def obj_func(x, t, dx):
                    self._add_grad(t, dx)
                    loss = float(closure()[0])
                    self._set_param(x)
                    return loss

                # Use -d for convention
                t = _armijo(obj_func, x_init, g, -d, group['lr'])
            else:
                t = group['lr']

            self.state[group_idx]['t'] = t

            # update parameters
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = d[ls:ls+np].view(p.shape)
                ls += np
                p.data.add_(-dp, alpha=t)

        self.n_iters += 1

        return loss, g

    def update_preconditioner(self, grad_tuple):
        """Update the Nystrom approximation of the Hessian."""
        # Flatten and concatenate the gradients
        gradsH = torch.cat([gradient.view(-1)
                           for gradient in grad_tuple if gradient is not None])

        # Generate test matrix (NOTE: This is transposed test matrix)
        p = gradsH.shape[0]
        Phi = torch.randn(
            (self.rank, p), device=gradsH.device) / (p ** 0.5)
        Phi = torch.linalg.qr(Phi.t(), mode='reduced')[0].t()

        Y = self._hvp_vmap(gradsH, self._params_list)(Phi)

        # Calculate shift
        shift = torch.finfo(Y.dtype).eps
        Y_shifted = Y + shift * Phi

        # Calculate Phi^T * H * Phi (w/ shift) for Cholesky
        choleskytarget = torch.mm(Y_shifted, Phi.t())

        # Perform Cholesky, if fails, do eigendecomposition
        try:
            C = torch.linalg.cholesky(choleskytarget)
        except:
            # eigendecomposition, eigenvalues and eigenvector matrix
            eigs, eigvectors = torch.linalg.eigh(choleskytarget)
            shift = shift + torch.abs(torch.min(eigs))
            # add shift to eigenvalues
            eigs = eigs + shift
            # put back the matrix for Cholesky by eigenvector * eigenvalues after shift * eigenvector^T
            C = torch.linalg.cholesky(
                torch.mm(eigvectors, torch.mm(torch.diag(eigs), eigvectors.T)))

        try:
            B = torch.linalg.solve_triangular(
                C, Y_shifted, upper=False, left=True)
        # temporary fix for issue @ https://github.com/pytorch/pytorch/issues/97211
        except:
            B = torch.linalg.solve_triangular(C.to('cpu'), Y_shifted.to(
                'cpu'), upper=False, left=True).to(C.device)
            
        # B = V * S * U^T b/c we have been using transposed sketch
        _, S, UT = torch.linalg.svd(B, full_matrices=False)
        self.U = UT.t()
        self.S = torch.max(torch.square(S) - shift, torch.tensor(0.0))

        self.rho = self.S[-1]

        if self.verbose:
            print(f'Approximate eigenvalues = {self.S}')

    def _hvp_vmap(self, grad_params, params):
        return vmap(lambda v: self._hvp(grad_params, params, v), in_dims=0, chunk_size=self.chunk_size)

    def _hvp(self, grad_params, params, v):
        Hv = torch.autograd.grad(grad_params, params, grad_outputs=v,
                                 retain_graph=True)
        Hv = tuple(Hvi.detach() for Hvi in Hv)
        return torch.cat([Hvi.reshape(-1) for Hvi in Hv])

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Avoid in-place operation by creating a new tensor
            p.data = p.data.add(
                update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            # Replace the .data attribute of the tensor
            p.data = pdata.data

# ===================== 原有代码部分 =====================

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
    error_u_vals = u_pred[:, 0] - ref_u1
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
    """应用硬边界条件"""
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    
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

# 强形式损失函数（支持返回梯度）
def compute_loss_strong(model, domain_points, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole, return_grads=False):
    """计算强形式损失函数，可选择性返回梯度"""
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
    
    # 2. 右边界应力条件: σ·n = [P, 0]
    xy_right = boundary_right.clone().requires_grad_(True)
    u_theta_right = model(xy_right)
    u_right = apply_hard_bc(xy_right, u_theta_right)
    
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
    u_theta_left = model(xy_left)
    u_left = apply_hard_bc(xy_left, u_theta_left)
    
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
    
    # 上边界误差计算:
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
    
    # 下边界误差计算:
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
    
    # 总损失（加权）
    w_pde = 1.0
    w_bc = 10.0  # 边界条件权重更高
    
    total_loss = w_pde * loss_pde + w_bc * (loss_bc_left+ loss_bc_right + loss_bc_top + loss_bc_bottom+ loss_bc_hole)
    
    if return_grads:
        # 计算梯度
        grads = torch.autograd.grad(total_loss, model.parameters(), create_graph=True, retain_graph=True)
        return total_loss, (loss_pde, loss_bc_left, loss_bc_right, loss_bc_top, loss_bc_bottom, loss_bc_hole), grads
    
    return total_loss, loss_pde, loss_bc_left, loss_bc_right, loss_bc_top, loss_bc_bottom, loss_bc_hole

def should_switch_to_nncg(loss_history, grad_norm, patience=100, grad_threshold=1e-3):
    """判断是否应该切换到NNCG"""
    if len(loss_history) < patience:
        return False
    
    # 检查损失是否停止下降
    recent_losses = loss_history[-patience:]
    loss_improvement = recent_losses[0] - recent_losses[-1]
    relative_improvement = loss_improvement / recent_losses[0] if recent_losses[0] > 0 else 0
    
    # 如果损失改善很小但梯度仍然较大，说明L-BFGS卡住了
    return grad_norm > grad_threshold##relative_improvement < 1e-6 and 

# 三阶段优化训练函数
def train_three_stage_optimization(n_domain=6000, n_boundary=500, 
                                 adam_epochs=5000, adam_lr=1e-3,
                                 lbfgs_max_iter=10000,
                                 nncg_max_iter=5000, nncg_rank=60, nncg_mu=1e-2,
                                 force_nncg=True, nncg_start_epoch=None):
    """
    三阶段优化策略：Adam预训练 + L-BFGS精调 + NNCG超精调
    
    Args:
        force_nncg: 是否强制启用NNCG（True=固定epoch切换，False=动态判断）
        nncg_start_epoch: NNCG开始的epoch，如果为None则自动设置为adam_epochs + lbfgs_max_iter
    """
    
    # 设置NNCG开始epoch
    if nncg_start_epoch is None:
        nncg_start_epoch = adam_epochs + lbfgs_max_iter
    
    # 读取参考数据
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data()
    
    # 检查点设置
    adam_check_points = [1000, 2500, 4000, 5000]
    lbfgs_check_points = [adam_epochs + 2000, adam_epochs + 5000, adam_epochs + 8000, adam_epochs + 10000]
    nncg_check_points = [adam_epochs + lbfgs_max_iter + 1000, adam_epochs + lbfgs_max_iter + 3000, adam_epochs + lbfgs_max_iter + 5000]
    
    # 初始化
    model = PINN_Network().to(DEVICE)
    domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole = generate_points(
        n_domain, n_boundary, n_boundary*2
    )
    
    # 训练历史
    loss_history = []
    
    # 训练计时
    start_time = time.time()
    
    print("=" * 60)
    print("Phase 1: Adam预训练阶段")
    print("=" * 60)
    
    # 第一阶段：Adam预训练
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
    print("Phase 2: L-BFGS精调阶段")
    print("=" * 60)
    
    # 第二阶段：L-BFGS精调
    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_max_iter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-7,
        tolerance_change=1e-12
    )
    
    lbfgs_iteration = [adam_epochs]
    grad_norm_history = []
    
    def closure():
        lbfgs_optimizer.zero_grad()
        loss, loss_pde, loss_bc_l, loss_bc_r, loss_bc_t, loss_bc_b, loss_bc_h = compute_loss_strong(
            model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole
        )
        loss.backward()
        
        # 计算梯度范数
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        grad_norm_history.append(grad_norm)
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # 检查是否需要输出结果
        if lbfgs_iteration[0] in lbfgs_check_points:
            current_time = time.time() - start_time
            print(f"\n训练时长: {current_time:.2f} 秒")
            compute_error_at_iteration(model, lbfgs_iteration[0], ref_x, ref_y, ref_u1, ref_u2, "L-BFGS ")
        
        if lbfgs_iteration[0] % 100 == 0:
            print(f"L-BFGS Iter {lbfgs_iteration[0]:4d}: Loss={loss_value:.6f}, "
                  f"PDE={loss_pde:.6f}, Grad_norm={grad_norm:.6f}")
        
        lbfgs_iteration[0] += 1
        return loss
    
    # 执行L-BFGS优化
    iterations_before = lbfgs_iteration[0]
    
    if force_nncg:
        # 固定epoch切换模式：L-BFGS只运行到指定epoch
        target_lbfgs_epochs = nncg_start_epoch - adam_epochs
        print(f"固定切换模式：L-BFGS将运行{target_lbfgs_epochs}个epoch后切换到NNCG")
        
        for _ in range(target_lbfgs_epochs):
            if lbfgs_iteration[0] >= nncg_start_epoch:
                break
            lbfgs_optimizer.step(closure)
            
        total_lbfgs_iterations = lbfgs_iteration[0] - iterations_before
        
    else:
        # 动态判断模式：让L-BFGS运行完整个过程
        lbfgs_optimizer.step(closure)
        total_lbfgs_iterations = lbfgs_iteration[0] - iterations_before
    
    lbfgs_time = time.time() - start_time
    final_grad_norm = grad_norm_history[-1] if grad_norm_history else 0.0
    
    print(f"\nL-BFGS精调完成，用时: {lbfgs_time - adam_time:.2f} 秒")
    print(f"L-BFGS最终损失: {loss_history[-1]:.6f}")
    print(f"L-BFGS最终梯度范数: {final_grad_norm:.6f}")
    print(f"L-BFGS迭代数: {total_lbfgs_iterations}")
    
    # 判断是否需要NNCG
    if force_nncg:
        # 固定切换模式：总是执行NNCG
        need_nncg = True
        print(f"\n固定切换模式：在epoch {nncg_start_epoch}强制启动NNCG")
    else:
        # 动态判断模式：根据L-BFGS表现决定
        need_nncg = should_switch_to_nncg(loss_history[-200:], final_grad_norm)
        if need_nncg:
            print(f"\n动态判断：检测到L-BFGS过早停止，启动NNCG")
        else:
            print(f"\n动态判断：L-BFGS收敛良好，无需NNCG")
    
    if need_nncg:
        print("\n" + "=" * 60)
        print("Phase 3: NNCG超精调阶段")
        print("检测到L-BFGS过早停止，启动NNCG继续优化")
        print("=" * 60)
        
        # 第三阶段：NNCG超精调
        # 调整参数以应对数值不稳定问题
        nncg_optimizer = NysNewtonCG(
            model.parameters(),
            lr=1.0,
            rank=min(nncg_rank, 30),  # 降低rank避免过拟合
            mu=max(nncg_mu, 0.1),  # 增加阻尼以提高稳定性
            cg_tol=1e-6,  # 放松PCG容差（从1e-16放松）
            cg_max_iters=100,  # 减少PCG最大迭代数
            line_search_fn='armijo',
            verbose=False  # 减少输出
        )
        
        nncg_iteration = lbfgs_iteration[0]
        
        def nncg_closure():
            loss, loss_components, grads = compute_loss_strong(
                model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole,
                return_grads=True
            )
            return loss, grads
        
        # 初始化预条件器
        print("初始化NNCG预条件器...")
        try:
            _, grads = nncg_closure()
            nncg_optimizer.update_preconditioner(grads)
            print("预条件器初始化成功")
        except Exception as e:
            print(f"预条件器初始化失败: {e}")
            print("跳过NNCG优化，使用当前L-BFGS结果")
            return model, loss_history, adam_epochs, total_lbfgs_iterations
        
        # 记录NNCG开始时的损失
        initial_nncg_loss = loss_history[-1]
        best_loss = initial_nncg_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        max_patience = 20  # 20次迭代没改善就停止
        
        for epoch in range(nncg_max_iter):
            try:
                # 每20步更新一次预条件器
                if epoch % 20 == 0 and epoch > 0:
                    try:
                        _, grads = nncg_closure()
                        nncg_optimizer.update_preconditioner(grads)
                    except Exception as e:
                        print(f"预条件器更新失败 (epoch {epoch}): {e}")
                        print("继续使用旧的预条件器...")
                
                # NNCG优化步
                loss, grads = nncg_optimizer.step(nncg_closure)
                
                loss_value = loss.item()
                
                # 检查损失是否合法
                if np.isnan(loss_value) or np.isinf(loss_value):
                    print(f"检测到非法损失值: {loss_value}")
                    print("恢复到最佳模型状态")
                    model.load_state_dict(best_model_state)
                    break
                
                # 检查损失是否爆炸（增加超过10倍）
                if loss_value > initial_nncg_loss * 10:
                    print(f"损失爆炸: {loss_value:.6f} > {initial_nncg_loss * 10:.6f}")
                    print("恢复到最佳模型状态")
                    model.load_state_dict(best_model_state)
                    break
                
                loss_history.append(loss_value)
                
                # 更新最佳模型
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 计算梯度范数
                grad_norm = torch.cat([g.view(-1) for g in grads]).norm().item()
                
                # 检查是否需要输出结果
                if nncg_iteration in nncg_check_points:
                    current_time = time.time() - start_time
                    print(f"\n训练时长: {current_time:.2f} 秒")
                    compute_error_at_iteration(model, nncg_iteration, ref_x, ref_y, ref_u1, ref_u2, "NNCG ")
                
                if epoch % 50 == 0:
                    print(f"NNCG Iter {nncg_iteration:4d}: Loss={loss_value:.6f}, "
                          f"Best={best_loss:.6f}, Grad_norm={grad_norm:.6f}")
                
                nncg_iteration += 1
                
                # 早停条件
                if grad_norm < 1e-5:
                    print(f"NNCG收敛：梯度范数 {grad_norm:.6e} < 1e-5")
                    break
                
                # 耐心用尽
                if patience_counter >= max_patience:
                    print(f"NNCG早停：{max_patience}次迭代无改善")
                    model.load_state_dict(best_model_state)
                    break
                    
            except Exception as e:
                print(f"NNCG优化失败 (epoch {epoch}): {e}")
                print("恢复到最佳模型状态")
                model.load_state_dict(best_model_state)
                break
        
        nncg_time = time.time() - start_time
        print(f"\nNNCG超精调完成，用时: {nncg_time - lbfgs_time:.2f} 秒")
        print(f"NNCG最佳损失: {best_loss:.6f}")
        print(f"NNCG改善: {((initial_nncg_loss - best_loss) / initial_nncg_loss * 100):.2f}%")
        total_iterations = nncg_iteration
        
    else:
        print("\n梯度范数已足够小，无需NNCG优化")
        total_iterations = lbfgs_iteration[0]
    
    training_time = time.time() - start_time
    print(f"\n总训练时间: {training_time:.2f} 秒")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"总迭代数: {total_iterations}")
    
    return model, loss_history, adam_epochs, total_lbfgs_iterations

# 可视化结果
def visualize_results(model):
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
    axes[0].set_title('  ')
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
    axes[1].set_title('  ')
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
    plt.savefig('final_displacement_three_stage.svg', format='svg', bbox_inches='tight')
    plt.show()

# 绘制三阶段损失曲线
def plot_loss_history_three_stage(loss_history, adam_epochs, lbfgs_iterations):
    """绘制三阶段优化的损失曲线"""
    plt.figure(figsize=(12, 6))
    
    iterations = range(len(loss_history))
    
    # 绘制完整的损失曲线
    plt.semilogy(iterations, loss_history, 'b-', linewidth=2, alpha=0.8)
    
    # 标记各阶段的分界线
    plt.axvline(x=adam_epochs, color='red', linestyle='--', linewidth=2, 
                label=f'Adam→L-BFGS (iter {adam_epochs})')
    
    if len(loss_history) > adam_epochs + lbfgs_iterations:
        plt.axvline(x=adam_epochs + lbfgs_iterations, color='green', linestyle='--', linewidth=2, 
                    label=f'L-BFGS→NNCG (iter {adam_epochs + lbfgs_iterations})')
    
    # 添加区域标注
    plt.fill_betweenx([min(loss_history), max(loss_history)], 0, adam_epochs, 
                      alpha=0.2, color='blue', label='Adam Phase')
    plt.fill_betweenx([min(loss_history), max(loss_history)], adam_epochs, 
                      min(adam_epochs + lbfgs_iterations, len(loss_history)), 
                      alpha=0.2, color='orange', label='L-BFGS Phase')
    
    if len(loss_history) > adam_epochs + lbfgs_iterations:
        plt.fill_betweenx([min(loss_history), max(loss_history)], 
                          adam_epochs + lbfgs_iterations, len(loss_history), 
                          alpha=0.2, color='green', label='NNCG Phase')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Three-Stage Optimization: Adam + L-BFGS + NNCG Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('loss_history_three_stage.svg', format='svg', bbox_inches='tight')
    plt.show()

# 主程序
if __name__ == "__main__":
    print("三阶段优化策略PINNs - 带圆孔板拉伸问题")
    print("=" * 50)
    print(f"问题参数：")
    print(f"板尺寸: {L} × {L} mm")
    print(f"圆孔半径: {R} mm")
    print(f"杨氏模量: {E} MPa")
    print(f"泊松比: {nu}")
    print(f"载荷: {P} MPa")
    print("=" * 50)
    
    # 三阶段优化参数设置
    params = {
        'n_domain': 6000,
        'n_boundary': 500,
        'adam_epochs': 10000,
        'adam_lr': 1e-3,
        'lbfgs_max_iter': 50000,  # 增加L-BFGS迭代数，让它充分优化
        'nncg_max_iter': 500,   # 减少NNCG迭代数，避免过度优化
        'nncg_rank': 20,         # 降低rank提高稳定性（从60降到20）
        'nncg_mu': 0.1,          # 增加阻尼参数提高稳定性（从0.01增到0.5）
        'force_nncg': False,      # 强制启用NNCG
        'nncg_start_epoch': 60000  # 让L-BFGS充分运行后再启动NNCG
    }
    
    print(f"\n三阶段优化参数：")
    print(f"Adam预训练: {params['adam_epochs']} epochs, lr={params['adam_lr']}")
    print(f"L-BFGS精调: 最大{params['lbfgs_max_iter']} 迭代")
    print(f"NNCG超精调: 最大{params['nncg_max_iter']} 迭代, rank={params['nncg_rank']}, mu={params['nncg_mu']}")
    print(f"切换模式: {'固定epoch切换' if params['force_nncg'] else '动态判断切换'}")
    if params['force_nncg']:
        print(f"NNCG启动epoch: {params['nncg_start_epoch']}")
    
    # 训练模型
    print(f"\n开始三阶段优化训练...")
    model, loss_history, adam_epochs, lbfgs_iterations = train_three_stage_optimization(**params)
    
    # 绘制损失曲线
    plot_loss_history_three_stage(loss_history, adam_epochs, lbfgs_iterations)
    
    # 可视化结果
    print("\n生成最终位移场...")
    visualize_results(model)
    
    # 计算并输出最终MSE、MAPE和R²
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
        
        # 计算相关系数R²
        r_u = np.corrcoef(ref_u1, u_pred[:, 0])[0, 1]
        r_v = np.corrcoef(ref_u2, u_pred[:, 1])[0, 1]
        
        print(f"\n最终误差指标结果：")
        print(f"水平位移(U) - MSE: {mse_u:.8e}, MAPE: {mape_u:.4f}%, R²: {r_u:.6f}")
        print(f"竖向位移(V) - MSE: {mse_v:.8e}, MAPE: {mape_v:.4f}%, R²: {r_v:.6f}")
    
    print("\n三阶段优化训练完成！")
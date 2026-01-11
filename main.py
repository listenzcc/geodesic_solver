from tqdm.auto import tqdm
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PerlinSurfaceGeodesic:
    def __init__(self, perlin_func, grad_func, hessian_func=None):
        """
        perlin_func: f(x,y) -> z
        grad_func: 返回 (df/dx, df/dy)
        hessian_func: 可选，返回二阶导数矩阵
        """
        self.f = perlin_func
        self.grad = grad_func
        self.hessian = hessian_func

    def metric_tensor(self, x, y):
        """计算度量张量 g_{ij} = δ_{ij} + f_i f_j"""
        f_x, f_y = self.grad(x, y)

        g = np.zeros((2, 2))
        g[0, 0] = 1 + f_x**2      # g_11
        g[0, 1] = f_x * f_y       # g_12
        g[1, 0] = f_x * f_y       # g_21
        g[1, 1] = 1 + f_y**2      # g_22

        return g

    def metric_inverse(self, x, y):
        """计算度量张量的逆 g^{ij}"""
        g = self.metric_tensor(x, y)
        return np.linalg.inv(g)

    def christoffel_symbols(self, x, y):
        """数值计算克里斯托费尔符号 Γ^k_ij"""
        eps = 1e-6

        # 计算基础度量
        g = self.metric_tensor(x, y)
        g_inv = self.metric_inverse(x, y)

        # 计算度量导数的有限差分近似
        # ∂g/∂x
        g_x_plus = self.metric_tensor(x + eps, y)
        g_x_minus = self.metric_tensor(x - eps, y)
        dg_dx = (g_x_plus - g_x_minus) / (2 * eps)

        # ∂g/∂y
        g_y_plus = self.metric_tensor(x, y + eps)
        g_y_minus = self.metric_tensor(x, y - eps)
        dg_dy = (g_y_plus - g_y_minus) / (2 * eps)

        # 计算克里斯托费尔符号
        Gamma = np.zeros((2, 2, 2))  # Gamma[k,i,j]

        for k in range(2):
            for i in range(2):
                for j in range(2):
                    # Γᵏᵢⱼ = ½ g^{kl} (∂g_{jl}/∂xⁱ + ∂g_{il}/∂xʲ - ∂g_{ij}/∂xˡ)
                    sum_val = 0.0
                    for l in range(2):
                        dg_jl_dxi = dg_dx[j, l] if i == 0 else dg_dy[j, l]
                        dg_il_dxj = dg_dx[i, l] if j == 0 else dg_dy[i, l]
                        dg_ij_dxl = dg_dx[i, j] if l == 0 else dg_dy[i, j]

                        term = dg_jl_dxi + dg_il_dxj - dg_ij_dxl
                        sum_val += g_inv[k, l] * term

                    Gamma[k, i, j] = 0.5 * sum_val

        return Gamma

    def geodesic_equation(self, t, state):
        """
        测地线方程：
        d²x/dt² + Γ¹₁₁ (dx/dt)² + 2Γ¹₁₂ dx/dt dy/dt + Γ¹₂₂ (dy/dt)² = 0
        d²y/dt² + Γ²₁₁ (dx/dt)² + 2Γ²₁₂ dx/dt dy/dt + Γ²₂₂ (dy/dt)² = 0

        state = [x, y, vx, vy]
        """
        x, y, vx, vy = state

        # 计算当前点的克里斯托费尔符号
        Gamma = self.christoffel_symbols(x, y)

        # 加速度 = -Γᵏᵢⱼ vⁱ vʲ
        ax = -(Gamma[0, 0, 0]*vx*vx + 2*Gamma[0, 0, 1]
               * vx*vy + Gamma[0, 1, 1]*vy*vy)
        ay = -(Gamma[1, 0, 0]*vx*vx + 2*Gamma[1, 0, 1]
               * vx*vy + Gamma[1, 1, 1]*vy*vy)

        return [vx, vy, ax, ay]

    def shoot_geodesic(self, initial_state, t_span=(0, 1), t_eval=None):
        """从初始状态发射测地线"""
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)

        sol = solve_ivp(
            self.geodesic_equation,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )

        return sol

    def geodesic_on_perlin_fast(self, A, B, num_points=100):
        """
        快速近似测地线：使用测地线方程的欧拉法
        直接在参数空间优化路径
        """
        A_xy = np.array([A[0], A[1]])
        B_xy = np.array([B[0], B[1]])

        # 方法1：使用离散变分法
        def path_energy(control_points_flat):
            """计算路径的能量"""
            # 控制点（不包括起点和终点）
            control_points = control_points_flat.reshape(-1, 2)

            # 构建完整路径（包括起点和终点）
            t = np.linspace(0, 1, len(control_points) + 2)
            path_xy = np.vstack([
                A_xy,
                control_points,
                B_xy
            ])

            # 使用样条插值获得光滑路径
            from scipy.interpolate import CubicSpline
            cs_x = CubicSpline(t, path_xy[:, 0])
            cs_y = CubicSpline(t, path_xy[:, 1])

            # 离散化计算能量
            eval_t = np.linspace(0, 1, num_points)
            xs = cs_x(eval_t)
            ys = cs_y(eval_t)

            # 计算能量（测地线能量 = 1/2 ∫ g_ij dx^i dx^j dt）
            energy = 0.0
            for i in range(len(eval_t)-1):
                x_mid = (xs[i] + xs[i+1]) / 2
                y_mid = (ys[i] + ys[i+1]) / 2
                dx = xs[i+1] - xs[i]
                dy = ys[i+1] - ys[i]
                dt = eval_t[i+1] - eval_t[i]

                # 计算中点处的度量
                g = self.metric_tensor(x_mid, y_mid)
                ds2 = g[0, 0]*dx*dx + 2*g[0, 1]*dx*dy + g[1, 1]*dy*dy
                energy += 0.5 * ds2 / dt

            return energy

        # 初始猜测：直线路径上的等分点
        num_control = 50  # 控制点数量，可调整
        initial_control = np.zeros((num_control, 2))
        for i in range(num_control):
            t = (i + 1) / (num_control + 1)
            initial_control[i] = A_xy + t * (B_xy - A_xy)

        # 快速优化（使用更少的迭代）
        result = minimize(path_energy, initial_control.flatten(),
                          method='L-BFGS-B',
                          bounds=[(0, 1)] * (2 * num_control),  # 限制在[0,1]区域内
                          options={'maxiter': 50, 'disp': False})

        # 重建路径
        control_points = result.x.reshape(-1, 2)
        t = np.linspace(0, 1, num_points)

        # 使用更高效的线性插值
        path_xy = []
        for ti in t:
            # 分段线性插值
            if ti == 0:
                xy = A_xy
            elif ti == 1:
                xy = B_xy
            else:
                # 找到对应的控制点段
                seg_idx = int(ti * (num_control + 1))
                if seg_idx == 0:
                    # 在第一段：A到第一个控制点
                    t_local = ti * (num_control + 1)
                    xy = A_xy + t_local * (control_points[0] - A_xy)
                elif seg_idx >= num_control:
                    # 在最后一段：最后一个控制点到B
                    t_local = ti * (num_control + 1) - num_control
                    xy = control_points[-1] + t_local * \
                        (B_xy - control_points[-1])
                else:
                    # 在中间段
                    t_local = ti * (num_control + 1) - seg_idx
                    xy = control_points[seg_idx-1] + t_local * \
                        (control_points[seg_idx] - control_points[seg_idx-1])
            path_xy.append(xy)

        path_xy = np.array(path_xy)

        # 计算z坐标
        path = []
        for xy in path_xy:
            z = self.f(xy[0], xy[1])
            path.append([xy[0], xy[1], z])

        return np.array(path)

    def geodesic_on_perlin_dijkstra(self, A, B, grid_size=50):
        """
        使用Dijkstra算法在离散网格上近似测地线
        """
        # 创建离散网格
        x_grid = np.linspace(0, 1, grid_size)
        y_grid = np.linspace(0, 1, grid_size)

        # 找到最近的网格点
        def find_nearest_grid_point(point_xy):
            idx_x = np.argmin(np.abs(x_grid - point_xy[0]))
            idx_y = np.argmin(np.abs(y_grid - point_xy[1]))
            return idx_x, idx_y

        start_idx = find_nearest_grid_point([A[0], A[1]])
        end_idx = find_nearest_grid_point([B[0], B[1]])

        # 计算网格点之间的距离（黎曼距离）
        import heapq

        # 预计算度量张量（加速）
        metric_cache = {}
        for i in range(grid_size):
            for j in range(grid_size):
                x = x_grid[i]
                y = y_grid[j]
                metric_cache[(i, j)] = self.metric_tensor(x, y)

        # Dijkstra算法
        dist = {(i, j): float('inf') for i in range(grid_size)
                for j in range(grid_size)}
        prev = {(i, j): None for i in range(grid_size)
                for j in range(grid_size)}
        dist[start_idx] = 0

        heap = [(0, start_idx)]

        # 8邻域连接
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1),  (1, 0), (1, 1)]

        while heap:
            current_dist, (i, j) = heapq.heappop(heap)

            if (i, j) == end_idx:
                break

            if current_dist > dist[(i, j)]:
                continue

            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    # 计算黎曼距离
                    dx = x_grid[ni] - x_grid[i]
                    dy = y_grid[nj] - y_grid[j]

                    # 使用平均度量
                    g_avg = (metric_cache[(i, j)] + metric_cache[(ni, nj)]) / 2
                    edge_length = np.sqrt(
                        g_avg[0, 0]*dx*dx + 2*g_avg[0, 1] *
                        dx*dy + g_avg[1, 1]*dy*dy
                    )

                    new_dist = current_dist + edge_length

                    if new_dist < dist[(ni, nj)]:
                        dist[(ni, nj)] = new_dist
                        prev[(ni, nj)] = (i, j)
                        heapq.heappush(heap, (new_dist, (ni, nj)))

        # 重建路径
        path_indices = []
        current = end_idx
        while current is not None:
            path_indices.append(current)
            current = prev[current]
        path_indices.reverse()

        # 转换为实际坐标
        path = []
        for i, j in path_indices:
            x = x_grid[i]
            y = y_grid[j]
            z = self.f(x, y)
            path.append([x, y, z])

        return np.array(path)

    def geodesic_on_perlin(self, A, B, initial_direction=None, optimize=True):
        """
        计算从A到B的测地线

        A, B: (x, y, z)坐标
        返回：测地线上的点列表
        """
        A_xy = np.array([A[0], A[1]])
        B_xy = np.array([B[0], B[1]])

        if initial_direction is None:
            # 初始猜测：指向B的直线方向
            direction = B_xy - A_xy
            direction = direction / np.linalg.norm(direction)
            initial_speed = np.linalg.norm(B_xy - A_xy)  # 近似总时间设为1
        else:
            direction = initial_direction
            initial_speed = np.linalg.norm(B_xy - A_xy)

        if optimize:
            # 使用优化方法调整初始方向以命中目标
            return self._geodesic_with_optimization(A_xy, B_xy, initial_speed)
        else:
            # 简单发射
            initial_state = [A_xy[0], A_xy[1],
                             direction[0]*initial_speed,
                             direction[1]*initial_speed]

            sol = self.shoot_geodesic(initial_state)

            # 获取路径上的点
            path = []
            for i in range(len(sol.t)):
                x, y = sol.y[0, i], sol.y[1, i]
                z = self.f(x, y)
                path.append([x, y, z])

            return np.array(path)

    def _geodesic_with_optimization(self, A_xy, B_xy, total_time=1.0):
        """优化初始方向以命中目标"""
        def shooting_error(theta_phi):
            """给定初始方向角，计算终点误差"""
            theta, phi = theta_phi
            # 将球坐标转换为方向向量
            v0 = total_time * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi)
            ])

            initial_state = [A_xy[0], A_xy[1], v0[0], v0[1]]
            sol = self.shoot_geodesic(initial_state, t_span=(0, 1))

            # 终点位置
            x_end, y_end = sol.y[0, -1], sol.y[1, -1]

            # 误差：到目标的距离
            error = np.linalg.norm([x_end - B_xy[0], y_end - B_xy[1]])
            return error

        # 初始猜测：直接指向B
        target_vec = B_xy - A_xy
        theta0 = np.pi/2  # 假设在平面上
        phi0 = np.arctan2(target_vec[1], target_vec[0])

        # 优化初始方向
        result = minimize(shooting_error, [theta0, phi0],
                          method='L-BFGS-B',
                          bounds=[(0.1, np.pi-0.1), (0, 2*np.pi)])

        theta_opt, phi_opt = result.x
        v0_opt = total_time * np.array([
            np.sin(theta_opt) * np.cos(phi_opt),
            np.sin(theta_opt) * np.sin(phi_opt)
        ])

        # 计算最优测地线
        initial_state = [A_xy[0], A_xy[1], v0_opt[0], v0_opt[1]]
        sol = self.shoot_geodesic(initial_state)

        # 获取路径
        path = []
        for i in range(len(sol.t)):
            x, y = sol.y[0, i], sol.y[1, i]
            z = self.f(x, y)
            path.append([x, y, z])

        return np.array(path)

    def path_length(self, path):
        """计算路径在曲面上的长度"""
        length = 0.0
        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]

            # 计算黎曼距离
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            x_mid = (p1[0] + p2[0]) / 2
            y_mid = (p1[1] + p2[1]) / 2

            # 计算中点处的度量
            g = self.metric_tensor(x_mid, y_mid)

            # ds² = g_ij dxⁱ dxʲ
            ds2 = g[0, 0]*dx*dx + 2*g[0, 1]*dx*dy + g[1, 1]*dy*dy
            length += np.sqrt(max(ds2, 0))  # 避免数值误差

        return length


# 示例：使用一个简单的Perlin噪声实现


class SimplePerlinNoise:
    """一个简化的Perlin噪声实现用于演示"""

    def __init__(self, seed=42):
        self.k = np.random.random()*10
        np.random.seed(seed)
        self.freq = 3.0
        self.amp = 0.3

    def noise(self, x, y):
        """简单的噪声函数"""
        return self.amp * (np.sin(self.freq * x) * np.cos(self.freq * y) +
                           0.5 * np.sin(2.5 * self.freq * x) * np.cos(1.7 * self.freq * y))

    def gradient(self, x, y):
        """数值梯度"""
        eps = 1e-6
        f_x = (self.noise(x + eps, y) - self.noise(x - eps, y)) / (2 * eps)
        f_y = (self.noise(x, y + eps) - self.noise(x, y - eps)) / (2 * eps)
        return f_x, f_y

    def __call__(self, x, y):
        return self.noise(x, y)


class SimplePerlinNoise:
    """一个改进的Perlin噪声实现，包含真正的随机性"""

    def __init__(self, seed=42):
        np.random.seed(seed)

        # 随机参数生成
        self.num_frequencies = np.random.randint(2, 5)  # 2-4个频率成分
        self.freqs = np.random.uniform(2.0, 8.0, self.num_frequencies)  # 随机频率
        self.amps = np.random.uniform(0.1, 0.5, self.num_frequencies)   # 随机振幅
        self.phases_x = np.random.uniform(
            0, 2*np.pi, self.num_frequencies)  # 随机相位
        self.phases_y = np.random.uniform(0, 2*np.pi, self.num_frequencies)

        # 添加一些随机缩放和偏移
        self.global_scale = np.random.uniform(0.8, 1.2)
        self.offset_x = np.random.uniform(-1.0, 1.0)
        self.offset_y = np.random.uniform(-1.0, 1.0)

        print(
            f"Generated noise with {self.num_frequencies} frequency components")
        print(f"Frequencies: {self.freqs}")
        print(f"Amplitudes: {self.amps}")

    def noise(self, x, y):
        """基于随机参数的噪声函数"""
        # 应用全局缩放和偏移
        x_scaled = x * self.global_scale + self.offset_x
        y_scaled = y * self.global_scale + self.offset_y

        value = 0.0
        for i in range(self.num_frequencies):
            # 每个频率成分有自己的相位和振幅
            value += self.amps[i] * (
                np.sin(self.freqs[i] * x_scaled + self.phases_x[i]) *
                np.cos(self.freqs[i] * y_scaled + self.phases_y[i])
            )

        return value

    def gradient(self, x, y):
        """数值梯度（中心差分）"""
        eps = 1e-6
        f_x = (self.noise(x + eps, y) - self.noise(x - eps, y)) / (2 * eps)
        f_y = (self.noise(x, y + eps) - self.noise(x, y - eps)) / (2 * eps)
        return f_x, f_y

    def __call__(self, x, y):
        return self.noise(x, y)


def visualize_comparison(A, B, geodesic_path, gd_path=None):
    """可视化结果"""
    fig = plt.figure(figsize=(15, 5))

    # 1. 3D视图
    ax1 = fig.add_subplot(131, projection='3d')

    # 创建曲面网格
    x_grid = np.linspace(0, 1, 50)
    y_grid = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = perlin.noise(X, Y)

    ax1.plot_surface(X, Y, Z, alpha=0.3, cmap='terrain')

    # 绘制测地线
    ax1.plot(geodesic_path[:, 0], geodesic_path[:, 1], geodesic_path[:, 2],
             'r-', linewidth=3, label='Geodesic')

    if gd_path is not None:
        ax1.plot(gd_path[:, 0], gd_path[:, 1], gd_path[:, 2],
                 'b--', linewidth=2, label='Gradient Descent')

    # 起点和终点
    ax1.scatter([A[0]], [A[1]], [A[2]], c='g',
                s=100, marker='o', label='Start')
    ax1.scatter([B[0]], [B[1]], [B[2]], c='r', s=100, marker='^', label='End')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View')
    ax1.legend()

    # 2. 俯视图
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(X, Y, Z, 20, cmap='terrain', alpha=0.7)
    ax2.plot(geodesic_path[:, 0], geodesic_path[:, 1], 'r-', linewidth=3)
    if gd_path is not None:
        ax2.plot(gd_path[:, 0], gd_path[:, 1], 'b--', linewidth=2)
    ax2.scatter(A[0], A[1], c='g', s=100)
    ax2.scatter(B[0], B[1], c='r', s=100)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (XY Projection)')
    ax2.axis('equal')

    # 3. 侧视图（高程）
    ax3 = fig.add_subplot(133)

    # 沿路径的参数化
    t = np.linspace(0, 1, len(geodesic_path))
    ax3.plot(t, geodesic_path[:, 2], 'r-', linewidth=3, label='Geodesic Z')

    # 计算欧氏直线在曲面上的投影
    straight_line = []
    for ti in t:
        x = A[0] + ti * (B[0] - A[0])
        y = A[1] + ti * (B[1] - A[1])
        z = perlin.noise(x, y)
        straight_line.append([x, y, z])
    straight_line = np.array(straight_line)

    ax3.plot(t, straight_line[:, 2], 'k--', linewidth=2, label='Straight Line')

    if gd_path is not None:
        t_gd = np.linspace(0, 1, len(gd_path))
        ax3.plot(t_gd, gd_path[:, 2], 'b--',
                 linewidth=2, label='Gradient Descent')

    ax3.set_xlabel('Normalized Path Parameter')
    ax3.set_ylabel('Elevation (Z)')
    ax3.set_title('Elevation Profile')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


# 对比函数：梯度下降路径


def gradient_descent_path(A, B, perlin_func, grad_func, steps=2000, lr=0.01):
    """梯度下降在曲面上的路径"""
    current = np.array([A[0], A[1], A[2]])
    path = [current.copy()]

    for i in tqdm(range(steps)):
        # 当前点的梯度
        f_x, f_y = grad_func(current[0], current[1])

        # 到B的方向（在XY平面）
        dir_to_target = np.array([B[0] - current[0], B[1] - current[1]])

        # 投影到切平面
        normal = np.array([-f_x, -f_y, 1.0])
        normal = normal / np.linalg.norm(normal)

        # 3D方向
        dir_3d = np.array([dir_to_target[0], dir_to_target[1], 0])

        # 投影到切平面
        tangent = dir_3d - np.dot(dir_3d, normal) * normal
        if np.linalg.norm(tangent) > 1e-10:
            tangent = tangent / np.linalg.norm(tangent)

        # 更新位置
        print(i, dir_3d, normal, tangent)
        new_xy = current[:2] + lr * tangent[:2]
        new_z = perlin_func(new_xy[0], new_xy[1])
        current = np.array([new_xy[0], new_xy[1], new_z])
        path.append(current.copy())
        if (np.linalg.norm(current - B) < 0.01):
            break

    return np.array(path)


# 主程序
if __name__ == "__main__":
    # 创建Perlin噪声函数
    # perlin = SimplePerlinNoise(seed=42)
    perlin = SimplePerlinNoise(seed=np.random.randint(20, 2000))

    # 定义起点和终点
    A = np.array([0.2, 0.2, perlin.noise(0.2, 0.2)])
    B = np.array([0.8, 0.8, perlin.noise(0.8, 0.8)])

    print(f"Start A: {A}")
    print(f"End B: {B}")

    # 创建测地线求解器
    geodesic_solver = PerlinSurfaceGeodesic(
        perlin_func=perlin.noise,
        grad_func=perlin.gradient
    )

    # 计算测地线
    print("\nComputing geodesic...")
    # geodesic_path = geodesic_solver.geodesic_on_perlin(A, B, optimize=True)
    geodesic_path = geodesic_solver.geodesic_on_perlin_fast(A, B, 100)
    # geodesic_path = geodesic_solver.geodesic_on_perlin_dijkstra(
    #     A, B, grid_size=100)

    # 计算梯度下降路径
    print("Computing gradient descent path...")
    gd_path = gradient_descent_path(
        A, B, perlin.noise, perlin.gradient, lr=0.005)

    # 计算路径长度
    geodesic_length = geodesic_solver.path_length(geodesic_path)
    gd_length = geodesic_solver.path_length(gd_path)
    straight_length = geodesic_solver.path_length(
        np.array([A, B])  # 仅计算端点
    )

    print(f"\nPath Lengths:")
    print(f"  Straight line (approx): {straight_length:.4f}")
    print(f"  Gradient Descent: {gd_length:.4f}")
    print(f"  Geodesic: {geodesic_length:.4f}")
    print(
        f"  Improvement: {(gd_length - geodesic_length)/gd_length*100:.1f}% shorter")

    # 可视化
    visualize_comparison(A, B, geodesic_path, gd_path)

    # 输出测地线的一些统计信息
    print(f"\nGeodesic Statistics:")
    print(f"  Number of points: {len(geodesic_path)}")
    print(
        f"  Mean curvature along path: {np.mean(np.abs(geodesic_path[:, 2])):.4f}")
    print(f"  Max elevation: {np.max(geodesic_path[:, 2]):.4f}")
    print(f"  Min elevation: {np.min(geodesic_path[:, 2]):.4f}")


# 快速使用
# perlin = SimplePerlinNoise()
# solver = PerlinSurfaceGeodesic(perlin.noise, perlin.gradient)

# A = [0.1, 0.1, perlin.noise(0.1, 0.1)]
# B = [0.9, 0.9, perlin.noise(0.9, 0.9)]

# path = solver.geodesic_on_perlin(A, B)
# length = solver.path_length(path)
# print(f"Geodesic found with length: {length}")

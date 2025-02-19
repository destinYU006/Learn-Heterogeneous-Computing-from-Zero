import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

def simple_cavity_flow(N=50, Re=100, max_iter=1000, tol=1e-5):
    """
    基于SIMPLE算法的二维方腔流求解器
    
    参数:
    N     : 网格数 (N x N)
    Re    : 雷诺数
    max_iter: 最大迭代次数
    tol   : 收敛容差
    
    返回:
    u, v, p: 速度场和压力场
    """
    # 几何参数
    L = 1.0           # 计算域尺寸
    U_wall = 1.0      # 顶盖速度
    
    # 网格参数
    dx = L / N
    dy = L / N
    x = np.linspace(0, L, N+1)
    y = np.linspace(0, L, N+1)
    
    # 初始化场变量 (使用交错网格)
    u = np.zeros((N+1, N+2))  # x速度 (存储于垂直面中心)
    print(u.shape)
    v = np.zeros((N+2, N+1))  # y速度 (存储于水平面中心)
    print(v.shape)
    p = np.zeros((N+2, N+2))  # 压力 (存储于单元中心)
    print(p.shape)
    
    # 设置顶盖速度
    u[:, -1] = U_wall
    
    # 松弛因子
    alpha_p = 0.7    # 压力松弛
    alpha_uv = 0.8   # 速度松弛

    # 构建泊松方程系数矩阵
    def build_poisson_matrix(N, dx, dy):
        """构建考虑Dirichlet边界条件的泊松方程矩阵"""
        N_inner = (N-2)**2  # 内部节点数
        main_diag = 2*(1/dx**2 + 1/dy**2) * np.ones(N_inner)
        
        # 相邻节点系数
        x_coeff = -1/dx**2
        y_coeff = -1/dy**2
        
        diagonals = []
        offsets = []
        
        # 南向连接 (垂直方向)
        south = y_coeff * np.ones(N_inner - (N-2))
        diagonals.append(south)
        offsets.append(-(N-2))
        
        # 西向连接 (水平方向)
        west = x_coeff * np.ones(N_inner - 1)
        diagonals.append(west)
        offsets.append(-1)
        
        # 主对角线
        diagonals.append(main_diag)
        offsets.append(0)
        
        # 东向连接
        east = x_coeff * np.ones(N_inner - 1)
        diagonals.append(east)
        offsets.append(1)
        
        # 北向连接
        north = y_coeff * np.ones(N_inner - (N-2))
        diagonals.append(north)
        offsets.append(N-2)
        
        return diags(diagonals, offsets, shape=(N_inner, N_inner), format='csr')



    # 迭代求解
    for iter in range(max_iter):
        # --- 步骤1: 求解动量方程 ---
        # x方向动量方程
        u_old = u.copy()
        for i in range(1, N):
            for j in range(1, N+1):
                # 对流项 (迎风格式)
                ue = 0.5*(u[i+1,j] + u[i,j])
                uw = 0.5*(u[i,j] + u[i-1,j])
                vn = 0.5*(v[i,j] + v[i+1,j])
                vs = 0.5*(v[i,j-1] + v[i+1,j-1])
                
                F_e = ue * dy
                F_w = uw * dy
                F_n = vn * dx
                F_s = vs * dx
                
                # 扩散项
                De = (1/Re) * dy/dx
                Dw = (1/Re) * dy/dx
                Dn = (1/Re) * dx/dy
                Ds = (1/Re) * dx/dy
                
                # 系数计算
                a_E = De + max(-F_e, 0)
                a_W = Dw + max(F_w, 0)
                a_N = Dn + max(-F_n, 0)
                a_S = Ds + max(F_s, 0)
                a_P = a_E + a_W + a_N + a_S + F_e - F_w + F_n - F_s
                
                # 压力梯度项
                dP = (p[i+1,j] - p[i,j]) * dy
                
                # 更新方程
                u[i,j] = (a_E*u[i+1,j] + a_W*u[i-1,j] + a_N*u[i,j+1] + a_S*u[i,j-1] - dP)/a_P
                u[i,j] = alpha_uv*u[i,j] + (1-alpha_uv)*u_old[i,j]
        
        # y方向动量方程
        v_old = v.copy()
        for i in range(1, N+1):
            for j in range(1, N):
                # 对流项 (迎风格式)
                ue = 0.5*(u[i,j+1] + u[i,j])
                uw = 0.5*(u[i-1,j+1] + u[i-1,j])
                vn = 0.5*(v[i,j+1] + v[i,j])
                vs = 0.5*(v[i,j] + v[i,j-1])
                
                F_e = ue * dy
                F_w = uw * dy
                F_n = vn * dx
                F_s = vs * dx
                
                # 扩散项
                De = (1/Re) * dy/dx
                Dw = (1/Re) * dy/dx
                Dn = (1/Re) * dx/dy
                Ds = (1/Re) * dx/dy
                
                # 系数计算
                a_E = De + max(-F_e, 0)
                a_W = Dw + max(F_w, 0)
                a_N = Dn + max(-F_n, 0)
                a_S = Ds + max(F_s, 0)
                a_P = a_E + a_W + a_N + a_S + F_e - F_w + F_n - F_s
                
                # 压力梯度项
                dP = (p[i,j+1] - p[i,j]) * dx
                
                # 更新方程
                v[i,j] = (a_E*v[i+1,j] + a_W*v[i-1,j] + a_N*v[i,j+1] + a_S*v[i,j-1] - dP)/a_P
                v[i,j] = alpha_uv*v[i,j] + (1-alpha_uv)*v_old[i,j]
        
        # --- 步骤2: 求解压力修正方程 ---
        # 构建泊松方程系数矩阵
        A = build_poisson_matrix(N, dx, dy)
        
        # 修改后的源项计算
        b = np.zeros((N-2, N-2))  # 仅内部节点
        for i in range(1, N-1):   # 内部节点范围
            for j in range(1, N-1):
                # 速度散度计算
                du_dx = (u[i+1,j] - u[i,j])/dx 
                dv_dy = (v[i,j+1] - v[i,j])/dy
                b[i-1,j-1] = du_dx + dv_dy

        b = b.ravel() * dx*dy
        b = b.reshape(N-2,N-2)
        # 添加边界条件到矩阵系统
        A = build_poisson_matrix(N, dx, dy)

        # 应用Dirichlet边界条件（压力修正量为0）
        b[0, :] = 0    # 底部边界
        b[-1, :] = 0   # 顶部边界
        b[:, 0] = 0    # 左侧边界
        b[:, -1] = 0   # 右侧边界

        b = b.ravel()
        
        # 求解压力修正
        p_prime_inner = spsolve(A, -b).reshape((N-2, N-2))

        # 初始化压力修正场
        p_prime = np.zeros((N, N))
        p_prime[1:-1, 1:-1] = p_prime_inner  # 内部节点赋值

        # 边界保持零值（Dirichlet条件）
        # --- 步骤3: 修正压力和速度 ---
        # 压力修正
        p[1:-1, 1:-1] += alpha_p * p_prime[1:-1, 1:-1]
        
        # 速度修正
        u[1:-1, 1:-1] -= (p_prime[2:,1:-1] - p_prime[1:-1,1:-1])/dx * dy
        v[1:-1, 1:-1] -= (p_prime[1:-1,2:] - p_prime[1:-1,1:-1])/dy * dx
        
        # --- 收敛检查 ---
        div = np.max(np.abs(b)/(dx*dy))
        if iter % 100 == 0:
            print(f"Iter: {iter:4d}, Divergence: {div:.2e}")
        if div < tol:
            print(f"收敛于 {iter} 次迭代")
            break
    
    return u, v, p

def plot_results(u, v, N=50):
    """ 可视化结果 """
    # 创建交错网格坐标
    x = np.linspace(0, 1, N+1)
    y = np.linspace(0, 1, N+1)
    X, Y = np.meshgrid(x, y)
    
    # 速度幅值
    u_center = 0.5*(u[:-1,1:-1] + u[1:,1:-1])
    v_center = 0.5*(v[1:-1,:-1] + v[1:-1,1:])
    speed = np.sqrt(u_center**2 + v_center**2)
    
    plt.figure(figsize=(12,5))
    
    # 流线图
    plt.subplot(121)
    plt.streamplot(X, Y, u_center.T, v_center.T, color=speed.T, density=2)
    plt.title('Streamlines')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Velocity Magnitude')
    
    # 中心速度剖面 (Ghia基准对比)
    plt.subplot(122)
    y_ghia = np.array([0.0,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,0.4531,0.5,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766,1.0])
    u_ghia = np.array([0.0,-0.03717,-0.04192,-0.04775,-0.06434,-0.10150,-0.15662,-0.21090,-0.20581,-0.13641,0.00332,0.23151,0.68717,0.73722,0.78871,0.84123,1.0])
    
    plt.plot(u_center[N//2,:], y[1:-1], 'r-', label='Current')
    plt.plot(u_ghia, y_ghia, 'ko', label='Ghia et al. (1982)')
    plt.title('Centerline Velocity (x=0.5)')
    plt.xlabel('u Velocity')
    plt.ylabel('y')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 运行求解器
u, v, p = simple_cavity_flow(N=50, Re=100)
plot_results(u, v)
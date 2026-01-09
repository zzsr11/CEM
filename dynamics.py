import numpy as np
from config import *
from utils import wrap_angle, great_circle_distance

class HFVDynamics:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.reset()

        # 惯性参数
        self.mass = M_REF
        self.S_ref = S_REF
        self.Ixx = 5000.0
        self.Iyy = 5000.0
        self.Izz = 8000.0

        # 舵效限制（弧度）
        self.de_max = np.deg2rad(20)  # 升降舵
        self.da_max = np.deg2rad(15)  # 副翼
        self.dr_max = np.deg2rad(20)  # 方向舵

    def reset(self, state=None):
        if state is not None:
            self.state = np.copy(state).astype(np.float32)
        else:
            self.state = np.array([
                R_EARTH + INIT_ALT,   # R
                0.0,                  # λ (lon)
                0.0,                  # φ (lat)
                INIT_V,               # V
                INIT_GAMMA,           # γ (flight path angle)
                INIT_CHI,             # χ (heading angle)
                0.0,                  # σ (bank angle)
                0.0, 0.0, 0.0         # p, q, r (角速度)
            ], dtype=np.float32)
        return self.state

    def get_altitude(self):
        return self.state[0] - R_EARTH

    def get_range(self):
        # 从经纬度计算地表距离（单位：米）
        lat = self.state[2]  # φ
        lon = self.state[1]  # λ

        # 球面余弦公式（更稳定用 haversine，但这里简化）
        central_angle = np.arccos(
            np.clip(
                np.sin(lat) ** 2 + np.cos(lat) ** 2 * np.cos(lon),
                -1.0, 1.0
            )
        )
        return R_EARTH * central_angle

    def _compute_aerodynamics(self, alpha, beta, de, da, dr, q):
        """计算气动力和力矩"""
        # 升力、阻力、侧力系数（简化线性模型）
        CL = 0.5 + 2.0 * alpha + 0.3 * de
        CD = 0.05 + 1.5 * alpha**2 + 0.1 * de**2
        CY = 0.5 * beta + 0.2 * dr

        # 力矩系数
        Cl = 0.1 * beta + 0.2 * da      # 滚转
        Cm = -0.5 * alpha + 0.6 * de    # 俯仰
        Cn = -0.1 * beta + 0.1 * dr     # 偏航

        L = q * self.S_ref * CL
        D = q * self.S_ref * CD
        Y = q * self.S_ref * CY

        Mx = q * self.S_ref * 1.0 * Cl
        My = q * self.S_ref * 1.0 * Cm
        Mz = q * self.S_ref * 1.0 * Cn

        return L, D, Y, Mx, My, Mz

    def _ode_rhs(self, state, action):
        R, lam, phi, V, gamma, chi, sigma, p, q, r = state

        # 映射动作到舵偏
        de = np.clip(action[0], -1, 1) * self.de_max
        da = np.clip(action[1], -1, 1) * self.da_max
        dr = np.clip(action[2], -1, 1) * self.dr_max

        # 大气密度
        h = max(R - R_EARTH, 0.0)
        rho = RHO_0 * np.exp(-h / H_SCALE)
        rho = max(rho, 1e-8)
        q_dyn = 0.5 * rho * V**2

        # === 关键：计算当前攻角 α 和侧滑角 β ===
        # 假设：机体滚转角 = sigma，且速度矢量与机体纵轴夹角由配平决定
        # 简化：α 由升力平衡重力分量决定（准平衡假设）
        g = G * (R_EARTH / R)**2
        L_req = self.mass * g * np.cos(gamma)  # 所需升力
        CL_req = L_req / (q_dyn * self.S_ref)
        alpha = np.clip((CL_req - 0.5) / 2.0, np.deg2rad(-10), np.deg2rad(10))  # 反推 α

        # 侧滑角 β 近似为 0（对称飞行），或由偏航力矩平衡估算
        beta = 0.0  # 简化处理；也可设为小值如 0.01

        # 计算气动力
        L, D, Y, Mx, My, Mz = self._compute_aerodynamics(alpha, beta, de, da, dr, q_dyn)

        # 运动方程（球坐标系）
        g_local = G * (R_EARTH / R)**2

        dR = V * np.sin(gamma)
        dlam = (V * np.cos(gamma) * np.sin(chi)) / (R * np.cos(phi) + 1e-8)
        dphi = (V * np.cos(gamma) * np.cos(chi)) / R
        dV = -D / self.mass - g_local * np.sin(gamma)
        dgamma = (L * np.cos(sigma) - Y * np.sin(sigma)) / (self.mass * V) - \
                 (g_local / V) * np.cos(gamma) + (V / R) * np.cos(gamma)
        dchi = (L * np.sin(sigma) + Y * np.cos(sigma)) / (self.mass * V * np.cos(gamma) + 1e-8) + \
               (V / R) * np.cos(gamma) * np.sin(chi) * np.tan(phi)

        # 角速度动力学
        dp = (Mx - (self.Iyy - self.Izz) * q * r) / self.Ixx
        dq = (My - (self.Izz - self.Ixx) * r * p) / self.Iyy
        dr = (Mz - (self.Ixx - self.Iyy) * p * q) / self.Izz

        # 滚转角变化率（σ_dot = p + ...，简化为 σ_dot ≈ p）
        dsigma = p  # 更精确模型可用欧拉角微分，此处简化

        return np.array([dR, dlam, dphi, dV, dgamma, dchi, dsigma, dp, dq, dr])

    def step(self, action):
        # RK4 积分
        k1 = self._ode_rhs(self.state, action)
        k2 = self._ode_rhs(self.state + 0.5 * self.dt * k1, action)
        k3 = self._ode_rhs(self.state + 0.5 * self.dt * k2, action)
        k4 = self._ode_rhs(self.state + self.dt * k3, action)
        new_state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 归一化角度
        new_state[1] = wrap_angle(new_state[1])  # λ
        new_state[2] = np.clip(new_state[2], -np.pi/2, np.pi/2)  # φ
        new_state[4] = np.clip(new_state[4], -np.pi/2, np.pi/2)  # γ
        new_state[5] = wrap_angle(new_state[5])  # χ
        new_state[6] = wrap_angle(new_state[6])  # σ

        # 数值保护
        if not np.isfinite(new_state).all():
            new_state = np.clip(new_state, -1e5, 1e5)

        self.state = new_state
        return self.state.copy()

    def is_terminal(self, target_lat, target_lon):
        alt = self.get_altitude()
        if alt < 0:
            return True  # 撞地

        # 计算到目标的大圆距离
        s_arc = great_circle_distance(self.state[2], self.state[1], target_lat, target_lon)
        if s_arc > 300000:  # 超出 200 km 范围
            return True


        return False

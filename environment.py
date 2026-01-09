import gymnasium as gym
import numpy as np
from config import *
from dynamics import HFVDynamics
from utils import geodetic_to_ecef, great_circle_distance


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
class HFVEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dyn = HFVDynamics(dt=0.1)
        self.target_lon = TARGET_LON
        self.target_lat = TARGET_LAT
        self.target_alt = TARGET_ALT

        self.target_ecef = geodetic_to_ecef(self.target_lat, self.target_lon, self.target_alt)

        # 动作空间: [de, da, dr] ∈ [-1, 1]^3
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # 观测空间: [s_arc, V, gamma, chi, sigma, p, q, r]
        low_obs = np.array([0, 100, -np.pi/2, -np.pi, -np.pi, -10, -10, -10, -np.pi, -np.pi, -5.0, -5.0], dtype=np.float32)
        high_obs = np.array([2e5, 3000, np.pi/2, np.pi, np.pi, 10, 10, 10, np.pi, np.pi, 5.0, 5.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.max_steps = 2000
        self.step_count = 0

        self.prev_heading_error = 0.0
        self.prev_elevation_error = 0.0

        # 在 __init__ 末尾添加
        self._cached_s_arc = 0.0
        self._cached_alt = 0.0
        self._cached_heading_error = 0.0
        self._cached_elevation_error = 0.0
        self._cached_heading_rate = 0.0
        self._cached_elevation_rate = 0.0



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 随机初始状态（±10%）
        alt = INIT_ALT * np.random.uniform(0.9, 1.1)
        V = INIT_V * np.random.uniform(0.95, 1.05)
        gamma = INIT_GAMMA + np.random.uniform(-0.05, 0.05)
        chi = INIT_CHI + np.random.uniform(-0.1, 0.1)

        self.dyn.reset(state=[
            R_EARTH + alt, 0.0, 0.0, V, gamma, chi, 0.0, 0.0, 0.0, 0.0
        ])
        self.step_count = 0

        # 👇 新增：重置误差历史
        self.prev_heading_error = 0.0
        self.prev_elevation_error = 0.0

        # 👇 首次调用，填充缓存（用于首次 _get_obs
        self._update_los_errors()

        self.min_s_3d = float('inf')
        self.prev_s_3d = float('inf')

        return self._get_obs(), {}

    def _update_los_errors(self):
        """计算并更新 LOS 相关误差和角速率"""
        R, lam, phi, V, gamma, chi, sigma, p, q, r = self.dyn.state
        s_arc = great_circle_distance(phi, lam, self.target_lat, self.target_lon)
        alt = self.dyn.get_altitude()

        # 水平航向误差
        delta_lon = self.target_lon - lam
        y = np.sin(delta_lon) * np.cos(self.target_lat)
        x = np.cos(phi) * np.sin(self.target_lat) - np.sin(phi) * np.cos(self.target_lat) * np.cos(delta_lon)
        bearing_to_target = np.arctan2(y, x)
        heading_error = wrap_angle(bearing_to_target - chi)

        # 垂直指向误差
        los_elevation = np.arctan2(-alt, s_arc + 1e-8)
        elevation_error = los_elevation - gamma

        # 计算角速率（差分）
        dt = self.dyn.dt
        heading_rate = (heading_error - self.prev_heading_error) / (dt + 1e-8)
        elevation_rate = (elevation_error - self.prev_elevation_error) / (dt + 1e-8)

        # 限幅
        heading_rate = np.clip(heading_rate, -5.0, 5.0)
        elevation_rate = np.clip(elevation_rate, -5.0, 5.0)

        # 更新历史（供下一步使用）
        self.prev_heading_error = heading_error
        self.prev_elevation_error = elevation_error

        # 缓存到 self，供 reward 和 obs 使用
        self._cached_s_arc = s_arc
        self._cached_alt = alt
        self._cached_heading_error = heading_error
        self._cached_elevation_error = elevation_error
        self._cached_heading_rate = heading_rate
        self._cached_elevation_rate = elevation_rate

    def _get_3d_distance(self):
        """计算导弹与目标之间的 3D 欧氏距离（米）"""
        _, lon, lat, _, _, _, _, _, _, _ = self.dyn.state
        alt = self.dyn.get_altitude()  # 必须返回海拔高度（米）

        missile_ecef = geodetic_to_ecef(lat, lon, alt)
        diff = missile_ecef - self.target_ecef
        return np.linalg.norm(diff)

    def _get_obs(self):
        R, lam, phi, V, gamma, chi, sigma, p, q, r = self.dyn.state

        # 直接使用已计算的缓存值
        s_arc = self._cached_s_arc
        heading_error = self._cached_heading_error
        elevation_error = self._cached_elevation_error
        heading_rate = self._cached_heading_rate
        elevation_rate = self._cached_elevation_rate

        #s_arc = great_circle_distance(phi, lam, self.target_lat, self.target_lon)
        #alt = self.dyn.get_altitude()
        # 水平航向误差
        #delta_lon = self.target_lon - lam
        #y = np.sin(delta_lon) * np.cos(self.target_lat)
        #x = np.cos(phi) * np.sin(self.target_lat) - np.sin(phi) * np.cos(self.target_lat) * np.cos(delta_lon)
        #bearing_to_target = np.arctan2(y, x)
        #heading_error = wrap_angle(bearing_to_target - chi)
        # 垂直指向误差：LOS 俯仰角 vs 飞行路径角
        #los_elevation = np.arctan2(-alt, s_arc + 1e-8)  # 目标在下方
        #elevation_error = los_elevation - gamma
        # ===== 新增：LOS 角速率（通过差分估计） =====
        #dt = self.dyn.dt  # 0.1 秒
        #heading_rate = (heading_error - self.prev_heading_error) / (dt + 1e-8)
        #elevation_rate = (elevation_error - self.prev_elevation_error) / (dt + 1e-8)
        # 更新历史（供下一步使用）
        #self.prev_heading_error = heading_error
        #self.prev_elevation_error = elevation_error
        # 归一化角速率（可选，但推荐）
        #heading_rate = np.clip(heading_rate, -5.0, 5.0)  # rad/s
        #elevation_rate = np.clip(elevation_rate, -5.0, 5.0)  # rad/s

        return np.array([
            s_arc, V, gamma, chi, sigma, p, q, r,
            heading_error,
            elevation_error,  # 新增！关键制导信号
            heading_rate,  # 新增！
            elevation_rate # 新增！
        ], dtype=np.float32)


    def _compute_reward(self, s_arc, prev_s_arc, done, success, heading_rate=0.0, elevation_rate=0.0, s_3d=None):
        V = self.dyn.state[3]
        gamma = self.dyn.state[4]
        alt = self.dyn.get_altitude()
        if s_3d is None:
            s_3d = self._get_3d_distance()
        if success:
            # 成功命中：基础 + 速度 + 俯冲角奖励
            speed_bonus = max(0, (V - 1000) * 0.01)  # 越快越好
            dive_bonus = max(0, -gamma - np.radians(30)) * 50  # 俯冲角 >30° 才奖励
            return 10000.0 + speed_bonus + dive_bonus
        elif done:
            # 失败：根据最终距离和是否低空接近给部分奖励
            distance_penalty = - (s_3d / 1000)
            if s_3d < self.min_s_3d + 100:
                return distance_penalty + 1000
            else:
                return distance_penalty - 10000
        else:
            reward = 0.0
            # 示例：在最后 N 步，若高度快速下降且 s 小，则加分
            if s_arc < 8000 and alt < 4000:
                reward += 10 * (1000 - alt) / 1000  # 鼓励降低高度
                reward += 5000 / (s_arc + 100)  # 强烈鼓励靠近
            if s_3d < self.min_s_3d + 10 :
            # 1. 鼓励靠近目标（增量）
                progress = (prev_s_arc - s_arc) * 0.2
                reward += progress
                if s_3d < 500:
                    reward += (1000 - s_3d * 1)
                elif s_3d < 3000:
                    reward += (550 - s_3d * 0.1)
                elif s_3d < 10000:
                    reward += (325 - s_3d * 0.025)
                #if s_3d < 5000:
                #    # 原来是线性衰减，现在用平方反比或指数
                #    reward += 1000.0 * np.exp(-s_3d / 500)  # 500m 内爆炸式奖励
            # 2. 鼓励大俯冲角（全程有效）
            if gamma < 0:
                reward += (-gamma) * 20.0  # -45° → +6.3
            else:
                reward -= 3.0  # 惩罚向上飞
            # === 3. 关键：惩罚“曾接近但正在飞离” ===
            if hasattr(self, 'min_s_3d') and self.min_s_3d < 8000:  # 曾进入 8km
                if s_3d > self.min_s_3d + 10:  # 现在比最近点远
                    # 越是飞远，惩罚越大
                    flyaway_penalty = - (s_3d - self.min_s_3d) * 0.1
                    reward += flyaway_penalty  # 负值
            # 3. 鼓励高速（>1500 m/s）
            if V < 1500:
                reward -= (1500 - V) * 0.01
            # 4. 小幅时间惩罚
            #reward -= 0.2
            # 新增：惩罚 LOS 抖动（稳定瞄准）
            reward -= 0.5 * (abs(heading_rate) + abs(elevation_rate))
            return reward

    #def _compute_reward(self, s_arc, prev_s_arc, done, success, heading_rate=0.0, elevation_rate=0.0, s_3d=None):
    #    if s_3d is None:
    #        s_3d = self._get_3d_distance()
    #    if success:
    #        V = self.dyn.state[3]
    #        gamma = self.dyn.state[4]
    #        speed_bonus = max(0, (V - 1000) * 0.01)
    #        dive_bonus = max(0, (-gamma - np.radians(30))) * 50
    #        return 1000.0 + speed_bonus + dive_bonus  # 👈 降低到 1000，避免量级失衡
    #    elif done:
            # 失败：按最终距离给负奖励
    #        return -s_3d / 100.0  # 100m → -1, 10km → -100
    #    else:
    #        reward = 0.0
            # 1. 鼓励靠近目标（基于 3D 距离减少）
    #        progress_3d = (self.prev_s_3d - s_3d) * 0.1  # 新增 self.prev_s_3d
    #        reward += progress_3d
            # 2. 距离越近，奖励越高（平滑）
    #        if s_3d < 5000:
    #            reward += 500.0 * np.exp(-s_3d / 500.0)
            # 3. 鼓励俯冲（但不过度惩罚）
    #        gamma = self.dyn.state[4]
    #        if gamma < 0:
    #            reward += (-gamma) * 10.0
    #        else:
    #            reward -= 0.5  # 轻微惩罚
            # 4. 鼓励高速
    #        V = self.dyn.state[3]
    #        if V > 1500:
    #            reward += (V - 1500) * 0.005
            # 5. 微弱惩罚抖动（仅在接近时）
    #        if s_3d < 10000:
    #            reward -= 0.05 * (abs(heading_rate) + abs(elevation_rate))
            # 更新 prev_s_3d
    #        self.prev_s_3d = s_3d
    #        return reward

    def step(self, action):
        #prev_s_arc = great_circle_distance(
        #    self.dyn.state[2], self.dyn.state[1],
        #    self.target_lat, self.target_lon
        #)

        prev_s_arc = self._cached_s_arc if hasattr(self, '_cached_s_arc') else \
            great_circle_distance(self.dyn.state[2], self.dyn.state[1], self.target_lat, self.target_lon)

        self.dyn.step(action)
        self.step_count += 1

        # 👇 关键：先更新 LOS 信息（用于 reward 和 obs）
        self._update_los_errors()

        # 计算当前水平距离
        #alt = self.dyn.get_altitude()
        #s_arc = great_circle_distance(
        #    self.dyn.state[2], self.dyn.state[1],
        #    self.target_lat, self.target_lon

        # 获取当前距离
        s_arc = self._cached_s_arc
        alt = self._cached_alt
        #s_3d = np.sqrt(s_arc ** 2 + alt ** 2)
        s_3d = self._get_3d_distance()
        self.min_s_3d = min(self.min_s_3d, s_3d)
        success = (s_3d < 1300.0)

        # ✅ 关键：用 3D 距离判断是否命中
        #s_3d = self._get_3d_distance()
        #success = (s_3d < 100.0)  # 100 米球形杀伤半径

        # 终止条件
        # 计算是否成功（必须在 is_terminal 之前判断！）
        #success = (s_arc < 100.0)# and (self.dyn.get_altitude() < 1000.0)

        # 终止条件：成功 或 动力学失败 或 超时
        #done = success or self.dyn.is_terminal(self.target_lat, self.target_lon) or (self.step_count >= self.max_steps)
        #done = self.dyn.is_terminal() or self.step_count >= self.max_steps
        #success = s_arc < 1000.0 and self.dyn.get_altitude() < 1000  # 1km 内命中
        done = success or self.dyn.is_terminal(self.target_lat, self.target_lon) or (self.step_count >= self.max_steps)

        # 现在可以安全使用缓存的 LOS 速率！
        reward = self._compute_reward(
            s_arc, prev_s_arc, done, success,
            heading_rate = self._cached_heading_rate,
            elevation_rate = self._cached_elevation_rate,
            s_3d = s_3d
        )



        #reward = self._compute_reward(s_arc, prev_s_arc, done, success)
        obs = self._get_obs()
        info = {
            "s_arc": s_arc,  # 水平距离（米）
            "s_3d": s_3d,    # 3D 空间距离（米）
            "success": success,
            "alt": alt, #self.dyn.get_altitude()
            "heading_error": self._cached_heading_error,
            "elevation_error": self._cached_elevation_error,
            "V": self.dyn.state[3],
            "gamma_deg": np.degrees(self.dyn.state[4])
        }

        return obs, reward, done, False, info
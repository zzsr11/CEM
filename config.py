import numpy as np

# 地球参数
R_EARTH = 6378137.0      # m
G = 9.80665              # m/s²

# 大气
RHO_0 = 1.225            # kg/m³ (海平面)
H_SCALE = 8500.0         # m (标高)

# 飞行器参考参数
M_REF = 1000.0           # kg
S_REF = 0.5              # m²

# 初始状态（高度 30km，速度 2000 m/s）
INIT_ALT = 30000.0       # m
INIT_V = 2000.0          # m/s
INIT_GAMMA = np.deg2rad(-16.7)   # 弹道倾角（俯冲进入）
INIT_CHI = np.deg2rad(45.0)    # 弹道偏航角（朝东北）

# 目标位置（经纬度，单位：弧度）
TARGET_LON = np.deg2rad(0.5)   # ～55.6 km east
TARGET_LAT = np.deg2rad(0.5)   # ～55.6 km north
TARGET_ALT = 0.0               # 垂直高度为0
# → 大圆距离 ≈ sqrt(55.6² + 55.6²) ≈ 78.6 km，但球面距离 ≈ 78.5 km

# 实际目标距离可通过 utils 计算
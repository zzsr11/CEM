# train_td3.py
import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from environment import HFVEnv  # 确保你的环境文件叫 environment.py

# ===== 配置 =====
SEED = 42
LOG_DIR = "./logs/TD3_HFV_10"
MODEL_SAVE_DIR = "./models"

os.makedirs(LOG_DIR, exist_ok=True)

# ===== 创建训练环境 =====
# 使用 DummyVecEnv 自动包装 Monitor（但 eval_env 需手动 Monitor）
env = make_vec_env(HFVEnv, n_envs=4, seed=SEED)

# ===== 创建评估环境（必须手动加 Monitor！）=====
eval_env = HFVEnv()
eval_env = Monitor(eval_env, filename=os.path.join(LOG_DIR, "eval"))  # 启用日志记录

# ===== 动作噪声（TD3 推荐）=====
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)  # 可调：0.1 ～ 0.2
)

# ===== 创建模型 =====
model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    learning_rate=1e-3,
    buffer_size=2000000,
    batch_size=256,
    gamma=0.99,
    tau=0.001,
    policy_delay=2,
    verbose=1,
    seed=SEED,
    device="cpu"  # 或 "cuda" 如果有 GPU
)

# ===== 回调：定期评估并保存最佳模型 =====
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_SAVE_DIR,
    log_path=LOG_DIR,
    eval_freq=5000,      # 每 5000 步评估一次（总步数 / n_envs）
    deterministic=True,
    render=False,
    n_eval_episodes=5    # 每次评估跑 5 个 episode
)

# ===== 开始训练 =====
print("🚀 开始 TD3 训练...")
model.learn(
    total_timesteps=1000000,
    callback=eval_callback,
    log_interval=100,
    progress_bar=True
)
# ===== 保存最终模型（非最佳，但完整训练结束状态）=====
model.save(os.path.join(MODEL_SAVE_DIR, "final_model"))
print(f"💾 最终模型已保存至: {os.path.join(MODEL_SAVE_DIR, 'final_model.zip')}")
print(f"✅ 训练完成！日志保存至: {LOG_DIR}")
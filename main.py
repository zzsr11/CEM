# test_td3.py
import numpy as np
from stable_baselines3 import TD3
from environment import HFVEnv

model = TD3.load("./models/best_model")
def main():
    env = HFVEnv()
    obs, info = env.reset()
    sum_reward = 0.0
    for step in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        sum_reward += reward  # 👈 累加奖励
        if step % 200 == 0:
            s_arc = info['s_arc']
            alt = info['alt']
            print(f"Step {step}: s_arc={s_arc/1000:.2f} km, alt={alt/1000:.1f} km")
        if done or truncated:
            break
    # 最终结果
    final_s_3d = info.get('s_3d', -1)
    success = info.get('success', False)
    print("\n🎯 最终结果:")
    print(f"  命中距离: {final_s_3d:.1f} 米")
    print(f"  是否成功: {success}")
    print(f"  总奖励:   {sum_reward:.1f}")
if __name__ == "__main__":
    main()


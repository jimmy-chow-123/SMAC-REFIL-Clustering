import sys
import os
os.environ['SC2PATH'] = '/Applications/StarCraft II'

# 尝试直接修改 pysc2 内部的地图注册列表，去除重复项
try:
    import pysc2.maps.lib as maps_lib
    if hasattr(maps_lib, "_REGISTERED_MAPS"):
        reg = maps_lib._REGISTERED_MAPS
        unique = {}
        for m in reg:
            # 如果已存在同名地图，则只保留第一次出现的
            if m.name not in unique:
                unique[m.name] = m
        maps_lib._REGISTERED_MAPS = list(unique.values())
        print("已移除重复的地图注册项")
    else:
        print("未找到 _REGISTERED_MAPS 属性，跳过重复检查")
except Exception as e:
    print("修改地图注册列表失败:", e)

from smac.smac.env import StarCraft2Env

class SMACEnvironment:
    def __init__(self, map_name="3m"):
        self.env = StarCraft2Env(map_name=map_name)
        self.env_info = self.env.get_env_info()
        print("环境信息:", self.env_info)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, actions):
        reward, terminated, info = self.env.step(actions)
        return reward, terminated, info
    
    def close(self):
        self.env.close()

if __name__ == "__main__":
    print("Python executable:", sys.executable)
    print("SC2PATH in Python:", os.environ.get('SC2PATH'))
    env = SMACEnvironment(map_name="3m")
    obs = env.reset()
    total_reward = 0
    terminated = False

    while not terminated:
        actions = [env.env.action_space.sample() for _ in range(env.env_info["n_agents"])]
        reward, terminated, _ = env.step(actions)
        total_reward += reward

    print("本轮总奖励:", total_reward)
    env.close()

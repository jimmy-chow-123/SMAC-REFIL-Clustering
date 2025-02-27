import sys
import os

# 自动获取当前工作目录
project_path = os.getcwd()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

os.environ['SC2PATH'] = '/Applications/StarCraft II'


from smac.env import StarCraft2Env

class SMACEnvironment:
    def __init__(self, map_name="2s3z"):
        self.env = StarCraft2Env(map_name=map_name)
        self.env_info = self.env.get_env_info()
        print("Environment info:", self.env_info)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, actions):
        reward, terminated, info = self.env.step(actions)
        return reward, terminated, info
    
    def close(self):
        self.env.close()

if __name__ == "__main__":
    env = SMACEnvironment(map_name="2s3z")
    obs = env.reset()
    total_reward = 0
    terminated = False

    while not terminated:
        actions = [env.env.action_space.sample() for _ in range(env.env_info["n_agents"])]
        reward, terminated, _ = env.step(actions)
        total_reward += reward

    print("Total reward this episode:", total_reward)
    env.close()
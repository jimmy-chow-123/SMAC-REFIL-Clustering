import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn

# 修改导入：使用 SMACEnvironment 而非 DynamicTeamSC2Env
from environment import SMACEnvironment

# 超参数（需与训练时一致）
OBS_DIM = 14             # 例如：2（位置）+ 1（血量）+ 1（团队ID归一化）+ 10（隐藏状态）
NUM_ACTIONS = 5          # 离散动作（例如：移动方向、no_op）
EMBED_DIM = 16           # 代理嵌入维度（用于辅助损失）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义代理网络（与 trainer.py 保持一致）
class AgentNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, embed_dim):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        # Q 值分支
        self.q_head = nn.Linear(64, num_actions)
        # 嵌入分支（用于组损失）
        self.embed_head = nn.Linear(64, embed_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_vals = self.q_head(x)               
        embeddings = self.embed_head(x)         
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return q_vals, embeddings

# Epsilon-greedy 动作选择，这里测试时 epsilon 设为 0（贪婪选择）
def select_actions(net, state, epsilon=0.0):
    num_agents = state.shape[0]
    state_tensor = torch.FloatTensor(state).to(DEVICE)
    q_vals, _ = net(state_tensor)
    actions = []
    for i in range(num_agents):
        if random.random() < epsilon:
            actions.append(random.randrange(NUM_ACTIONS))
        else:
            actions.append(torch.argmax(q_vals[i]).item())
    return np.array(actions)

def test_model(model_path, num_test_episodes=20):
    # 加载保存的模型权重
    net = AgentNetwork(OBS_DIM, NUM_ACTIONS, EMBED_DIM).to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()
    
    # 初始化环境（使用 SMACEnvironment）
    env = SMACEnvironment(map_name="2s3z")
    
    test_rewards = []
    for episode in range(1, num_test_episodes + 1):
        state = env.reset()  # 返回形状 (num_agents, OBS_DIM)
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            actions = select_actions(net, state, epsilon=0.0)
            # SMACEnvironment.step 返回 (reward, done, info)
            reward, done, info = env.step(actions)
            # 通过 get_obs() 获取下一个状态
            next_state = env.env.get_obs()
            total_reward += reward
            state = next_state
            step_count += 1
        print(f"Test Episode {episode}: steps = {step_count}, total reward = {total_reward:.2f}")
        test_rewards.append(total_reward)
    
    env.close()
    avg_reward = np.mean(test_rewards)
    print(f"Average Test Reward over {num_test_episodes} episodes: {avg_reward:.2f}")
    
    # 绘制测试奖励曲线
    plt.figure()
    plt.plot(range(1, num_test_episodes + 1), test_rewards, marker='o', label="Test Reward")
    plt.xlabel("Test Episode")
    plt.ylabel("Total Reward")
    plt.title("Test Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 模型检查点文件路径（确保文件存在）
    model_path = "best_model.pth"
    test_model(model_path, num_test_episodes=20)

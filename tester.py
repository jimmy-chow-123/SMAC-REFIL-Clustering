import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn

from environment import SMACEnvironment

# 超参数（需与训练时一致）
OBS_DIM = 80             # 环境的 obs_shape
NUM_ACTIONS = 11         # 环境的 n_actions
EMBED_DIM = 16           # 代理嵌入维度

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义代理网络（与 trainer.py 保持一致）
class AgentNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, embed_dim):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q_head = nn.Linear(64, num_actions)
        self.embed_head = nn.Linear(64, embed_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_vals = self.q_head(x)
        embeddings = self.embed_head(x)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return q_vals, embeddings

# 更新后的 epsilon-greedy 动作选择函数，考虑可用动作
def select_actions(net, state, epsilon=0.0, avail_actions=None):
    num_agents = state.shape[0]
    state_tensor = torch.FloatTensor(state).to(DEVICE)
    q_vals, _ = net(state_tensor)
    actions = []
    for i in range(num_agents):
        if avail_actions is not None:
            avail = avail_actions[i]  # 可用动作：二值列表，长度为 NUM_ACTIONS
            avail_indices = [j for j, flag in enumerate(avail) if flag == 1]
            if random.random() < epsilon:
                action = random.choice(avail_indices)
            else:
                # 对不可用动作 Q 值置为 -∞
                masked_q = q_vals[i].clone()
                for j in range(NUM_ACTIONS):
                    if avail[j] == 0:
                        masked_q[j] = -float('inf')
                action = torch.argmax(masked_q).item()
        else:
            if random.random() < epsilon:
                action = random.randrange(NUM_ACTIONS)
            else:
                action = torch.argmax(q_vals[i]).item()
        actions.append(action)
    return np.array(actions)

def test_model(model_path, num_test_episodes=20):
    # 加载训练好的模型权重
    net = AgentNetwork(OBS_DIM, NUM_ACTIONS, EMBED_DIM).to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()
    
    # 初始化环境，使用 SMACEnvironment
    env = SMACEnvironment(map_name="2s3z")
    
    test_rewards = []
    for episode in range(1, num_test_episodes + 1):
        env.reset()
        state = np.array(env.env.get_obs())
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            avail_actions = env.env.get_avail_actions()  # 获取当前可用动作
            actions = select_actions(net, state, epsilon=0.0, avail_actions=avail_actions)
            reward, done, info = env.step(actions)
            state = np.array(env.env.get_obs())
            total_reward += reward
            step_count += 1
        print(f"Test Episode {episode}: steps = {step_count}, total reward = {total_reward:.2f}")
        test_rewards.append(total_reward)
    
    env.close()
    avg_reward = np.mean(test_rewards)
    print(f"Average Test Reward over {num_test_episodes} episodes: {avg_reward:.2f}")
    
    plt.figure()
    plt.plot(range(1, num_test_episodes + 1), test_rewards, marker='o', label="Test Reward")
    plt.xlabel("Test Episode")
    plt.ylabel("Total Reward")
    plt.title("Test Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    model_path = "best_model.pth"
    test_model(model_path, num_test_episodes=20)

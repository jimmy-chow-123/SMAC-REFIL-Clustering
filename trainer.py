import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# 修改导入：使用 environment.py 中的 SMACEnvironment
from environment import SMACEnvironment

# 超参数设置
OBS_DIM = 14             # 例如：2（位置）+ 1（血量）+ 1（归一化团队ID）+ 10（隐藏状态）
NUM_ACTIONS = 5          # 离散动作（例如：移动方向和 no_op）
EMBED_DIM = 16           # 代理嵌入维度，用于组损失
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 16
BUFFER_CAPACITY = 5000
NUM_EPISODES = 200
TARGET_UPDATE_FREQ = 10  # 每 TARGET_UPDATE_FREQ 轮更新一次目标网络
LAMBDA = 0.5             # 辅助组损失权重

# Epsilon-greedy 参数
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# 组嵌入损失参数：对于不同组，要求余弦相似度低于该阈值
SIMILARITY_MARGIN = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 代理网络：输出 Q 值和嵌入向量
class AgentNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, embed_dim):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q_head = nn.Linear(64, num_actions)      # Q 值分支
        self.embed_head = nn.Linear(64, embed_dim)      # 嵌入分支
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_vals = self.q_head(x)
        embeddings = self.embed_head(x)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return q_vals, embeddings

# 经验回放缓冲区，存储转换及团队信息
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, actions, reward, next_state, done, teams):
        self.buffer.append((state, actions, reward, next_state, done, teams))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, teams = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.uint8),
                teams)
    
    def __len__(self):
        return len(self.buffer)

# 计算当前状态下所有代理所选动作的 Q 值之和
def compute_total_q(net, state, actions):
    q_vals, _ = net(state)  # (num_agents, NUM_ACTIONS)
    chosen_q = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
    return chosen_q.sum()

# 计算下个状态下所有代理的最大 Q 值之和
def compute_total_max_q(net, next_state):
    q_vals, _ = net(next_state)
    max_q, _ = q_vals.max(dim=1)
    return max_q.sum()

# 每个代理的 epsilon-greedy 动作选择
def select_actions(net, state, epsilon):
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

# 计算组嵌入损失：同组希望余弦相似度接近 1，不同组则低于 SIMILARITY_MARGIN
def compute_group_loss(embeddings, team_assignments):
    loss = 0.0
    count = 0
    num_agents = embeddings.shape[0]
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            cos_sim = nn.functional.cosine_similarity(embeddings[i].unsqueeze(0),
                                                      embeddings[j].unsqueeze(0))
            if team_assignments[i] == team_assignments[j]:
                loss += (1 - cos_sim)
            else:
                loss += torch.relu(cos_sim - SIMILARITY_MARGIN)
            count += 1
    if count > 0:
        return loss / count
    else:
        return torch.tensor(0.0).to(DEVICE)

def main():
    # 初始化环境，这里使用 SMACEnvironment（注意：不再传入 step_mul, render, team_capacity 参数）
    env = SMACEnvironment(map_name="2s3z")
    
    # 初始化在线网络和目标网络
    net = AgentNetwork(OBS_DIM, NUM_ACTIONS, EMBED_DIM).to(DEVICE)
    target_net = AgentNetwork(OBS_DIM, NUM_ACTIONS, EMBED_DIM).to(DEVICE)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    
    epsilon = EPS_START
    episode_rewards = []
    
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()  # 初始状态：形状 (num_agents, OBS_DIM)
        done = False
        total_reward = 0.0
        step_count = 0
        current_team_assignments = None

        while not done:
            num_agents = state.shape[0]
            actions = select_actions(net, state, epsilon)
            # SMACEnvironment.step 返回 (reward, done, info)
            reward, done, info = env.step(actions)
            # 通过 get_obs() 获取下一个状态
            next_state = env.env.get_obs()
            total_reward += reward
            step_count += 1
            # 获取团队分配信息，如不存在则默认为所有智能体同一团队
            current_team_assignments = info.get("team_assignments", [0] * num_agents)
            replay_buffer.push(state, actions, reward, next_state, done, current_team_assignments)
            state = next_state
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode} finished in {step_count} steps with reward {total_reward:.2f}")
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        
        if len(replay_buffer) >= BATCH_SIZE:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_team_info = replay_buffer.sample(BATCH_SIZE)
            loss_total = 0.0
            for i in range(BATCH_SIZE):
                state_tensor = torch.FloatTensor(batch_states[i]).to(DEVICE)
                actions_tensor = torch.LongTensor(batch_actions[i]).to(DEVICE)
                next_state_tensor = torch.FloatTensor(batch_next_states[i]).to(DEVICE)
                reward = batch_rewards[i]
                done_flag = batch_dones[i]
                team_info = batch_team_info[i]
                
                current_q = compute_total_q(net, state_tensor, actions_tensor)
                with torch.no_grad():
                    next_max_q = compute_total_max_q(target_net, next_state_tensor)
                    target_q = reward + GAMMA * next_max_q * (1 - done_flag)
                td_loss = (current_q - target_q) ** 2
                
                _, embeddings = net(state_tensor)
                group_loss = compute_group_loss(embeddings, team_info)
                
                loss = td_loss + LAMBDA * group_loss
                loss_total += loss
            
            loss_total = loss_total / BATCH_SIZE
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
        
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(net.state_dict())
            print("Updated target network.")
    
    env.close()
    torch.save(net.state_dict(), "best_model.pth")
    
    plt.figure()
    plt.plot(episode_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Convergence Curve")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

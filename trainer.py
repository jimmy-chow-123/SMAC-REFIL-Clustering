import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Import our custom environment (which implements bilateral team matching)
from environment import DynamicTeamSC2Env

# Hyperparameters
OBS_DIM = 14             # Example: 2 (position) + 1 (health) + 1 (team id normalized) + 10 (hidden state)
NUM_ACTIONS = 5          # Discrete actions (e.g., move directions and no_op)
EMBED_DIM = 16           # Dimension of agent embedding for group loss
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 16
BUFFER_CAPACITY = 5000
NUM_EPISODES = 200
TARGET_UPDATE_FREQ = 10  # Episodes between target network updates
LAMBDA = 0.5             # Weighting for the auxiliary group loss

# Epsilon-greedy parameters
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# For group embedding loss
SIMILARITY_MARGIN = 0.5  # For agents in different groups, we want cosine similarity below this

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Agent Network: outputs both Q-values and an embedding vector.
class AgentNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, embed_dim):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        # Q-value head
        self.q_head = nn.Linear(64, num_actions)
        # Embedding head (the "encoder" producing hidden representations for group factorization)
        self.embed_head = nn.Linear(64, embed_dim)
    
    def forward(self, x):
        # x: (num_agents, input_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_vals = self.q_head(x)         # (num_agents, NUM_ACTIONS)
        embeddings = self.embed_head(x)   # (num_agents, embed_dim)
        # Normalize embeddings for cosine similarity
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return q_vals, embeddings

# Replay Buffer storing transitions along with team assignments
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, actions, reward, next_state, done, teams):
        # state, next_state: np.array (num_agents, OBS_DIM)
        # actions: np.array (num_agents,)
        # reward: scalar, done: bool, teams: list of team assignments per agent
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

# Compute total Q by summing each agent’s chosen Q values
def compute_total_q(net, state, actions):
    # state: tensor (num_agents, OBS_DIM)
    # actions: tensor (num_agents,)
    q_vals, _ = net(state)  # (num_agents, NUM_ACTIONS)
    chosen_q = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # (num_agents,)
    return chosen_q.sum()

def compute_total_max_q(net, next_state):
    q_vals, _ = net(next_state)  # (num_agents, NUM_ACTIONS)
    max_q, _ = q_vals.max(dim=1)
    return max_q.sum()

# Epsilon-greedy action selection per agent
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

# Group embedding loss L_SD: Encourages same-group embeddings to be similar, and different-group embeddings to be below a margin.
def compute_group_loss(embeddings, team_assignments):
    loss = 0.0
    count = 0
    num_agents = embeddings.shape[0]
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            cos_sim = nn.functional.cosine_similarity(embeddings[i].unsqueeze(0),
                                                      embeddings[j].unsqueeze(0))
            if team_assignments[i] == team_assignments[j]:
                # For same team: want cos_sim close to 1
                loss += (1 - cos_sim)
            else:
                # For different teams: penalize if similarity is above margin
                loss += torch.relu(cos_sim - SIMILARITY_MARGIN)
            count += 1
    if count > 0:
        return loss / count
    else:
        return torch.tensor(0.0).to(DEVICE)

def main():
    # Initialize environment.
    # Our environment implements dynamic team formation and returns team assignments in info['team_assignments'].
    env = DynamicTeamSC2Env(map_name="2s3z", step_mul=8, render=True, team_capacity=3)
    
    # Initialize online and target networks.
    net = AgentNetwork(OBS_DIM, NUM_ACTIONS, EMBED_DIM).to(DEVICE)
    target_net = AgentNetwork(OBS_DIM, NUM_ACTIONS, EMBED_DIM).to(DEVICE)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    
    epsilon = EPS_START
    episode_rewards = []
    
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()  # state: np.array (num_agents, OBS_DIM)
        done = False
        total_reward = 0.0
        step_count = 0
        current_team_assignments = None

        while not done:
            num_agents = state.shape[0]
            actions = select_actions(net, state, epsilon)
            next_state, reward, done, info = env.step(actions)
            total_reward += reward
            step_count += 1
            # Obtain team assignments from environment info.
            current_team_assignments = info.get("team_assignments", [0] * num_agents)
            replay_buffer.push(state, actions, reward, next_state, done, current_team_assignments)
            state = next_state
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode} finished in {step_count} steps with reward {total_reward:.2f}")
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        
        # Training updates when enough transitions have been stored.
        if len(replay_buffer) >= BATCH_SIZE:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_team_info = replay_buffer.sample(BATCH_SIZE)
            loss_total = 0.0
            for i in range(BATCH_SIZE):
                state_tensor = torch.FloatTensor(batch_states[i]).to(DEVICE)
                actions_tensor = torch.LongTensor(batch_actions[i]).to(DEVICE)
                next_state_tensor = torch.FloatTensor(batch_next_states[i]).to(DEVICE)
                reward = batch_rewards[i]
                done_flag = batch_dones[i]
                team_info = batch_team_info[i]  # list of team assignments for this transition

                current_q = compute_total_q(net, state_tensor, actions_tensor)
                with torch.no_grad():
                    next_max_q = compute_total_max_q(target_net, next_state_tensor)
                    target_q = reward + GAMMA * next_max_q * (1 - done_flag)
                td_loss = (current_q - target_q) ** 2
                
                # Auxiliary group loss from agent embeddings.
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
    
    # Plot reward convergence curve.
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

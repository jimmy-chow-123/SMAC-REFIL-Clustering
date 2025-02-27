import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn

# Import our custom environment which implements dynamic team formation
from environment import DynamicTeamSC2Env

# Hyperparameters (must match those used during training)
OBS_DIM = 14             # e.g., 2 (position) + 1 (health) + 1 (team id normalized) + 10 (hidden state)
NUM_ACTIONS = 5          # Discrete actions (e.g., move directions, no_op)
EMBED_DIM = 16           # Dimension of agent embeddings (for auxiliary loss)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the agent network (same as in trainer.py)
class AgentNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, embed_dim):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        # Q-value head
        self.q_head = nn.Linear(64, num_actions)
        # Embedding head (produces agent representation for group loss)
        self.embed_head = nn.Linear(64, embed_dim)
    
    def forward(self, x):
        # x shape: (num_agents, input_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_vals = self.q_head(x)               # (num_agents, NUM_ACTIONS)
        embeddings = self.embed_head(x)         # (num_agents, embed_dim)
        # Normalize embeddings to unit norm (for cosine similarity)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return q_vals, embeddings

# Epsilon-greedy action selection; here we use epsilon = 0 for testing (greedy)
def select_actions(net, state, epsilon=0.0):
    num_agents = state.shape[0]
    state_tensor = torch.FloatTensor(state).to(DEVICE)
    q_vals, _ = net(state_tensor)
    actions = []
    for i in range(num_agents):
        # For testing, epsilon=0; always choose the greedy action.
        if random.random() < epsilon:
            actions.append(random.randrange(NUM_ACTIONS))
        else:
            actions.append(torch.argmax(q_vals[i]).item())
    return np.array(actions)

def test_model(model_path, num_test_episodes=20):
    # Load the trained model weights from the checkpoint file.
    net = AgentNetwork(OBS_DIM, NUM_ACTIONS, EMBED_DIM).to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()
    
    # Initialize the environment; ensure the environment is configured to use bilateral matching.
    env = DynamicTeamSC2Env(map_name="2s3z", step_mul=8, render=True, team_capacity=3)
    
    test_rewards = []
    for episode in range(1, num_test_episodes + 1):
        state = env.reset()  # Returns np.array of shape (num_agents, OBS_DIM)
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            actions = select_actions(net, state, epsilon=0.0)  # Pure greedy action selection
            next_state, reward, done, info = env.step(actions)
            total_reward += reward
            state = next_state
            step_count += 1
        print(f"Test Episode {episode}: steps = {step_count}, total reward = {total_reward:.2f}")
        test_rewards.append(total_reward)
    
    env.close()
    avg_reward = np.mean(test_rewards)
    print(f"Average Test Reward over {num_test_episodes} episodes: {avg_reward:.2f}")
    
    # Plot the reward convergence curve over test episodes.
    plt.figure()
    plt.plot(range(1, num_test_episodes + 1), test_rewards, marker='o', label="Test Reward")
    plt.xlabel("Test Episode")
    plt.ylabel("Total Reward")
    plt.title("Test Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Path to the saved model checkpoint (ensure this file exists)
    model_path = "best_model.pth"
    test_model(model_path, num_test_episodes=20)

import numpy as np
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dueling DQN Architecture for Better Stability
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3_adv = nn.Linear(128, action_dim)  # Advantage stream
        self.fc3_val = nn.Linear(128, 1)  # Value stream

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        adv = self.fc3_adv(x)
        val = self.fc3_val(x)
        return val + adv - adv.mean()

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.memory = deque(maxlen=50000)

        self.model = DuelingDQN(state_dim, action_dim).to(device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
        target_q_values = rewards + self.gamma * self.target_model(next_states).gather(1, next_actions).squeeze() * (1 - dones)
        
        current_q_values = self.model(states).gather(1, actions).squeeze()
        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train the agent with visualization
def train_dqn(episodes=1200, early_stopping_threshold=200):
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    rewards_history = deque(maxlen=100)
    episode_rewards = []
    epsilons = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        trajectory = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            trajectory.append(state[:2])
        
        if episode % 10 == 0:
            agent.update_target_network()

        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history)
        episode_rewards.append(total_reward)
        epsilons.append(agent.epsilon)

        print(f"Episode {episode}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

        if episode % 500 == 0:
            torch.save(agent.model.state_dict(), f"lunarlander_dqn_{episode}.pth")

        if avg_reward > early_stopping_threshold:
            print(f"🎉 Early stopping triggered at Episode {episode} with Avg Reward: {avg_reward:.2f}!")
            break

        if episode % 100 == 0:
            visualize_landing_trajectory(trajectory, episode)

    env.close()
    plot_training_progress(episode_rewards, epsilons)
    return agent

# Plot training progress
def plot_training_progress(rewards, epsilons):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epsilons, label="Epsilon")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon Value")
    plt.title("Epsilon Decay")
    plt.legend()
    plt.show()

# Visualize Lunar Lander trajectory
def visualize_landing_trajectory(trajectory, episode):
    trajectory = np.array(trajectory)
    plt.figure(figsize=(5, 5))
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-', markersize=3, alpha=0.7)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Landing Trajectory (Episode {episode})")
    plt.xlim([-1.5, 1.5])
    plt.ylim([-0.5, 1.5])
    plt.grid()
    plt.show()

# Run training with visualization
if __name__ == "__main__":
    trained_agent = train_dqn(episodes=1200)

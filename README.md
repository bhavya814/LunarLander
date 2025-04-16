# LunarLander
# 🚀 Lunar Lander with Enhanced Deep Q-Learning (DQN)

This project presents an enhanced Deep Q-Learning (DQN) agent that learns to land a spacecraft in the **LunarLander-v2** environment from OpenAI Gym. By integrating improvements such as **Dueling Network Architecture**, **Batch Normalization**, and optimized training hyperparameters, the model achieves faster convergence and higher stability.

---

## 📌 Project Overview

The objective is to train an agent to successfully land a lunar module using Reinforcement Learning. The agent observes an 8-dimensional state and chooses from 4 discrete actions to maximize cumulative rewards.

---

## 🧠 Key Features 

- ✅ **Dueling DQN Architecture**: Separates the estimation of state value and advantage for better Q-value representation.
- ✅ **Deeper Neural Network**: Three fully connected layers for richer feature extraction.
- ✅ **Batch Normalization**: Improves training stability and convergence.
- ✅ **Lower Learning Rate** (`1e-4`): Helps in smoother and more stable learning.
- ✅ **Increased Replay Buffer** (`1e6`) and **Batch Size** (`128`): Promotes diverse and effective learning.
- ✅ **More Frequent Network Updates** (`UPDATE_EVERY = 2`): Enables faster learning.
- ✅ **Soft Update Strategy**: Smoothly updates the target network weights to reduce variance.

---

## 📊 Results –

| Episode | Average Score |
|---------|----------------|
| 100     | -158.08        |
| 200     | -61.15         |
| 300     | 91.25          |
| 400     | 142.53         |
| 491     | **200.01**     |

✅ **Environment solved in 491 episodes** (average score ≥ 200 over 100 episodes)

---

## 📁 Installation

Install the required libraries using pip:

```bash
pip install gym numpy torch matplotlib

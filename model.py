import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Improved Q-Network with more layers and Dueling Architecture."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)  # Batch Normalization

        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)  # Batch Normalization

        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)  # Batch Normalization

        # Dueling Network Architecture
        self.fc_value = nn.Linear(fc3_units, 1)
        self.fc_advantage = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.leaky_relu(self.bn1(self.fc1(state)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))

        value = self.fc_value(x)
        advantage = self.fc_advantage(x)

        return value + (advantage - advantage.mean(dim=1, keepdim=True))
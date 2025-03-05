import torch
from torch import nn, F
import numpy as np


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_filters=256):
        super().__init__()

        self.state_dim = state_dim
        self.n_hidden_filters = n_hidden_filters
        self.action_dim = action_dim

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1
        )

        self.fc1 = nn.Linear(
            64 * state_dim[0] * state_dim[1] * state_dim[2] + 2, self.n_hidden_filters
        )

        # self.fc1 = nn.Linear(
        #     np.array(env.single_observation_space.shape).prod()
        #     + np.prod(env.single_action_space.shape),
        #     256,
        # )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, a):
        x = F.relu(self.conv1(states.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds, n_hidden_filters=256):
        super().__init__()
        self.state_dim = state_dim
        self.n_hidden_filters = n_hidden_filters
        self.action_dim = action_dim
        self.action_bounds = action_bounds

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1
        )

        self.fc1 = nn.Linear(
            64 * state_dim[0] * state_dim[1] * state_dim[2], self.n_hidden_filters
        )
        # self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, 2)
        self.fc_logstd = nn.Linear(256, 2)

    def forward(self, states):
        x = F.relu(self.conv1(states.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SAC:
    pass

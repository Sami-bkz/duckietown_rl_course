import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# Function to initialize weights
def init_weight(layer, initializer="xavier uniform"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.n_hidden_filters = n_hidden_filters

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
            64 * self.state_dim[0] * self.state_dim[1] * self.state_dim[2],
            out_features=self.n_hidden_filters,
        )
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()

        self.hidden2 = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.n_hidden_filters
        )
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.conv1(states.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.hidden2(x))
        return self.value(x)


class QvalueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
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
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()

        self.hidden2 = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.n_hidden_filters
        )
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states, actions):
        x = F.relu(self.conv1(states.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat([x, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.hidden2(x))
        return self.q_value(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
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
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()

        self.hidden2 = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.n_hidden_filters
        )
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.action_dim
        )
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(
            in_features=self.n_hidden_filters, out_features=self.action_dim
        )
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.conv1(states.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def get_action(self, states):
        dist = self(states)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        log_prob -= torch.log(1 - action**2 + 1e-6)
        log_prob -= 1
        return (action * self.action_bounds[1]).clamp_(
            self.action_bounds[0], self.action_bounds[1]
        ), log_prob

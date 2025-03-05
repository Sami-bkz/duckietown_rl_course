import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy


class CNNPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=state_dim[0],
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(
            64 * (state_dim[1] * state_dim[2] + action_dim), 256
        )  # Adjust input size based on the output of conv layers
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_log_std = nn.Linear(256, action_dim)

    def forward(self, x, action):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = torch.cat([x, action], dim=1)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        return mu, log_std


class SAC:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4):
        self.policy = CNNPolicy(state_dim, action_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.critic1 = self._create_critic(state_dim)
        self.critic2 = self._create_critic(state_dim)

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def _create_critic(self, state_dim):
        return nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * state_dim[1] * state_dim[2], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def select_action(self, state):
        # state = torch.FloatTensor(state).unsqueeze(0)
        mu, log_std = self.policy(state)

        # Compute scale (standard deviation) with a minimum value
        std = torch.exp(log_std).clamp(
            min=1e-6, max=1
        )  # Use a small positive value to prevent issues

        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()

        # Clip action between -1 and 1
        action = torch.tanh(action) * torch.tensor(1)

        return numpy.array(action.cpu().detach()[0].tolist())

    # Update function remains unchanged...
    def update(self, replay_buffer, batch_size=64):
        # Sample from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # state = torch.FloatTensor(state)
        # next_state = torch.FloatTensor(next_state)
        # action = torch.FloatTensor(action)
        # reward = reward.unsqueeze(0)
        # done = done.unsqueeze(0)

        # Update the critic networks
        with torch.no_grad():
            next_mu, next_log_std = self.policy(next_state)

            # Compute scale (standard deviation) with a minimum value
            next_std = torch.exp(next_log_std).clamp(
                min=1e-6
            )  # Use a small positive value to prevent issues

            next_dist = torch.distributions.Normal(next_mu, next_std)
            next_action = next_dist.rsample()  # Reparameterization trick
            q1_next = self.critic1(next_state).detach()
            q2_next = self.critic2(next_state).detach()
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + (1 - done) * self.gamma * (
                q_next
                - self.alpha * next_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            )

        q1 = self.critic1(state)
        q2 = self.critic2(state)

        critic1_loss = nn.functional.mse_loss(q1, target_q)
        critic2_loss = nn.functional.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update the policy network
        mu, log_std = self.policy(state)
        std = torch.exp(log_std).clamp(min=1e-6)

        dist = torch.distributions.Normal(mu, std)
        policy_loss = (
            self.alpha * dist.log_prob(action).sum(dim=-1, keepdim=True)
            - self.critic1(state)
        ).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def soft_update(self, target_network, source_network):
        for target_param, param in zip(
            target_network.parameters(), source_network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

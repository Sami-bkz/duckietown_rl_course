import numpy as np

from duckietownrl.algorithms.model import PolicyNetwork, QvalueNetwork, ValueNetwork

import torch

import os.path as op
from torch import from_numpy
from torch.optim.adam import Adam


class SAC:
    def __init__(
        self,
        env_name,
        state_dim,
        action_dim,
        gamma=0.99,
        alpha=0.5,
        lr=3e-4,
        action_bounds=(-1, 1),
        reward_scale=1,
        replay_buffer=None,
        device="cpu",
    ):
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.action_bounds = action_bounds
        self.reward_scale = reward_scale
        self.memory = replay_buffer

        self.device = device

        self.policy_network = PolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_bounds=self.action_bounds,
        ).to(self.device)

        self.q_value_network1 = QvalueNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)

        self.q_value_network2 = QvalueNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)

        self.value_network = ValueNetwork(state_dim=self.state_dim).to(self.device)

        self.value_target_network = ValueNetwork(state_dim=self.state_dim).to(
            self.device
        )
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

        self.value_loss = torch.nn.MSELoss()
        self.q_value_loss = torch.nn.MSELoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.lr)
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.lr)

    def store(self, state, reward, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        done = torch.Tensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, reward, done, action, next_state)

    def train(self):
        if not self.memory.can_sample():
            return 0, 0, 0
        else:
            print("training...")
            states, actions, rewards, next_states, dones = self.memory.sample()

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(
                states
            )
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.alpha * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.value_loss(value, target_value)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = (
                    self.reward_scale * rewards
                    + self.gamma * self.value_target_network(next_states) * (1 - dones)
                )
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.q_value_loss(q1, target_q)
            q2_loss = self.q_value_loss(q2, target_q)

            policy_loss = (self.alpha * log_probs - q).mean()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.soft_update_target_network(
                self.value_network, self.value_target_network
            )
            print("...done!")
            return (
                value_loss.item(),
                0.5 * (q1_loss + q2_loss).item(),
                policy_loss.item(),
            )

    def select_action(self, states):
        action, _ = self.policy_network.sample_or_likelihood(states)
        return np.array(action.detach().cpu().tolist())

    @staticmethod
    def soft_update_target_network(local_network, target_network, tau=0.005):
        for target_param, local_param in zip(
            target_network.parameters(), local_network.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1 - tau) * target_param.data
            )

    def save(self, path, folder_name, episodes):
        import os

        file_path = op.join(path, folder_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        torch.save(
            self.policy_network.state_dict(),
            os.path.join(file_path, self.env_name + f"_policy_{episodes}_weights.pth"),
        )
        torch.save(
            self.q_value_network1.state_dict(),
            os.path.join(
                file_path,
                self.env_name + f"_qvalue_net1_{episodes}_weights.pth",
            ),
        )
        torch.save(
            self.q_value_network2.state_dict(),
            os.path.join(
                file_path,
                self.env_name + f"_qvalue_net2_{episodes}_weights.pth",
            ),
        )
        torch.save(
            self.value_network.state_dict(),
            os.path.join(
                file_path,
                self.env_name + f"_value_net_{episodes}_weights.pth",
            ),
        )

    def load_weights(self, path, folder_name, n_episodes):
        self.policy_network.load_state_dict(
            torch.load(
                op.join(
                    path,
                    "models",
                    folder_name,
                    self.env_name + "_policy_" + str(n_episodes) + "_weights.pth",
                )
            )
        )

    def set_to_eval_mode(self):
        self.policy_network.eval()

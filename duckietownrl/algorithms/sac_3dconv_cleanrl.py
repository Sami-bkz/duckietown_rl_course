import numpy as np

from duckietownrl.algorithms.model_3dconv import (
    PolicyNetwork,
    QvalueNetwork,
    ValueNetwork,
)

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
        target_network_frequency=1,
        policy_frequency=2,
        tau=0.005,
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
        self.target_network_frequency = target_network_frequency
        self.policy_frequency = policy_frequency
        self.tau = tau

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

        self.q_value_network1_target = QvalueNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)

        self.q_value_network2_target = QvalueNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)

        self.q_value_loss = torch.nn.MSELoss()

        self.q_optimizer = Adam(
            list(self.q_value_network1.parameters())
            + list(self.q_value_network2.parameters()),
            lr=self.lr,
        )
        self.actor_optimizer = Adam(list(self.policy_network.parameters()), lr=self.lr)

    def store(self, state, reward, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        done = torch.Tensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, reward, done, action, next_state)

    def train(self, global_step):
        if not self.memory.can_sample():
            return 0, 0, 0
        else:
            print("training...")
            states, actions, rewards, next_states, dones = self.memory.sample()
            with torch.no_grad():
                # Calculating the value target
                next_state_actions, next_state_log_probs = (
                    self.policy_network.get_action(states)
                )
                q1 = self.q_value_network1(states, next_state_actions)
                q2 = self.q_value_network2(states, next_state_actions)
                next_q_value = torch.min(q1, q2) - self.alpha * next_state_log_probs

            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = self.q_value_loss(q1, next_q_value)
            q2_loss = self.q_value_loss(q2, next_q_value)
            q_loss = q1_loss + q2_loss

            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            if global_step % self.policy_frequency == 0:
                pi, log_pi = self.policy_network.get_action(states)
                self.q_value_network1_pi = self.q_value_network1(states, pi)
                self.q_value_network2_pi = self.q_value_network2(states, pi)
                min_qf_pi = torch.min(
                    self.q_value_network1_pi, self.q_value_network2_pi
                )
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # update the target networks
            if global_step % self.target_network_frequency == 0:
                for param, target_param in zip(
                    self.q_value_network1.parameters(),
                    self.q_value_network1_target.parameters(),
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.q_value_network2.parameters(),
                    self.q_value_network2_target.parameters(),
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

    def select_action(self, states):
        action, _ = self.policy_network.get_action(states)
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

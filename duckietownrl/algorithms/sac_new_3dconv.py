import numpy as np

from duckietownrl.algorithms.model_3dconv import (
    PolicyNetwork,
    QvalueNetwork,
    ValueNetwork,
)

import torch
import copy
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
        alpha=0.2,
        actor_lr=1e-4,
        critic_lr=1e-3,
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
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

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

        self.qnet1 = QvalueNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)

        self.qnet2 = QvalueNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)

        self.qnet1_target = copy.deepcopy(self.qnet1).to(self.device)

        self.qnet2_target = copy.deepcopy(self.qnet2).to(self.device)

        self.q_value_loss = torch.nn.MSELoss()

        self.q_optimizer = Adam(
            list(self.qnet1.parameters()) + list(self.qnet2.parameters()),
            lr=self.critic_lr,
        )
        self.actor_optimizer = Adam(
            list(self.policy_network.parameters()), lr=self.actor_lr
        )

    def store(self, state, reward, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        reward = torch.Tensor([reward]).to("cpu")
        done = torch.Tensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, reward, done, action, next_state)

    def train(self, global_step, device):
        if not self.memory.can_sample():
            return 0, 0, 0
        else:
            print("training...")
            states, actions, rewards, next_states, dones = self.memory.sample(
                device=device
            )
            # with torch.no_grad():
            # Calculating the value target
            action_values1 = self.qnet1(states, actions)
            action_values2 = self.qnet2(states, actions)

            target_actions, target_actions_log_probs = self.policy_network.get_action(
                next_states
            )
            next_action_values = torch.min(
                self.qnet1(next_states, target_actions),
                self.qnet2(next_states, target_actions),
            )

            expected_action_values = rewards.unsqueeze(-1) + self.gamma * (
                next_action_values - self.alpha * target_actions_log_probs
            )

            action_values1_loss = self.q_value_loss(
                action_values1, expected_action_values
            )
            action_values2_loss = self.q_value_loss(
                action_values2, expected_action_values
            )
            q_loss = action_values1_loss + action_values2_loss

            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            if global_step % self.policy_frequency == 0:
                actions, log_actions_values = self.policy_network.get_action(states)

                actions_values = torch.min(
                    self.qnet1(states, actions), self.qnet2(states, actions)
                )

                actor_loss = ((self.alpha * log_actions_values) - actions_values).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # update the target networks
            if global_step % self.target_network_frequency == 0:
                for param, target_param in zip(
                    self.qnet1.parameters(),
                    self.qnet1_target.parameters(),
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.qnet2.parameters(),
                    self.qnet2_target.parameters(),
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

    def save(self, path, folder_name, eactions_valuesodes):
        import os

        file_path = op.join(path, folder_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        torch.save(
            self.policy_network.state_dict(),
            os.path.join(
                file_path, self.env_name + f"_policy_{eactions_valuesodes}_weights.pth"
            ),
        )

    def load_weights(self, path, folder_name, n_eactions_valuesodes):
        self.policy_network.load_state_dict(
            torch.load(
                op.join(
                    path,
                    "models",
                    folder_name,
                    self.env_name
                    + "_policy_"
                    + str(n_eactions_valuesodes)
                    + "_weights.pth",
                )
            )
        )

    def set_to_eval_mode(self):
        self.policy_network.eval()

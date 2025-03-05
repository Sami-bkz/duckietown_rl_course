import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from duckietown_rl_course.duckietownrl.gym_duckietown import envs
from duckietown_rl_course.duckiesim.rl.munchausen import QNetwork
import pandas as pd

from tqdm import tqdm

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "duckie_imitation"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    file_dataset: str = str(Path(__file__).resolve().parents[2])+'/duckiesim/manual/dataset/expert_data_28530.parquet'
    """Path to the dataset file (Parquet) to initialize the replay buffer"""
    env_id = "DuckietownDiscrete-v0"
    """The environment id"""
    buffer_size: int = int(1e6)
    """The size of the replay buffer"""

### Algo specific arguments
    learning_rate: float = 5e-4
    """learning rate of the model"""
    batch_size: int = 512
    """batch size"""
    batch_size_compute_value: int = 2048
    """batch size for the value computation"""
    num_epochs: int = 32
    """number of epochs"""
    inter_epoch: int = 2
    """number of inter epochs"""
    tau_soft: float = 1.0 #0.03 dans munchausen
    """temperature for the softmax"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    frac_data_per_epoch = 0.10
    """fraction of data to generate per epoch"""


class Classifier(nn.Module):
    def __init__(self, action_space, observation_space):
        super().__init__()
        self.gate = nn.Tanh()
        self.conv1 = nn.Conv2d(observation_space.shape[-1], 32, 4, stride=2)  # Modification : 3 canaux pour RGB (s)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        # compute the flatten size
        self.flatten_size = self._get_flatten_size(observation_space.shape)
        self.fc1 = nn.Linear(self.flatten_size + 1, 512) # action + state   
        self.fc2 = nn.Linear(512, 1)

    def forward(self, s, a):
        x = s.to(dtype=torch.float32)
        x = self.gate(self.conv1(x.permute(0, 3, 1, 2)/255.0))  # ReLU appliqué directement
        x = self.gate(self.conv2(x))
        x = self.gate(self.conv3(x))
        x = torch.flatten(x, start_dim=1)  # Aplatissement à partir de la dimension 1
        x = torch.cat((x, a), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

    def _get_flatten_size(self, input_shape):
        """
        Compute the size of the output of the network, given the input shape
        """
        dummy_input = torch.zeros(1, *input_shape)  # Batch size = 1
        x = self.conv1(dummy_input.permute(0, 3, 1, 2))  # ReLU appliqué directement
        x = self.conv2(x)
        x = self.conv3(x)
        return x.numel()  # Taille totale de la sortie aplatie



if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    
    # ########## TRAINING #########
    env = gym.make(args.env_id)
    df = pd.read_parquet(args.file_dataset)
    # Conversion des colonnes en numpy
    s_data = np.array(df["s"].tolist())
    s_data = s_data.reshape(len(s_data), *env.observation_space.shape)[:, None, :, :, :]
    a_data = df["a"].to_numpy()[:, None, None]
    r_data = df["r"].to_numpy()[:, None]
    d_data = df["d"].to_numpy()[:, None]
    next_s_data = np.array(df["next_s"].tolist())
    next_s_data = next_s_data.reshape(len(next_s_data), *env.observation_space.shape)[:, None, :, :, :]
    
    
    # replay buffer expert
    rb_expert = ReplayBuffer(
        len(s_data),
        env.observation_space,
        env.action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=False,
    )
    rb_expert.observations[:len(s_data)] = s_data
    rb_expert.actions[:len(a_data)] = a_data
    rb_expert.rewards[:len(r_data)] = r_data
    rb_expert.dones[:len(d_data)] = d_data
    rb_expert.next_observations[:len(next_s_data)] = next_s_data
    rb_expert.pos = len(s_data)
    
    # Initialize the networks
    q_network = QNetwork(env.action_space, env.observation_space).to(device)
    classifier = Classifier(env.action_space, env.observation_space).to(device)
    optimizer_q = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    mini_epoch = int(len(s_data)/args.batch_size)
    size_data_gen = int(len(s_data)*args.frac_data_per_epoch)
    # Training
    for epoch in range(args.num_epochs):
        total_loss_classifier = 0.0
        total_loss_agent = 0.0
        total_entropy = 0.0
        
        # generate enough data for the classifier
        # replay buffer agent 
        rb_agent = ReplayBuffer(
            size_data_gen,
            env.observation_space,
            env.action_space,
            device,
            optimize_memory_usage=False,
            handle_timeout_termination=False,
        )
        obs, _ = env.reset(seed=args.seed)
        for step in tqdm(range(size_data_gen)):
            with torch.no_grad():
                q_values = q_network(torch.Tensor(obs).to(device).unsqueeze(0))
                policy = F.softmax(q_values / args.tau_soft, dim=-1)
                actions = np.array(torch.multinomial(policy, 1).squeeze(-1).cpu().tolist())
            next_obs, rewards, terminations, truncations, infos = env.step(actions[0])
            real_next_obs = next_obs.copy()
            rb_agent.add(obs, real_next_obs, actions, rewards, terminations, infos)
            if terminations or truncations:
                obs, _ = env.reset(seed=args.seed)
            obs = next_obs
            
        # train the classifier 
        b_inds_agent = np.arange(size_data_gen) 
        b_inds_expert = np.arange(len(s_data))
        for mini_epoch in tqdm(range(args.inter_epoch)):
            np.random.shuffle(b_inds_agent)
            np.random.shuffle(b_inds_expert)
            for start in tqdm(range(0, size_data_gen, args.batch_size)):
                end = min(start + args.batch_size, size_data_gen)
                batch_idx_agent = b_inds_agent[start:end]
                batch_idx_expert = b_inds_expert[start:end]
                # mb expert data
                observations_exp = torch.Tensor(rb_expert.observations[batch_idx_expert]).to(device).squeeze(1)
                actions_exp = torch.Tensor(rb_expert.actions[batch_idx_expert]).to(device).long().squeeze(-1)
                rewards_exp = torch.Tensor(rb_expert.rewards[batch_idx_expert]).to(device)
                # next_observations_exp = torch.Tensor(rb_expert.observations[batch_idx+1]).to(device).squeeze(1)
                dones_exp = torch.Tensor(rb_expert.dones[batch_idx_expert]).to(device)
                # mb agent data
                observations_agent = torch.Tensor(rb_agent.observations[batch_idx_agent]).to(device).squeeze(1)
                actions_agent = torch.Tensor(rb_agent.actions[batch_idx_agent]).to(device).long().squeeze(-1)
                rewards_agent = torch.Tensor(rb_agent.rewards[batch_idx_agent]).to(device)
                # next_observations_agent = torch.Tensor(rb_agent.observations[batch_idx+1]).to(device).squeeze(1)
                dones_agent = torch.Tensor(rb_agent.dones[batch_idx_agent]).to(device)
                
                batch_ones = torch.ones(batch_idx_agent.size, 1).long().to(device)
                q_values = q_network(observations_agent)
                policy = F.softmax(q_values / args.tau_soft, dim=-1)
                # classifier training
                negative_obj = 0.0
                for i in range(env.action_space.n) : 
                    negative_obj += torch.log(1 - classifier(observations_agent, i*batch_ones))*torch.gather(policy.detach(), 1, i*batch_ones)
                positive_obj = torch.log(classifier(observations_exp, actions_exp))
                loss_classifier = -torch.mean(negative_obj + positive_obj)
                optimizer_classifier.zero_grad()
                loss_classifier.backward()
                optimizer_classifier.step()
                total_loss_classifier += loss_classifier.item()
            
        # compute values for agent 
        idx_mb = 0
        rewards_int = np.zeros(size_data_gen)
        for start in tqdm(range(0, size_data_gen, args.batch_size_compute_value)):
            end = min(start + args.batch_size, size_data_gen)
            rewards_int[start:end] = np.array(classifier(torch.Tensor(s_data[start:end]).to(device).squeeze(1), torch.Tensor(a_data[start:end]).to(device).long().squeeze(-1)).cpu().squeeze(-1).detach().tolist())
        
        # compute advantages and returns
        with torch.no_grad():
            advantages = torch.zeros(rb_agent.rewards.shape).to(device)
            last_advantage = 0  # on commence avec 0 puisqu'on n'a pas de values

            for t in tqdm(reversed(range(rb_agent.rewards.size))):
                if t == rb_agent.rewards.size - 1:
                    nextnonterminal = 1.0 - (terminations or truncations)
                else:
                    nextnonterminal = 1.0 - rb_agent.dones[t + 1][0]
                    
                advantages[t] = rewards_int[t] + args.gamma * args.gae_lambda * nextnonterminal * last_advantage
                last_advantage = advantages[t]

            returns = advantages  # returns est directement égal aux advantages puisqu'il n'y a pas de values

            
        b_inds = np.arange(size_data_gen)
        old_q_network = QNetwork(env.action_space, env.observation_space).to(device)
        old_q_network.load_state_dict(q_network.state_dict())
        for mini_epoch in tqdm(range(args.inter_epoch)):
            np.random.shuffle(b_inds)
            for start in tqdm(range(0, size_data_gen, args.batch_size)):
                end = min(start + args.batch_size, size_data_gen)
                batch_idx = b_inds[start:end]
                # mb agent data
                observations_agent = torch.Tensor(rb_agent.observations[batch_idx]).to(device).squeeze(1)
                actions_agent = torch.Tensor(rb_agent.actions[batch_idx]).to(device).long().squeeze(-1)
                logpolicy = torch.gather(F.log_softmax(q_network(observations_agent) / args.tau_soft, dim=-1), 1, actions_agent)
                old_logpolicy = torch.gather(F.log_softmax(old_q_network(observations_agent) / args.tau_soft, dim=-1), 1, actions_agent)
                logratio = logpolicy - old_logpolicy.detach()
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = advantages[torch.Tensor(batch_idx).long()].to(device)
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    
                entropy = -(F.softmax(q_network(observations_agent) / args.tau_soft, dim=-1) * F.log_softmax(q_network(observations_agent) / args.tau_soft, dim=-1)).sum(1).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss 
                optimizer_q.zero_grad()
                loss.backward()
                optimizer_q.step()
                total_loss_agent += loss.item()
                total_entropy += -entropy_loss.item()
        
        # log 
        print(f"Epoch {epoch} : loss_agent = {total_loss_agent}, loss_classifier = {total_loss_classifier}, entropy = {total_entropy}")
        
        if args.track:
            wandb.log({
                "loss_agent": total_loss_agent,
                "loss_classifier": total_loss_classifier,
                "entropy": total_entropy
            })
            
            
    # Save the model
    model_path = f"models_imitation/{args.exp_name}"
    # create path if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(q_network.state_dict(), model_path)
    print(f"model saved to {model_path}")

        
        
        
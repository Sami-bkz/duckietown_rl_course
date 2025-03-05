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
    file_dataset: str = str(Path(__file__).resolve().parents[2])+'/duckiesim/manual/dataset/expert_data_36591.parquet'
    """Path to the dataset file (Parquet) to initialize the replay buffer"""
    env_id = "DuckietownDiscrete-v0"
    """The environment id"""
    
### Algo specific arguments
    learning_rate: float = 1e-4
    """learning rate of the model"""
    batch_size: int = 256
    """batch size"""
    num_epochs: int = 8
    """number of epochs"""
    tau_soft: float = 1.0 #0.03 dans munchausen
    """temperature for the softmax"""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    min_counts: int = 500







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
    
    # Calculons le nombre d'occurrences de chaque action
    action_counts = np.bincount(a_data[:, 0, 0].astype(int), minlength=9)

    # Sélectionnons les actions qui sont représentées au moins args.min_counts fois
    selected_actions = np.where(action_counts >= args.min_counts)[0]

    # Calculons le nombre maximum d'occurrences pour les actions sélectionnées
    max_count = np.min(action_counts[selected_actions])

    # Effectuons le sous-échantillonnage des actions sélectionnées
    new_s_data = []
    new_a_data = []
    new_r_data = []
    new_d_data = []
    new_next_s_data = []

    for i in selected_actions:
        indices = np.where(a_data[:, 0, 0].astype(int) == i)[0]
        indices = np.random.choice(indices, size=min(len(indices), max_count), replace=False)
        new_s_data.append(s_data[indices])
        new_a_data.append(a_data[indices])
        new_r_data.append(r_data[indices])
        new_d_data.append(d_data[indices])
        new_next_s_data.append(next_s_data[indices])

    # Concaténons les données sous-échantillonnées
    new_s_data = np.concatenate(new_s_data, axis=0)
    new_a_data = np.concatenate(new_a_data, axis=0)
    new_r_data = np.concatenate(new_r_data, axis=0)
    new_d_data = np.concatenate(new_d_data, axis=0)
    new_next_s_data = np.concatenate(new_next_s_data, axis=0)

    # Mettons à jour le dataset
    rb_expert.observations[:len(new_s_data)] = new_s_data
    rb_expert.actions[:len(new_a_data)] = new_a_data
    rb_expert.rewards[:len(new_r_data)] = new_r_data
    rb_expert.dones[:len(new_d_data)] = new_d_data
    rb_expert.next_observations[:len(new_next_s_data)] = new_next_s_data
    rb_expert.pos = len(new_s_data)
    print(f"Expert data size: {len(new_s_data)}")
    # Initialize the networks
    q_network = QNetwork(env.action_space, env.observation_space).to(device)
    optimizer_q = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    # Training
    b_inds = np.arange(len(new_s_data)) 
    for epoch in tqdm(range(args.num_epochs)):
        total_behavioral_loss = 0
        total_entropy = 0
        np.random.shuffle(b_inds)
        for start in tqdm(range(0, len(new_s_data), args.batch_size)):
            end = min(start + args.batch_size, len(new_s_data))
            batch_idx = b_inds[start:end]
            s = torch.tensor(new_s_data[batch_idx], device=device, dtype=torch.float32).squeeze(dim=1)
            a = torch.tensor(new_a_data[batch_idx], device=device, dtype=torch.long)
            r = torch.tensor(new_r_data[batch_idx], device=device, dtype=torch.float32)
            d = torch.tensor(new_d_data[batch_idx], device=device, dtype=torch.float32)
            next_s = torch.tensor(next_s_data[batch_idx], device=device, dtype=torch.float32)
            
            
            # behavioral loss 
            policy =  F.softmax(q_network(s) / args.tau_soft, dim=-1)
            labels = torch.zeros_like(policy)
            labels[torch.arange(len(labels)), a.squeeze()] = 1
            binary_loss = -torch.mean( torch.sum(labels * torch.log(policy), dim=-1))
            entropy = -torch.mean(torch.sum(policy * torch.log(policy), dim=-1))
            behavioral_loss = binary_loss - args.ent_coef * entropy
            optimizer_q.zero_grad()
            behavioral_loss.backward()
            optimizer_q.step()
            total_behavioral_loss += binary_loss.item()
            total_entropy += entropy.item()
            
        # log 
        print(f"Epoch {epoch} : loss = {total_behavioral_loss}, entropy = {total_entropy}")
        
        if args.track:
            wandb.log({
                "behavioral_loss": total_behavioral_loss,
                "entropy": total_entropy
            })
            
            
    # Save the model
    model_path = f"models_imitation/{args.exp_name}.pt"
    # create path if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(q_network.state_dict(), model_path)
    print(f"model saved to {model_path}")

        
        
        
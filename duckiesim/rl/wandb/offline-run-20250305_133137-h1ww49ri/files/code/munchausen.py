# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51_ataripy
import os
import random
import time
from dataclasses import dataclass
import plotext as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
import pandas as pd
from duckiesim.rl.process_data_with_reward import process_dataset
# from stable_baselines3.common.atari_wrappers import (
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )
from stable_baselines3.common.buffers import ReplayBuffer
from duckietownrl.gym_duckietown import envs
from duckiesim.rl.rl_utils import evaluate
import wandb
import plotext


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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "DuckieTownRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "DuckietownDiscrete-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 4
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10_000
    """timestep to start learning"""
    train_frequency: int = 8
    """the frequency of training"""

    # Munchausen specific arguments
    tau_soft: float = 0.3 #0.03
    """the temperature parameter for the soft-max policy as well as entropy regularization : tau = lambda_kl + lambda_entropy"""
    alpha: float = 1./3 # 0.9
    """the ppo term weight : alpha = lambda_kl / (lambda_kl + lambda_entropy)"""
    l_0: float = -1.0
    """the lower bound of the weighted log probability"""
    epsilon_tar: float = 1e-6
    """the epsilon term for numerical stability"""
    polyak: float = 0.995
    """the polyak averaging coefficient"""

    # Dataset argument
    load_dataset : bool = True
    """Whether to load a dataset from a file"""
    file_dataset: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/manual/dataset/expert_data_36591.parquet'
    """Path to the dataset file (Parquet) to initialize the replay buffer"""
    save_dataset: bool = False
    """Whether to save the dataset to a file"""
    file_name: str = "training_data"
    """Name of the Parquet file to save the dataset"""
    
    # plotext debbug 
    plotext: bool = True
    """if toggled, plotext will be enabled"""
    per_data: float = 1.0
    """Percentage of data to save in the dataset"""



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        # env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

def load_dataset_to_buffer(file_path, replay_buffer, observation_space, env_id):
    """Charge le dataset depuis un fichier et initialise le ReplayBuffer."""
    df = pd.read_parquet(file_path)

    # Conversion des colonnes en numpy
    s_data = np.array(df["s"].tolist())
    s_data = s_data.reshape(len(s_data), *observation_space.shape)[:, None, :, :, :]
    a_data = df["a"].to_numpy()[:, None, None]
    r_data = df["r"].to_numpy()[:, None]
    d_data = df["d"].to_numpy()[:, None]
    next_s_data = np.array(df["next_s"].tolist())
    next_s_data = next_s_data.reshape(len(next_s_data), *observation_space.shape)[:, None, :, :, :]
    
    # process reward function
    s_data, a_data, r_data, d_data, next_s_data = process_dataset(s_data, a_data, r_data, d_data, next_s_data, gym.make(env_id))
    
    replay_buffer.observations[:len(s_data)] = s_data
    replay_buffer.actions[:len(a_data)] = a_data
    replay_buffer.rewards[:len(r_data)] = r_data
    replay_buffer.dones[:len(d_data)] = d_data
    replay_buffer.next_observations[:len(next_s_data)] = next_s_data
    replay_buffer.pos = len(s_data)
    return replay_buffer

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, action_space, observation_space):
        super().__init__()
        self.gate = nn.Tanh()
        self.conv1 = nn.Conv2d(observation_space.shape[-1], 32, 4, stride=2)  # Modification : 3 canaux pour RGB
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        # compute the flatten size
        self.flatten_size = self._get_flatten_size(observation_space.shape)
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, action_space.n)

    def forward(self, x):
        x = x.to(dtype=torch.float32).permute(0, 3, 1, 2)/255.0
        x = self.gate(self.conv1(x))  # ReLU appliqué directement
        x = self.gate(self.conv2(x))
        x = self.gate(self.conv3(x))
        x = torch.flatten(x, start_dim=1)  # Aplatissement à partir de la dimension 1
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_flatten_size(self, input_shape):
        """
        Compute the size of the output of the network, given the input shape
        """
        dummy_input = torch.zeros(1, *input_shape)  # Batch size = 1
        x = self.conv1(dummy_input.permute(0, 3, 1, 2))  # ReLU appliqué directement
        x = self.conv2(x)
        x = self.conv3(x)
        return x.numel()  # Taille totale de la sortie aplatie


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs.single_action_space, envs.single_observation_space).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs.single_action_space, envs.single_observation_space).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=False,
    )
    # Charge le dataset si l'argument file_dataset est défini
    if args.load_dataset:
        print(f"Chargement du dataset depuis : {args.file_dataset}")
        rb = load_dataset_to_buffer(args.file_dataset, rb, envs.single_observation_space, args.env_id)
        
        
    # MODEL PATH 
    # Vérifie si le répertoire 'model' existe, sinon le crée
    if not os.path.exists('model'):
        os.makedirs('model')
    # Récupère la liste des expériences existantes
    exps = [d for d in os.listdir('model') if d.startswith('exp_')]
    # Si aucun dossier exp_ n'existe, on commence à exp_0
    if not exps:
        run_name = 'exp_0'
    else:
        # Sinon, on récupère le numéro de l'expérience la plus élevée
        last_exp_num = max(int(d.split('_')[1]) for d in exps)
        # On incrémente le numéro de l'expérience pour obtenir le nouveau nom
        run_name = f'exp_{last_exp_num + 1}'

    # Utilise le nom d'expérience pour sauvegarder le modèle
    model_path = f"model/{run_name}"
    # Crée le répertoire si il n'existe pas
    os.makedirs(model_path, exist_ok=True)
    
    
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    actions_name = [f"a{i}" for i in range(envs.single_action_space.n)]
    stats_actions= np.zeros(envs.single_action_space.n)
    obs, _ = envs.reset(seed=args.seed)
    best_eval_score = -1e9
    episodic_return = 0.0
    best_episodic_return = -1e9
    for global_step in range(1,args.total_timesteps+1):
         # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            with torch.no_grad():
                q_values = q_network(torch.Tensor(obs).to(device))
                policy = F.softmax(q_values / args.tau_soft, dim=-1)
                actions = np.array(torch.multinomial(policy, 1).squeeze(-1).cpu().tolist())
        # add plotext
        stats_actions[actions[0]] += 1

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for termination in terminations:
            if termination:
                    wandb.log({"charts/episodic_return": infos["episodic_return"], "charts/episodic_length": infos["episodic_length"]}, step=global_step) if args.track else None
                    episodic_return = infos["episodic_return"]
                    print(f"Episodic return: {infos['episodic_return']}, Episodic length: {infos['episodic_length']}")
                    # plotext debug
                    percentages = stats_actions / stats_actions.sum() * 100
                    # print(f"Step: {global_step}, Actions: {stats_actions}, Percentages: {percentages}")
                    stats_actions = np.zeros(envs.single_action_space.n)
                    plt.simple_bar(actions_name, percentages, width = 100, title = 'Actions', color = 'cyan')
                    plt.show()

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                batch_idx = np.random.randint(0, rb.pos, args.batch_size) if not rb.full else np.random.randint(0, rb.buffer_size, args.batch_size)
                observations = torch.Tensor(rb.observations[batch_idx]).to(device).squeeze(1)
                actions = torch.Tensor(rb.actions[batch_idx]).to(device).long().squeeze(-1)
                rewards = torch.Tensor(rb.rewards[batch_idx]).to(device)
                next_observations = torch.Tensor(rb.observations[batch_idx+1]).to(device).squeeze(1)
                dones = torch.Tensor(rb.dones[batch_idx]).to(device)
                with torch.no_grad():
                    target_q_values = target_network(observations)
                    target_policy = F.softmax(target_q_values / args.tau_soft, dim=-1)
                    target_next_q_values = target_network(next_observations)
                    target_next_policy = F.softmax(target_next_q_values / args.tau_soft, dim=-1)
                    red_term = args.alpha * (args.tau_soft * torch.log(target_policy.gather(1, actions)) + args.epsilon_tar).clamp(args.l_0, 0.0)  
                    blue_term = -args.tau_soft * torch.log(target_next_policy + args.epsilon_tar)
                    munchausen_target = (rewards + red_term + args.gamma * (1 - dones)* (target_next_policy * (target_next_q_values + blue_term)).sum(dim=-1).unsqueeze(-1))
                    td_target = munchausen_target.squeeze()
                old_val = q_network(observations).gather(1, actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    # writer.add_scalar("losses/td_loss", loss, global_step)
                    # writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    wandb.log({"losses/td_loss": loss, 
                               "losses/q_values": old_val.mean().item(), 
                               "losses/entropy": red_term.mean().item(),
                               "charts/mean_reward_batch": rewards.mean().item(),
                               "charts/SPS": int(global_step / (time.time() - start_time)),
                               }, step=global_step) if args.track else None

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update target network
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        (1-args.polyak) * q_network_param.data + args.polyak * target_network_param.data
                    )
        if episodic_return > best_episodic_return or global_step % 10_000 == 0:
            best_episodic_return = episodic_return
            eval_score = evaluate(q_network, args.env_id, seed=args.seed, tau=args.tau_soft)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                if args.save_model:
                    torch.save(q_network.state_dict(), model_path+f"/{args.exp_name}_{global_step}_{eval_score}.pt")
                    print(f"model saved to {model_path}")

            print(f"Step: {global_step}, Eval score: {eval_score}")

        if args.save_dataset and global_step % 50_000 == 0:
            output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset_training")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Le dossier '{output_dir}' a été créé.")

            max_size = rb.observations.shape[0] if rb.full else rb.pos
            subset_size = int(max_size * args.per_data)
            selected_indices = np.random.choice(max_size, subset_size, replace=False)
            observations = rb.observations[selected_indices].reshape(subset_size, -1).tolist()
            next_observations = rb.next_observations[selected_indices].reshape(subset_size, -1).tolist()
            actions = rb.actions[selected_indices].reshape(subset_size, -1).tolist()
            rewards = rb.rewards[selected_indices].flatten().tolist()
            dones = rb.dones[selected_indices].flatten().tolist()
            data = {
                "s": observations,
                "a": actions,
                "r": rewards,
                "d": dones,
                "next_s": next_observations
            }
            df = pd.DataFrame(data)
            output_file = os.path.join(output_dir, args.file_name+'_'+str(max_size)+".parquet")
            df.to_parquet(output_file, engine="pyarrow", index=False)
   

    envs.close()

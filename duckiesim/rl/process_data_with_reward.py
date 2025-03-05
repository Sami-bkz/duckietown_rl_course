from stable_baselines3.common.buffers import ReplayBuffer
from duckietown_rl_course.duckiesim.rl.custom_reward_function import compute_custom_reward
from duckietown_rl_course.duckietownrl.gym_duckietown import envs
from duckietown_rl_course.duckietownrl.gym_duckietown.simulator import REWARD_INVALID_POSE
import pandas as pd
from tqdm import tqdm
import os
from dataclasses import dataclass
from pathlib import Path
import gymnasium as gym
import numpy as np
import tyro

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    file_dataset: str = str(Path(__file__).resolve().parents[2])+'/duckiesim/manual/dataset/expert_data_36591.parquet'
    """Path to the dataset file (Parquet) to initialize the replay buffer"""
    env_id = "DuckietownDiscrete-v0"
    """The environment id"""
    
    
def process_dataset(s_data, a_data, r_data, d_data, next_s_data, env):
    for i in tqdm(range(len(s_data))):
        obs = s_data[i, 0, : , : , -3:]
        a = a_data[i].item()
        d = d_data[i]
        reward = compute_custom_reward(obs, env.unwrapped.actions[a]) if not d else r_data[i]
        r_data[i] = reward if not (d and reward<=-10.0) else REWARD_INVALID_POSE
    return s_data, a_data, r_data, d_data, next_s_data
    

if __name__ == "__main__":
    args = tyro.cli(Args)
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

    for i in tqdm(range(len(s_data))):
        obs = s_data[i, 0, : , : , -3:]
        a = a_data[i].item()
        d = d_data[i]
        reward = compute_custom_reward(obs, env.unwrapped.actions[a]) if not d else r_data[i]
        r_data[i] = reward
        
    
    


#!/usr/bin/env python3
import argparse
import torch
from datetime import datetime
import os
import time

import gym
import pickle
import numpy as np
import pyglet
from pyglet.window import key  # do not remove, otherwhise render issue


from duckietownrl.utils.utils import (
    ReplayBuffer,
    load_replay_buffer,
    parse_arguments_from_ini,
    read_file_if_modified,
    add_environment,
)


from duckietownrl.algorithms.sac_new_2dconv import SAC

import wandb
import os.path as op


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="duckietownrl-sac-conv2d",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "architecture": "CNN",
        "dataset": "duckietown-simulator",
        "epochs": 0,
    },
)


file_config_path = op.join(__file__[: -len("parallel_training.py")], "config.ini")
args = parse_arguments_from_ini(file_config_path)
last_mod_time = os.path.getmtime(file_config_path)


n_frames = args["n_frames"]
n_chans = args["n_chans"]  # 1 for B/W images 3 for RGBs
n_envs = args["n_envs"]
resize_shape = (args["width_frame"], args["height_frame"])  # (width,height)"]
envs = []
k = 0
for i in range(n_envs):
    add_environment(envs, args, k)
    k += 1

device = "cuda"

# assemble first obervation
l_obs = []
obs, _, _, _ = envs[0].step(np.array([0, 0], dtype=np.float32))

for _ in envs:
    l_obs.append(np.asarray(obs))
obs = np.stack(l_obs, axis=0)

# create replay buffer
batch_size = args["batch_size"]
state_dim = (n_frames * n_chans, *resize_shape)  # Shape of state input (4, 84, 84)
action_dim = 2
replay_buffer = ReplayBuffer(
    args["replay_buffer_size"],
    batch_size,
    state_dim,
    action_dim,
    normalize_rewards=False,
    device=device,
)
# replay_buffer = load_replay_buffer()

# define an agent
agent = SAC(
    "DuckieTown",
    state_dim,  # envs[0].observation_space.shape[:3],
    envs[0].action_space.shape[0],
    replay_buffer=replay_buffer,
    device=device,
    actor_lr=args["actor_lr"],
    critic_lr=args["critic_lr"],
    tau=args["tau"],
    alpha=args["alpha"],
)

tot_episodes = 0
timesteps = 0
probability_training = 1.0
save_on_episodes = args["save_on_episode"]
running_avg_reward = 0

folder_name = os.path.join("models", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
path = args["path"]

eps_returns = np.zeros(n_envs)
once = True
collect_random_timesteps = args["collect_random_steps"]


def update(dt):
    global obs
    global tot_episodes
    global timesteps
    global probability_training
    global running_avg_reward
    global eps_returns
    global once
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    next_observations = []
    rewards = []
    dones = []

    if timesteps < collect_random_timesteps:
        action = 2 * np.random.rand(n_envs, 2) - 1

    else:
        action, entropy = agent.select_action(
            torch.tensor(obs, dtype=torch.float32).to(device)
        )
        wandb.log({"entropy": entropy})

    for i, env in enumerate(envs):
        next_obs, reward, done, info = env.step(action[i])
        next_observations.append(next_obs)
        rewards.append(reward)
        dones.append(done)
        wandb.log(
            {
                "actor_loss": agent.actor_loss_value,
                "q_loss": agent.q_loss_value,
                "speed": env.speed,
                "lp.dot_dir": env.lp.dot_dir,
                "lp.dist": env.lp.dist,
            }
        )

    next_obs = np.stack(next_observations, axis=0)
    reward = np.stack(rewards, axis=0)
    done = np.stack(dones, axis=0)

    next_observations.clear()
    rewards.clear()
    dones.clear()

    avg_reward = reward.sum(0) / n_envs
    running_avg_reward += (avg_reward - running_avg_reward) / (timesteps + 1)
    print(
        f"eps = {tot_episodes} step_count = {timesteps}, avg_reward={avg_reward:.3f}, runn_avg_reward={running_avg_reward:.3f}"
    )

    wandb.log(
        {
            "avg_reward": avg_reward,
            "runn_avg_reward": running_avg_reward,
        }
    )

    eps_returns += reward
    replay_buffer.add(obs, next_obs, action, reward, done)

    # Train with a certain probability for computing efficiency
    if (
        np.random.random() < probability_training
        and timesteps >= collect_random_timesteps
    ):
        entropy = agent.train(timesteps, device)

    obs = next_obs

    if tot_episodes > 0 and tot_episodes % save_on_episodes == 0 and once:
        agent.save(path, folder_name, tot_episodes)
        once = False

    if True in done:
        idx = np.where(done)[0]
        idx_envs = idx % n_envs
        for id in idx_envs:
            env = envs[id]
            print(f"env N.{id} done!")
            wandb.log({"ep_return": eps_returns[id], "step_count": env.step_count})
            obs_env = env.reset()
            env.render()
            obs[id] = np.stack(obs_env[0], axis=0)

            # env.render()
            tot_episodes += 1
            eps_returns[id] = 0.0
            once = True

    for env in envs[-4:]:
        env.render(mode="human")

    timesteps += 1


dt = 0.001
t = time.time()
while True:
    last_mod_time, args, has_changed = read_file_if_modified(
        args, file_config_path, last_mod_time
    )
    if has_changed:

        if agent.replay_buffer.batch_size != args["batch_size"]:
            agent.replay_buffer.change_bacth_size(args["batch_size"])
            agent.replay_buffer.batch_size = args["batch_size"]
            print(f"new batch_size {args['batch_size']}")

        optimizer = agent.actor_optimizer
        if agent.actor_lr != args["actor_lr"]:
            for param_group in optimizer.param_groups:
                # Set new learning rate
                param_group["lr"] = args["actor_lr"]
            agent.actor_lr = args["actor_lr"]
            print(f"new actor_lr {args['actor_lr']}")

        optimizer = agent.q_optimizer
        if agent.critic_lr != args["critic_lr"]:
            for param_group in optimizer.param_groups:
                # Set new learning rate
                param_group["lr"] = args["critic_lr"]
            agent.critic_lr = args["critic_lr"]
            print(f"new critic_lr {args['critic_lr']}")

        if agent.tau != args["tau"]:
            agent.tau = args["tau"]
            print(f"new tau {args['tau']}")

        if agent.alpha != args["alpha"]:
            agent.alpha = args["alpha"]
            print(f"new alpha {args['alpha']}")

        if collect_random_timesteps != args["collect_random_steps"]:
            collect_random_timesteps = args["collect_random_steps"]
            print(f"new collect_random_steps {args['collect_random_steps']}")

        if save_on_episodes != args["save_on_episode"]:
            save_on_episodes = args["save_on_episode"]
            print(f"new save_on_episode {args['save_on_episode']}")

    update(dt)

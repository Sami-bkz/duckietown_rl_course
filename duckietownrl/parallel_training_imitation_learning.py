#!/usr/bin/env python3
import argparse
from sympy import false
import torch
from datetime import datetime
import os
import time

import gym
import pickle
import numpy as np
import pyglet
from pyglet.window import key  # do not remove, otherwhise render issue


from duckietownrl.gym_duckietown.envs import DuckietownEnv
from duckietownrl.utils.utils import (
    ReplayBuffer,
    load_replay_buffer,
    saturate_replay_buffer,
)
from duckietownrl.utils.wrappers import (
    Wrapper_BW,
    Wrapper_NormalizeImage,
    Wrapper_Resize,
    Wrapper_StackObservation,
    Wrapper_YellowWhiteMask,
)

from duckietownrl.algorithms.sac_new_2dconv import SAC

import wandb

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

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="small_loop")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument(
    "--draw-curve", action="store_true", help="draw the lane following curve"
)
parser.add_argument(
    "--domain-rand", action="store_true", help="enable domain randomization"
)
parser.add_argument(
    "--return_masked_obs",
    action="store_true",
    default=False,
    help="to use yellow and white wrapper",
)

parser.add_argument(
    "--max_steps", default=600, help="number of steps per episode ", type=int
)

parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--replay_buffer_size", default=50_000, type=int)
parser.add_argument("--save_on_episode", default=100, type=int)
parser.add_argument("--width_frame", default=28, type=int)
parser.add_argument("--height_frame", default=28, type=int)
parser.add_argument("--width_preview", default=800, type=int)
parser.add_argument("--height_preview", default=600, type=int)

parser.add_argument("--n_chans", default=3, type=int)
parser.add_argument("--n_frames", default=3, type=int)
parser.add_argument("--n_envs", default=1, type=int)
parser.add_argument("--tau", default=0.001, type=float)
parser.add_argument("--reward_invalid_pose", default=-77, type=int)
parser.add_argument("--alpha", default=0.05, type=float)
parser.add_argument("--collect_random_steps", default=1000, type=int)


args = parser.parse_args()

yellow_mask = True
n_frames = args.n_frames
n_chans = args.n_chans
n_envs = args.n_envs
resize_shape = (args.width_frame, args.height_frame)  # (width,height)
envs = []
k = 0
for i in range(n_envs):
    print(f"creating env N.{i}...")
    env = DuckietownEnv(
        map_name=args.map_name,
        distortion=args.distortion,
        domain_rand=args.domain_rand,
        max_steps=args.max_steps,
        seed=args.seed + k,
        window_width=args.width_preview,
        window_height=args.height_preview,
        camera_width=resize_shape[0],
        camera_height=resize_shape[1],
        reward_invalid_pose=args.reward_invalid_pose,
        user_tile_start=(2, 0),
        start_pose=(0.34220727, 0, 0.58371305),
        start_angle=np.pi / 2,
    )
    k += 1
    # wrapping the environment
    env = Wrapper_StackObservation(env, n_frames, n_chans=n_chans)
    # env.append_wrapper(Wrapper_Resize(env, shape=resize_shape))
    if n_chans == 1:
        env.append_wrapper(Wrapper_BW(env))
    env.append_wrapper(Wrapper_NormalizeImage(env))
    if yellow_mask:
        return_mask = args.return_masked_obs
        env.append_wrapper(Wrapper_YellowWhiteMask(env, return_mask))
        if return_mask is False:
            n_chans += 3
            env.n_chans += 3

    env.reset()
    env.render(mode="rgb_array")
    envs.append(env)

env.render(mode="human")

device = "cuda"

# assemble first obervation
l_obs = []
obs, _, _, _ = env.step(np.array([0, 0], dtype=np.float32))

for _ in envs:
    l_obs.append(obs)
obs = np.stack(l_obs, axis=0)

# create replay buffer
batch_size = args.batch_size
state_dim = (
    n_frames * n_chans,
    *resize_shape,
)  # Shape of state input (4, 84, 84)
action_dim = 2
if os.path.isfile("replay_buffer_not"):
    replay_buffer = load_replay_buffer()
    saturate_replay_buffer(replay_buffer)
else:
    replay_buffer = ReplayBuffer(
        args.replay_buffer_size,
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
    actor_lr=0.001,
    tau=args.tau,
)

# # Load model
# folder_name = "20241108_172952"
# fl = "6400"
# path = "/media/g.ferraro/DONNEES"
# agent.load_weights(path, folder_name, fl)

tot_episodes = 0
timesteps = 0
probability_training = 1.0
save_on_episodes = args.save_on_episode
running_avg_reward = 0

folder_name = os.path.join("models", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
path = "/media/g.ferraro/DONNEES"

eps_returns = np.zeros(n_envs)


def action_from_joystick():
    x = round(joystick.y, 2)
    z = round(joystick.x, 2)
    y = round(joystick.z, 2)

    action = np.array([-x, -z])
    # Giuseppe's modification for going forward while turning
    # if action[0] == 0 and (action[1] == 1 or action[1] == -1):
    #     action[0] = 0.2
    #     action[1] *= 1.5
    # print([x, y, z])
    return np.expand_dims(action, axis=0)


stop_collecting = False
import sys


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global stop_collecting

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render("rgb_array")
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
        env.render("rgb_array")
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    elif symbol == key.J:
        stop_collecting = not stop_collecting


once = True
collect_random_timesteps = args.collect_random_steps


def update(dt):
    global obs
    global tot_episodes
    global timesteps
    global probability_training
    global running_avg_reward
    global eps_returns
    global stop_collecting
    global once
    global collect_random_timesteps
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    next_observations = []
    rewards = []
    dones = []

    if stop_collecting is not True:
        # No actions took place
        if abs(round(joystick.x, 2)) <= 0.1 and abs(round(joystick.y, 2)) <= 0.1:
            return
        else:
            action = action_from_joystick()

    else:
        if timesteps < collect_random_timesteps:
            action = 2 * torch.rand(n_envs, 2) - 1
        else:
            action, entropy = agent.select_action(
                torch.tensor(obs, dtype=torch.float32).to(device)
            )
            wandb.log({"entropy": entropy})

    # noise = np.random.randint(-300, 300, (n_envs, 2)) * 0.0001
    # noisy_action = action + noise

    # add noise to actions
    for i, env in enumerate(envs):
        # next_obs, reward, done, info = env.step(noisy_action[i])
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
    # action = noisy_action

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
            env.render("rgb_array")
            obs[id] = np.stack(obs_env[0], axis=0)

            # env.render("rgb_array")
            tot_episodes += 1
            eps_returns[id] = 0.0
            once = True

    for env in envs[-1:]:
        env.render(mode="human")

    timesteps += 1


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Registers joysticks and recording controls
joysticks = pyglet.input.get_joysticks()
assert joysticks, "No joystick device is connected"
joystick = joysticks[0]
joystick.open()
# joystick.push_handlers(on_joybutton_press)

# Enter main event loop
pyglet.app.run()


dt = 0.001
t = time.time()
while True:
    # if time.time() - t > dt:
    update(dt)
    # t = time.time()

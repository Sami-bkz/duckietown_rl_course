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


from duckietownrl.gym_duckietown.envs import DuckietownEnv
from duckietownrl.utils.utils import ReplayBuffer
from duckietownrl.utils.wrappers import (
    Wrapper_BW,
    Wrapper_NormalizeImage,
    Wrapper_Resize,
    Wrapper_StackObservation,
)

from duckietownrl.algorithms.sac_3dconv import SAC

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
    "--max-steps", action="store_true", default=1500, help="number of steps per episode"
)

args = parser.parse_args()


def load_replay_buffer(filename="replay_buffer"):
    print("loading replay buffer...")
    file = open(filename, "rb")
    replay_buffer = pickle.load(file)
    file.close()
    print("Done")
    print(len(replay_buffer))
    return replay_buffer


n_frames = 4
resize_shape = (32, 24)  # (width,height)
envs = []
env = DuckietownEnv(
    map_name=args.map_name,
    distortion=args.distortion,
    domain_rand=args.domain_rand,
    max_steps=args.max_steps,
    seed=args.seed,
)

# wrapping the environment
env = Wrapper_StackObservation(env, n_frames)
env.append_wrapper(Wrapper_Resize(env, shape=resize_shape))
env.append_wrapper(Wrapper_BW(env))
env.append_wrapper(Wrapper_NormalizeImage(env))

env.reset()
env.render(mode="rgb_array")

# initialize stack
# initialize stack
for _ in range(n_frames):
    obs, _, _, _ = env.step([0, 0])

# create replay buffer
batch_size = 256

# define an agent
state_dim = (n_frames, *resize_shape)  # Shape of state input (4, 84, 84)
action_dim = 2
agent = SAC(
    "DuckieTown",
    state_dim,  # envs[0].observation_space.shape[:3],
    env.action_space.shape[0],
    replay_buffer=None,
)
# set the agent in evaluate mode
agent.set_to_eval_mode()

folder_name = "20241018_112848"
path = "/media/g.ferraro/DONNEES"

# load model
agent.load_weights(path, folder_name, 3)

tot_episodes = 0
timesteps = 0


def update(dt):
    global obs
    global tot_episodes
    global timesteps
    global probability_training
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = agent.select_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
    next_obs, reward, done, info = env.step(action[0])

    print(f"step_count = {env.unwrapped.step_count}, rewards={sum(reward)}")

    obs = next_obs
    env.render(mode="human")

    if True in done:
        obs, _, _, _ = env.reset()
        env.render()
        tot_episodes += 1

    timesteps += 1


dt = 0.01
t = time.time()
while True:
    if time.time() - t > dt:
        update(dt)
        t = time.time()

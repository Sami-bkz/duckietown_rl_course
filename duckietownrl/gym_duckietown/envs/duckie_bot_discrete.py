from gymnasium.spaces import Discrete, Box
import gymnasium as gym
from enum import Enum
import os
import time
import random
import socket
import curses
import numpy as np
from duckietownrl.gym_duckietown.api import DuckiebotAPI


class DuckieBotDiscrete(gym.Env, DuckiebotAPI):

    """
    DuckieBot environment with discrete actions.
    """

    class Actions(Enum):
        FORWARD = 0
        BACKWARD = 1
        LEFT = 2
        RIGHT = 3


    def __init__(self, **params):
        """
        Instantiate a discrete action environment.
        Args:
            robot_name (str): The name of the robot you are using.
            fixed_linear_velocity (float): The linear velocity that will be used for linear actions (forward, backward).
                The velocity used for the backward action is - fixed_linear_velocity.
                Default = 0.1.
            fixed_angular_velocity (float): The angular velocity that will be used for angular actions (left, right).
                The velocity used for the left action is - fixed_angular_velocity.
                Default = 0.1.
            action_duration (float): The duration of the action in seconds.
                Default = 1.
            stochasticity (float): The probability to take a discrete action. If a different actions is chosen, the
                action actually taken is chosen uniformly among the remaining actions.
                Default = 0.
        """
        print("  > Initializing environment... ")
        DuckiebotAPI.__init__(self, **params)
        gym.Env.__init__(self)
        self.observation_space = Box(low=0, high=255, shape=(100, 200, 3), dtype=np.uint8)  # Images observation space
        self.action_space = Discrete(4)     # Action space with four possible actions (from 0 to 3 included)

        self.fixed_linear_velocity: float = params.get("fixed_linear_velocity", 0.25)
        self.fixed_angular_velocity: float = params.get("fixed_angular_velocity", 0.12)
        self.action_duration: float = params.get("action_duration", 0.3)
        self.stochasticity: float = params.get("stochasticity", 0.0)      # Probability to take a different action,

        print("  > Environment initialized.")

    def step(self, action):

        assert self.action_space.contains(action)
        original_action = action
        if self.stochasticity > 0.0 and random.random() < self.stochasticity:
            available_actions = list(set(range(self.action_space.n)) - {int(action)})
            action = random.choice(available_actions)

        # print(f"Action chosen: {original_action} -> {action} (after stochasticity)")  # Debugging log
        if isinstance(action, np.ndarray):
            action = int(action)
        if action == self.Actions.FORWARD.value:
            self.apply_action(linear_velocity=self.fixed_linear_velocity)
        elif action == self.Actions.BACKWARD.value:
            self.apply_action(linear_velocity=-self.fixed_linear_velocity)
        elif action == self.Actions.LEFT.value:
            self.apply_action(angular_velocity=self.fixed_angular_velocity)
        elif action == self.Actions.RIGHT.value:
            self.apply_action(angular_velocity=-self.fixed_angular_velocity)

        # Now the action is performed, return the observation
        # NB: Because you have to design your own reward, the reward isn't computed here.
        # Compute it after the call to this function.
        observation = self.get_observation()

        print("Observation:")
        print(" > Type:", type(observation))
        print(" > dtype:", observation.dtype)
        print(" > shape:", observation.shape)
        reward = 0.0
        done = False
        trunc = False
        info = {}
        return observation, reward, done, trunc, info

    def reset(self, seed = 0, option = None):
        print("    ########################################    ")
        print("    ###      RESET FUNCTION CALLED.      ###    ")
        print("    ###  Pick up the robot and place it  ###    ")
        print("    ###  in a valid initial position     ###    ")
        print("    ###  and then press any key to       ###    ")
        print("    ###  continue.                       ###    ")
        print("    ########################################    ")
        input("Press any key to continue ...")
        time.sleep(0.2)
        return self.get_observation(), {}


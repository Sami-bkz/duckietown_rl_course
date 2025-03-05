import os
import time
import random
import socket
import curses
import numpy as np
import rospy
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Header
from cv_bridge import CvBridge
import keyboard
import time
from duckietown_rl_course.duckietownrl.gym_duckietown.envs.duckie_bot_discrete import DuckieBotDiscrete
from duckietownrl.gym_duckietown.api import DuckiebotAPI
import gymnasium


def read_arrow_keys(stdscr):
    # Disable cursor and enable keypad
    curses.curs_set(0)
    stdscr.nodelay(1)

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP:
            return 0
        elif key == curses.KEY_DOWN:
            return 1
        elif key == curses.KEY_LEFT:
            return 2
        elif key == curses.KEY_RIGHT:
            return 3
        elif key == 67 or key == 99:  # 'C' key (uppercase)
            return -1
        time.sleep(0.1)


if __name__ == '__main__':
    # Initialize the environment
    # env = DuckieBotDiscrete(robot_name="paperino", fixed_linear_velocity=0.3, fixed_angular_velocity=0.1,
    #                         action_duration=0.3, stochasticity=0.1)

    env = gymnasium.make("DuckieBotDiscrete-V1")

    # Reset the environment and get the initial observation
    observation = env.reset()

    # Print the initial observation
    print("Initial Observation:", observation)

    # Main loop to select actions
    while True:
        try:
            # Start the curses screen
            action = curses.wrapper(read_arrow_keys)
            print(f"Action selected: {action}")
            if action == -1:
                print("\nExiting...")
                break
            observation = env.step(action)

            # Optionally: you can add a small sleep to control the action speed
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nExiting...")
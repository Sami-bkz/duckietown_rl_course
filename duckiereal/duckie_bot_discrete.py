import cv2
from gymnasium.spaces import Discrete, Box
from enum import Enum
import os
import time
import random
import socket
import curses
import numpy as np
import rospy
from duckietown_msgs.msg import WheelsCmdStamped
from matplotlib import pyplot as plt
from pandas.core.sample import process_sampling_size
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Header
from cv_bridge import CvBridge
from gymnasium import Env
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import threading
from environments.real_world_environment.utils import process_image
import pickle


class DuckieBotDiscrete(Env):

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

        print()
        print("    ______________________________________________________    ")
        print()
        print("   ___                 _            _   _       ____  _     _ ")
        print("  |_ _|_ __         __| | ___ _ __ | |_| |__   |  _ \| |   | |")
        print("   | || '_ \ _____ / _` |/ _ \ '_ \| __| '_ \  | |_) | |   | |")
        print("   | || | | |_____| (_| |  __/ |_) | |_| | | | |  _ <| |___|_|")
        print("  |___|_| |_|      \__,_|\___| .__/ \__|_| |_| |_| \_\_____(_)")
        print("                             |_|                              ")
        print("    ______________________________________________________    ")
        print()
        print()

        print("  > Initializing environment... ")
        super().__init__()
        self.robot_name: str = params.get("robot_name", os.environ['ROBOT_NAME'])
        
        # Ros stuff
        rospy.init_node('robot_discrete_environment', anonymous=True)   # Initialise the ros node
        self.actions_publisher = rospy.Publisher('/' + str(self.robot_name) + '/discrete_action', Int32, queue_size=10) # Create a publisher actions
        self._image_bridge = CvBridge()
        self.last_observation = None    # Not important, it will be instantiated when we receive the first observation.
        self.observation_subscriber = rospy.Subscriber(
            f"/{self.robot_name}/observation",
            CompressedImage,
            self.observation_callback
        )
        self.observation_event = threading.Event()  # Used to wait for an observation message

        # Env stuff
        self.observation_space = Box(low=0, high=255, shape=(330, 640, 3), dtype=np.uint8)  # Images observation space
        self.action_space = Discrete(4)     # Action space with four possible actions (from 0 to 3 included)

        self.stochasticity: float = params.get("stochasticity", 0.0)      # Probability to take a different action,

        print("  > Environment initialized.")


    def observation_callback(self, observation_message):
        """
        This function is called everytime an observation is received.
        Returns: None
        """
        try:
            self.last_observation = self._image_bridge.compressed_imgmsg_to_cv2(observation_message)[150:]
            self.last_observation = self.last_observation[:, :, ::-1]  # Convert image to BGR
            self.observation_event.set()  # Signal the step function to continue
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def step(self, action):

        assert self.action_space.contains(action)
        if self.stochasticity > 0.0 and random.random() < self.stochasticity:
            print("RANDOM ACTION")
            available_actions = list(set(range(self.action_space.n)) - {int(action)})
            action = random.choice(available_actions)

        print("sending action", action, "to robot", self.robot_name)

        # print(f"Action chosen: {original_action} -> {action} (after stochasticity)")  # Debugging log
        if isinstance(action, np.ndarray):
            action = int(action)


        self.observation_event.clear()              # Reset the event before waiting for the new observation
        self.actions_publisher.publish(action)

        # Wait until a new observation is received
        self.observation_event.wait()               # Blocks execution until the event is set

        # WARNING the following code call reset which call step: possible infinite recursive calls
        # Verify if we lost the center line
        x_blue_center = process_image(self.last_observation)[0]
        reward = 0
        terminate = x_blue_center is None
        if x_blue_center is None:
            reward = -100
            self.reset()

        return self.last_observation, reward, terminate, terminate, {}

    def reset(self, seed=None, options=None):
        
        return self.last_observation, {}

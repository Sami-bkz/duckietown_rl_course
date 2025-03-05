"""
API between your code and the duckiebot ros topics.
"""

import os
import time
import random
import socket
import curses
import numpy as np
import rospy
from enum import Enum
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from std_msgs.msg import Int32

"""
#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32

def callback(data):
    # Process the data
    rospy.loginfo(f"Receiver consumed: {data.data}")

def receiver():
    # Initialize the ROS node
    rospy.init_node('number_receiver', anonymous=True)

    # Subscribe to the '/number' topic
    rospy.Subscriber('/number', Int32, callback)

    # Keep the node running and listening to the topic
    rospy.spin()

if __name__ == '__main__':
    try:
        receiver()
    except rospy.ROSInterruptException:
        pass

"""


class DuckieBotAPI(object):
    """
    API between the code and the duckiebot ros topics.
    This class is an interface that defines usefull functions, used by the discrete actions and continuous
    actions environments.
    """

    class Actions(Enum):
        FORWARD = 0
        BACKWARD = 1
        LEFT = 2
        RIGHT = 3

    def __init__(self, **params):
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
        self.robot_name = params.get("robot_name", "paperino")  # Duckiebot name
        self.fixed_linear_velocity: float = params.get("fixed_linear_velocity", 0.25)
        self.fixed_angular_velocity: float = params.get("fixed_angular_velocity", 0.12)
        self.action_duration: float = params.get("action_duration", 0.3)

        # Init a node for this api
        # self.node = rospy.init_node('api_from_' + socket.gethostname(), anonymous=True)
        self.nb_messages_received = 0
        print("  > Initializing node...")
        self.node = rospy.init_node('actions_converter', anonymous=True)
        print("  > Node initialized.")

        # Setup ros command publisher
        self.commands_publisher = rospy.Publisher('/' + str(self.robot_name) + '/wheels_driver_node/wheels_cmd',
                                                  WheelsCmdStamped, queue_size=10)
        print("  > Commands publisher initialized.")

        # Setup ros command publisher
        self.observations_publisher = rospy.Publisher('/' + str(self.robot_name) + '/observation', CompressedImage,
                                                      queue_size=10)
        print("  > Observation publisher initialized.")
        self.last_observation_message = None

        # Set up the observation update process
        # - The robot will send us a lot of images. Because we don't want when the get_observation method will be
        #   called, we will update the last observation every time, and return it when get_observation is called.
        self.last_observation_message = None  # Not important, it will be instantiated when we receive the first observation.
        self.observation_subscriber = rospy.Subscriber(
            f"/{self.robot_name}/camera_node/image/compressed",
            CompressedImage,
            self.observation_callback
        )

        # Setup action listener (listen to integers from 0 to 3 included on a discrete action topics, then convert them as a robot velocity.)
        self.actions_subscriber = rospy.Subscriber('/' + str(self.robot_name) + '/discrete_action', Int32,
                                                   self.actions_callback)

        time.sleep(0.5)  # Wait for the publisher and subscriber to be registered.
        print("  > Api initialized.")
        rospy.spin()

    def observation_callback(self, observation_message):
        """
        This function is called everytime an observation is received.
        Returns: None
        """
        try:
            # self.last_observation = self._image_bridge.compressed_imgmsg_to_cv2(observation_message)[150:]
            self.last_observation_message = observation_message
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def actions_callback(self, data):
        action = int(data.data)
        print("    [api] Received action", action)
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

    def set_velocity_raw(self, left_wheel_velocity=0.0, right_wheel_velocity=0.0):
        print("    [api] setting vel raw to ", left_wheel_velocity, ", ", right_wheel_velocity)
        msg = WheelsCmdStamped()

        # Set message parameters
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.vel_left = left_wheel_velocity
        msg.vel_right = right_wheel_velocity

        # Publish the message at 10 Hz
        rate = rospy.Rate(10)
        if not rospy.is_shutdown():
            self.commands_publisher.publish(msg)
            rate.sleep()

    def set_velocity(self, linear_velocity=0.0, angular_velocity=0.0):
        self.set_velocity_raw(
            linear_velocity - angular_velocity,
            linear_velocity + angular_velocity)

    def apply_action(self, linear_velocity=0.0, angular_velocity=0.0):
        print("    [api] applying action with velocity ", linear_velocity, ", ", angular_velocity)
        self.set_velocity(linear_velocity=linear_velocity, angular_velocity=angular_velocity)  # Send the action
        time.sleep(self.action_duration)  # Let the action run for a fixed duration
        self.set_velocity()  # Stop the robot at the end of the timer (default velocities are 0.0)

        # Send the last observation received
        print("    [api] Sending observation ...")
        rate = rospy.Rate(10)
        if not rospy.is_shutdown():
            print("debug: publishing")
            self.observations_publisher.publish(self.last_observation_message)
            print("debug: calling rate.sleep()")
            rate.sleep()
        print("debug: observation sent")


if __name__ == "__main__":
    DuckieBotAPI(robot_name="gastone")
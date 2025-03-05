
"""
API between your code and the duckiebot ros topics.
"""


import os
import time
import random
import socket
import curses
import numpy as np



class DuckiebotAPI(object):
    """
    API between the code and the duckiebot ros topics.
    This class is an interface that defines usefull functions, used by the discrete actions and continuous
    actions environments.
    """

    def __init__(self, **params):
        import rospy
        from duckietown_msgs.msg import WheelsCmdStamped
        from sensor_msgs.msg import CompressedImage
        from cv_bridge import CvBridge
        from std_msgs.msg import Header
        from cv_bridge import CvBridge
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
        self.robot_name = params.get("robot_name", "paperino")      # Duckiebot name

        # Init a node for this api
        #self.node = rospy.init_node('api_from_' + socket.gethostname(), anonymous=True)
        self.nb_messages_received = 0
        print("  > Initializing node...")
        self.node = rospy.init_node('api', anonymous=True)
        print("  > Node initialized.")

        # Setup ros command publisher
        self.commands_publisher = rospy.Publisher('/' + str(self.robot_name) + '/wheels_driver_node/wheels_cmd',
                                                  WheelsCmdStamped, queue_size=10)
        print("  > Commands publisher initialized.")

        # Set up the observation update process
        # - The robot will send us a lot of images. Because we don't want when the get_observation method will be
        #   called, we will update the last observation every time, and return it when get_observation is called.
        self.last_observation = None    # Not important, it will be instantiated when we receive the first observation.
        self.observation_subscriber = rospy.Subscriber(
            f"/{self.robot_name}/camera_node/image/compressed",
            CompressedImage,
            self.observation_callback
        )
        self._image_bridge = CvBridge()

        time.sleep(0.5)  # Wait for the publisher and subscriber to be registered.
        print("  > Api initialized.")

    def observation_callback(self, observation_message):
        """
        This function is called everytime an observation is received.
        Returns: None
        """
        try:
            self.nb_messages_received += 1
            # print(" > Received", self.nb_messages_received)
            self.last_observation = self._image_bridge.compressed_imgmsg_to_cv2(observation_message)[150:]
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def set_velocity_raw(self, left_wheel_velocity=0.0, right_wheel_velocity=0.0):
        msg = WheelsCmdStamped()
        print("Which mean left:", left_wheel_velocity, ", right:", right_wheel_velocity)

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
        print("Setting velocity to", linear_velocity, ",", angular_velocity, ":")
        print("Which mean left:", linear_velocity - angular_velocity, ", right:", linear_velocity + angular_velocity)
        self.set_velocity_raw(
            linear_velocity - angular_velocity,
            linear_velocity + angular_velocity)

    def apply_action(self, linear_velocity=0.0, angular_velocity=0.0):
        self.set_velocity(linear_velocity=linear_velocity, angular_velocity=angular_velocity)  # Send the action
        time.sleep(self.action_duration)    # Let the action run for a fixed duration
        self.set_velocity()                 # Stop the robot at the end of the timer (default velocities are 0.0)

    def stop_robot(self):
        self.set_velocity(0.0, 0.0)

    def get_observation(self):
        return self.last_observation



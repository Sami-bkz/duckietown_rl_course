#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage

def image_callback(msg):
    rospy.loginfo("Received an image!")

def main():
    rospy.init_node("image_listener", anonymous=True)
    rospy.Subscriber("/paperino/camera_node/image/compressed", CompressedImage, image_callback)
    rospy.spin()

if __name__ == "__main__":
    main()

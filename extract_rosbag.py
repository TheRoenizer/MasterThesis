from __future__ import division, print_function

import cv2 as cv
import cv_bridge
import geometry_msgs.msg
import image_geometry
import itertools
import matplotlib.pyplot as plt
import numpy as np
import quaternion  # https://github.com/moble/quaternion / pip install --user numpy-quaternion
import rosbag
import rospy
import tf2_py as tf2
import warnings

#path = '/home/christoffer/Documents/rosbags/cool_2019-04-21-02-27-42_0.bag'
#path = '/home/christoffer/Documents/rosbags/cool_2019-04-21-02-29-36_0.bag'
#path = '/home/christoffer/Documents/rosbags/grasp_2019-04-21-00-31-48_0.bag'
path = '/home/christoffer/Documents/rosbags/grasp_2019-04-21-00-33-18_0.bag'

cv_bridge = cv_bridge.CvBridge()
tf_buffer = tf2.BufferCore()
psm1_msgs = []
cam_info = [None, None]
img_msg = [None, None]

with rosbag.Bag(path) as bag:
    for topic, msg, stamp in bag.read_messages(topics=['/tf']):
        for tf in msg.transforms:
            tf_buffer.set_transform(fix_tf_msg(tf), 'default_authority')

    # camera info msgs do not change during the recording, so we save only the first one
    cam_info[0] = next(bag.read_messages(topics=['/basler_stereo/left/camera_info']))[1]
    cam_info[1] = next(bag.read_messages(topics=['/basler_stereo/right/camera_info']))[1]

    # Get the 400th left camera image message in the bag
    img_msg[0] = nth(bag.read_messages(topics=['/basler_stereo/left/image_rect_color/compressed']), 400)[1]

    # Find the corresponding (equal time stamp) right camera image message
    for topic, msg, stamp in bag.read_messages(topics=['/basler_stereo/right/image_rect_color/compressed']):
        if msg.header.stamp == img_msg[0].header.stamp:
            img_msg[1] = msg
            break
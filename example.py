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

# image_geometry uses the numpy.matrix which is pending deprecation in numpy
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

# kls@schwaner-desktop:~/tmp/for-haris$ rosbag info ./cool_2019-04-21-02-27-42_0.bag
# path:        ./cool_2019-04-21-02-27-42_0.bag
# version:     2.0
# duration:    59.9s
# start:       Apr 21 2019 02:27:42.22 (1555806462.22)
# end:         Apr 21 2019 02:28:42.14 (1555806522.14)
# size:        334.9 MB
# messages:    182834
# compression: none [425/425 chunks]
# types:       geometry_msgs/PointStamped  [c63aecb41bfdfd6b7e1fac37c7cbe7bf]
#              geometry_msgs/PoseStamped   [d3812c3cbc69362b77dc0b19b345f8f5]
#              geometry_msgs/TwistStamped  [98d34b0043a2093cf9d9345ab6eef12e]
#              geometry_msgs/WrenchStamped [d78d3cb249ce23087ade7e7d0c40cfa7]
#              rosgraph_msgs/Log           [acffd30cd6b6de30f120938c17c593fb]
#              sensor_msgs/CameraInfo      [c9a58c1b0b154e0e6da7578cb991d214]
#              sensor_msgs/CompressedImage [8f7a12909da2c9d3332d540a0977563f]
#              sensor_msgs/JointState      [3066dcd76a6cfaef579bd0f34173e9fd]
#              std_msgs/Bool               [8b94c1b53db61fb6aed406028ad6332a]
#              std_msgs/Float64MultiArray  [4b7d974086d4060e7db4613a7e6c3ba4]
#              std_msgs/String             [992ce8a1687cec8c8bd883ec73ca41d1]
#              tf2_msgs/TFMessage          [94810edda583a504dfda3829e70d7eec]
# topics:      /basler_stereo/left/camera_info                      1794 msgs    : sensor_msgs/CameraInfo
#              /basler_stereo/left/image_rect_color/compressed      1794 msgs    : sensor_msgs/CompressedImage
#              /basler_stereo/right/camera_info                     1794 msgs    : sensor_msgs/CameraInfo
#              /basler_stereo/right/image_rect_color/compressed     1775 msgs    : sensor_msgs/CompressedImage
#              /dvrk/PSM1/current_state                                1 msg     : std_msgs/String
#              /dvrk/PSM1/desired_state                                1 msg     : std_msgs/String
#              /dvrk/PSM1/error                                        1 msg     : std_msgs/String
#              /dvrk/PSM1/goal_reached                                 5 msgs    : std_msgs/Bool
#              /dvrk/PSM1/jacobian_body                             5942 msgs    : std_msgs/Float64MultiArray
#              /dvrk/PSM1/jacobian_spatial                          5942 msgs    : std_msgs/Float64MultiArray
#              /dvrk/PSM1/position_cartesian_current                5939 msgs    : geometry_msgs/PoseStamped
#              /dvrk/PSM1/position_cartesian_desired                5939 msgs    : geometry_msgs/PoseStamped
#              /dvrk/PSM1/state_jaw_current                         5939 msgs    : sensor_msgs/JointState
#              /dvrk/PSM1/state_jaw_desired                         5939 msgs    : sensor_msgs/JointState
#              /dvrk/PSM1/state_joint_current                       5939 msgs    : sensor_msgs/JointState
#              /dvrk/PSM1/state_joint_desired                       5939 msgs    : sensor_msgs/JointState
#              /dvrk/PSM1/status                                       8 msgs    : std_msgs/String
#              /dvrk/PSM1/twist_body_current                        5939 msgs    : geometry_msgs/TwistStamped
#              /dvrk/PSM1/warning                                      1 msg     : std_msgs/String
#              /dvrk/PSM1/wrench_body_current                       5938 msgs    : geometry_msgs/WrenchStamped
#              /rosout                                                34 msgs    : rosgraph_msgs/Log           (15 connections)
#              /rosout_agg                                            11 msgs    : rosgraph_msgs/Log
#              /tf                                                110672 msgs    : tf2_msgs/TFMessage          (2 connections)
#              /tracker/grasp_pose                                  1377 msgs    : geometry_msgs/PoseStamped
#              /tracker/needle_pose                                 1377 msgs    : geometry_msgs/PoseStamped
#              /tracker/point0                                      1451 msgs    : geometry_msgs/PointStamped
#              /tracker/point1                                      1343 msgs    : geometry_msgs/PointStamped


def nth(iterable, n, default=None):
    """Returns the n'th item or a default value
    """
    return next(itertools.islice(iterable, n, None), default)


def find_if(iterable, predicate):
    """Returns the first element in iterable for which predicate returns True
    """
    # return next((x for x in iterable if predicate(x)), None)
    return next(((i, x) for i, x in enumerate(iterable) if predicate(x)), None)


# bisect comparator return values
BISECT_LOWER, BISECT_STOP, BISECT_HIGHER = (-1, 0, 1)


def bisect(sequence, comparator, lo=0, hi=None):
    if hi is None:
        hi = len(sequence)

    if lo < 0 or hi > len(sequence):
        raise ValueError

    while lo < hi:
        mid = (lo + hi) // 2

        if comparator(sequence[mid]) == BISECT_STOP:
            return mid
        elif comparator(sequence[mid]) == BISECT_HIGHER:
            lo = mid + 1
        else:
            hi = mid

    return lo


def find_nearest_by_stamp(sequence, stamp):
    comp = lambda m: BISECT_HIGHER if m.header.stamp < stamp else BISECT_LOWER
    i = bisect(sequence, comp)

    # Found first or last item
    if i == 0:
        return (i, sequence[i])
    elif i == len(sequence):
        return (i-1, sequence[i-1])

    # Return the item whose header.stamp is closest to stamp
    if (stamp - sequence[i-1].header.stamp) < (sequence[i].header.stamp - stamp):
        return (i-1, sequence[i-1])
    else:
        return (i, sequence[i])


def msg2tf(m):
    tf = np.identity(4)

    if m._type == 'geometry_msgs/Transform':
        tf[:3,3] = (m.translation.x, m.translation.y, m.translation.z)
        tf[:3,:3] = quaternion.as_rotation_matrix(np.quaternion(m.rotation.w, m.rotation.x, m.rotation.y, m.rotation.z))
    elif m._type == 'geometry_msgs/Pose':
        tf[:3,3] = (m.position.x, m.position.y, m.position.z)
        tf[:3,:3] = quaternion.as_rotation_matrix(np.quaternion(m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z))
    else:
        raise ValueError("Bad msg type '{}'".format(m._type))

    return tf


def fix_tf_msg(x):
    """Copy bag message type to system ROS message type (those are different types!)
    """
    y = geometry_msgs.msg.TransformStamped()
    y.header.stamp = x.header.stamp
    y.header.seq = x.header.seq
    y.header.frame_id = x.header.frame_id
    y.child_frame_id = x.child_frame_id
    y.transform.translation.x = x.transform.translation.x
    y.transform.translation.y = x.transform.translation.y
    y.transform.translation.z = x.transform.translation.z
    y.transform.rotation.x = x.transform.rotation.x
    y.transform.rotation.y = x.transform.rotation.y
    y.transform.rotation.z = x.transform.rotation.z
    y.transform.rotation.w = x.transform.rotation.w
    return y


path = 'cool_2019-04-21-02-27-42_0.bag'
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

    # Read all PSM1 pose messages (instrument TCP wrt. base frame)
    psm1_msgs = [msg for topic, msg, stamp in bag.read_messages(topics=['/dvrk/PSM1/position_cartesian_current'])]

# Find PSM1 pose message correponding (nearest time stamp) to the camera frames
psm1_msg = find_nearest_by_stamp(psm1_msgs, img_msg[0].header.stamp)[1]

# Set up stereo camera model from the image_geometry distributed with ROS
stereo_model = image_geometry.StereoCameraModel()
stereo_model.fromCameraInfo(*cam_info)

# Get robot base to optical (left camera of stereo pair) transformation (the
# 'optical' frame seen wrt. the PSM1 robot 'base' frame)
t_base_optical = msg2tf(tf_buffer.lookup_transform_core('PSM1_base', 'stereo_optical', rospy.Time()).transform)
t_optical_base = np.linalg.inv(t_base_optical)

# t_base_tcp = np.array([msg2tf(m.pose) for m in psm1_msgs])
# t_optical_tcp = np.array([t_optical_base.dot(t) for t in t_base_tcp])
t_base_tcp = msg2tf(psm1_msg.pose)
t_optical_tcp = t_optical_base.dot(t_base_tcp)

# de-compress images
imgs = [cv_bridge.compressed_imgmsg_to_cv2(m) for m in img_msg]

# Project 3D point to left/right images
p = stereo_model.project3dToPixel(t_optical_tcp[:3, 3])

print(p)
'''
# Draw circles at projected points
def f2i(iterable):
    return tuple(int(round(x)) for x in iterable)


cv.circle(imgs[0], f2i(p[0]), 10, (0, 0, 255), 3)
cv.circle(imgs[1], f2i(p[1]), 10, (0, 0, 255), 3)

# Show images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(cv.cvtColor(imgs[0], cv.COLOR_BGR2RGB))
ax[1].imshow(cv.cvtColor(imgs[1], cv.COLOR_BGR2RGB))
plt.savefig("rosbag_test.png")
'''

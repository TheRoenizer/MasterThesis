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
from tempfile import TemporaryFile


def nth(iterable, n, default=None):
    """Returns the n'th item or a default value
    """
    return next(itertools.islice(iterable, n, None), default)


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


# path = '/home/jsteeen/PycharmProjects/MasterThesis/bagfiles/cool_2019-04-21-02-27-42_0.bag'
# path = '/home/jsteeen/PycharmProjects/MasterThesis/bagfiles/cool_2019-04-21-02-29-36_0.bag'
path = '/home/jsteeen/PycharmProjects/MasterThesis/bagfiles/grasp_2019-04-21-00-31-48_0.bag'
# path = '/home/jsteeen/PycharmProjects/MasterThesis/bagfiles/grasp_2019-04-21-00-33-18_0.bag'

cv_bridge = cv_bridge.CvBridge()
tf_buffer = tf2.BufferCore()
psm1_msgs = []
cam_info = [None, None]
img_msg = [None, None]

outfile = TemporaryFile()

with rosbag.Bag(path) as bag:
    for topic, msg, stamp in bag.read_messages(topics=['/tf']):
        for tf in msg.transforms:
            tf_buffer.set_transform(fix_tf_msg(tf), 'default_authority')

    # camera info msgs do not change during the recording, so we save only the first one
    cam_info[0] = next(bag.read_messages(topics=['/basler_stereo/left/camera_info']))[1]
    cam_info[1] = next(bag.read_messages(topics=['/basler_stereo/right/camera_info']))[1]

    # Set up stereo camera model from the image_geometry distributed with ROS
    stereo_model = image_geometry.StereoCameraModel()
    stereo_model.fromCameraInfo(*cam_info)

    # Get robot base to optical (left camera of stereo pair) transformation (the
    # 'optical' frame seen wrt. the PSM1 robot 'base' frame)
    t_base_optical = msg2tf(tf_buffer.lookup_transform_core('PSM1_base', 'stereo_optical', rospy.Time()).transform)
    t_optical_base = np.linalg.inv(t_base_optical)

    # Read all PSM1 pose messages (instrument TCP wrt. base frame) PSM = patient side manipulator
    psm1_msgs = [msg for topic, msg, stamp in bag.read_messages(topics=['/dvrk/PSM1/position_cartesian_current'])]

    poses = np.zeros((4, 4))

    for i in range(120, 264, 2):
        # Get the i'th right camera image message in the bag
        img_msg[1] = nth(bag.read_messages(topics=['/basler_stereo/right/image_rect_color/compressed']), i)[1]

        # Find the corresponding (equal time stamp) left camera image message
        for topic, msg, stamp in bag.read_messages(topics=['/basler_stereo/left/image_rect_color/compressed']):
            if msg.header.stamp == img_msg[1].header.stamp:
                img_msg[0] = msg
                break

        # de-compress images
        imgs = [cv_bridge.compressed_imgmsg_to_cv2(m) for m in img_msg]

        img_left = imgs[0]  # cv.cvtColor(imgs[0], cv.COLOR_BGR2RGB)
        img_right = imgs[1]  # cv.cvtColor(imgs[1], cv.COLOR_BGR2RGB)

        cv.imwrite("/home/jsteeen/Pictures/rosbag_pictures/grasp1/img{}_left.png".format(int((i-120)/2)), img_left)
        cv.imwrite("/home/jsteeen/Pictures/rosbag_pictures/grasp1/img{}_right.png".format(int((i-120)/2)), img_right)

        # Find PSM1 pose message corresponding (nearest time stamp) to the camera frames
        psm1_msg = find_nearest_by_stamp(psm1_msgs, img_msg[0].header.stamp)[1]

        # t_base_tcp = np.array([msg2tf(m.pose) for m in psm1_msgs])
        # t_optical_tcp = np.array([t_optical_base.dot(t) for t in t_base_tcp])
        t_base_tcp = msg2tf(psm1_msg.pose)
        t_optical_tcp = t_optical_base.dot(t_base_tcp)

        # print(t_optical_tcp)
        poses = np.concatenate([np.atleast_3d(poses), np.atleast_3d(t_optical_tcp)], axis=2)
        # print(poses.shape)


poses = np.delete(poses, 0, axis=-1)
print(poses.shape)
np.save("/home/jsteeen/Pictures/rosbag_pictures/grasp1/pose_arr.npy", poses)
'''
print("--------------------- Poses from rosbag ---------------------\n")
print(poses.shape)
print(poses[:, :, 0])

_ = outfile.seek(0)
poses = np.load("/home/jsteeen/Pictures/rosbag_pictures/pose_arr.npy")

print("--------------------- Poses from file ---------------------\n")
print(poses.shape)
print(poses[:, :, 0])
'''
print("DONE!")

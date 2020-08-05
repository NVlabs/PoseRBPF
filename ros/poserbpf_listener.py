# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import rospy
import tf
import message_filters
import cv2
import numpy as np
import torch
import torch.nn as nn
import threading
import sys

from Queue import Queue
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROS_Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import JointState
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from scipy.optimize import minimize
from geometry_msgs.msg import PoseStamped, PoseArray
from rospy.numpy_msg import numpy_msg
import matplotlib.pyplot as plt
from config.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file

from pose_rbpf.pose_rbpf import *
import scipy.io
from scipy.spatial import distance_matrix as scipy_distance_matrix
import random

lock = threading.Lock()

from random import shuffle
import tf.transformations as tra
import time

def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T

class ImageListener:

    def __init__(self, object_list, cfg_list, checkpoint_list, codebook_list,
                 modality, cad_model_dir, category='ycb', refine=True):

        print(' *** Initializing PoseRBPF ROS Node ... ')

        # variables
        self.cv_bridge = CvBridge()
        self.count = 0
        self.objects = []
        self.frame_names = []
        self.frame_lost = []
        self.renders = dict()
        self.n_renders = 0
        self.num_lost = 50
        self.queue_size = 10
        self.scene = 1

        # initialize poserbpf with cfg_file
        self.object_list = object_list
        self.cfg_list = cfg_list
        self.ckpt_list = checkpoint_list
        self.codebook_list = codebook_list

        self.pose_rbpf = PoseRBPF(self.object_list, self.cfg_list, self.ckpt_list,
                                  self.codebook_list, category,
                                  modality, cad_model_dir, refine=refine)

        # target list
        self.target_list = range(len(self.pose_rbpf.obj_list))
        self.target_object = cfg.TEST.OBJECTS[0]
        self.target_cfg = self.pose_rbpf.set_target_obj(self.target_object)
        self.class_info = torch.ones((1, 1, 128, 128), dtype=torch.float32)
        self.init_failure_steps = 0
        self.input_rgb = None
        self.input_depth = None
        self.input_seg = None
        self.input_rois = None
        self.input_stamp = None
        self.input_frame_id = None
        self.input_joint_states = None
        self.input_robot_joint_states = None
        self.main_thread_free = True
        self.kf_time_stamp = None
        # roi information
        self.prefix = '00_'

        # initialize a node
        rospy.init_node('poserbpf_image_listener')
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        self.pose_pub = rospy.Publisher('poserbpf_image', ROS_Image, queue_size=1)

        # target detection
        self.flag_detected = False

        # forward kinematics (base to camera link transformation)
        self.Tbr_now = np.eye(4, dtype=np.float32)
        self.Tbr_prev = np.eye(4, dtype=np.float32)
        self.Trc = np.load('./ros/extrinsics_D415.npy')
        self.Tbc_now = np.eye(4, dtype=np.float32)

        # dry run poserbpf to initialize the numba modules
        self.run_poserbpf_initial()
        print(' *** PoseRBPF Ready ... ')

        # subscriber for camera information
        msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
        K = np.array(msg.K).reshape(3, 3)
        self.intrinsic_matrix = K
        self.pose_rbpf.set_intrinsics(K, 640, 480)
        print('Intrinsics matrix : ')
        print(self.intrinsic_matrix)

        # subscriber for rgbd images and joint states
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', ROS_Image, queue_size=1)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', ROS_Image, queue_size=1)
        queue_size = 1
        slop_seconds = 0.01
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

    def visualize_poses(self, rgb):
        image_rgb = rgb.astype(np.float32) / 255.0
        Tco_list = []
        object_initialized = []
        for i in range(len(self.pose_rbpf.rbpf_list)):
            if self.pose_rbpf.rbpf_ok_list[i]:
                Tco = np.eye(4, dtype=np.float32)
                Tco[:3, :3] = self.pose_rbpf.rbpf_list[i].rot_bar
                Tco[:3, 3] = self.pose_rbpf.rbpf_list[i].trans_bar

                Tco_list.append(Tco.copy())
                object_initialized.append(i)

        image_est_render, _, _ = self.pose_rbpf.renderer.render_pose_multiple(self.intrinsic_matrix,
                                                                               Tco_list,
                                                                               object_initialized)
        image_est_disp = image_est_render[0].permute(1, 2, 0).cpu().numpy()
        image_disp = 0.4 * image_rgb + 0.6 * image_est_disp
        image_disp = np.clip(image_disp, 0, 1.0)
        return image_disp

    def run_poserbpf_initial(self):
        image_rgb = torch.zeros((640, 480, 3), dtype=torch.float32)
        image_depth = torch.zeros((640, 480, 1), dtype=torch.float32)
        target_obj = self.pose_rbpf.obj_list[0]
        roi = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.pose_rbpf.set_target_obj(target_obj)
        self.pose_rbpf.pose_estimation_single(target_obj, roi, image_rgb, image_depth, dry_run=True)

    def process_data(self):
        # callback data
        with lock:
            input_stamp = self.input_stamp
            input_rgb = self.input_rgb.copy()
            input_depth = self.input_depth.copy()

        # subscribe the transformation
        try:
            source_frame = 'measured/camera_color_optical_frame'
            target_frame = 'measured/base_link'
            trans, rot = self.listener.lookupTransform(target_frame, source_frame, input_stamp)
            Tbc = ros_qt_to_rt(rot, trans)
            self.Tbc_now = Tbc.copy()
            self.Tbr_now = Tbc.dot(np.linalg.inv(self.Trc))
            if np.linalg.norm(self.Tbr_prev[:3, 3]) == 0:
                self.pose_rbpf.T_c1c0 = np.eye(4, dtype=np.float32)
            else:
                Tbc0 = np.matmul(self.Tbr_prev, self.Trc)
                Tbc1 = np.matmul(self.Tbr_now, self.Trc)
                self.pose_rbpf.T_c1c0 = np.matmul(np.linalg.inv(Tbc1), Tbc0)
            self.Tbr_prev = self.Tbr_now.copy()
        except:
            # print('missing forward kinematics info')
            return

        # todo: call object detection
        # ...
        if len(self.pose_rbpf.obj_list) == 1:
            rois = np.array([[0, 0, 447, 167, 447, 167]], dtype=np.float32)
        else:
            rois = np.array([[0, 0, 332, 150, 332, 150],
                             [0, 1, 480, 193, 480, 193],
                             [0, 2, 407, 370, 407, 370]], dtype=np.float32)

        # call pose estimation function
        self.pose_estimation(input_rgb, input_depth, rois)


        for idx_tf in range(len(self.pose_rbpf.rbpf_list)):
            if not self.pose_rbpf.rbpf_ok_list[idx_tf]:
                continue
            Tco = np.eye(4, dtype=np.float32)
            Tco[:3, :3] = self.pose_rbpf.rbpf_list[idx_tf].rot_bar
            Tco[:3, 3] = self.pose_rbpf.rbpf_list[idx_tf].trans_bar
            Tbo = self.Tbc_now.dot(Tco)
            # publish tf
            t_bo = Tbo[:3, 3]
            q_bo = mat2quat(Tbo[:3, :3])
            self.br.sendTransform(t_bo, [q_bo[1], q_bo[2], q_bo[3], q_bo[0]], self.input_stamp,
                                  'poserbpf/' + self.pose_rbpf.obj_list[idx_tf], 'measured/base_link')

        # visualization
        image_disp = self.visualize_poses(input_rgb) * 255.0
        image_disp = image_disp.astype(np.uint8)
        image_disp = np.clip(image_disp, 0, 255)
        pose_msg = self.cv_bridge.cv2_to_imgmsg(image_disp)
        pose_msg.header.stamp = self.input_stamp
        pose_msg.header.frame_id = self.input_frame_id
        pose_msg.encoding = 'rgb8'
        self.pose_pub.publish(pose_msg)

    def callback(self, rgb, depth):
        # decode image
        if depth is not None:
            if depth.encoding == '32FC1':
                depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
            elif depth.encoding == '16UC1':
                depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
            else:
                rospy.logerr_throttle(1,
                                      'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(depth.encoding))
                return
        else:
            depth_cv = None
        with lock:
            self.input_depth = depth_cv
            # rgb image used for posecnn detection
            self.input_rgb = self.cv_bridge.imgmsg_to_cv2(rgb, 'rgb8')
            # other information
            self.input_stamp = rgb.header.stamp
            self.input_frame_id = rgb.header.frame_id

    def pose_estimation(self, rgb, depth, rois):
        image_rgb = torch.from_numpy(rgb).float() / 255.0
        image_depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(2).float() / 1000.0

        # propagate particles for all the initialized objects
        for i in range(len(self.pose_rbpf.obj_list)):
            if self.pose_rbpf.rbpf_ok_list[i]:
                target_obj = self.pose_rbpf.obj_list[i]
                self.pose_rbpf.propagate_with_forward_kinematics(target_obj)

        # initialize the uninitialized but detected objects (initialize and refine)
        obj_list_detected = []
        obj_list_init = []
        Tco_list_init = []
        sim_list_init = []
        for i in range(rois.shape[0]):
            roi = rois[i]
            obj_idx = int(roi[1])
            obj_list_detected.append(self.pose_rbpf.obj_list[obj_idx])

            target_obj = self.pose_rbpf.obj_list[obj_idx]

            Tco, max_sim = self.pose_rbpf.pose_estimation_single(target_obj, roi, image_rgb, image_depth, visualize=False)

            Tco_list_init.append(Tco.copy())
            obj_list_init.append(target_obj)
            sim_list_init.append(max_sim)
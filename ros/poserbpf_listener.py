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
import posecnn_cuda

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
from utils.nms import nms
from utils.cython_bbox import bbox_overlaps

from pose_rbpf.pose_rbpf import *
from pose_rbpf.sdf_multiple_optimizer import sdf_multiple_optimizer
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
                 modality, cad_model_dir, category='ycb', refine_single=True, refine_multiple=False, use_depth=True,
                 use_self_supervised_ckpts=True):

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
        self.use_depth = use_depth

        self.posecnn_classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', \
                         '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', \
                         '011_banana', '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', \
                         '036_wood_block', '037_scissors', '040_large_marker', '052_extra_large_clamp', '061_foam_brick')

        # initialize poserbpf with cfg_file
        self.object_list = object_list
        self.cfg_list = cfg_list
        self.ckpt_list = checkpoint_list
        self.codebook_list = codebook_list

        self.pose_rbpf = PoseRBPF(self.object_list, self.cfg_list, self.ckpt_list,
                                  self.codebook_list, category,
                                  modality, cad_model_dir, refine=refine_single)

        self.ssv_ckpts = use_self_supervised_ckpts

        # initial sdf refinement for multiple objects at once
        self.refine_multiple = refine_multiple
        if refine_multiple:
            print('loading SDFs')
            sdf_files = []
            for cls in self.object_list:
                sdf_file = '{}/ycb_models/{}/textured_simple_low_res.pth'.format(cad_model_dir, cls)
                sdf_files.append(sdf_file)
            reg_trans = 1000.0
            reg_rot = 10.0
            self.sdf_optimizer = sdf_multiple_optimizer(self.object_list, sdf_files, reg_trans, reg_rot)

        # target list
        self.target_list = range(len(self.pose_rbpf.obj_list))
        self.target_object = cfg.TEST.OBJECTS[0]
        self.pose_rbpf.add_object_instance(self.target_object)
        self.target_cfg = self.pose_rbpf.set_target_obj(0)
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
        self.target_frame = 'measured/base_link'

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
        # subscriber for posecnn label
        label_sub = message_filters.Subscriber('/posecnn_label', ROS_Image, queue_size=1)
        queue_size = 10
        slop_seconds = 0.2
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, label_sub], queue_size, slop_seconds)
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
                object_initialized.append(self.pose_rbpf.obj_list.index(self.pose_rbpf.instance_list[i]))

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
        roi = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.pose_rbpf.set_target_obj(0)
        self.pose_rbpf.pose_estimation_single(0, roi, image_rgb, image_depth, dry_run=True)
        # reset lists
        self.pose_rbpf.rbpf_list = []
        self.pose_rbpf.rbpf_ok_list = []
        self.pose_rbpf.instance_list = []
        self.pose_rbpf.mope_Tbo_list = []
        self.pose_rbpf.mope_pc_b_list = []


    def query_posecnn_detection(self, classes):

        # detection information of the target object
        rois_est = np.zeros((0, 7), dtype=np.float32)
        # TODO look for multiple object instances
        max_objects = 5
        for i in range(len(classes)):

            for object_id in range(max_objects):

                # check posecnn frame
                cls = classes[i]
                suffix_frame = '_%02d_roi' % (object_id)
                source_frame = 'posecnn/' + cls + suffix_frame

                try:
                    # print('look for posecnn detection ' + source_frame)
                    trans, rot = self.listener.lookupTransform(self.target_frame, source_frame, rospy.Time(0))
                    n = trans[0]
                    secs = trans[1]
                    now = rospy.Time.now()
                    if abs(now.secs - secs) > 1.0:
                        print 'posecnn pose for %s time out %f %f' % (source_frame, now.secs, secs)
                        continue
                    roi = np.zeros((1, 7), dtype=np.float32)
                    roi[0, 0] = 0
                    roi[0, 1] = i
                    roi[0, 2] = rot[0] * n
                    roi[0, 3] = rot[1] * n
                    roi[0, 4] = rot[2] * n
                    roi[0, 5] = rot[3] * n
                    roi[0, 6] = trans[2]
                    rois_est = np.concatenate((rois_est, roi), axis=0)
                    print('find posecnn detection ' + source_frame)
                except:
                    continue

        if rois_est.shape[0] > 0:
            # non-maximum suppression within class
            index = nms(rois_est, 0.2)
            rois_est = rois_est[index, :]

        return rois_est


    def callback(self, rgb, depth, label):
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
            self.input_seg = self.cv_bridge.imgmsg_to_cv2(label, 'mono8')
            # other information
            self.input_stamp = rgb.header.stamp
            self.input_frame_id = rgb.header.frame_id


    def process_data(self):
        # callback data
        with lock:
            input_stamp = self.input_stamp
            input_rgb = self.input_rgb.copy()
            input_depth = self.input_depth.copy()
            input_seg = self.input_seg.copy()

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

        # call object detection
        '''
        if len(self.pose_rbpf.obj_list) == 1:
            rois = np.array([[0, 0, 447, 167, 447, 167]], dtype=np.float32)
        else:
            rois = np.array([[0, 0, 332, 150, 332, 150],
                             [0, 1, 480, 193, 480, 193],
                             [0, 2, 407, 370, 407, 370]], dtype=np.float32)
        '''

        rois = self.query_posecnn_detection(self.pose_rbpf.obj_list)
        rois = rois[:, :6]

        # call pose estimation function
        self.pose_estimation(input_rgb, input_depth, input_seg, rois)

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
            instance_count = 0
            for i in range(idx_tf):
                if self.pose_rbpf.instance_list[i] == self.pose_rbpf.instance_list[idx_tf]:
                    instance_count += 1
            self.br.sendTransform(t_bo, [q_bo[1], q_bo[2], q_bo[3], q_bo[0]], self.input_stamp,
                                  'poserbpf/{}_{}'.format(self.pose_rbpf.instance_list[idx_tf],
                                                          instance_count), 'measured/base_link')

        # visualization
        image_disp = self.visualize_poses(input_rgb) * 255.0
        image_disp = image_disp.astype(np.uint8)
        image_disp = np.clip(image_disp, 0, 255)
        pose_msg = self.cv_bridge.cv2_to_imgmsg(image_disp)
        pose_msg.header.stamp = self.input_stamp
        pose_msg.header.frame_id = self.input_frame_id
        pose_msg.encoding = 'rgb8'
        self.pose_pub.publish(pose_msg)


    def pose_estimation(self, rgb, depth, label, rois):
        image_rgb = torch.from_numpy(rgb).float() / 255.0
        image_depth = torch.from_numpy(depth.astype(np.float32)).float() / 1000.0
        im_depth = image_depth.cuda()
        image_depth = image_depth.unsqueeze(2)
        im_label = torch.from_numpy(label).cuda()

        if self.ssv_ckpts:
            image_input = image_rgb[:, :, [2, 1, 0]]
        else:
            image_input = image_rgb

        if not self.use_depth:
            image_depth = None

        # propagate particles for all the initialized objects
        for i in range(len(self.pose_rbpf.instance_list)):
            if self.pose_rbpf.rbpf_ok_list[i]:
                self.pose_rbpf.propagate_with_forward_kinematics(i)

        # collect rois from rbpfs
        rois_rbpf = np.zeros((0, 6), dtype=np.float32)
        index_rbpf = []
        for i in range(len(self.pose_rbpf.instance_list)):
            if self.pose_rbpf.rbpf_ok_list[i]:
                roi = self.pose_rbpf.rbpf_list[i].roi
                rois_rbpf = np.concatenate((rois_rbpf, roi), axis=0)
                index_rbpf.append(i)
                self.pose_rbpf.rbpf_list[i].roi_assign = None

        # data association based on bounding box overlap
        num_rois = rois.shape[0]
        num_rbpfs = rois_rbpf.shape[0]
        assigned_rois = np.zeros((num_rois, ), dtype=np.int32)
        if num_rbpfs > 0 and num_rois > 0:
            # overlaps: (rois x gt_boxes) (batch_id, x1, y1, x2, y2)
            overlaps = bbox_overlaps(np.ascontiguousarray(rois_rbpf[:, (1, 2, 3, 4, 5)], dtype=np.float),
                np.ascontiguousarray(rois[:, (1, 2, 3, 4, 5)], dtype=np.float))

            # assign rois to rbpfs
            assignment = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)
            unassigned = []
            for i in range(num_rbpfs):
                if max_overlaps[i] > 0.2:
                    self.pose_rbpf.rbpf_list[index_rbpf[i]].roi_assign = rois[assignment[i]]
                    assigned_rois[assignment[i]] = 1
                else:
                    unassigned.append(i)

            # check if there are un-assigned rois
            index = np.where(assigned_rois == 0)[0]

            # if there is un-assigned rbpfs
            if len(unassigned) > 0 and len(index) > 0:
                for i in range(len(unassigned)):
                    for j in range(len(index)):
                        if assigned_rois[index[j]] == 0 and self.pose_rbpf.rbpf_list[index_rbpf[unassigned[i]]].roi[0, 1] == rois[index[j], 1]:
                            self.pose_rbpf.rbpf_list[index_rbpf[unassigned[i]]].roi_assign = rois[index[j]]
                            assigned_rois[index[j]] = 1
        elif num_rbpfs == 0 and num_rois == 0:
            return

        # filter tracked objects
        for i in range(len(self.pose_rbpf.instance_list)):
            if self.pose_rbpf.rbpf_ok_list[i]:
                roi = self.pose_rbpf.rbpf_list[i].roi_assign
                Tco, max_sim = self.pose_rbpf.pose_estimation_single(i, roi, image_input, image_depth, visualize=False)

        # initialize new object
        for i in range(num_rois):
            if assigned_rois[i]:
                continue
            roi = rois[i]
            obj_idx = int(roi[1])
            target_obj = self.pose_rbpf.obj_list[obj_idx]
            add_new_instance = True
            for j in range(len(self.pose_rbpf.instance_list)):
                if self.pose_rbpf.instance_list[j] == target_obj and self.pose_rbpf.rbpf_ok_list[j] == False:
                    add_new_instance = False
                    Tco, max_sim = self.pose_rbpf.pose_estimation_single(j, roi,
                                                                         image_input,
                                                                         image_depth, visualize=False)
            if add_new_instance:
                self.pose_rbpf.add_object_instance(target_obj)
                Tco, max_sim = self.pose_rbpf.pose_estimation_single(len(self.pose_rbpf.instance_list)-1, roi, image_input,
                                                                     image_depth, visualize=False)

        # SDF refinement for multiple objects
        if self.refine_multiple and self.use_depth:
            # backproject depth
            fx = self.intrinsic_matrix[0, 0]
            fy = self.intrinsic_matrix[1, 1]
            px = self.intrinsic_matrix[0, 2]
            py = self.intrinsic_matrix[1, 2]
            im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, im_depth)[0]

            index_sdf = []
            for i in range(len(self.pose_rbpf.instance_list)):
                if self.pose_rbpf.rbpf_ok_list[i]:
                    index_sdf.append(i)
            if len(index_sdf) > 0:
                self.pose_rbpf.pose_refine_multiple(self.sdf_optimizer, self.posecnn_classes, index_sdf, im_depth, im_pcloud, im_label, steps=50)

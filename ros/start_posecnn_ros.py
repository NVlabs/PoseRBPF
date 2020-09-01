#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import tf
import rosnode
import message_filters
import cv2
import torch.nn as nn
import threading
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import rospy
import matplotlib.pyplot as plt
import copy
import posecnn_cuda
import _init_paths
import networks

from cv_bridge import CvBridge, CvBridgeError
from config.config_posecnn import cfg, cfg_from_file
from datasets.factory import get_dataset
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.render_utils import render_image_detection
from utils.se3 import *
from utils.nms import nms
from utils.blob import pad_im

lock = threading.Lock()

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

def get_relative_pose_from_tf(listener, source_frame, target_frame):
    first_time = True
    while True:
        try:
            stamp = rospy.Time.now()
            init_trans, init_rot = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            break
        except Exception as e:
            if first_time:
                print(str(e))
                first_time = False
            continue
    return ros_qt_to_rt(init_rot, init_trans), stamp


class ImageListener:

    def __init__(self, network, dataset):

        self.net = network
        self.dataset = dataset
        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None

        suffix = '_%02d' % (cfg.instance_id)
        prefix = '%02d_' % (cfg.instance_id)
        self.suffix = suffix
        self.prefix = prefix

        # initialize a node
        rospy.init_node('posecnn_rgb')
        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()
        self.label_pub = rospy.Publisher('posecnn_label', Image, queue_size=10)
        self.rgb_pub = rospy.Publisher('posecnn_rgb', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('posecnn_depth', Image, queue_size=10)
        self.posecnn_pub = rospy.Publisher('posecnn_detection', Image, queue_size=10)

        # create pose publisher for each known object class
        self.pubs = []
        for i in range(1, self.dataset.num_classes):
            if self.dataset.classes[i][3] == '_':
                cls = self.dataset.classes[i][4:]
            else:
                cls = self.dataset.classes[i]
            self.pubs.append(rospy.Publisher('/objects/prior_pose/' + cls, PoseStamped, queue_size=10))

        print('***PoseCNN ready, waiting for camera images***')

        if cfg.TEST.ROS_CAMERA == 'D415':
            # use RealSense D415
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.camera_frame = 'measured/camera_color_optical_frame'
            self.target_frame = self.base_frame
            self.viz_pub = rospy.Publisher('/obj/mask_estimates/realsense', MarkerArray, queue_size=1)
        elif cfg.TEST.ROS_CAMERA == 'Azure':
            # use RealSense Azure
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
            self.viz_pub = rospy.Publisher('/obj/mask_estimates/azure', MarkerArray, queue_size=1)
        else:
            # use kinect
            self.base_frame = '%s_rgb_optical_frame' % (cfg.TEST.ROS_CAMERA)
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (cfg.TEST.ROS_CAMERA), Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (cfg.TEST.ROS_CAMERA), Image, queue_size=10)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (cfg.TEST.ROS_CAMERA), CameraInfo)
            self.camera_frame = '%s_rgb_optical_frame' % (cfg.TEST.ROS_CAMERA)
            self.target_frame = self.base_frame
            self.viz_pub = rospy.Publisher('/obj/mask_estimates/%s' % (cfg.TEST.ROS_CAMERA), MarkerArray, queue_size=1)

        # camera to base transformation
        self.Tbc_now = np.eye(4, dtype=np.float32)

        # update camera intrinsics
        K = np.array(msg.K).reshape(3, 3)
        dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)

        # use fake label blob
        num_classes = dataset.num_classes
        height = cfg.TRAIN.SYN_HEIGHT
        width = cfg.TRAIN.SYN_WIDTH
        label_blob = np.zeros((1, num_classes, height, width), dtype=np.float32)
        pose_blob = np.zeros((1, num_classes, 9), dtype=np.float32)
        gt_boxes = np.zeros((1, num_classes, 5), dtype=np.float32)

        # construct the meta data
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros((1, 18), dtype=np.float32)
        meta_data_blob[0, 0:9] = K.flatten()
        meta_data_blob[0, 9:18] = Kinv.flatten()

        self.label_blob = torch.from_numpy(label_blob).cuda()
        self.meta_data_blob = torch.from_numpy(meta_data_blob).cuda()
        self.extents_blob = torch.from_numpy(dataset._extents).cuda()
        self.gt_boxes_blob = torch.from_numpy(gt_boxes).cuda()
        self.poses_blob = torch.from_numpy(pose_blob).cuda()
        self.points_blob = torch.from_numpy(dataset._point_blob).cuda()
        self.symmetry_blob = torch.from_numpy(dataset._symmetry).cuda()


    def callback_rgbd(self, rgb, depth):

        self.Tbc_now, self.Tbc_stamp = get_relative_pose_from_tf(self.listener, self.camera_frame, self.base_frame)

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        # rescale image if necessary
        if cfg.TEST.SCALES_BASE[0] != 1:
            im_scale = cfg.TEST.SCALES_BASE[0]
            im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
            depth_cv = pad_im(cv2.resize(depth_cv, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp


    def run_network(self):

        with lock:
            if listener.im is None:
              return
            im_color = self.im.copy()
            depth_cv = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp
            input_Tbc = self.Tbc_now.copy()
            input_Tbc_stamp = self.Tbc_stamp

        print('===========================================')

        # compute image blob
        im = im_color.astype(np.float32, copy=True)
        im -= cfg.PIXEL_MEANS
        height = im.shape[0]
        width = im.shape[1]
        im = np.transpose(im / 255.0, (2, 0, 1))
        im = im[np.newaxis, :, :, :]
        inputs = torch.from_numpy(im).cuda()

        # tranform to gpu
        Tbc = torch.from_numpy(input_Tbc).cuda().float()

        # backproject depth
        depth = torch.from_numpy(depth_cv).cuda()
        fx = self.dataset._intrinsic_matrix[0, 0]
        fy = self.dataset._intrinsic_matrix[1, 1]
        px = self.dataset._intrinsic_matrix[0, 2]
        py = self.dataset._intrinsic_matrix[1, 2]
        im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, depth)[0]

        # compare the depth
        depth_meas_roi = im_pcloud[:, :, 2]
        mask_depth_meas = depth_meas_roi > 0
        mask_depth_valid = torch.isfinite(depth_meas_roi)

        # forward
        cfg.TRAIN.POSE_REG = False
        out_label, out_vertex, rois, out_pose = self.net(inputs, self.label_blob, self.meta_data_blob, 
                                                         self.extents_blob, self.gt_boxes_blob, self.poses_blob, self.points_blob, self.symmetry_blob)
        label_tensor = out_label[0]
        labels = label_tensor.detach().cpu().numpy()

        # filter out detections
        rois = rois.detach().cpu().numpy()
        index = np.where(rois[:, -1] > cfg.TEST.DET_THRESHOLD)[0]
        rois = rois[index, :]

        # non-maximum suppression within class
        index = nms(rois, 0.2)
        rois = rois[index, :]

        # render output image
        im_label = render_image_detection(self.dataset, im_color, rois, labels)
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(im_label, 'rgb8')
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.posecnn_pub.publish(rgb_msg)

        # publish segmentation mask
        label_msg = self.cv_bridge.cv2_to_imgmsg(labels.astype(np.uint8))
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.label_pub.publish(label_msg)

        # visualization
        if cfg.TEST.VISUALIZE:
            fig = plt.figure()
            ax = fig.add_subplot(2, 3, 1)
            plt.imshow(im_color)
            ax.set_title('input image')

            ax = fig.add_subplot(2, 3, 2)
            plt.imshow(im_label)

            ax = fig.add_subplot(2, 3, 3)
            plt.imshow(labels)

            # show predicted vertex targets
            vertex_pred = out_vertex.detach().cpu().numpy()
            vertex_target = vertex_pred[0, :, :, :]
            center = np.zeros((3, height, width), dtype=np.float32)

            for j in range(1, dataset._num_classes):
                index = np.where(labels == j)
                if len(index[0]) > 0:
                    center[0, index[0], index[1]] = vertex_target[3*j, index[0], index[1]]
                    center[1, index[0], index[1]] = vertex_target[3*j+1, index[0], index[1]]
                    center[2, index[0], index[1]] = np.exp(vertex_target[3*j+2, index[0], index[1]])

            ax = fig.add_subplot(2, 3, 4)
            plt.imshow(center[0,:,:])
            ax.set_title('predicted center x')
            ax = fig.add_subplot(2, 3, 5)
            plt.imshow(center[1,:,:])
            ax.set_title('predicted center y')
            ax = fig.add_subplot(2, 3, 6)
            plt.imshow(center[2,:,:])
            ax.set_title('predicted z')
            plt.show()

        if not rois.shape[0]:
            return

        indexes = np.zeros((self.dataset.num_classes, ), dtype=np.int32)
        index = np.argsort(rois[:, 2])
        rois = rois[index, :]
        now = rospy.Time.now()
        markers = []
        for i in range(rois.shape[0]):
            roi = rois[i]
            cls = int(roi[1])
            cls_name = self.dataset._classes_test[cls]
            if cls > 0 and roi[-1] > cfg.TEST.DET_THRESHOLD:

                # compute mask translation
                w = roi[4] - roi[2]
                h = roi[5] - roi[3]
                x1 = max(int(roi[2]), 0)
                y1 = max(int(roi[3]), 0)
                x2 = min(int(roi[4]), width - 1)
                y2 = min(int(roi[5]), height - 1)

                labels = torch.zeros_like(label_tensor)
                labels[y1:y2, x1:x2] = label_tensor[y1:y2, x1:x2]
                mask_label = labels == cls
                mask = mask_label * mask_depth_meas * mask_depth_valid
                pix_index = torch.nonzero(mask)
                n = pix_index.shape[0]
                print('[%s] points : %d' % (cls_name, n))
                if n == 0:
                    '''
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 2, 1)
                    plt.imshow(depth.cpu().numpy())
                    ax.set_title('depth')

                    ax = fig.add_subplot(1, 2, 2)
                    plt.imshow(im_label.cpu().numpy())
                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))
                    ax.set_title('label')
                    plt.show()
                    '''
                    continue
                points = im_pcloud[pix_index[:, 0], pix_index[:, 1], :]

                # filter points
                m = torch.mean(points, dim=0, keepdim=True)
                mpoints = m.repeat(n, 1)
                distance = torch.norm(points - mpoints, dim=1)
                extent = np.mean(self.dataset._extents_test[cls, :])
                points = points[distance < 1.5 * extent, :]
                if points.shape[0] == 0:
                    continue
            
                # transform points to base
                ones = torch.ones((points.shape[0], 1), dtype=torch.float32, device=0)
                points = torch.cat((points, ones), dim=1)
                points = torch.mm(Tbc, points.t())
                location = torch.mean(points[:3, :], dim=1).cpu().numpy()
                if location[2] > 2.5:
                    continue
                print('[%s] detection score: %f' % (cls_name, roi[-1]))
                print('[%s] location mean: %f, %f, %f' % (cls_name, location[0], location[1], location[2]))

                # extend the location away from camera a bit
                c = Tbc[:3, 3].cpu().numpy()
                d = location - c
                d = d / np.linalg.norm(d)
                location = location + (extent / 2) * d

                # publish tf raw
                self.br.sendTransform(location, [0, 0, 0, 1], now, cls_name + '_raw', self.target_frame)

                # project location to base plane
                location[2] = extent / 2
                print('[%s] location mean on table: %f, %f, %f' % (cls_name, location[0], location[1], location[2]))
                print('-------------------------------------------')

                # publish tf
                self.br.sendTransform(location, [0, 0, 0, 1], now, cls_name, self.target_frame)

                # publish tf detection
                indexes[cls] += 1
                name = cls_name + '_%02d' % (indexes[cls])
                tf_name = os.path.join("posecnn", name)

                # send another transformation as bounding box (mis-used)
                n = np.linalg.norm(roi[2:6])
                x1 = roi[2] / n
                y1 = roi[3] / n
                x2 = roi[4] / n
                y2 = roi[5] / n
                self.br.sendTransform([n, now.secs, roi[6]], [x1, y1, x2, y2], now, tf_name + '_roi', self.target_frame)

                # publish marker
                marker = Marker()
                marker.header.frame_id = self.target_frame
                marker.header.stamp = now
                marker.id = cls
                marker.type = Marker.SPHERE;
                marker.action = Marker.ADD;
                marker.pose.position.x = location[0]
                marker.pose.position.y = location[1]
                marker.pose.position.z = location[2]
                marker.pose.orientation.x = 0.
                marker.pose.orientation.y = 0.
                marker.pose.orientation.z = 0.
                marker.pose.orientation.w = 1.
                marker.scale.x = .05
                marker.scale.y = .05
                marker.scale.z = .05

                if cfg.TEST.ROS_CAMERA == 'Azure':
                    marker.color.a = .3
                elif cfg.TEST.ROS_CAMERA == 'D415':
                    marker.color.a = 1.
                marker.color.r = self.dataset._class_colors_test[cls][0] / 255.0
                marker.color.g = self.dataset._class_colors_test[cls][1] / 255.0
                marker.color.b = self.dataset._class_colors_test[cls][2] / 255.0
                markers.append(marker)
        self.viz_pub.publish(MarkerArray(markers))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--instance', dest='instance_id', help='PoseCNN instance id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.device = torch.device('cuda:{:d}'.format(0))
    print('GPU device {:d}'.format(args.gpu_id))
    cfg.gpu_id = args.gpu_id
    cfg.instance_id = args.instance_id

    # dataset
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[0]).cuda(device=cfg.device)
    cudnn.benchmark = True

    # image listener
    network.eval()
    listener = ImageListener(network, dataset)
    while not rospy.is_shutdown():
       listener.run_network()

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from __future__ import print_function
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
import os, time, sys
import os.path
import numpy as np
from transforms3d.euler import quat2euler
from transforms3d.quaternions import *
from imgaug import augmenters as iaa
import cv2
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from ycb_render.ycb_renderer import *
import torch.nn.functional as F
from config.config import cfg


class ycb_video_dataset(data.Dataset):
    def __init__(self, class_ids, object_names, class_model_num, path, list_file,
                 detection_path='./detections/posecnn_detections/'):
        self.dataset_type = 'ycb'
        self.path = path

        list_file = open(list_file)
        name_list = list_file.read().splitlines()

        file_start = name_list[0]
        file_end = name_list[1]

        start_num = int(file_start.split('/')[1])
        end_num = int(file_end.split('/')[1])

        file_num = np.arange(start_num, end_num+1)

        file_list = []
        for i in range(len(file_num)):
            file_list.append(file_start.split('/')[0]+'/{:06}'.format(file_num[i]))

        self.file_list = file_list

        self.files = [path+item+'-color.png' for item in file_list]

        kf_path = path[:-5] + 'keyframes/keyframe.txt'
        self.kfs = [line.rstrip('\n') for line in open(kf_path)]

        print('***CURRENT SEQUENCE INCLUDES {} IMAGES***'.format(len(self.files)))

        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            self.class_names_all = class_name_file.read().split('\n')

        assert len(object_names)==1, "current only support loading the information for one object !!!"
        self.object_names = object_names
        self.class_ids = class_ids
        self.class_model_number = class_model_num

        # object list
        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            self.object_name_list = class_name_file.read().split('\n')

        self.obj_idx = self.object_name_list.index(self.object_names[0])

        # posecnn for initialization
        self.posecnn_results_dir = detection_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, depth, pose, intrinsics, mask = self.load(self.files[idx])

        image = torch.from_numpy(image).float()/255.0
        depth = torch.from_numpy(depth)
        mask = torch.from_numpy(mask)

        instance_mask = torch.zeros(3 * self.class_model_number)
        instance_mask[self.class_ids[0]*3 : self.class_ids[0]*3 + 3] = 1
        class_mask = (instance_mask==1)

        # check if this frame is keyframe
        image_id = self.file_list[idx]
        indexes_kf = [i for i, x in enumerate(self.kfs) if x == image_id]
        is_kf = False
        if len(indexes_kf) > 0:
            is_kf = True

        # use posecnn results for initialization
        center = np.array([0, 0])
        z = 0
        t_est = np.array([0, 0, 0], dtype=np.float32)
        q_est = np.array([1, 0, 0, 0], dtype=np.float32)
        if is_kf:
            posecnn_file = self.posecnn_results_dir + '{:06}.mat'.format(indexes_kf[0])
            posecnn_result = loadmat(posecnn_file)
            rois = posecnn_result['rois']

            poses = posecnn_result['poses']

            obj_list = rois[:, 1].astype(np.int).tolist()

            clsId_obj = [i for i, x in enumerate(obj_list) if x == (self.obj_idx+1)]

            if len(clsId_obj) > 0:
                center[0] = (rois[clsId_obj, 2] + rois[clsId_obj, 4]) / 2
                center[1] = (rois[clsId_obj, 3] + rois[clsId_obj, 5]) / 2
                z = poses[clsId_obj, 6]
                t_est = poses[clsId_obj, 4:][0]
                q_est = poses[clsId_obj, :4][0]

        return image, depth, pose, intrinsics, class_mask, self.file_list[idx], is_kf, center, z, t_est, q_est, mask

    def load(self, fn):
        # make sure the object is in the frame
        meta_name = fn.replace('color.png', 'meta.mat')
        dataset_info = loadmat(meta_name)
        class_ids_dataset = np.squeeze(dataset_info['cls_indexes'])
        object_names_dataset = []
        for class_id_dataset in class_ids_dataset:
            object_names_dataset.append(self.class_names_all[class_id_dataset-1])
        for object_name in self.object_names:
            assert object_name in object_names_dataset, \
                "specified object is not in this sequence, try another sequence !!!"

        img = np.array(Image.open(fn))

        # read semantic labels
        mask_name = fn.replace('color', 'label')
        mask = np.array(Image.open(mask_name))
        mask = np.expand_dims(mask, 2)

        # read the depth image
        depth_name = fn.replace('color', 'depth')
        depth = np.array(Image.open(depth_name))
        cam_scale = dataset_info['factor_depth'][0][0].astype(np.float32)
        depth = depth / cam_scale
        depth = np.expand_dims(depth, 2)

        # get bounding box
        box = fn.replace('color.png', 'box.txt')
        dtype = {'names': ('obj_cat', 'ul_x', 'ul_y', 'dr_x', 'dr_y'),
                 'formats': ('S40', 'f', 'f', 'f', 'f')}
        box_file = np.loadtxt(box, dtype=dtype, delimiter=' ')
        row_object = np.where(box_file['obj_cat'] == self.object_names[0].encode()) # todo: multiple objects
        bbox_object = np.squeeze(np.array([np.floor(box_file['ul_x'][row_object]),
                             np.floor(box_file['ul_y'][row_object]),
                             np.ceil(box_file['dr_x'][row_object]),
                             np.ceil(box_file['dr_y'][row_object])
                             ]).astype(np.int), axis=1)

        bbox_object += [-10, -10, 10, 10]
        if bbox_object[0] < 0:
            bbox_object[0] = 0

        if bbox_object[1] < 0:
            bbox_object[1] = 0

        if bbox_object[2] > img.shape[1]:
            bbox_object[2] = img.shape[1]

        if bbox_object[3] > img.shape[0]:
            bbox_object[3] = img.shape[0]

        # load GT pose
        poses = dataset_info['poses']
        pose = poses[:, :, row_object[0][0]]

        # load camera intrinsics
        intrinsics = dataset_info['intrinsic_matrix']

        return img, depth, pose, intrinsics, mask


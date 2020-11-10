import os, sys
import os.path
import cv2
import random
import glob
import torch
import torch.utils.data as data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import datasets
import yaml
import scipy
try:
    import cPickle  # Use cPickle on Python 2.7
except ImportError:
    import pickle as cPickle
import imgaug.augmenters as iaa

from config.config import cfg
from transforms3d.quaternions import mat2quat, quat2mat
from utils.se3 import *
from utils.pose_error import *
from utils.cython_bbox import bbox_overlaps
from utils.RoIAlign.layer_utils.roi_layers import ROIAlign
from datasets.render_ycb_dataset import chromatic_transform, add_noise_cuda


_SUBJECTS = [
    '20200709-weiy',
    '20200813-ceppner',
    '20200820-amousavian',
    '20200903-ynarang',
    '20200908-yux',
    '20200918-ftozetoramos',
    '20200928-ahanda',
    '20201002-dieterf',
    '20201015-ychao',
    '20201022-lmanuelli',
]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

_MANO_JOINTS = [
    'wrist',
    'thumb_mcp',
    'thumb_pip',
    'thumb_dip',
    'thumb_tip',
    'index_mcp',
    'index_pip',
    'index_dip',
    'index_tip',
    'middle_mcp',
    'middle_pip',
    'middle_dip',
    'middle_tip',
    'ring_mcp',
    'ring_pip',
    'ring_dip',
    'ring_tip',
    'little_mcp',
    'little_pip',
    'little_dip',
    'little_tip'
]

_MANO_JOINT_CONNECT = [
    [0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
    [0,  5], [ 5,  6], [ 6,  7], [ 7,  8],
    [0,  9], [ 9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]

_BOP_EVAL_SUBSAMPLING_FACTOR = 4


class dex_ycb_encoder_dataset(data.Dataset):

    def __init__(self, setup, split, obj_list, renderer):

        self._setup = setup
        self._split = split
        self._color_format = "color_{:06d}.jpg"
        self._depth_format = "aligned_depth_to_color_{:06d}.png"
        self._label_format = "labels_{:06d}.npz"
        self._height = 480
        self._width = 640
        self.h = cfg.TRAIN.RENDER_SZ
        self.w = cfg.TRAIN.RENDER_SZ
        self.output_size=cfg.TRAIN.INPUT_IM_SIZE
        self.target_size = 128
        self.renderer = renderer

        # paths
        self._name = 'dex_ycb_' + setup + '_' + split
        self._image_set = split
        self._dex_ycb_path = self._get_default_path()
        path = os.path.join(self._dex_ycb_path, 'data')
        self._data_dir = path
        self._calib_dir = os.path.join(self._data_dir, "calibration")
        self._model_dir = os.path.join(self._data_dir, "models")

        self._obj_file = {
            k: os.path.join(self._model_dir, v, "textured_simple.obj")
            for k, v in _YCB_CLASSES.items()
        }

        # define all the classes
        self._classes_all = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
        self._num_classes_all = len(self._classes_all)
        self._class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192)]
        self._extents_all = self._load_object_extents()
        self._posecnn_class_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21]

        self.render_dist_list = cfg.TRAIN.RENDER_DIST
        self.bbox_3d_sz_list = []
        for render_dist in self.render_dist_list:
            bbox_sz = 2 * cfg.TRAIN.U0 / cfg.TRAIN.FU * render_dist
            self.bbox_3d_sz_list.append(bbox_sz)

        self.dropout_seq = iaa.Sequential([iaa.CoarseDropout(p=(0, 0.3), size_percent=(0.05, 0.15))])

        # compute class index
        class_index = []
        for name in obj_list:
            for i in range(self._num_classes_all):
                if name == self._classes_all[i]:
                    class_index.append(i)
                    break
        print('class index:', class_index)
        self._class_index = class_index

        # select a subset of classes
        self._classes = obj_list
        self._num_classes = len(self._classes)
        self._class_colors = [self._class_colors_all[i] for i in class_index]
        self._extents = self._extents_all[class_index]
        self._points, self._points_all = self._load_object_points(self._classes, self._extents)

        # Seen subjects, camera views, grasped objects.
        if self._setup == 's0':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 != 4]
            if self._split == 'val':
                subject_ind = [0, 1]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 == 4]
            if self._split == 'test':
                subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i % 5 == 4]

        # Unseen subjects.
        if self._setup == 's1':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))
            if self._split == 'val':
                subject_ind = [6]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))
            if self._split == 'test':
                subject_ind = [7, 8]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = list(range(100))

        # Unseen camera views.
        if self._setup == 's2':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5]
                sequence_ind = list(range(100))
            if self._split == 'val':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [6]
                sequence_ind = list(range(100))
            if self._split == 'test':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [7]
                sequence_ind = list(range(100))

        # Unseen grasped objects.
        if self._setup == 's3':
            if self._split == 'train':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [
                    i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)
                ]
            if self._split == 'val':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
            if self._split == 'test':
                subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
                sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]

        self._subjects = [_SUBJECTS[i] for i in subject_ind]
        self._serials = [_SERIALS[i] for i in serial_ind]
        self._intrinsics = []
        for s in self._serials:
            intr_file = os.path.join(self._calib_dir, "intrinsics", "{}_{}x{}.yml".format(s, self._width, self._height))
            with open(intr_file, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
            intr = intr['color']
            self._intrinsics.append(intr)

        # build mapping
        self._sequences = []
        self._mapping = []
        self._ycb_ids = []
        offset = 0
        for n in self._subjects:
            seq = sorted(os.listdir(os.path.join(self._data_dir, n)))
            seq = [os.path.join(n, s) for s in seq]
            assert len(seq) == 100
            seq = [seq[i] for i in sequence_ind]
            for _, q in enumerate(seq):
                meta_file = os.path.join(self._data_dir, q, "meta.yml")
                with open(meta_file, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                # skip videos without the target
                ycb_ids = np.array(meta['ycb_ids'])
                index = np.where(ycb_ids == self._class_index[0] + 1)[0]
                if len(index) == 0:
                    continue
                c = np.arange(len(self._serials))
                f = np.arange(meta['num_frames'])
                f, c = np.meshgrid(f, c)
                c = c.ravel()
                f = f.ravel()
                s = offset * np.ones_like(c)
                m = np.vstack((s, c, f)).T
                self._mapping.append(m)
                self._ycb_ids.append(meta['ycb_ids'])
                self._sequences += [q]
                offset += 1
        self._mapping = np.vstack(self._mapping)

        # sample a subset for training
        if split == 'train':
            self._mapping = self._mapping[::10]

        # dataset size
        self._size = len(self._mapping)
        print('dataset %s with images %d' % (self._name, self._size))

        self.lb_shift = cfg.TRAIN.SHIFT_MIN
        self.ub_shift = cfg.TRAIN.SHIFT_MAX
        self.lb_scale = cfg.TRAIN.SCALE_MIN
        self.ub_scale = cfg.TRAIN.SCALE_MAX


    def __len__(self):
        return self._size


    def get_bop_id_from_idx(self, idx):
        s, c, f = map(lambda x: x.item(), self._mapping[idx])
        scene_id = s * len(self._serials) + c
        im_id = f
        return scene_id, im_id


    def __getitem__(self, idx):
        s, c, f = self._mapping[idx]

        is_testing = f % _BOP_EVAL_SUBSAMPLING_FACTOR == 0
        if self._split == 'test' and not is_testing:
            sample = {'is_testing': is_testing}
            return sample

        scene_id, im_id = self.get_bop_id_from_idx(idx)
        video_id = '%04d' % (scene_id)
        image_id = '%06d' % (im_id)

        d = os.path.join(self._data_dir, self._sequences[s], self._serials[c])
        roidb = {
            'color_file': os.path.join(d, self._color_format.format(f)),
            'depth_file': os.path.join(d, self._depth_format.format(f)),
            'label_file': os.path.join(d, self._label_format.format(f)),
            'intrinsics': self._intrinsics[c],
            'ycb_ids': self._ycb_ids[s]
        }

        image, image_target, pose_cam, mask, shift, scale, affine, depth_input, depth_target= self._get_image_item(roidb)

        # shift = torch.from_numpy(np.asarray(shift)).float()
        shift = (shift - self.lb_shift)/(self.ub_shift - self.lb_shift)  # normalize to [0, 1]
        # scale = torch.from_numpy(np.asarray(scale)).float()
        scale = (scale - self.lb_scale)/(self.ub_scale - self.lb_scale)  # normalize to [0, 1]

        # pad image if necessary
        if image.size(0) != self.output_size[0] or image.size(1) != self.output_size[1]:
            out_h = self.output_size[1]
            out_w = self.output_size[0]
            c1_n = int(np.floor(out_h / 2.0 - image.size(1) / 2.0))
            c1_p = int(out_h - image.size(1) - c1_n)
            c2_n = int(np.floor(out_w / 2.0 - image.size(0) / 2.0))
            c2_p = int(out_w - image.size(0) - c2_n)
            image = F.pad(image.permute(2, 0, 1), [c2_n, c2_p, c1_n, c1_p]).permute(1, 2, 0)
            depth_input = F.pad(depth_input.permute(2, 0, 1), [c2_n, c2_p, c1_n, c1_p]).permute(1, 2, 0)
            mask = F.pad(mask.permute(2, 0, 1), [c2_n, c2_p, c1_n, c1_p]).permute(1, 2, 0)

        # and pad the target
        if image_target.size(0) != self.target_size or image_target.size(1) != self.target_size:
            out_h = self.target_size
            out_w = self.target_size
            c1_n = int(np.floor(out_h / 2.0 - image_target.size(1) / 2.0))
            c1_p = int(out_h - image_target.size(1) - c1_n)
            c2_n = int(np.floor(out_w / 2.0 - image_target.size(0) / 2.0))
            c2_p = int(out_w - image_target.size(0) - c2_n)
            image_target = F.pad(image_target.permute(2, 0, 1), [c2_n, c2_p, c1_n, c1_p]).permute(1, 2, 0)
            depth_target = F.pad(depth_target.permute(2, 0, 1), [c2_n, c2_p, c1_n, c1_p]).permute(1, 2, 0)

        # add noise to the input image
        image_cpu = torch.clamp(image * 255.0, 0, 255.0).byte().cpu().numpy()
        image_cpu = chromatic_transform(image_cpu)
        image = torch.from_numpy(image_cpu).cuda().float() / 255.0
        image = add_noise_cuda(image)
        image = torch.clamp(image, min=0.0, max=1.0)

        # add noise to the input depth
        depth_input = add_noise_cuda(depth_input)
        depth_input = torch.clamp(depth_input, min=0.0, max=1.0)
        depth_input = torch.from_numpy(self.dropout_seq(images=depth_input.unsqueeze(0).cpu().numpy())[0]).cuda()

        # randomized location in the image
        roi_center = np.array([np.random.uniform(cfg.TRAIN.ROI_CENTER_RANGE[0], cfg.TRAIN.ROI_CENTER_RANGE[2]),
                               np.random.uniform(cfg.TRAIN.ROI_CENTER_RANGE[1], cfg.TRAIN.ROI_CENTER_RANGE[3])], dtype=np.float32)
        roi_center = torch.from_numpy(roi_center)
        roi_size = np.random.uniform(cfg.TRAIN.ROI_SIZE_RANGE[0], cfg.TRAIN.ROI_SIZE_RANGE[1])

        scales_u = self.target_size * 1.0 / roi_size
        scales_v = self.target_size * 1.0 / roi_size
        affine_roi = torch.zeros(2, 3).float()
        affine_roi[0, 2] = (self.output_size[0] / 2 - roi_center[0]) / self.output_size[0] * 2 * scales_u
        affine_roi[1, 2] = (self.output_size[1] / 2 - roi_center[1]) / self.output_size[1] * 2 * scales_v
        affine_roi[0, 0] = scales_u
        affine_roi[1, 1] = scales_v

        return image.permute(2, 0, 1), image_target.permute(2, 0, 1), pose_cam, \
               mask.permute(2, 0, 1).float(), shift, scale, affine, \
               roi_center, roi_size, affine_roi, \
               depth_input.permute(2, 0, 1), depth_target.permute(2, 0, 1)

    def _get_image_item(self, roidb):

        # rgba
        rgba = cv2.imread(roidb['color_file'], cv2.IMREAD_UNCHANGED)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba
        im_color = im.astype('float') / 255.0
        im_color = im_color[:, :, (2, 1, 0)]
        im_cuda = torch.from_numpy(im_color).cuda()

        # depth image
        im_depth = cv2.imread(roidb['depth_file'], cv2.IMREAD_UNCHANGED)
        im_depth = im_depth.astype('float') / 1000.0
        im_depth_cuda = torch.from_numpy(im_depth).unsqueeze(2).cuda()

        # parse data
        cls_indexes = np.array(roidb['ycb_ids']).flatten()
        classes = np.array(self._class_index)
        fx_data = roidb['intrinsics']['fx']
        fy_data = roidb['intrinsics']['fy']
        px_data = roidb['intrinsics']['ppx']
        py_data = roidb['intrinsics']['ppy']
        intrinsic_matrix = np.eye(3, dtype=np.float32)
        intrinsic_matrix[0, 0] = fx_data
        intrinsic_matrix[1, 1] = fy_data
        intrinsic_matrix[0, 2] = px_data
        intrinsic_matrix[1, 2] = py_data
        label = np.load(roidb['label_file'])

        # label image
        im_label = label['seg']
        im_label_cuda = torch.from_numpy(im_label).float().unsqueeze(2).cuda()

        # foreground mask
        seg = torch.from_numpy((im_label != 0).astype(np.float32))
        mask = seg.unsqueeze(2).float().cuda()

        # poses
        poses = label['pose_y']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (1, 3, 4))
        num = poses.shape[0]
        assert num == len(cls_indexes), 'number of poses not equal to number of objects'

        # render poses to get the target image
        render_dist = cfg.TRAIN.RENDER_DIST[0]
        bbox_3d_sz = self.bbox_3d_sz_list[0]
        poses_all = []
        qt = np.zeros((7, ), dtype=np.float32)
        ind = np.where(cls_indexes == self._class_index[0] + 1)[0][0]
        RT = poses[ind, :, :]
        qt[0] = 0
        qt[1] = 0
        qt[2] = render_dist
        qt[3:] = mat2quat(RT[:, :3])
        poses_all.append(qt)
        pose_cam = qt.copy()

        # define tensors
        frames_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        seg_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        pc_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
            
        # rendering
        self.renderer.set_poses(poses_all)
        self.renderer.set_light_pos(cfg.TRAIN.TARGET_LIGHT1_POS)
        target_color = [cfg.TRAIN.TARGET_INTENSITY, cfg.TRAIN.TARGET_INTENSITY, cfg.TRAIN.TARGET_INTENSITY]
        self.renderer.set_light_color(target_color)
        self.renderer.render([0], frames_target_cuda, seg_target_cuda, pc2_tensor=pc_target_cuda)
        frames_target_cuda = frames_target_cuda.flip(0)
        frames_target_cuda = frames_target_cuda[:, :, :3]
        seg_target_cuda = seg_target_cuda.flip(0)
        seg_target = seg_target_cuda[:, :, 0].clone().cpu().numpy()
        pc_target_cuda = pc_target_cuda.flip(0)
        pc_target_cuda = pc_target_cuda[:, :, :3]
        depth_target = pc_target_cuda[:, :, [2]]
        depth_target[depth_target > 0] = (depth_target[depth_target > 0] - render_dist) / bbox_3d_sz + 0.5

        # project the 3D translation to get the center
        uv = np.zeros((1, 3), dtype=np.float32)
        uv[0, 0] = fx_data * RT[0, 3] / RT[2, 3] + px_data
        uv[0, 1] = fy_data * RT[1, 3] / RT[2, 3] + py_data
        uv[0, 2] = 1
        z = np.zeros((1, 1), dtype=np.float32)
        z[0, 0] = RT[2, 3]

        # crop the rois from input image
        frames_cuda, scale_roi = self.trans_zoom_uvz_cuda(im_cuda, uv, z, fx_data, fy_data, target_distance=render_dist)
        depth_input, scale_roi = self.trans_zoom_uvz_cuda(im_depth_cuda, uv, z, fx_data, fy_data, target_distance=render_dist)
        label_input, scale_roi = self.trans_zoom_uvz_cuda(im_label_cuda, uv, z, fx_data, fy_data, target_distance=render_dist)
        mask_input, scale_roi = self.trans_zoom_uvz_cuda(mask, uv, z, fx_data, fy_data, target_distance=render_dist)
        mask_input[:, :, 0] = 1.0
        depth_input[depth_input > 0] = (depth_input[depth_input > 0] - RT[2, 3]) / bbox_3d_sz + 0.5

        label_numpy = label_input[:, :, 0].cpu().numpy()
        non_occluded = np.sum(np.logical_and(seg_target > 0, label_numpy == self._class_index[0] + 1)).astype(np.float)
        occluded_ratio = 1 - non_occluded / np.sum(seg_target>0).astype(np.float)
        if occluded_ratio > 0.9:
            frames_target_cuda *= 0
            pc_target_cuda *= 0
            depth_target *= 0

        # affine transformation
        shift = np.float32([np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)])
        scale = np.random.uniform(self.lb_scale, self.ub_scale)
        affine_matrix = np.float32([[scale, 0, shift[0] / self._width], [0, scale, shift[1] / self._height]])

        shift = [np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)]
        shift = np.array(shift, dtype=np.float)
        scale = np.random.uniform(self.lb_scale, self.ub_scale)
        affine = torch.from_numpy(np.float32([[scale, 0, shift[0]/128.0], [0, scale, shift[1]/128.0]])).float()

        return frames_cuda, frames_target_cuda, pose_cam.astype(np.float32), mask_input, shift, scale, affine, \
               depth_input, depth_target


    def trans_zoom_uvz_cuda(self, image, uvs, zs, pf_fu, pf_fv, target_distance=2.5, out_size=128):

        bbox_u = target_distance * (1 / zs) / cfg.TRAIN.FU * pf_fu * out_size
        bbox_v = target_distance * (1 / zs) / cfg.TRAIN.FV * pf_fv * out_size
        boxes = np.zeros((uvs.shape[0], 5), dtype=np.float32)
        boxes[:, 1] = uvs[:, 0] - bbox_u[:, 0] / 2.0
        boxes[:, 2] = uvs[:, 1] - bbox_v[:, 0] / 2.0
        boxes[:, 3] = uvs[:, 0] + bbox_u[:, 0] / 2.0
        boxes[:, 4] = uvs[:, 1] + bbox_v[:, 0] / 2.0
        boxes = torch.from_numpy(boxes).cuda()

        image = image.permute(2, 0, 1).float().unsqueeze(0).cuda()
        out = self.CropAndResizeFunction(image, boxes)[0].permute(1, 2, 0)
        uv_scale = target_distance * (1 / zs) / cfg.TRAIN.FU * pf_fu

        '''
        for i in range(out.shape[0]):
            roi = out[i].permute(1, 2, 0).cpu().numpy()
            roi = np.clip(roi, 0, 1)
            im = roi * 255
            im = im.astype(np.uint8)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.imshow(im[:, :, (2, 1, 0)])
            plt.show()
        #'''

        return out, uv_scale


    def CropAndResizeFunction(self, image, rois):
        return ROIAlign((128, 128), 1.0, 0)(image, rois)


    def _get_default_path(self):
        """
        Return the default path where YCB_Video is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'DEX_YCB')


    def _load_object_extents(self):
        extents = np.zeros((self._num_classes_all, 3), dtype=np.float32)
        for i in range(self._num_classes_all):
            point_file = os.path.join(self._model_dir, self._classes_all[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points = np.loadtxt(point_file)
            extents[i, :] = 2 * np.max(np.absolute(points), axis=0)
        return extents


    def _load_object_points(self, classes, extents):

        points = [[] for _ in range(len(classes))]
        num = np.inf
        num_classes = len(classes)
        for i in range(num_classes):
            point_file = os.path.join(self._model_dir, classes[i], 'points.xyz')
            print(point_file)
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((num_classes, num, 3), dtype=np.float32)
        for i in range(num_classes):
            points_all[i, :, :] = points[i][:num, :]

        return points, points_all

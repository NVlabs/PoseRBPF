import os
import sys
import yaml
import numpy as np
import torch
import torch.utils.data as data
import numpy as np
import numpy.random as npr
import cv2
import copy
import glob
import scipy

import datasets
from config.config import cfg
from transforms3d.quaternions import mat2quat, quat2mat
from utils.se3 import *
from utils.pose_error import *
from utils.cython_bbox import bbox_overlaps

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

class dex_ycb_dataset(data.Dataset):

    def __init__(self, setup, split, obj_list):
        self._setup = setup
        self._split = split
        self._color_format = "color_{:06d}.jpg"
        self._depth_format = "aligned_depth_to_color_{:06d}.png"
        self._label_format = "labels_{:06d}.npz"
        self._height = 480
        self._width = 640

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
            self._sequences += seq
            for i, q in enumerate(seq):
                meta_file = os.path.join(self._data_dir, q, "meta.yml")
                with open(meta_file, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                c = np.arange(len(self._serials))
                f = np.arange(meta['num_frames'])
                f, c = np.meshgrid(f, c)
                c = c.ravel()
                f = f.ravel()
                s = (offset + i) * np.ones_like(c)
                m = np.vstack((s, c, f)).T
                self._mapping.append(m)
                self._ycb_ids.append(meta['ycb_ids'])
            offset += len(seq)
        self._mapping = np.vstack(self._mapping)

        # sample a subset for training
        if split == 'train':
            self._mapping = self._mapping[::10]

        # dataset size
        self._size = len(self._mapping)
        print('dataset %s with images %d' % (self._name, self._size))


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
        scene_id, im_id = self.get_bop_id_from_idx(idx)
        video_id = '%04d' % (scene_id)
        image_id = '%06d' % (im_id)

        # posecnn result path
        posecnn_result_path = os.path.join(self._dex_ycb_path, 'results_posecnn', self._name, video_id + '_' + image_id + '.mat')

        d = os.path.join(self._data_dir, self._sequences[s], self._serials[c])
        roidb = {
            'color_file': os.path.join(d, self._color_format.format(f)),
            'depth_file': os.path.join(d, self._depth_format.format(f)),
            'label_file': os.path.join(d, self._label_format.format(f)),
            'intrinsics': self._intrinsics[c],
            'ycb_ids': self._ycb_ids[s],
            'posecnn': posecnn_result_path,
        }

        # Get the input image blob
        im_color, im_depth = self._get_image_blob(roidb['color_file'], roidb['depth_file'])

        # build the label blob
        im_label, intrinsic_matrix, poses, gt_boxes, poses_result, rois_result, labels_result \
            = self._get_label_blob(roidb, self._num_classes)

        is_syn = 0
        im_scale = 1.0
        im_info = np.array([im_color.shape[1], im_color.shape[2], im_scale, is_syn], dtype=np.float32)

        sample = {'image_color': im_color[:, :, (2, 1, 0)],
                  'image_depth': im_depth,
                  'label': im_label,
                  'intrinsic_matrix': intrinsic_matrix,
                  'gt_poses': poses,
                  'gt_boxes': gt_boxes,
                  'poses_result': poses_result,
                  'rois_result': rois_result,
                  'labels_result': labels_result,
                  'extents': self._extents,
                  'points': self._points_all,
                  'im_info': im_info,
                  'video_id': video_id,
                  'image_id': image_id}

        if self._split == 'test':
            sample['is_testing'] = is_testing

        return sample


    def _get_image_blob(self, color_file, depth_file):    

        # rgba
        rgba = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:,:,:3])
            alpha = rgba[:,:,3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba
        im_color = im.astype('float') / 255.0

        # depth image
        im_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        im_depth = im_depth.astype('float') / 1000.0

        return im_color, im_depth


    def _get_label_blob(self, roidb, num_classes):
        """ build the label blob """

        # parse data
        cls_indexes = roidb['ycb_ids']
        classes = np.array(self._class_index)
        fx = roidb['intrinsics']['fx']
        fy = roidb['intrinsics']['fy']
        px = roidb['intrinsics']['ppx']
        py = roidb['intrinsics']['ppy']
        intrinsic_matrix = np.eye(3, dtype=np.float32)
        intrinsic_matrix[0, 0] = fx
        intrinsic_matrix[1, 1] = fy
        intrinsic_matrix[0, 2] = px
        intrinsic_matrix[1, 2] = py
        label = np.load(roidb['label_file'])

        # label image
        im_label = label['seg']

        # poses
        poses = label['pose_y']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (1, 3, 4))
        num = poses.shape[0]
        assert num == len(cls_indexes), 'number of poses not equal to number of objects'

        # bounding boxes
        gt_boxes = np.zeros((num, 5), dtype=np.float32)
        for i in range(num):
            cls = int(cls_indexes[i]) - 1
            ind = np.where(classes == cls)[0]
            if len(ind) > 0:
                R = poses[i, :, :3]
                T = poses[i, :, 3]

                # compute box
                x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
                x3d[0, :] = self._points_all[ind,:,0]
                x3d[1, :] = self._points_all[ind,:,1]
                x3d[2, :] = self._points_all[ind,:,2]
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = R
                RT[:, 3] = T
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        
                gt_boxes[i, 0] = np.min(x2d[0, :])
                gt_boxes[i, 1] = np.min(x2d[1, :])
                gt_boxes[i, 2] = np.max(x2d[0, :])
                gt_boxes[i, 3] = np.max(x2d[1, :])
                gt_boxes[i, 4] = ind

        # load posecnn result if available
        if os.path.exists(roidb['posecnn']):
            result = scipy.io.loadmat(roidb['posecnn'])
            n = result['poses'].shape[0]
            poses_result = np.zeros((n, 9), dtype=np.float32)
            poses_result[:, 0] = 1
            poses_result[:, 1] = result['rois'][:, 1]
            poses_result[:, 2:] = result['poses']
            rois_result = result['rois'].copy()
            labels_result = result['labels'].copy()

            # select the classes
            index = []
            for i in range(poses_result.shape[0]):
                cls = self._posecnn_class_indexes[int(poses_result[i, 1])] - 1
                ind = np.where(classes == cls)[0]
                if len(ind) > 0:
                    index.append(i)
                    poses_result[i, 1] = ind
                    rois_result[i, 1] = ind
            poses_result = poses_result[index, :]
            rois_result = rois_result[index, :]
        else:
            print('no posecnn result %s' % (roidb['posecnn']))
            poses_result = np.zeros((0, 9), dtype=np.float32)
            rois_result = np.zeros((0, 7), dtype=np.float32)
            labels_result = np.zeros((0, 1), dtype=np.float32)

        poses = poses.transpose((1, 2, 0))
        return im_label, intrinsic_matrix, poses, gt_boxes, poses_result, rois_result, labels_result


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


    def write_dop_results(self, output_dir):
        # only write the result file
        filename = os.path.join(output_dir, 'posecnn_' + self.name + '.csv')
        f = open(filename, 'w')
        f.write('scene_id,im_id,obj_id,score,R,t,time\n')

        if cfg.TEST.POSE_REFINE:
            filename_refined = os.path.join(output_dir, 'posecnn_' + self.name + '_refined.csv')
            f1 = open(filename_refined, 'w')
            f1.write('scene_id,im_id,obj_id,score,R,t,time\n')

        # list the mat file
        images_color = []
        filename = os.path.join(output_dir, '*.mat')
        files = sorted(glob.glob(filename))

        # for each image
        for i in range(len(files)):
            filename = os.path.basename(files[i])

            # parse filename
            pos = filename.find('_')
            scene_id = int(filename[:pos])
            im_id = int(filename[pos+1:-4])

            # load result
            print(files[i])
            result = scipy.io.loadmat(files[i])
            if len(result['rois']) == 0:
                continue

            rois = result['rois']
            num = rois.shape[0]
            for j in range(num):
                obj_id = cfg.TRAIN.CLASSES[int(rois[j, 1])]
                if obj_id == 0:
                    continue
                score = rois[j, -1]
                run_time = -1

                # pose from network
                R = quat2mat(result['poses'][j, :4].flatten())
                t = result['poses'][j, 4:]
                line = '{scene_id},{im_id},{obj_id},{score},{R},{t},{time}\n'.format(
                    scene_id=scene_id,
                    im_id=im_id,
                    obj_id=obj_id,
                    score=score,
                    R=' '.join(map(str, R.flatten().tolist())),
                    t=' '.join(map(str, t.flatten().tolist())),
                    time=run_time)
                f.write(line)

                if cfg.TEST.POSE_REFINE:
                    R = quat2mat(result['poses_refined'][j, :4].flatten())
                    t = result['poses_refined'][j, 4:]
                    line = '{scene_id},{im_id},{obj_id},{score},{R},{t},{time}\n'.format(
                        scene_id=scene_id,
                        im_id=im_id,
                        obj_id=obj_id,
                        score=score,
                        R=' '.join(map(str, R.flatten().tolist())),
                        t=' '.join(map(str, t.flatten().tolist())),
                        time=run_time)
                    f1.write(line)

        # close file
        f.close()
        if cfg.TEST.POSE_REFINE:
            f1.close()


    # compute box
    def compute_box(self, cls, intrinsic_matrix, RT):
        classes = np.array(cfg.TRAIN.CLASSES)
        ind = np.where(classes == cls)[0]
        x3d = np.ones((4, self._points_all.shape[1]), dtype=np.float32)
        x3d[0, :] = self._points_all[ind,:,0]
        x3d[1, :] = self._points_all[ind,:,1]
        x3d[2, :] = self._points_all[ind,:,2]
        x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
        x1 = np.min(x2d[0, :])
        y1 = np.min(x2d[1, :])
        x2 = np.max(x2d[0, :])
        y2 = np.max(x2d[1, :])
        return [x1, y1, x2, y2]


    def evaluation(self, output_dir):
        self.write_dop_results(output_dir)

        filename = os.path.join(output_dir, 'results_posecnn.mat')
        if os.path.exists(filename):
            results_all = scipy.io.loadmat(filename)
            print('load results from file')
            print(filename)
            distances_sys = results_all['distances_sys']
            distances_non = results_all['distances_non']
            errors_rotation = results_all['errors_rotation']
            errors_translation = results_all['errors_translation']
            results_seq_id = results_all['results_seq_id'].flatten()
            results_frame_id = results_all['results_frame_id'].flatten()
            results_object_id = results_all['results_object_id'].flatten()
            results_cls_id = results_all['results_cls_id'].flatten()
        else:
            # save results
            num_max = 200000
            num_results = 2
            distances_sys = np.zeros((num_max, num_results), dtype=np.float32)
            distances_non = np.zeros((num_max, num_results), dtype=np.float32)
            errors_rotation = np.zeros((num_max, num_results), dtype=np.float32)
            errors_translation = np.zeros((num_max, num_results), dtype=np.float32)
            results_seq_id = np.zeros((num_max, ), dtype=np.float32)
            results_frame_id = np.zeros((num_max, ), dtype=np.float32)
            results_object_id = np.zeros((num_max, ), dtype=np.float32)
            results_cls_id = np.zeros((num_max, ), dtype=np.float32)

            # for each image
            count = -1
            for i in range(len(self._mapping)):

                s, c, f = self._mapping[i]
                is_testing = f % _BOP_EVAL_SUBSAMPLING_FACTOR == 0
                if not is_testing:
                    continue

                # intrinsics
                intrinsics = self._intrinsics[c]
                intrinsic_matrix = np.eye(3, dtype=np.float32)
                intrinsic_matrix[0, 0] = intrinsics['fx']
                intrinsic_matrix[1, 1] = intrinsics['fy']
                intrinsic_matrix[0, 2] = intrinsics['ppx']
                intrinsic_matrix[1, 2] = intrinsics['ppy']

                # parse keyframe name
                scene_id, im_id = self.get_bop_id_from_idx(i)

                # load result
                filename = os.path.join(output_dir, '%04d_%06d.mat' % (scene_id, im_id))
                print(filename)
                result = scipy.io.loadmat(filename)

                # load gt
                d = os.path.join(self._data_dir, self._sequences[s], self._serials[c])
                label_file = os.path.join(d, self._label_format.format(f))
                label = np.load(label_file)
                cls_indexes = np.array(self._ycb_ids[s]).flatten()

                # poses
                poses = label['pose_y']
                if len(poses.shape) == 2:
                    poses = np.reshape(poses, (1, 3, 4))
                num = poses.shape[0]
                assert num == len(cls_indexes), 'number of poses not equal to number of objects'

                # instance label
                im_label = label['seg']
                instance_ids = np.unique(im_label)
                if instance_ids[0] == 0:
                    instance_ids = instance_ids[1:]
                if instance_ids[-1] == 255:
                    instance_ids = instance_ids[:-1]

                # for each gt poses
                for j in range(len(instance_ids)):
                    cls = instance_ids[j]

                    # find the number of pixels of the object
                    pixels = np.sum(im_label == cls)
                    if pixels < 200:
                        continue
                    count += 1

                    # find the pose
                    object_index = np.where(cls_indexes == cls)[0][0]
                    RT_gt = poses[object_index, :, :]
                    box_gt = self.compute_box(cls, intrinsic_matrix, RT_gt)

                    results_seq_id[count] = scene_id
                    results_frame_id[count] = im_id
                    results_object_id[count] = object_index
                    results_cls_id[count] = cls

                    # network result
                    roi_index = []
                    if len(result['rois']) > 0:
                        for k in range(result['rois'].shape[0]):
                            ind = int(result['rois'][k, 1])
                            if cls == cfg.TRAIN.CLASSES[ind]:
                                roi_index.append(k)

                    # select the roi
                    if len(roi_index) > 1:
                        # overlaps: (rois x gt_boxes)
                        roi_blob = result['rois'][roi_index, :]
                        roi_blob = roi_blob[:, (0, 2, 3, 4, 5, 1)]
                        gt_box_blob = np.zeros((1, 5), dtype=np.float32)
                        gt_box_blob[0, 1:] = box_gt
                        overlaps = bbox_overlaps(
                            np.ascontiguousarray(roi_blob[:, :5], dtype=np.float),
                            np.ascontiguousarray(gt_box_blob, dtype=np.float)).flatten()
                        assignment = overlaps.argmax()
                        roi_index = [roi_index[assignment]]

                    if len(roi_index) > 0:
                        RT = np.zeros((3, 4), dtype=np.float32)
                        ind = int(result['rois'][roi_index, 1])
                        points = self._points[ind]

                        # pose from network
                        RT[:3, :3] = quat2mat(result['poses'][roi_index, :4].flatten())
                        RT[:, 3] = result['poses'][roi_index, 4:]
                        distances_sys[count, 0] = adi(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                        distances_non[count, 0] = add(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                        errors_rotation[count, 0] = re(RT[:3, :3], RT_gt[:3, :3])
                        errors_translation[count, 0] = te(RT[:, 3], RT_gt[:, 3])

                        # pose after depth refinement
                        if cfg.TEST.POSE_REFINE:
                            RT[:3, :3] = quat2mat(result['poses_refined'][roi_index, :4].flatten())
                            RT[:, 3] = result['poses_refined'][roi_index, 4:]
                            distances_sys[count, 1] = adi(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                            distances_non[count, 1] = add(RT[:3, :3], RT[:, 3],  RT_gt[:3, :3], RT_gt[:, 3], points)
                            errors_rotation[count, 1] = re(RT[:3, :3], RT_gt[:3, :3])
                            errors_translation[count, 1] = te(RT[:, 3], RT_gt[:, 3])
                        else:
                            distances_sys[count, 1] = np.inf
                            distances_non[count, 1] = np.inf
                            errors_rotation[count, 1] = np.inf
                            errors_translation[count, 1] = np.inf
                    else:
                        distances_sys[count, :] = np.inf
                        distances_non[count, :] = np.inf
                        errors_rotation[count, :] = np.inf
                        errors_translation[count, :] = np.inf

            distances_sys = distances_sys[:count+1, :]
            distances_non = distances_non[:count+1, :]
            errors_rotation = errors_rotation[:count+1, :]
            errors_translation = errors_translation[:count+1, :]
            results_seq_id = results_seq_id[:count+1]
            results_frame_id = results_frame_id[:count+1]
            results_object_id = results_object_id[:count+1]
            results_cls_id = results_cls_id[:count+1]

            results_all = {'distances_sys': distances_sys,
                       'distances_non': distances_non,
                       'errors_rotation': errors_rotation,
                       'errors_translation': errors_translation,
                       'results_seq_id': results_seq_id,
                       'results_frame_id': results_frame_id,
                       'results_object_id': results_object_id,
                       'results_cls_id': results_cls_id }

            filename = os.path.join(output_dir, 'results_posecnn.mat')
            scipy.io.savemat(filename, results_all)

        # print the results
        # for each class
        import matplotlib.pyplot as plt
        max_distance = 0.1
        index_plot = [0, 1]
        color = ['r', 'b']
        leng = ['PoseCNN', 'PoseCNN refined']
        num = len(leng)
        ADD = np.zeros((self._num_classes_all, num), dtype=np.float32)
        ADDS = np.zeros((self._num_classes_all, num), dtype=np.float32)
        TS = np.zeros((self._num_classes_all, num), dtype=np.float32)
        classes = list(copy.copy(self._classes_all))
        classes[0] = 'all'
        for k in range(self._num_classes_all):
            fig = plt.figure(figsize=(16.0, 10.0))
            if k == 0:
                index = range(len(results_cls_id))
            else:
                index = np.where(results_cls_id == k)[0]

            if len(index) == 0:
                continue
            print('%s: %d objects' % (classes[k], len(index)))

            # distance symmetry
            ax = fig.add_subplot(2, 3, 1)
            lengs = []
            for i in index_plot:
                D = distances_sys[index, i]
                ind = np.where(D > max_distance)[0]
                D[ind] = np.inf
                d = np.sort(D)
                n = len(d)
                accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
                plt.plot(d, accuracy, color[i], linewidth=2)
                ADDS[k, i] = VOCap(d, accuracy)
                lengs.append('%s (%.2f)' % (leng[i], ADDS[k, i] * 100))
                print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

            ax.legend(lengs)
            plt.xlabel('Average distance threshold in meter (symmetry)')
            plt.ylabel('accuracy')
            ax.set_title(classes[k])

            # distance non-symmetry
            ax = fig.add_subplot(2, 3, 2)
            lengs = []
            for i in index_plot:
                D = distances_non[index, i]
                ind = np.where(D > max_distance)[0]
                D[ind] = np.inf
                d = np.sort(D)
                n = len(d)
                accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
                plt.plot(d, accuracy, color[i], linewidth=2)
                ADD[k, i] = VOCap(d, accuracy)
                lengs.append('%s (%.2f)' % (leng[i], ADD[k, i] * 100))
                print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

            ax.legend(lengs)
            plt.xlabel('Average distance threshold in meter (non-symmetry)')
            plt.ylabel('accuracy')
            ax.set_title(classes[k])

            # translation
            ax = fig.add_subplot(2, 3, 3)
            lengs = []
            for i in index_plot:
                D = errors_translation[index, i]
                ind = np.where(D > max_distance)[0]
                D[ind] = np.inf
                d = np.sort(D)
                n = len(d)
                accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
                plt.plot(d, accuracy, color[i], linewidth=2)
                TS[k, i] = VOCap(d, accuracy)
                lengs.append('%s (%.2f)' % (leng[i], TS[k, i] * 100))
                print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

            ax.legend(lengs)
            plt.xlabel('Translation threshold in meter')
            plt.ylabel('accuracy')
            ax.set_title(classes[k])

            # rotation histogram
            count = 4
            for i in index_plot:
                ax = fig.add_subplot(2, 3, count)
                D = errors_rotation[index, i]
                ind = np.where(np.isfinite(D))[0]
                D = D[ind]
                ax.hist(D, bins=range(0, 190, 10), range=(0, 180))
                plt.xlabel('Rotation angle error')
                plt.ylabel('count')
                ax.set_title(leng[i])
                count += 1

            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            filename = output_dir + '/' + classes[k] + '.png'
            # plt.show()
            plt.savefig(filename)

        # print ADD
        print('==================ADD======================')
        for k in range(len(classes)):
            print('%s: %f' % (classes[k], ADD[k, 0]))
        for k in range(len(classes)-1):
            print('%f' % (ADD[k+1, 0]))
        print('%f' % (ADD[0, 0]))
        print(cfg.TRAIN.SNAPSHOT_INFIX)
        print('===========================================')

        # print ADD-S
        print('==================ADD-S====================')
        for k in range(len(classes)):
            print('%s: %f' % (classes[k], ADDS[k, 0]))
        for k in range(len(classes)-1):
            print('%f' % (ADDS[k+1, 0]))
        print('%f' % (ADDS[0, 0]))
        print(cfg.TRAIN.SNAPSHOT_INFIX)
        print('===========================================')

        # print ADD
        print('==================ADD refined======================')
        for k in range(len(classes)):
            print('%s: %f' % (classes[k], ADD[k, 1]))
        for k in range(len(classes)-1):
            print('%f' % (ADD[k+1, 1]))
        print('%f' % (ADD[0, 1]))
        print(cfg.TRAIN.SNAPSHOT_INFIX)
        print('===========================================')

        # print ADD-S
        print('==================ADD-S refined====================')
        for k in range(len(classes)):
            print('%s: %f' % (classes[k], ADDS[k, 1]))
        for k in range(len(classes)-1):
            print('%f' % (ADDS[k+1, 1]))
        print('%f' % (ADDS[0, 1]))
        print(cfg.TRAIN.SNAPSHOT_INFIX)
        print('===========================================')

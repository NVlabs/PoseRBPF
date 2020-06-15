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

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    resize_ratio = 0.5 * (w/new_w + h/new_h)

    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img, resize_ratio

class YcbVideoDataset(data.Dataset):
    def __init__(self, class_ids, object_names, class_model_num, path):
        self.path = path
        files = os.listdir(self.path)
        self.files = [item for item in files if item[-9:] == 'color.png']
        self.files = [os.path.join(self.path, item) for item in self.files]

        meta_filename = self.files[0].replace('color.png', 'meta.mat')
        dataset_info = loadmat(meta_filename)
        class_ids_dataset = np.squeeze(dataset_info['cls_indexes'])
        object_names_dataset = []

        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            class_names_all = class_name_file.read().split('\n')
        for class_id_dataset in class_ids_dataset:
            object_names_dataset.append(class_names_all[class_id_dataset-1])

        for object_name in object_names:
            assert object_name in object_names_dataset, \
                "specified object is not in this sequence, try another sequence !!!"

        # todo: implementation of multiple object, may just extend the dataset size
        assert len(object_names)==1, "current only support loading the information for one object !!!"
        self.object_names = object_names
        self.class_ids = class_ids
        self.class_model_number = class_model_num

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, pose, intrinsics, img_origin = self.load(self.files[idx])
        image = torch.from_numpy(image)
        image_origin = torch.from_numpy(img_origin)

        class_info = torch.zeros(image.size(1), image.size(2), self.class_model_number)
        class_info[:, :, self.class_ids[0]] = 1
        instance_mask = torch.zeros(3 * self.class_model_number)
        instance_mask[self.class_ids[0]*3 : self.class_ids[0]*3 + 3] = 1
        class_mask = (instance_mask==1)

        return image, pose, intrinsics, image_origin, class_info.permute(2, 0, 1), class_mask, self.class_ids[0]

    def load(self, fn):
        label = fn.replace('color', 'label')
        img = np.array(Image.open(fn))
        # seg = np.array(Image.open(label))

        # img = cv2.resize(img[:,80:560,:], (256,256))
        # seg = cv2.resize(seg[:,80:560], (256,256), interpolation = cv2.INTER_NEAREST)

        # seg = (seg == 2) * 1 + (seg ==1) * 2 + (seg == 10) * 3

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

        image_cropped = img[(bbox_object[1]):(bbox_object[3]), (bbox_object[0]):(bbox_object[2]), :]

        # image_cropped = resizeAndPad(image_cropped, (128, 128))

        # load GT pose
        meta_name = fn.replace('color.png', 'meta.mat')
        poses = loadmat(meta_name)['poses']

        intrinsics = loadmat(meta_name)['intrinsic_matrix']

        pose = poses[:,:,row_object[0][0]]

        img = img.transpose(2,0,1).astype(np.float32) / 255.0

        image_cropped = image_cropped.transpose(2,0,1).astype(np.float32) / 255.0

        return image_cropped, pose, intrinsics, img

# use translation information to crop and resize original image to transform the croped image to the training space
def trans_zoom_cpu(image, translation, intrinsics, target_distance=2.5, out_size=128):
    obj_center = np.matmul(intrinsics, translation)
    if obj_center.shape[0] == 1:
        obj_center = obj_center[0]
    obj_center_xy = obj_center / obj_center[2]
    cbox_size = out_size * target_distance / translation[2]

    if obj_center_xy[0] <  0 or obj_center_xy[1] < 0 or \
        obj_center_xy[1] > image.shape[0] or obj_center_xy[0] > image.shape[1]:
        return np.zeros((128, 128, 3))

    cbox_x = np.asarray([obj_center_xy[0] - cbox_size / 2, obj_center_xy[0] + cbox_size / 2]).astype(np.int)
    cbox_x_clip = np.clip(cbox_x, 0, image.shape[1]-1).astype(np.int)
    cbox_y = np.asarray([obj_center_xy[1] - cbox_size / 2, obj_center_xy[1] + cbox_size / 2]).astype(np.int)
    cbox_y_clip = np.clip(cbox_y, 0, image.shape[0]-1).astype(np.int)

    image_cropped = np.zeros((cbox_y[1] - cbox_y[0], cbox_x[1] - cbox_x[0], 3), dtype=np.uint8)
    image_cropped[(cbox_y_clip[0] - cbox_y[0]) : (cbox_y_clip[0] - cbox_y[0] - cbox_y_clip[0] + cbox_y_clip[1]),
                (cbox_x_clip[0] - cbox_x[0]) : (cbox_x_clip[0] - cbox_x[0] - cbox_x_clip[0] + cbox_x_clip[1]),
                :] = image[cbox_y_clip[0]:cbox_y_clip[1],
                           cbox_x_clip[0]:cbox_x_clip[1],
                            :]

    image_cropped = cv2.resize(image_cropped, (128, 128))

    return image_cropped

class YcbVideoTestDataset(data.Dataset):
    def __init__(self, class_ids, object_names, class_model_num, path, list_file):
        self.path = path

        list_file = open(list_file)
        file_list = list_file.read().splitlines()

        self.files = [path+item+'-color.png' for item in file_list]

        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            self.class_names_all = class_name_file.read().split('\n')

        # todo: implementation of multiple object, may just extend the dataset size
        assert len(object_names)==1, "current only support loading the information for one object !!!"
        self.object_names = object_names
        self.class_ids = class_ids
        self.class_model_number = class_model_num

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, pose, intrinsics, img_origin, resize_ratio, image_c_cropped = self.load(self.files[idx])
        image = torch.from_numpy(image)
        image_origin = img_origin
        image_c_cropped = torch.from_numpy(image_c_cropped)

        class_info = torch.zeros(image.size(1), image.size(2), self.class_model_number)
        class_info[:, :, self.class_ids[0]] = 1
        instance_mask = torch.zeros(3 * self.class_model_number)
        instance_mask[self.class_ids[0]*3 : self.class_ids[0]*3 + 3] = 1
        class_mask = (instance_mask==1)

        return image, pose, intrinsics, image_origin, class_info.permute(2, 0, 1), class_mask, self.class_ids[0], resize_ratio, \
                image_c_cropped

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

        image_cropped = img[(bbox_object[1]):(bbox_object[3]), (bbox_object[0]):(bbox_object[2]), :]

        # image_cropped, resize_ratio = resizeAndPad(image_cropped, (128, 128))
        resize_ratio = 1.0

        # load GT pose
        poses = dataset_info['poses']

        intrinsics = dataset_info['intrinsic_matrix']

        pose = poses[:,:,row_object[0][0]]

        obj_t = pose[:, 3]

        image_c_cropped = trans_zoom_cpu(img, obj_t, intrinsics)

        image_c_cropped = image_c_cropped.transpose(2,0,1).astype(np.float32) / 255.0

        image_cropped = image_cropped.transpose(2,0,1).astype(np.float32) / 255.0

        return image_cropped, pose, intrinsics, img, resize_ratio, image_c_cropped

class YcbVideoSeqDataset(data.Dataset):
    def __init__(self, class_ids, object_names, class_model_num, path, list_file):
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

        print('***CURRENT SEQUENCE INCLUDES {} IMAGES WITH {} KFS***'.format(len(self.files), len(self.kfs)))

        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            self.class_names_all = class_name_file.read().split('\n')

        # todo: implementation of multiple object, may just extend the dataset size
        assert len(object_names)==1, "current only support loading the information for one object !!!"
        self.object_names = object_names
        self.class_ids = class_ids
        self.class_model_number = class_model_num

        # object list
        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            self.object_name_list = class_name_file.read().split('\n')

        self.obj_idx = self.object_name_list.index(self.object_names[0])

        # posecnn for initialization
        self.use_posecnn = False
        self.posecnn_results_dir = '../results_PoseCNN_RSS2018/'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, pose, intrinsics, img_origin, resize_ratio, image_c_cropped, depth, mask = self.load(self.files[idx])
        image = torch.from_numpy(image)
        image_origin = img_origin
        image_c_cropped = torch.from_numpy(image_c_cropped)

        depth = torch.from_numpy(depth)
        mask = torch.from_numpy(mask)

        class_info = torch.zeros(image.size(1), image.size(2), self.class_model_number)
        class_info[:, :, self.class_ids[0]] = 1
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
        if self.use_posecnn:
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

            return image, pose, intrinsics, image_origin, class_info.permute(2, 0, 1), class_mask, self.class_ids[
                0], resize_ratio, \
                   image_c_cropped, self.file_list[idx], is_kf, center, z, t_est, q_est, depth, mask

        return image, pose, intrinsics, image_origin, class_info.permute(2, 0, 1), class_mask, self.class_ids[0], resize_ratio, \
                image_c_cropped, self.file_list[idx], is_kf, depth, mask

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

        # image_cropped = img[(bbox_object[1]):(bbox_object[3]), (bbox_object[0]):(bbox_object[2]), :]

        # image_cropped, resize_ratio = resizeAndPad(image_cropped, (128, 128))
        image_cropped = np.zeros((128, 128, 3))
        resize_ratio = 1.0

        # load GT pose
        poses = dataset_info['poses']

        intrinsics = dataset_info['intrinsic_matrix']

        pose = poses[:,:,row_object[0][0]]

        obj_t = pose[:, 3]

        image_c_cropped = trans_zoom_cpu(img, obj_t, intrinsics)

        image_c_cropped = image_c_cropped.transpose(2,0,1).astype(np.float32) / 255.0

        image_cropped = image_cropped.transpose(2,0,1).astype(np.float32) / 255.0

        return image_cropped, pose, intrinsics, img, resize_ratio, image_c_cropped, depth, mask

# dataset for training STN
import glob
from config.config import cfg, cfg_from_file
from utils_dpf.DPF_utils import trans_zoom_uvz_cpu
from datasets.render_ycb_dataset import add_noise

class YCB_Video_Trans_Dataset(data.Dataset):
    def __init__(self, ycb_root_folder, training_cfg):
        self.ycb_root_folder = ycb_root_folder
        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            self.object_name_list = class_name_file.read().split('\n')

        cfg_from_file(training_cfg)

        self.sequences = []
        for s in range(48):
            self.sequences.append('{:04}'.format(s))
        for s in np.arange(60, 92):
            self.sequences.append('{:04}'.format(s))

        self.object = cfg.TRAIN.OBJECTS

        # load all the files
        self.files = []

        filenames_dir = './datasets/YCB_Video_Dataset_Train/Filenames/{}.txt'.format(self.object[0])
        if os.path.exists(filenames_dir):
            with open(filenames_dir, 'r') as filenames:
                self.files = filenames.read().split('\n')
                self.files = self.files[:-2]
        else:
            for sequence in self.sequences:
                ycb_data_folder = ycb_root_folder + sequence + '/'
                for filename in glob.iglob(ycb_data_folder + "/*.mat"):
                    dataset_info = loadmat(filename)
                    class_ids_dataset = np.squeeze(dataset_info['cls_indexes'])
                    object_names_seq = []
                    for class_id_dataset in class_ids_dataset:
                        object_names_seq.append(self.object_name_list[class_id_dataset - 1])
                    if self.object[0] in object_names_seq:
                        self.files.append(filename)
                    else:
                        continue
                    if len(self.files) % 100 == 0:
                        print('current obj: {}, sequence: {}, {} images generated'. format(self.object[0], sequence, len(self.files) + 1))
            with open('./datasets/YCB_Video_Dataset_Train/Filenames/{}.txt'.format(self.object[0]), 'w') as f:
                for item in self.files:
                    f.write("%s\n" % item)

        self.uv_min = cfg.TRAIN.TRANS_SHIFT_MIN
        self.uv_max = cfg.TRAIN.TRANS_SHIFT_MAX
        self.scale_min = cfg.TRAIN.TRANS_SCALE_MIN
        self.scale_max = cfg.TRAIN.TRANS_SCALE_MAX

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        im_target, im_random, uv_shift, scale = self.load(self.files[idx])

        im_random = add_noise(im_random)

        im_target_cuda = torch.from_numpy(im_target).permute(2, 0, 1).float() / 255.0
        im_random_cuda = torch.from_numpy(im_random).permute(2, 0, 1).float() / 255.0

        scale_normalized = (scale - self.scale_min) / (self.scale_max - self.scale_min)
        shift_normalized = (uv_shift - self.uv_min) / (self.uv_max - self.uv_min)

        return im_target_cuda, im_random_cuda, scale_normalized, shift_normalized

    def load(self, fn):
        filename = fn
        image_name = filename[:-8] + 'color.png'
        bbox_name = image_name.replace('color.png', 'box.txt')

        # check if the object exist
        dataset_info = loadmat(filename)
        class_ids_dataset = np.squeeze(dataset_info['cls_indexes'])
        object_names_seq = []
        for class_id_dataset in class_ids_dataset:
            object_names_seq.append(self.object_name_list[class_id_dataset - 1])

        object_data_idx = object_names_seq.index(self.object[0])

        object_center = dataset_info['center'][object_data_idx]

        intrinsics = dataset_info['intrinsic_matrix']

        poses = dataset_info['poses']
        pose = poses[:, :, object_data_idx]
        obj_t = pose[:, 3]

        img = np.array(Image.open(image_name))
        im_target = trans_zoom_uvz_cpu(img, [object_center[0], object_center[1]],
                                        obj_t[2],
                                        intrinsics[0, 0],
                                        intrinsics[1, 1],
                                        u_shift=0,
                                        v_shift=0,
                                        target_distance=cfg.TRAIN.RENDER_DIST[0])

        uv_shift = np.random.uniform(self.uv_min, self.uv_max, size=(2,))
        scale = np.random.uniform(self.scale_min, self.scale_max)

        im_random = trans_zoom_uvz_cpu(img, [object_center[0], object_center[1]],
                                        obj_t[2] * scale,
                                        intrinsics[0, 0],
                                        intrinsics[1, 1],
                                        u_shift=uv_shift[0],
                                        v_shift=uv_shift[1],
                                        target_distance=cfg.TRAIN.RENDER_DIST[0])


        return im_target, im_random, uv_shift, scale


def K2P(w, h, fu, fv, u0, v0, zNear = 0.01, zFar = 100.0):
    L = -(u0) * zNear / fu
    R = (w - u0) * zNear / fu
    T = -v0 * zNear / fv
    B = (h - v0) * zNear / fv

    P = np.zeros((4,4)).astype(np.float32)
    P[0,0] = 2 * zNear / (R - L)
    P[1,1] = 2 * zNear / (T - B)
    P[2,0] = (R+L)/(L-R)
    P[2,1] = (T+B)/(B-T)
    P[2,2] = (zFar + zNear)/(zFar - zNear)
    P[2,3] = 1.0
    P[3,2] = (2*zFar*zNear)/(zNear-zFar)

    return P

# for debugging
import matplotlib.colors as colors
class YcbVideoDataset_Render(data.Dataset):
    def __init__(self, path):
        self.renderer = YCBTensorRenderer(640, 480)
        models = [
            # "025_mug",
            "003_cracker_box",
            # "011_banana",
        ]
        model_dir = '/home/xinke/Documents/modelAndpose'
        obj_paths = ['{}/models/{}/textured_simple.obj'.format(model_dir, item) for item in models]
        texture_paths = ['{}/models/{}/texture_map.png'.format(model_dir, item) for item in models]
        self.renderer.load_objects(obj_paths, texture_paths)
        self.renderer.set_camera([0, 1, 1], [0, 0, 0], [0, 1, 0])
        # self.renderer.set_fov(48.6)
        self.renderer.set_light_pos([0, 0, 0])
        self.renderer.set_light_color([1.4, 1.4, 1.4])
        self.renderer_cam_pos = [0, 0, 0]

        self.path = path
        files = os.listdir(self.path)
        self.files = [item for item in files if item[-9:] == 'color.png']
        self.files = [os.path.join(self.path, item) for item in self.files]

    # def render_with_pose_cuda(self, pose):
    #     """
    #     :param pose: pose estimation of the object 1x7
    #     :return: rendered image
    #     """
    #     self.renderer.set_poses([pose])
    #     self.renderer.render(self.frames_cuda, self.seg_cuda)
    #     frames_cuda = self.frames_cuda.flip(0)
    #     # frames_cuda
    #     frames_cuda = frames_cuda[:, :, :3].float() / 255.0
    #     pose_cam = self.renderer.get_poses()
    #
    #     return frames_cuda

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, pose, intrinsics, img_origin = self.load(self.files[idx])

        image = torch.from_numpy(image)
        image_origin = torch.from_numpy(img_origin)
        return image, pose, intrinsics, image_origin

    def load(self, fn):
        label = fn.replace('color', 'label')
        img = np.array(Image.open(fn))
        seg = np.array(Image.open(label))

        # img = cv2.resize(img[:,80:560,:], (256,256))
        # seg = cv2.resize(seg[:,80:560], (256,256), interpolation = cv2.INTER_NEAREST)

        # seg = (seg == 2) * 1 + (seg ==1) * 2 + (seg == 10) * 3

        # get bounding box
        box = fn.replace('color.png', 'box.txt')
        dtype = {'names': ('obj_cat', 'ul_x', 'ul_y', 'dr_x', 'dr_y'),
                 'formats': ('S40', 'f', 'f', 'f', 'f')}
        box_file = np.loadtxt(box, dtype=dtype, delimiter=' ')
        row_mug = np.where(box_file['obj_cat'] == b'025_mug')
        # row_mug = np.where(box_file['obj_cat'] == b'003_cracker_box')
        bbox_mug = np.squeeze(np.array([np.floor(box_file['ul_x'][row_mug]),
                             np.floor(box_file['ul_y'][row_mug]),
                             np.ceil(box_file['dr_x'][row_mug]),
                             np.ceil(box_file['dr_y'][row_mug])
                             ]).astype(np.int), axis=1)

        # load GT pose
        meta_name = fn.replace('color.png', 'meta.mat')
        poses = loadmat(meta_name)['poses']

        intrinsics = loadmat(meta_name)['intrinsic_matrix']

        pose = poses[:, :, row_mug[0][0]]

        img = img.astype(np.float32) / 255.0

        # P = K2P(w=128, h=128, fu=1066.778, fv=1056.487, u0=312.987, v0=241.311)
        P = K2P(w=640, h=480, fu=1066.778, fv=1056.487, u0=312.987, v0=241.311)

        self.renderer.set_perspective(P)

        Rc1o1 = pose[:3, :3]
        qco = mat2quat(Rc1o1)
        tco = pose[:3, 3]

        pose_gt = [0.0, 0.0, 1.25, 1, 0, 0, 0]
        pose_gt[:3] = tco
        pose_gt[3:] = qco

        frames_cuda = torch.cuda.ByteTensor(480, 640, 4)
        seg_cuda = torch.cuda.ByteTensor(480, 640, 4)
        self.renderer.set_poses([pose_gt])
        self.renderer.render(frames_cuda, seg_cuda)
        frames_cuda = frames_cuda.flip(0)
        # frames_cuda
        frames_cuda = frames_cuda[:, :, :3].float() / 255.0

        image_original = img.copy()
        image_render = frames_cuda.cpu().numpy()

        cup_real = img.copy()
        cup_real[seg!=14] = 0
        image_cropped_real = cup_real[bbox_mug[1]:bbox_mug[3], bbox_mug[0]:bbox_mug[2], :]
        # image_cropped_real = resizeAndPad(image_cropped_real, (128, 128))
        real_hsv = colors.rgb_to_hsv(image_cropped_real)

        cup_render = image_render.copy()
        image_cropped_render = cup_render[bbox_mug[1]:bbox_mug[3], bbox_mug[0]:bbox_mug[2], :]
        # image_cropped_render = resizeAndPad(image_cropped_render, (128, 128))
        render_hsv = colors.rgb_to_hsv(image_cropped_render)

        plt.figure(0)
        plt.subplot(251)
        lu1_real = real_hsv[:, :, 0].flatten()
        plt.hist(lu1_real * 180, bins=180, range=(0.0, 180.0), histtype='stepfilled', color='r', label='H')
        plt.subplot(252)
        lu2_real = real_hsv[:, :, 1].flatten()
        plt.hist(lu2_real * 256, bins=256, range=(0.0, 256.0), histtype='stepfilled', color='r', label='S')
        plt.subplot(253)
        lu3_real = real_hsv[:, :, 2].flatten()
        plt.hist(lu3_real * 180, bins=180, range=(0.0, 180.0), histtype='stepfilled', color='r', label='V')
        plt.subplot(254)
        plt.imshow(real_hsv)
        plt.subplot(255)
        plt.imshow(image_cropped_real)
        plt.subplot(256)
        lu1_render = render_hsv[:, :, 0].flatten()
        plt.hist(lu1_render * 180, bins=180, range=(0.0, 180.0), histtype='stepfilled', color='r', label='H')
        plt.subplot(257)
        lu2_render = render_hsv[:, :, 1].flatten()
        plt.hist(lu2_render * 256, bins=256, range=(0.0, 256.0), histtype='stepfilled', color='r', label='S')
        plt.subplot(258)
        lu3_render = render_hsv[:, :, 2].flatten()
        plt.hist(lu3_render * 256, bins=256, range=(0.0, 256.0), histtype='stepfilled', color='r', label='V')
        plt.subplot(259)
        plt.imshow(render_hsv)
        plt.subplot(2,5,10)
        plt.imshow(image_cropped_render)
        plt.show()


        weight_real = 0.2
        img[frames_cuda.cpu().numpy()>0] *= weight_real
        img += (1-weight_real)*frames_cuda.cpu().numpy()

        image_mix = img

        image_disp = np.concatenate((image_original, image_render, image_mix), axis=1)

        plt.figure(0)
        plt.imshow(image_disp)
        plt.show()

        image_cropped = img[bbox_mug[1]:bbox_mug[3], bbox_mug[0]:bbox_mug[2], :]

        # image_cropped = resizeAndPad(image_cropped, (128, 128))

        image_cropped = image_cropped.transpose(2,0,1).astype(np.float32)

        return image_cropped, pose, intrinsics, img


if __name__ == "__main__":
    dir = '/home/xinke/Documents/YCB_Video_Dataset/data/0039/'
    dataset = YcbVideoDataset(dir)
    l = (len(dataset))

    image, seg, image_cropped = dataset[0]

    image_disp = image.transpose(1, 2, 0)
    image_cropped_disp = image_cropped.transpose(1, 2, 0)
    plt.figure(0)
    plt.imshow(image_cropped_disp)

    plt.figure(1)
    plt.imshow(image_disp)

    plt.show()


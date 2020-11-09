# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md
from ycb_render.ycb_renderer import *
from config.config import cfg
import torch.utils.data as data
import torchvision
from PIL import Image
import os
import os.path
from transforms3d.quaternions import *
from transforms3d.euler import *
import torch
import torch.nn as nn
import cv2
import random
import torch.nn.functional as F
import glob
import time
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

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

def rgb_to_hls_cuda(image_float):
    V_max, C_max = torch.max(image_float, 2)
    V_min, C_min = torch.min(image_float, 2)

    R = image_float[:, :, 0]
    G = image_float[:, :, 1]
    B = image_float[:, :, 2]

    L = 0.5 * (V_max + V_min)

    S = (V_max - V_min)/(1 - torch.abs(V_max + V_min - 1))
    S[torch.isnan(S)] = 0

    H = torch.zeros_like(L).cuda()
    idx1 = C_max == 0
    idx2 = C_max == 1
    idx3 = C_max == 2
    H1 = 60 * (G - B) / (V_max - V_min)
    H2 = 120 + 60 * (B - R) / (V_max - V_min)
    H3 = 240 + 60 * (R - G) / (V_max - V_min)

    H[idx1] = H1[idx1]
    H[idx2] = H2[idx2]
    H[idx3] = H3[idx3]

    H[torch.isnan(H)] = 0

    H[H < 0] += 360
    H = (H/2).byte()
    S = (S*255.0).byte()
    L = (L*255.0).byte()

    return torch.cat((H.unsqueeze(2), L.unsqueeze(2), S.unsqueeze(2)), 2)

def hls_to_rgb_cuda(image_byte):
    H = image_byte[:, :, 0].float() * 2
    S = image_byte[:, :, 2].float() / 255.0
    L = image_byte[:, :, 1].float() / 255.0

    C = (1 - torch.abs(2 * L - 1)) * S
    H_ = H/60
    X = C * (1 - torch.abs(torch.fmod(H_, 2) - 1))

    R_ = torch.zeros_like(H)
    G_ = torch.zeros_like(H)
    B_ = torch.zeros_like(H)

    idx = ((H_ >= 0) + (H_ < 1) == 2)
    R_[idx] = C[idx]
    G_[idx] = X[idx]

    idx = ((H_ >= 1) + (H_ < 2) == 2)
    G_[idx] = C[idx]
    R_[idx] = X[idx]

    idx = ((H_ >= 2) + (H_ < 3) == 2)
    G_[idx] = C[idx]
    B_[idx] = X[idx]

    idx = ((H_ >= 3) + (H_ < 4) == 2)
    B_[idx] = C[idx]
    G_[idx] = X[idx]

    idx = ((H_ >= 4) + (H_ < 5) == 2)
    B_[idx] = C[idx]
    R_[idx] = X[idx]

    idx = ((H_ >= 5) + (H_ < 6) == 2)
    R_[idx] = C[idx]
    B_[idx] = X[idx]

    m = L - C/2

    R = R_ + m
    G = G_ + m
    B = B_ + m


    rgb = torch.cat((R.unsqueeze(2), G.unsqueeze(2), B.unsqueeze(2)), 2)

    rgb = torch.clamp(rgb, min=0.0, max=1.0)

    return rgb

def chromatic_transform_cuda(im_rgb, h_level=0.1, s_level=0.3, l_level=0.3, label=None):
    # make sure the im_rgb is in cuda and has been normalized

    torch.cuda.synchronize()
    time_start = time.time()
    im_hls = rgb_to_hls_cuda(im_rgb).float()
    torch.cuda.synchronize()
    time_elapse = time.time() - time_start

    print('rgb 2 hls cuda: ', time_elapse)

    torch.cuda.synchronize()
    time_start = time.time()
    d_h = (torch.rand(1) - 0.5) * h_level * 180
    d_l = (torch.rand(1) - 0.5) * s_level * 256
    d_s = (torch.rand(1) - 0.5) * l_level * 256

    im_hls[:, :, 0] = torch.fmod((im_hls[:,:,0] + d_h.cuda()), 180)
    im_hls[:, :, 1] = torch.clamp((d_l.cuda() + im_hls[:, :, 1]), 0, 255)
    im_hls[:, :, 2] = torch.clamp((d_s.cuda() + im_hls[:, :, 2]), 0, 255)

    im_hls = im_hls.byte()
    torch.cuda.synchronize()
    time_elapse = time.time() - time_start
    print('inject noise: ', time_elapse)


    torch.cuda.synchronize()
    time_start = time.time()
    im_rgb_new = hls_to_rgb_cuda(im_hls)
    torch.cuda.synchronize()
    time_elapse = time.time() - time_start

    print('hls 2 rgb cuda: ', time_elapse)

    return im_rgb_new

def add_noise_cuda(image, normalize=True):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.8:
        noise_level = random.uniform(0, 0.05)
        gauss = torch.randn_like(image) * noise_level
        noisy = image + gauss
        if normalize:
            noisy = torch.clamp(noisy, 0, 1.0)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = torch.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = torch.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = torch.ones(size)
        kernel_motion_blur = kernel_motion_blur.cuda() / size
        kernel_motion_blur = kernel_motion_blur.view(1, 1, size, size)
        kernel_motion_blur = kernel_motion_blur.repeat(image.size(2), 1,  1, 1)

        motion_blur_filter = nn.Conv2d(in_channels=image.size(2),
                                       out_channels=image.size(2),
                                       kernel_size=size,
                                       groups=image.size(2),
                                       bias=False,
                                       padding=int(size/2))

        motion_blur_filter.weight.data = kernel_motion_blur
        motion_blur_filter.weight.requires_grad = False
        noisy = motion_blur_filter(image.permute(2, 0, 1).unsqueeze(0))
        noisy = noisy.squeeze(0).permute(1, 2, 0)

    return noisy

def chromatic_transform(im, h_level=0.1, s_level=0.3, l_level=0.3, label=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    d_h = (np.random.rand(1) - 0.5) * h_level * 180
    d_l = (np.random.rand(1) - 0.5) * s_level * 256
    d_s = (np.random.rand(1) - 0.5) * l_level * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2RGB)

    if label is not None:
        I = np.where(label > 0)
        new_im[I[0], I[1], :] = im[I[0], I[1], :]
    return new_im

def add_noise(image):

    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.8:
        row,col,ch= image.shape
        mean = 0
        var = np.random.rand(1) * 0.2 * 256
        sigma = var**0.5
        gauss = sigma * np.random.randn(row,col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy

class DistractorDataset(data.Dataset):
    def __init__(self, distractor_dir, chrom_rand_level, size_crop=(128, 128)):
        self.dis_fns = []
        if distractor_dir is not None:
            for fn in os.listdir(distractor_dir):
                self.dis_fns.append(os.path.join(distractor_dir, fn))
        self.num_dis = len(self.dis_fns)

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(360),
                torchvision.transforms.RandomResizedCrop(size_crop, scale=(0.08, 1.5)),
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            ]
        )

        self.chrom_rand_level = chrom_rand_level

    def __len__(self):
        return self.num_dis

    def __getitem__(self, idx):
        return self.load(self.dis_fns[idx])

    def load(self, fn):

        image_dis = Image.open(fn).convert("RGB")
        image_dis = self.transform(image_dis)
        distractor = np.array(image_dis)

        distractor = cv2.cvtColor(distractor, cv2.COLOR_RGB2BGR)
        distractor = add_noise(chromatic_transform(distractor,
                                                   self.chrom_rand_level[0],
                                                   self.chrom_rand_level[1],
                                                   self.chrom_rand_level[2])
                               ).astype(np.uint8)
        distractor = cv2.cvtColor(distractor, cv2.COLOR_BGR2RGB)

        distractor = distractor.transpose(
            2, 0, 1).astype(
            np.float32) / 255.0

        return distractor

class ycb_multi_render_dataset(torch.utils.data.Dataset):
    def __init__(self, model_dir, model_names, render_size=128,
                 output_size=(128, 128),
                 target_size=128,
                 chrom_rand_level=cfg.TRAIN.CHM_RAND_LEVEL):

        self.render_size = render_size
        self.renderer = YCBRenderer(self.render_size, self.render_size, cfg.GPU_ID)

        self.h = self.render_size
        self.w = self.render_size
        self.models = model_names[:]
        self.main_models = model_names[:]

        self.output_size = output_size
        self.target_size = target_size

        # load all the models for occlusion
        self.use_occlusion = cfg.TRAIN.USE_OCCLUSION
        if self.use_occlusion:
            with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
                class_names_all = class_name_file.read().split('\n')
                for class_name in class_names_all:
                    if class_name not in self.models:
                        self.models.append(class_name)
            self.n_occluder = cfg.TRAIN.N_OCCLUDERS

        obj_paths = ['{}/ycb_models/{}/textured_simple.obj'.format(model_dir, item) for item in self.models]
        texture_paths = ['{}/ycb_models/{}/texture_map.png'.format(model_dir, item) for item in self.models]
        self.renderer.load_objects(obj_paths, texture_paths)

        # renderer properties
        self.renderer.set_camera_default()
        self.renderer.set_projection_matrix(self.w, self.h, fu=cfg.TRAIN.FU, fv=cfg.TRAIN.FU,
                                            u0=cfg.TRAIN.U0, v0=cfg.TRAIN.V0, znear=0.01, zfar=10)
        self.renderer.set_light_pos([0, 0, 0])

        # put in the configuration file
        self.lb_shift = cfg.TRAIN.SHIFT_MIN
        self.ub_shift = cfg.TRAIN.SHIFT_MAX
        self.lb_scale = cfg.TRAIN.SCALE_MIN
        self.ub_scale = cfg.TRAIN.SCALE_MAX

        # for lighting condition
        self.light_intensity_min = cfg.TRAIN.LIGHT_INT_MIN
        self.light_intensity_max = cfg.TRAIN.LIGHT_INT_MAX
        self.light_r_min = cfg.TRAIN.LIGHT_R_MIN
        self.light_r_max = cfg.TRAIN.LIGHT_R_MAX
        self.light_color_var = cfg.TRAIN.LIGHT_COLOR_VAR
        self.chrom_rand_level = chrom_rand_level

        # invalid scale region
        self.invalid_scale_prob = cfg.TRAIN.INVALID_SCALE_PROB
        self.invalid_scale_low = [cfg.TRAIN.INVALID_SCALE_LOW, cfg.TRAIN.SCALE_MIN]
        self.invalid_scale_up = [cfg.TRAIN.SCALE_MAX, cfg.TRAIN.INVALID_SCALE_UP]

        # normalized coordinate for depth
        self.embed_depth = cfg.TRAIN.DEPTH_EMBEDDING
        self.use_normalize_depth = cfg.TRAIN.NORMALIZE_DEPTH
        self.render_dist_list = cfg.TRAIN.RENDER_DIST

        self.bbox_3d_sz_list = []
        for render_dist in self.render_dist_list:
            bbox_sz = 2 * cfg.TRAIN.U0 / cfg.TRAIN.FU * render_dist
            self.bbox_3d_sz_list.append(bbox_sz)

        _, self.pose_list = torch.load('./data_files/poses_train.pth')
        self.dropout_seq = iaa.Sequential([
                                            iaa.CoarseDropout(p=(0, 0.3), size_percent=(0.05, 0.15))
                                            ])

    # @profile
    def __getitem__(self, image_index):

        image, image_target, pose_cam, mask, class_id, shift, scale, affine, depth_input, depth_target = self.load(image_index)

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

        class_info = torch.zeros(image.size(0), image.size(1), len(self.main_models))
        class_info[:, :, class_id] = 1
        instance_mask = torch.zeros(3 * len(self.main_models))
        instance_mask[class_id*3 : class_id*3 + 3] = 1
        class_mask = (instance_mask==1)

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

    def load(self, index):
        # render instances
        instance = random.sample(set(list(range(0, len(self.main_models)))), 1)
        render_dist = cfg.TRAIN.RENDER_DIST[instance[0]]
        bbox_3d_sz = self.bbox_3d_sz_list[instance[0]]

        poses = [self.pose_list[index].cpu().numpy()]
        poses[0][0:3] = [0, 0, cfg.TRAIN.RENDER_DIST[instance[0]]]

        # perturb this pose
        d_euler = np.random.uniform(-10*np.pi/180, 10*np.pi/180, (3,))
        d_quat = euler2quat(d_euler[0], d_euler[1], d_euler[2])
        poses[0][3:] = qmult(poses[0][3:], d_quat)

        # poses for object and occluder
        if self.use_occlusion and np.random.uniform(0, 1) < cfg.TRAIN.OC_PROB:

            # remove the main object from the occluders
            occluder_candidate_list = list(range(len(self.models)))
            occluder_candidate_list.remove(instance[0])
            occluder = random.sample(set(occluder_candidate_list), self.n_occluder)
            instance += occluder

            for i in range(self.n_occluder):
                pose_occ = poses[0].copy()
                pose_occ[:2] += np.random.uniform(cfg.TRAIN.OC_XY_RANGE[0], cfg.TRAIN.OC_XY_RANGE[1], size=(2,))
                pose_occ[2] -= np.random.uniform(cfg.TRAIN.OC_Z_RANGE[0], cfg.TRAIN.OC_Z_RANGE[1])
                pose_occ[3:] = euler2quat(np.random.uniform(-np.pi, np.pi),
                                                np.random.uniform(-np.pi, np.pi),
                                                np.random.uniform(-np.pi, np.pi))
                poses.append(pose_occ.copy())

        theta = np.random.uniform(-np.pi/2, np.pi/2, size=(3,))
        phi = np.random.uniform(0, np.pi/2, size=(3,))
        r = np.random.uniform(self.light_r_min, self.light_r_max, size=(3,))

        light1_pos = [r[0] * np.sin(theta[0]) * np.sin(phi[0]), r[0] * np.cos(phi[0]) + np.random.uniform(-2, 2), r[0] * np.cos(theta[0]) * np.sin(phi[0])] \
                     + cfg.TRAIN.TARGET_LIGHT1_POS
        self.renderer.set_light_pos(light1_pos)

        intensity = np.random.uniform(self.light_intensity_min, self.light_intensity_max, size=(3,))
        light1_color = [intensity[0] * item for item in [np.random.uniform(1-self.light_color_var, 1+self.light_color_var),
                                                     np.random.uniform(1-self.light_color_var, 1+self.light_color_var),
                                                     np.random.uniform(1-self.light_color_var, 1+self.light_color_var)]]
        self.renderer.set_light_color(light1_color)

        # declare cuda tensor
        frames_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        seg_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        pc_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)

        frames_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        seg_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        pc_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)

        self.renderer.set_poses(poses)
        # pose = self.renderer.get_poses()
        poses_cam = poses[0]

        self.renderer.render(instance, frames_cuda, seg_cuda, pc2_tensor=pc_cuda)
        frames_cuda = frames_cuda.flip(0)
        seg_cuda = seg_cuda.flip(0)
        frames_cuda = frames_cuda[:, :, :3]  # get rid of normalization for adding noise
        seg = seg_cuda[:, :, 0]
        shift = [np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)]
        shift = np.array(shift, dtype=np.float)
        pc_cuda = pc_cuda.flip(0)
        pc_cuda = pc_cuda[:, :, :3]
        scale = np.random.uniform(self.lb_scale, self.ub_scale)
        affine = torch.from_numpy(np.float32([[scale, 0, shift[0]/128.0], [0, scale, shift[1]/128.0]])).float()

        seg_input = seg.clone().cpu().numpy()

        # for fixed condition:
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

        depth_input = pc_cuda[:, :, [2]]
        depth_input[depth_input > 0] = (depth_input[depth_input > 0] - render_dist) / bbox_3d_sz + 0.5
        depth_target = pc_target_cuda[:, :, [2]]
        depth_target[depth_target > 0] = (depth_target[depth_target > 0] - render_dist) / bbox_3d_sz + 0.5

        non_occluded = np.sum(np.logical_and(seg_target > 0, seg_target == seg_input)).astype(np.float)

        occluded_ratio = 1 - non_occluded / np.sum(seg_target>0).astype(np.float)

        if occluded_ratio > 0.9:
            frames_target_cuda *= 0
            pc_target_cuda *= 0
            depth_target *= 0

        mask = (seg != 0).unsqueeze(2)

        return frames_cuda, frames_target_cuda, poses_cam.astype(np.float32), mask, instance[0], shift, scale, affine, \
               depth_input, depth_target

    def __len__(self):
        return len(self.pose_list)

# render on the fly
class ycb_codebook_online_generator(torch.utils.data.Dataset):
    def __init__(self, model_dir, model_names, render_dist, output_size=(128, 128),
                 fu=1066.778, fv=1056.487, u0=312.987/640*128, v0=241.311/480*128,
                 gpu_id=0, ts=15):
        self.renderer = YCBRenderer(128, 128, gpu_id)
        self.h = 128
        self.w = 128
        self.models = model_names
        if model_names[0][0] == '0':
            obj_paths = ['{}/ycb_models/{}/textured_simple.obj'.format(model_dir, item) for item in self.models]
            texture_paths = ['{}/ycb_models/{}/texture_map.png'.format(model_dir, item) for item in self.models]
        else:
            obj_paths = ['{}/ycb_models/{}/textured_simple.ply'.format(model_dir, item) for item in self.models]
            texture_paths = [''.format(model_dir, item) for item in self.models]

        self.renderer.load_objects(obj_paths, texture_paths)
        self.renderer.set_light_pos([0, 0, 0])
        self.renderer.set_light_color([1.2, 1.2, 1.2])
        self.renderer_cam_pos = [0, 0, 0]
        self.renderer.set_camera_default()
        self.renderer.set_projection_matrix(self.w, self.h,
                                            fu, fv,
                                            u0, v0, 0.01, 10)

        self.render_dist = render_dist
        self.output_size = output_size
        self.ts = ts
        self.bbox_sz = 2 * cfg.TRAIN.U0 / cfg.TRAIN.FU * render_dist
        self.pose_list = torch.load('./data_files/poses_codebook.pth')

    def __getitem__(self, image_index):

        image_target, pose_cam, depth_target = self.load(image_index)
        if image_target.size(0) != self.output_size[0] or image_target.size(1) != self.output_size[1]:
            out_h = self.output_size[1]
            out_w = self.output_size[0]
            c1_n = int(np.floor(out_h / 2.0 - image_target.size(1) / 2.0))
            c1_p = int(out_h - image_target.size(1) - c1_n)
            c2_n = int(np.floor(out_w / 2.0 - image_target.size(0) / 2.0))
            c2_p = int(out_w - image_target.size(0) - c2_n)
            image_target = F.pad(image_target.permute(2, 0, 1), [c2_n, c2_p, c1_n, c1_p]).permute(1, 2, 0)
            depth_target = F.pad(depth_target.permute(2, 0, 1), [c2_n, c2_p, c1_n, c1_p]).permute(1, 2, 0)
        image_target = image_target.permute(2, 0, 1)
        depth_target = depth_target.permute(2, 0, 1)

        return image_target, pose_cam, depth_target

    # @profile
    def load(self, index):
        # randomize
        while True:

            # end randomize
            poses = [self.pose_list[index].cpu().numpy()]

            # declare cuda tensor
            frames_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
            seg_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
            pc_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
            poses[0][0:3] = [0, 0, self.render_dist]
            self.renderer.set_poses(poses)
            pose = self.renderer.get_poses()
            poses_cam = np.array(pose)

            self.renderer.render([0], frames_target_cuda, seg_target_cuda, pc2_tensor=pc_target_cuda)
            frames_target_cuda = frames_target_cuda.flip(0)
            frames_target_cuda = frames_target_cuda[:, :, :3]
            seg_target_cuda = seg_target_cuda.flip(0)
            pc_target_cuda = pc_target_cuda.flip(0)
            pc_target_cuda = pc_target_cuda[:, :, :3]

            depth_target = pc_target_cuda[:, :, [2]]
            depth_target[depth_target > 0] = (depth_target[depth_target > 0] - self.render_dist) / self.bbox_sz + 0.5

            if torch.max(seg_target_cuda).data > 0:
                break

        return frames_target_cuda, poses_cam.astype(np.float32), depth_target

    def __len__(self):
        return len(self.pose_list)
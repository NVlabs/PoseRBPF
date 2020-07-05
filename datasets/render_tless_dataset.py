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
from datasets.render_ycb_dataset import *
from ycb_render.tless_renderer_tensor import *
import imgaug.augmenters as iaa

class tless_multi_render_dataset(torch.utils.data.Dataset):
    def __init__(self, model_dir, model_names, render_size=128, output_size=(128, 128),
                 target_size=128,
                 chrom_rand_level=cfg.TRAIN.CHM_RAND_LEVEL):

        self.render_size = render_size
        self.renderer = TLessTensorRenderer(self.render_size, self.render_size)

        self.h = render_size
        self.w = render_size
        self.models = model_names[:]
        self.main_models = model_names[:]
        self.output_size = output_size
        self.target_size = target_size
        self.use_occlusion = cfg.TRAIN.USE_OCCLUSION

        class_txt = './datasets/tless_classes.txt'
        class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                            (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64),
                            (155, 0, 0), (0, 155, 0), (0, 0, 155), (155, 155, 0), (155, 0, 155), (0, 155, 155),
                            (200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0),
                            (200, 0, 200), (0, 200, 200)
                            ]

        # load all the models
        if self.use_occlusion:
            with open(class_txt, 'r') as class_name_file:
                class_names_all = class_name_file.read().split('\n')
                for class_name in class_names_all:
                    if class_name not in self.models:
                        self.models.append(class_name)
            self.n_occluder = cfg.TRAIN.N_OCCLUDERS

        obj_paths = ['{}/tless_models/{}.ply'.format(model_dir, item) for item in self.models]
        texture_paths = ['' for cls in self.models]
        self.renderer.load_objects(obj_paths, texture_paths, class_colors_all)

        # renderer properties
        self.renderer.set_projection_matrix(self.w, self.h, cfg.TRAIN.FU, cfg.TRAIN.FU,
                                            render_size/2.0, render_size/2.0, 0.01, 10)
        self.renderer.set_camera_default()
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

        _, self.pose_list = torch.load('./data_files/poses_train.pth')

        # normalized coordinate for depth
        self.embed_depth = cfg.TRAIN.DEPTH_EMBEDDING
        self.use_normalize_depth = cfg.TRAIN.NORMALIZE_DEPTH
        self.render_dist_list = cfg.TRAIN.RENDER_DIST

        self.bbox_3d_sz_list = []
        for render_dist in self.render_dist_list:
            bbox_sz = 2 * cfg.TRAIN.U0 / cfg.TRAIN.FU * render_dist
            self.bbox_3d_sz_list.append(bbox_sz)

        self.dropout_seq = iaa.Sequential([
                                            iaa.CoarseDropout(p=(0, 0.3), size_percent=(0.05, 0.15))
                                            ])

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
            mask = F.pad(mask.permute(2, 0, 1), [c2_n, c2_p, c1_n, c1_p]).permute(1, 2, 0)
            depth_input = F.pad(depth_input.permute(2, 0, 1), [c2_n, c2_p, c1_n, c1_p]).permute(1, 2, 0)

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
        image = add_noise_cuda(image)
        image = torch.clamp(image, min=0.0, max=1.0)

        depth_input = add_noise_cuda(depth_input, normalize=self.use_normalize_depth)
        # add noise to the input depth
        if self.use_normalize_depth:
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
               mask.permute(2, 0, 1).float(), \
               shift, scale, affine, \
               roi_center, roi_size, affine_roi, \
               depth_input.permute(2, 0, 1), depth_target.permute(2, 0, 1)

    def load(self, index):
        instance = random.sample(set(list(range(0, len(self.main_models)))), 1)
        self.renderer.instances = instance
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
            self.renderer.instances += occluder
            poses = np.repeat(poses, self.n_occluder + 1, axis=0)
            # occluder translation
            poses[1:, :2] += np.random.uniform(cfg.TRAIN.OC_XY_RANGE[0], cfg.TRAIN.OC_XY_RANGE[1], size=(self.n_occluder,2))
            poses[1:, 2] -= np.random.uniform(cfg.TRAIN.OC_Z_RANGE[0], cfg.TRAIN.OC_Z_RANGE[1], size=(self.n_occluder,))
            # occluder rotation
            for i_ocr in range(self.n_occluder):
                poses[i_ocr+1, 3:] = euler2quat(np.random.uniform(-np.pi, np.pi),
                                                np.random.uniform(-np.pi, np.pi),
                                                np.random.uniform(-np.pi, np.pi))


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
        frames_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        seg_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        pc_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        pc_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)

        self.renderer.set_poses(poses)
        pose = self.renderer.get_poses()
        poses_cam = np.array(pose[0])

        cls_indexes = range(len(self.renderer.instances))
        self.renderer.render(cls_indexes, frames_cuda, seg_cuda, pc2_tensor=pc_cuda)
        frames_cuda = frames_cuda.flip(0)
        seg_cuda = seg_cuda.flip(0)
        frames_cuda = frames_cuda[:, :, :3]  # get rid of normalization for adding noise
        seg = seg_cuda[:, :, :3]
        shift = np.asarray([np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)], dtype=np.float32)
        scale = np.random.uniform(self.lb_scale, self.ub_scale)
        affine = torch.from_numpy(np.float32([[scale, 0, shift[0]/128.0], [0, scale, shift[1]/128.0]])).float()
        pc_cuda = pc_cuda.flip(0)
        pc_cuda = pc_cuda[:, :, :3]
        seg_input = seg[:, :, 0].clone().cpu().numpy()

        # for fixed condition:
        cls_indexes = [0]
        self.renderer.set_light_pos(cfg.TRAIN.TARGET_LIGHT1_POS)
        target_color = [cfg.TRAIN.TARGET_INTENSITY, cfg.TRAIN.TARGET_INTENSITY, cfg.TRAIN.TARGET_INTENSITY]
        self.renderer.set_light_color(target_color)
        self.renderer.render(cls_indexes, frames_target_cuda, seg_target_cuda, pc2_tensor=pc_target_cuda)
        frames_target_cuda = frames_target_cuda.flip(0)
        frames_target_cuda = frames_target_cuda[:, :, :3].float()
        seg_target_cuda = seg_target_cuda.flip(0)
        seg_target = seg_target_cuda[:, :, 0].clone().cpu().numpy()
        pc_target_cuda = pc_target_cuda.flip(0)
        pc_target_cuda = pc_target_cuda[:, :, :3]

        non_occluded = np.sum(np.logical_and(seg_target>0, seg_target==seg_input)).astype(np.float)

        occluded_ratio = 1 - non_occluded / np.sum(seg_target>0).astype(np.float)

        depth_input = pc_cuda[:, :, [2]]
        depth_target = pc_target_cuda[:, :, [2]]

        if self.use_normalize_depth:
            depth_input[depth_input > 0] = (depth_input[depth_input > 0] - render_dist) / bbox_3d_sz + 0.5
            depth_target[depth_target > 0] = (depth_target[depth_target > 0] - render_dist) / bbox_3d_sz + 0.5

        if occluded_ratio > 0.9:
            frames_target_cuda *= 0
            pc_target_cuda *= 0
            depth_target *= 0

        mask = (torch.sum(seg, dim=2) != 0).unsqueeze(2)

        return frames_cuda, frames_target_cuda, poses_cam.astype(np.float32), mask, instance[0], shift, scale, affine, \
               depth_input, depth_target

    def __len__(self):
        return len(self.pose_list)

# render on the fly
class tless_codebook_online_generator(torch.utils.data.Dataset):
    def __init__(self, model_dir, model_names, render_dist, output_size=(128, 128), gpu_id=0,ts=15):
        self.renderer = TLessTensorRenderer(128, 128, gpu_id=gpu_id)
        self.h = 128
        self.w = 128
        self.models = model_names
        obj_paths = ['{}/tless_models/{}.ply'.format(model_dir, item) for item in self.models]
        texture_paths = ['' for cls in self.models]

        self.renderer.load_objects(obj_paths, texture_paths)
        self.renderer.set_camera_default()
        self.renderer.set_projection_matrix(self.w, self.h, 1076.74064739, 1075.17825536, 64.0, 64.0, 0.01, 100)
        self.renderer.set_light_pos([0, 0, 0])

        self.render_dist = render_dist
        self.output_size = output_size

        self.ts = ts

        self.bbox_sz = 2 * cfg.TRAIN.U0 / cfg.TRAIN.FU * render_dist

        self.pose_list = torch.load('./data_files/poses_codebook.pth')

        self.use_normalize_depth = cfg.TRAIN.NORMALIZE_DEPTH

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


        # image_target = image_target[:, :, [2, 1, 0]]
        image_target = image_target.permute(2, 0, 1)
        depth_target = depth_target.permute(2, 0, 1)


        return image_target, pose_cam, depth_target

    # @profile
    def load(self, index):
        # randomize
        while True:
            self.renderer.set_light_pos([0, 0, 0])
            target_color = [1.0, 1.0, 1.0]
            self.renderer.set_light_color(target_color)

            # end randomize
            poses = [self.pose_list[index].cpu().numpy()]

            # declare cuda tensor
            frames_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
            seg_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
            pc_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
            poses[0][0:3] = [0, 0, self.render_dist]
            self.renderer.set_poses(poses)
            pose = self.renderer.get_poses()

            if np.max(np.abs(pose)) > 10:
                pose = [np.array([0,0,0,1,0,0,0]), np.array([0,0,0,1,0,0,0]), np.array([0,0,0,1,0,0,0])]
            poses_cam = np.array(pose)

            self.renderer.render([0], frames_target_cuda, seg_target_cuda, pc2_tensor=pc_target_cuda)
            frames_target_cuda = frames_target_cuda.flip(0)
            frames_target_cuda = frames_target_cuda[:, :, :3].float()
            seg_target_cuda = seg_target_cuda.flip(0)
            seg = seg_target_cuda[:, :, 0]

            pc_target_cuda = pc_target_cuda.flip(0)
            pc_target_cuda = pc_target_cuda[:, :, :3]
            depth_target = pc_target_cuda[:, :, [2]]
            if self.use_normalize_depth:
                depth_target[depth_target > 0] = (depth_target[depth_target > 0] - self.render_dist) / self.bbox_sz + 0.5

            if torch.max(seg_target_cuda).data > 0:
                break

        return frames_target_cuda, poses_cam.astype(np.float32), depth_target

    def __len__(self):
        return len(self.pose_list)







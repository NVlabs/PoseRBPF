# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from networks.aae_models import *
import numpy.ma as ma
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pdb
import glob
import copy
import scipy
from config.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
import pprint
from transforms3d.axangles import *
from .particle_filter import *
from .render_wrapper import *
from datasets.render_ycb_dataset import *
from datasets.render_tless_dataset import *
import matplotlib.patches as patches
import os
from functools import partial
import pickle
from .sdf_optimizer import *

class PoseRBPF:
    def __init__(self, obj_list, cfg_list, ckpt_list, codebook_list, obj_ctg, modality, cad_model_dir, visualize=True, refine=False, gpu_id=0):

        self.visualize = visualize
        self.obj_list = obj_list
        self.obj_ctg = obj_ctg

        # ycb class names
        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            self.obj_list_all = class_name_file.read().split('\n')

        # load the object information
        self.cfg_list = cfg_list
        self.gpu_id = gpu_id

        # load encoders and poses
        self.aae_list = []
        self.codebook_list = []
        self.codebook_list_depth = []
        self.instance_list = []
        self.rbpf_list = []
        self.rbpf_ok_list = []

        self.modality = modality

        for ckpt_file, codebook_file, obj in zip(ckpt_list, codebook_list, obj_list):
            print(ckpt_file)
            print(codebook_file)
            self.aae_full = AAE([obj], modality)
            self.aae_full.encoder.eval()
            self.aae_full.decoder.eval()
            for param in self.aae_full.encoder.parameters():
                param.requires_grad = False
            for param in self.aae_full.decoder.parameters():
                param.requires_grad = False
            checkpoint = torch.load(ckpt_file)
            self.aae_full.load_ckpt_weights(checkpoint['aae_state_dict'])
            self.aae_list.append(copy.deepcopy(self.aae_full.encoder))

            # compute codebook file if necessary
            if os.path.exists(codebook_file):
                print('Found codebook in : ' + codebook_file)
                self.codebook_list.append(torch.load(codebook_file)[0])
                if modality == 'rgbd':
                    self.codebook_list_depth.append(torch.load(codebook_file)[2])
            else:
                print('Cannot find codebook in : ' + codebook_file)
                print('Start computing codebook ...')
                if self.obj_ctg == 'ycb':
                    if modality == 'rgbd':
                        dataset_code = ycb_codebook_online_generator(cad_model_dir, [obj],
                                                                     self.cfg_list[self.obj_list.index(obj)].TRAIN.RENDER_DIST[0],
                                                                     output_size=(256, 256),
                                                                     fu=self.cfg_list[
                                                                         self.obj_list.index(obj)].TRAIN.FU,
                                                                     fv=self.cfg_list[
                                                                         self.obj_list.index(obj)].TRAIN.FV,
                                                                     u0=self.cfg_list[
                                                                         self.obj_list.index(obj)].TRAIN.U0,
                                                                     v0=self.cfg_list[
                                                                         self.obj_list.index(obj)].TRAIN.V0,
                                                                     gpu_id=self.cfg_list[self.obj_list.index(obj)].GPU_ID)
                    else:
                        dataset_code = ycb_codebook_online_generator(cad_model_dir, [obj],
                                                                     self.cfg_list[
                                                                     self.obj_list.index(obj)].TRAIN.RENDER_DIST[0],
                                                                     output_size=(128, 128),
                                                                     fu=self.cfg_list[
                                                                         self.obj_list.index(obj)].TRAIN.FU,
                                                                     fv=self.cfg_list[
                                                                         self.obj_list.index(obj)].TRAIN.FV,
                                                                     u0=self.cfg_list[
                                                                         self.obj_list.index(obj)].TRAIN.U0,
                                                                     v0=self.cfg_list[
                                                                         self.obj_list.index(obj)].TRAIN.V0,
                                                                     gpu_id=self.cfg_list[
                                                                     self.obj_list.index(obj)].GPU_ID)
                elif self.obj_ctg == 'tless':
                    if modality == 'rgbd':
                        dataset_code = tless_codebook_online_generator(cad_model_dir, [obj],
                                                                     self.cfg_list[self.obj_list.index(obj)].TRAIN.RENDER_DIST[0],
                                                                     output_size=(256, 256),
                                                                     gpu_id=self.cfg_list[self.obj_list.index(obj)].GPU_ID)
                    else:
                        dataset_code = tless_codebook_online_generator(cad_model_dir, [obj],
                                                                       self.cfg_list[
                                                                           self.obj_list.index(obj)].TRAIN.RENDER_DIST[
                                                                           0],
                                                                       output_size=(128, 128),
                                                                       gpu_id=self.cfg_list[
                                                                           self.obj_list.index(obj)].GPU_ID)
                if modality == 'rgbd':
                    self.aae_full.compute_codebook_rgbd(dataset_code, codebook_file, save=True)
                else:
                    self.aae_full.compute_codebook(dataset_code, codebook_file, save=True)

                self.codebook_list.append(torch.load(codebook_file)[0])
                if self.cfg_list[0].TRAIN.DEPTH_EMBEDDING:
                    self.codebook_list_depth.append(torch.load(codebook_file)[2])

            self.rbpf_codepose = torch.load(codebook_file)[1].cpu().numpy()  # all are identical

        # renderer
        self.intrinsics = np.array([[self.cfg_list[0].PF.FU, 0, self.cfg_list[0].PF.U0],
                               [0, self.cfg_list[0].PF.FV, self.cfg_list[0].PF.V0],
                               [0, 0, 1.]], dtype=np.float32)

        self.renderer = render_wrapper(self.obj_list, self.intrinsics, gpu_id=self.gpu_id,
                                       model_dir=cad_model_dir,
                                       model_ctg=self.obj_ctg,
                                       im_w=int(self.cfg_list[0].PF.W),
                                       im_h=int(self.cfg_list[0].PF.H),
                                       initialize_render=True)

        # target object property
        self.target_obj = None
        self.target_obj_idx = None
        self.target_obj_encoder = None
        self.target_obj_codebook = None
        self.target_obj_cfg = None
        self.target_box_sz = None

        self.max_sim_rgb = 0
        self.max_sim_depth = 0
        self.max_vis_ratio = 0

        # initialize the particle filters
        self.rbpf = particle_filter(self.cfg_list[0].PF, n_particles=self.cfg_list[0].PF.N_PROCESS)
        self.rbpf_ok = False

        # pose rbpf for initialization
        self.rbpf_init_max_sim = 0

        # data properties
        self.data_with_gt = False
        self.data_with_est_bbox = False
        self.data_with_est_center = False
        self.data_intrinsics = np.ones((3, 3), dtype=np.float32)

        # initialize the PoseRBPF variables
        # ground truth information
        self.gt_available = False
        self.gt_t = [0, 0, 0]
        self.gt_rotm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.gt_bbox_center = np.zeros((3,))
        self.gt_bbox_size = 0
        self.gt_uv = np.array([0, 0, 1], dtype=np.float32)
        self.gt_z = 0

        # estimated states
        self.est_bbox_center = np.zeros((2, self.cfg_list[0].PF.N_PROCESS))
        self.est_bbox_size = np.zeros((self.cfg_list[0].PF.N_PROCESS,))
        self.est_bbox_weights = np.zeros((self.cfg_list[0].PF.N_PROCESS,))

        # for logging
        self.log_err_t = []
        self.log_err_tx = []
        self.log_err_ty = []
        self.log_err_tz = []
        self.log_err_rx = []
        self.log_err_ry = []
        self.log_err_rz = []
        self.log_err_r = []
        self.log_err_t_star = []
        self.log_err_r_star = []
        self.log_max_sim = []
        self.log_dir = './'
        self.log_created = False
        self.log_pose = None
        self.log_error = None

        # posecnn prior
        self.prior_uv = [0, 0, 1]
        self.prior_z = 0
        self.prior_t = np.array([0, 0, 0], dtype=np.float32)
        self.prior_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        # flags for experiments
        self.exp_with_mask = True
        self.step = 0
        self.iskf = False
        self.init_step = False
        self.save_uncertainty = False
        self.show_prior = False

        # motion model
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.T_c0o = np.eye(4, dtype=np.float32)
        self.T_c1o = np.eye(4, dtype=np.float32)
        self.Tbr1 = np.eye(4, dtype=np.float32)
        self.Tbr0 = np.eye(4, dtype=np.float32)
        self.Trc = np.eye(4, dtype=np.float32)

        # multiple object pose estimation (mope)
        self.mope_Tbo_list = []
        self.mope_pc_b_list = []
        # for i in range(len(self.rbpf_list)):
        #     self.mope_Tbo_list.append(np.eye(4, dtype=np.float32))
        #     self.mope_pc_b_list.append(None)

        # evaluation module
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        # relative poses
        self.T_ob_o = []
        self.rel_pose_flag = []
        for i in range(len(self.obj_list)):
            self.T_ob_o.append(np.eye(4, dtype=np.float32))
            self.rel_pose_flag.append(False)

        # for visualization
        self.uv_init = None
        self.z_init = None

        self.image_recon = None
        self.skip = False

        # SDF refine
        self.refine = refine
        self.sdf_list = []
        self.sdf_limits_list = []
        self.points_list = []
        self.extents = np.zeros((len(self.obj_list), 3), dtype=np.float32)
        self.sdf_refiner = None
        self.target_sdf = None
        self.target_sdf_limits = None
        self.target_points = None

        # load points
        if self.obj_ctg == 'ycb':
            for i in range(len(self.obj_list)):
                object = self.obj_list[i]
                points_file = '{}/ycb_models/{}/points.xyz'.format(cad_model_dir, object)
                points = np.loadtxt(points_file).astype(np.float32)
                self.extents[i, :] = 2 * np.max(np.absolute(points), axis=0)
                points = np.concatenate((points, np.ones((points.shape[0], 1), dtype=np.float32)), axis=1)
                self.points_list.append(points)
        else:
            for i in range(len(self.obj_list)):
                object = self.obj_list[i]
                points_file = '{}/tless_models/{}.pth'.format(cad_model_dir, object)
                points = np.loadtxt(points_file).astype(np.float32) / 1000
                self.extents[i, :] = 2 * np.max(np.absolute(points), axis=0)
                points = np.concatenate((points, np.ones((points.shape[0], 1), dtype=np.float32)), axis=1)
                self.points_list.append(points)

        # load sdf if need to refine
        if self.refine:
            # load the objects
            if self.obj_ctg == 'ycb':
                for object in self.obj_list:
                    sdf_file = '{}/ycb_models/{}/textured_simple_low_res.pth'.format(cad_model_dir, object)
                    sdf_torch, sdf_limits = load_sdf(sdf_file)
                    self.sdf_list.append(sdf_torch)
                    self.sdf_limits_list.append(sdf_limits)
            else:
                for object in self.obj_list:
                    sdf_file = '{}/tless_models/{}.pth'.format(cad_model_dir, object)
                    sdf_torch, sdf_limits = load_sdf(sdf_file)
                    self.sdf_list.append(sdf_torch)
                    self.sdf_limits_list.append(sdf_limits)
            # define the optimizer
            self.sdf_refiner = sdf_optimizer(lr=0.005, use_gpu=True)

    # add object instance
    def add_object_instance(self, object_name):
        assert object_name in self.obj_list, "object {} is not in the list of test objects".format(
            object_name)
        idx_obj = self.obj_list.index(object_name)
        self.rbpf_list.append(particle_filter(self.cfg_list[idx_obj].PF, n_particles=self.cfg_list[idx_obj].PF.N_PROCESS))
        self.rbpf_ok_list.append(False)
        self.instance_list.append(object_name)
        self.mope_Tbo_list.append(np.eye(4, dtype=np.float32))
        self.mope_pc_b_list.append(None)

    # reset
    def reset_poserbpf(self):
        self.rbpf_list = []
        self.rbpf_ok_list = []
        self.instance_list = []
        self.mope_Tbo_list = []
        self.mope_pc_b_list = []

    # specify the target object for tracking
    def set_target_obj(self, target_instance_idx):
        target_object = self.instance_list[target_instance_idx]
        assert target_object in self.obj_list, "target object {} is not in the list of test objects".format(target_object)

        # set target object property
        self.target_obj = target_object
        self.target_obj_idx = self.obj_list.index(target_object)
        self.target_obj_encoder = self.aae_list[self.target_obj_idx]
        self.target_obj_codebook = self.codebook_list[self.target_obj_idx]
        self.target_obj_cfg = self.cfg_list[self.target_obj_idx]

        if self.modality == 'rgbd':
            self.target_obj_codebook_depth = self.codebook_list_depth[self.target_obj_idx]

        self.target_box_sz = 2 * self.target_obj_cfg.TRAIN.U0 / self.target_obj_cfg.TRAIN.FU * \
                             self.target_obj_cfg.TRAIN.RENDER_DIST[0]

        self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
        self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
        self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
        self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]

        # reset particle filter
        self.rbpf = self.rbpf_list[target_instance_idx]
        self.rbpf_ok = self.rbpf_ok_list[target_instance_idx]
        self.rbpf_init_max_sim = 0

        if self.refine:
            self.target_sdf = self.sdf_list[self.target_obj_idx]
            self.target_sdf_limits = self.sdf_limits_list[self.target_obj_idx]
            self.target_points = self.points_list[self.target_obj_idx]
            self.Tco_list = []
            self.pc_list = []

        # estimated states
        self.est_bbox_center = np.zeros((2, self.target_obj_cfg.PF.N_PROCESS))
        self.est_bbox_size = np.zeros((self.target_obj_cfg.PF.N_PROCESS,))
        self.est_bbox_weights = np.zeros((self.target_obj_cfg.PF.N_PROCESS,))

        # for logging
        self.log_err_t = []
        self.log_err_tx = []
        self.log_err_ty = []
        self.log_err_tz = []
        self.log_err_rx = []
        self.log_err_ry = []
        self.log_err_rz = []
        self.log_err_r = []
        self.log_err_t_star = []
        self.log_err_r_star = []
        self.log_max_sim = []
        self.log_dir = './'
        self.log_created = False
        self.log_pose = None
        self.log_error = None

        # prior
        self.prior_uv = [0, 0, 1]
        self.prior_z = 0
        self.prior_t = np.array([0, 0, 0], dtype=np.float32)
        self.prior_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        # motion model
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.T_c0o = np.eye(4, dtype=np.float32)
        self.T_c1o = np.eye(4, dtype=np.float32)
        self.Tbr1 = np.eye(4, dtype=np.float32)
        self.Tbr0 = np.eye(4, dtype=np.float32)
        self.Trc = np.eye(4, dtype=np.float32)

        # print
        print('target object is set to {}'.format(self.target_obj_cfg.PF.TRACK_OBJ))

    def switch_target_obj(self, target_instance_idx):
        target_object = self.instance_list[target_instance_idx]

        # set target object property
        self.target_obj = target_object
        self.target_obj_idx = self.obj_list.index(target_object)
        self.target_obj_encoder = self.aae_list[self.target_obj_idx]
        self.target_obj_codebook = self.codebook_list[self.target_obj_idx]
        self.target_obj_cfg = self.cfg_list[self.target_obj_idx]

        if self.modality == 'rgbd':
            self.target_obj_codebook_depth = self.codebook_list_depth[self.target_obj_idx]

        self.target_box_sz = 2 * self.target_obj_cfg.TRAIN.U0 / self.target_obj_cfg.TRAIN.FU * \
                             self.target_obj_cfg.TRAIN.RENDER_DIST[0]

        self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
        self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
        self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
        self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]

        # reset particle filter
        self.rbpf = self.rbpf_list[target_instance_idx]
        self.rbpf_ok = self.rbpf_ok_list[target_instance_idx]
        self.rbpf_init_max_sim = 0

        if self.refine:
            self.target_sdf = self.sdf_list[self.target_obj_idx]
            self.target_sdf_limits = self.sdf_limits_list[self.target_obj_idx]
            self.target_points = self.points_list[self.target_obj_idx]
            self.Tco_list = []
            self.pc_list = []

    def set_intrinsics(self, intrinsics, w, h):
        self.intrinsics = intrinsics
        self.data_intrinsics = intrinsics
        if self.target_obj_cfg is not None:
            self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
            self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
            self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
            self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]
        if self.renderer.renderer is not None:
            self.renderer.set_intrinsics(self.intrinsics, w, h)

    def visualize_roi(self, image, uv, z, step, phase='tracking', show_gt=False, error=True, uncertainty=False, show=False, color='g', skip=False):
        image_disp = image

        # plt.figure()
        fig, ax = plt.subplots(1)

        ax.imshow(image_disp)

        plt.gca().set_axis_off()

        if skip:
            plt.axis('off')
            save_name = self.log_dir + '/{:06}_{}.png'.format(step, phase)
            plt.savefig(save_name)
            return

        if error == False:
            # set the margin to 0
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

        if show_gt:
            gt_bbox_center = self.gt_uv
            gt_bbox_size = 128 * self.target_obj_cfg.TRAIN.RENDER_DIST[0] / self.gt_z * self.target_obj_cfg.PF.FU / self.target_obj_cfg.TRAIN.FU * 0.15

        est_bbox_center = uv
        est_bbox_size = 128 * self.target_obj_cfg.TRAIN.RENDER_DIST[0] * np.ones_like(z) / z * self.target_obj_cfg.PF.FU / self.target_obj_cfg.TRAIN.FU

        if error:
            plt.plot(uv[:, 0], uv[:, 1], 'co', markersize=2)
            plt.plot(gt_bbox_center[0], gt_bbox_center[1], 'ro', markersize=5)
            plt.axis('off')

        t_v = self.rbpf.trans_bar
        center = project(t_v, self.data_intrinsics)
        plt.plot(center[0], center[1], 'co', markersize=5)

        if show_gt:
            rect = patches.Rectangle((gt_bbox_center[0] - 0.5*gt_bbox_size,
                                      gt_bbox_center[1] - 0.5*gt_bbox_size),
                                      gt_bbox_size,
                                      gt_bbox_size, linewidth=5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        for bbox_center, bbox_size in zip(est_bbox_center, est_bbox_size):
            rect = patches.Rectangle((bbox_center[0] - 0.5 * bbox_size,
                                      bbox_center[1] - 0.5 * bbox_size),
                                      bbox_size,
                                      bbox_size, linewidth=0.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        if error:
            z_show = [- 0.035, -0.07, -0.105, -0.14]
            plt.annotate('step {} - t err:        {:.3f} cm'.format(step, self.log_err_t[-1] * 100), xy=(0.15, z_show[0]),
                         xycoords='axes fraction')
            plt.annotate('step {} - R err:        {:.3f} deg'.format(step, self.log_err_r[-1]), xy=(0.15, z_show[1]),
                         xycoords='axes fraction')
            plt.annotate('step {} - max similarity:   {:.2f}'.format(step, self.log_max_sim[-1]), xy=(0.15, z_show[2]),
                         xycoords='axes fraction')

        plt.axis('off')

        save_name = self.log_dir + '/{:06}_{}.png'.format(step, phase)
        plt.savefig(save_name)

        if show:
            plt.show()
            # raw_input('show image')

        if uncertainty:
            ### save for uncertainty visualization ###
            rot_max = torch.sum(self.rbpf.rot.view(self.rbpf.n_particles, -1), dim=0) / self.rbpf.n_particles
            rot_max_c = rot_max.clone()
            rot_max = rot_max.view(1, 37, 72, 72)
            rot_max, _ = torch.max(rot_max, dim=3)
            np.savez(self.log_dir + '/rot_{:06}.npz'.format(step), rot=rot_max.cpu().numpy(),
                     rot_gt=mat2quat(self.gt_rotm))

    # logging
    def display_result(self, step, steps):
        qco = mat2quat(self.gt_rotm)

        filter_rot_error_bar = abs(single_orientation_error(qco, mat2quat(self.rbpf.rot_bar)))

        trans_star_f = self.rbpf.trans_star
        trans_bar_f = self.rbpf.trans_bar
        rot_star_f = self.rbpf.rot_star

        filter_trans_error_star = np.linalg.norm(trans_star_f - self.gt_t)
        filter_trans_error_bar = np.linalg.norm(trans_bar_f - self.gt_t)
        filter_rot_error_star = abs(single_orientation_error(qco, mat2quat(rot_star_f)))

        self.log_err_t_star.append(filter_trans_error_star)
        self.log_err_t.append(filter_trans_error_bar)
        self.log_err_r_star.append(filter_rot_error_star * 57.3)
        self.log_err_r.append(filter_rot_error_bar * 57.3)

        self.log_err_tx.append(np.abs(trans_bar_f[0] - self.gt_t[0]))
        self.log_err_ty.append(np.abs(trans_bar_f[1] - self.gt_t[1]))
        self.log_err_tz.append(np.abs(trans_bar_f[2] - self.gt_t[2]))

        rot_err_axis = single_orientation_error_axes(qco, mat2quat(self.rbpf.rot_bar))
        self.log_err_rx.append(np.abs(rot_err_axis[0]))
        self.log_err_ry.append(np.abs(rot_err_axis[1]))
        self.log_err_rz.append(np.abs(rot_err_axis[2]))

        print('     step {}/{}: translation error (filter)   = {:.4f} cm'.format(step+1, int(steps),
                                                                                filter_trans_error_bar * 100))
        print('     step {}/{}: RGB Similarity   = {:.3f}'.format(step + 1, int(steps), self.max_sim_rgb))
        if self.modality == 'rgbd':
            print('     step {}/{}: Depth Similarity   = {:.3f}'.format(step + 1, int(steps), self.max_sim_depth))
        print('     step {}/{}: uvz error (filter)           = ({:.4f}, {:.4f}, {:.4f})'.format(step + 1, int(steps),
                                                                                                 self.rbpf.uv_bar[0] - self.gt_uv[0],
                                                                                                 self.rbpf.uv_bar[1] - self.gt_uv[1],
                                                                                                 (trans_bar_f[2] - self.gt_t[2]) * 100))
        print('     step {}/{}: xyz error (filter)           = ({:.4f}, {:.4f}, {:.4f})'.format(step + 1, int(steps),
                                                                                                self.log_err_tx[-1
                                                                                                ] * 1000,
                                                                                                self.log_err_ty[
                                                                                                    -1] * 1000,
                                                                                                self.log_err_tz[
                                                                                                    -1] * 1000))
        print('     step {}/{}: xyz rotation err (filter)    = ({:.4f}, {:.4f}, {:.4f})'.format(step + 1, int(steps),
                                                                                                self.log_err_rx[-1
                                                                                                ] * 57.3,
                                                                                                self.log_err_ry[
                                                                                                    -1] * 57.3,
                                                                                                self.log_err_rz[
                                                                                                    -1] * 57.3))
        print('     step {}/{}: translation error (best)     = {:.4f} cm'.format(step+1, int(steps),
                                                                                filter_trans_error_star * 100))
        print('     step {}/{}: rotation error (filter)      = {:.4f} deg'.format(step+1, int(steps),
                                                                                filter_rot_error_bar * 57.3))
        print('     step {}/{}: rotation error (best)        = {:.4f} deg'.format(step+1, int(steps),
                                                                                filter_rot_error_star * 57.3))

        return filter_rot_error_bar * 57.3

    def save_log(self, sequence, filename, with_gt=True, tless=False):
        if not self.log_created:
            self.log_pose = open(self.log_dir + "/Pose_{}_seq{}.txt".format(self.target_obj_cfg.PF.TRACK_OBJ, sequence), "w+")
            if with_gt:
                self.log_pose_gt = open(self.log_dir + "/Pose_GT_{}_seq{}.txt".format(self.target_obj_cfg.PF.TRACK_OBJ, sequence),
                                     "w+")
                self.log_error = open(self.log_dir + "/Error_{}_seq{}.txt".format(self.target_obj_cfg.PF.TRACK_OBJ, sequence), "w+")
            self.log_created = True

        q_log = mat2quat(self.rbpf.rot_bar)
        self.log_pose.write('{} {} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} \n'.format(self.target_obj_cfg.PF.TRACK_OBJ,
                                                                                               filename[0],
                                                                                               self.rbpf.trans_bar[0],
                                                                                               self.rbpf.trans_bar[1],
                                                                                               self.rbpf.trans_bar[2],
                                                                                               q_log[0],
                                                                                               q_log[1],
                                                                                               q_log[2],
                                                                                               q_log[3]))

        if with_gt:
            q_log_gt = mat2quat(self.gt_rotm)
            self.log_pose_gt.write(
                '{} {} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} \n'.format(self.target_obj_cfg.PF.TRACK_OBJ,
                                                                                   filename[0],
                                                                                   self.gt_t[0],
                                                                                   self.gt_t[1],
                                                                                   self.gt_t[2],
                                                                                   q_log_gt[0],
                                                                                   q_log_gt[1],
                                                                                   q_log_gt[2],
                                                                                   q_log_gt[3]))
            self.log_error.write('{} {} {:.5f} {:.5f} \n'.format(self.target_obj_cfg.PF.TRACK_OBJ,
                                                                filename[0],
                                                                self.log_err_t[-1] * 100,
                                                                self.log_err_r[-1] * 57.3))

        # for tless dataset, save the results as sixd challenge format
        if tless:
            obj_id_sixd = self.target_obj_cfg.PF.TRACK_OBJ[-2:]
            seq_id_sixd = filename[0][:2]
            img_id_sixd = filename[0][-4:]
            save_folder_sixd = self.target_obj_cfg.PF.SAVE_DIR[
                               :-7] + 'dpf_tless_primesense_all/'  # .format(obj_id_sixd)
            save_folder_sixd += seq_id_sixd + '/'
            if not os.path.exists(save_folder_sixd):
                os.makedirs(save_folder_sixd)
            filename_sixd = img_id_sixd + '_' + obj_id_sixd + '.yml'
            pose_log_sixd = open(save_folder_sixd + filename_sixd, "w+")
            pose_log_sixd.write('run_time: -1 \n')
            pose_log_sixd.write('ests: \n')
            str_score = '- {score: 1.00000000, '
            str_R = 'R: [{:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}], ' \
                .format(self.rbpf.rot_bar[0, 0], self.rbpf.rot_bar[0, 1], self.rbpf.rot_bar[0, 2],
                        self.rbpf.rot_bar[1, 0], self.rbpf.rot_bar[1, 1], self.rbpf.rot_bar[1, 2],
                        self.rbpf.rot_bar[2, 0], self.rbpf.rot_bar[2, 1], self.rbpf.rot_bar[2, 2])
            str_t = 't: [{:.8f}, {:.8f}, {:.8f}]'.format(self.rbpf.trans_bar[0] * 1000.0,
                                                         self.rbpf.trans_bar[1] * 1000.0,
                                                         self.rbpf.trans_bar[2] * 1000.0)
            pose_log_sixd.write(str_score + str_R + str_t + '}')
            pose_log_sixd.close()

    def display_overall_result(self):
        print('filter trans closest error = ', np.mean(np.asarray(self.log_err_t_star)))
        print('filter trans mean error = ', np.mean(np.asarray(self.log_err_t)))
        print('filter trans RMSE (x) = ', np.sqrt(np.mean(np.asarray(self.log_err_tx) ** 2)) * 1000)
        print('filter trans RMSE (y) = ', np.sqrt(np.mean(np.asarray(self.log_err_ty) ** 2)) * 1000)
        print('filter trans RMSE (z) = ', np.sqrt(np.mean(np.asarray(self.log_err_tz) ** 2)) * 1000)
        print('filter rot RMSE (x) = ', np.sqrt(np.mean(np.asarray(self.log_err_rx) ** 2)) * 57.3)
        print('filter rot RMSE (y) = ', np.sqrt(np.mean(np.asarray(self.log_err_ry) ** 2)) * 57.3)
        print('filter rot RMSE (z) = ', np.sqrt(np.mean(np.asarray(self.log_err_rz) ** 2)) * 57.3)
        print('filter rot closest error = ', np.mean(np.asarray(self.log_err_r_star)))
        print('filter rot mean error = ', np.mean(np.asarray(self.log_err_r)))

    def use_detection_priors(self, n_particles):
        self.rbpf.uv[-n_particles:] = np.repeat([self.prior_uv], n_particles, axis=0)
        self.rbpf.uv[-n_particles:, :2] += np.random.uniform(-self.target_obj_cfg.PF.UV_NOISE_PRIOR,
                                                             self.target_obj_cfg.PF.UV_NOISE_PRIOR,
                                                             (n_particles, 2))

    # initialize PoseRBPF
    def initialize_poserbpf(self, image, intrinsics, uv_init, n_init_samples, z_init=None, depth=None):
        # sample around the center of bounding box
        uv_h = np.array([uv_init[0], uv_init[1], 1])
        uv_h = np.repeat(np.expand_dims(uv_h, axis=0), n_init_samples, axis=0)
        uv_h[:, :2] += np.random.uniform(-self.target_obj_cfg.PF.INIT_UV_NOISE, self.target_obj_cfg.PF.INIT_UV_NOISE,
                                         (n_init_samples, 2))
        uv_h[:, 0] = np.clip(uv_h[:, 0], 0, image.shape[1])
        uv_h[:, 1] = np.clip(uv_h[:, 1], 0, image.shape[0])

        self.uv_init = uv_h.copy()

        if self.modality=='rgbd':
            uv_h_int = uv_h.astype(int)
            uv_h_int[:, 0] = np.clip(uv_h_int[:, 0], 0, image.shape[1] - 1)
            uv_h_int[:, 1] = np.clip(uv_h_int[:, 1], 0, image.shape[0] - 1)
            depth_np = depth.numpy()
            z = depth_np[uv_h_int[:, 1], uv_h_int[:, 0], 0]
            z = np.expand_dims(z, axis=1)
            z[z > 0] += np.random.uniform(-0.3, 0.3, z[z > 0].shape)
            z[z == 0] = np.random.uniform(0.5, 1.5, z[z == 0].shape)
        else:
            # sample around z
            if z_init == None:
                z = np.random.uniform(0.5, 1.5, (n_init_samples, 1))
            else:
                z = np.random.uniform(z_init - 0.2, z_init + 0.2, (n_init_samples, 1))

        self.z_init = z.copy()
        # evaluate translation
        if self.modality == 'rgbd':
            distribution = self.evaluate_particles_rgbd(image, uv_h, z,
                                                       self.target_obj_cfg.TRAIN.RENDER_DIST[0], 0.1, depth=depth,
                                                       initialization=True)
        else:
            distribution = self.evaluate_particles_rgb(image, uv_h, z,
                                                        self.target_obj_cfg.TRAIN.RENDER_DIST[0], 0.1, depth=depth,
                                                       initialization=True)

        # find the max pdf from the distribution matrix
        index_star = my_arg_max(distribution)
        uv_star = uv_h[index_star[0], :]  # .copy()
        z_star = z[index_star[0], :]  # .copy()
        self.rbpf.update_trans_star_uvz(uv_star, z_star, intrinsics)
        distribution[index_star[0], :] /= torch.sum(distribution[index_star[0], :])
        self.rbpf.rot = distribution[index_star[0], :].view(1, 1, 37, 72, 72).repeat(self.rbpf.n_particles, 1, 1, 1, 1)

        self.rbpf.update_rot_star_R(quat2mat(self.rbpf_codepose[index_star[1]][3:]))
        self.rbpf.rot_bar = self.rbpf.rot_star
        self.rbpf.uv_bar = uv_star
        self.rbpf.z_bar = z_star
        self.rbpf_init_max_sim = self.log_max_sim[-1]

        # compute roi
        pose = np.zeros((7,), dtype=np.float32)
        pose[4:] = self.rbpf.trans_bar
        pose[:4] = mat2quat(self.rbpf.rot_bar)
        points = self.points_list[self.target_obj_idx]
        box = self.compute_box(pose, points)
        self.rbpf.roi = np.zeros((1, 6), dtype=np.float32)
        self.rbpf.roi[0, 1] = self.target_obj_idx
        self.rbpf.roi[0, 2:6] = box


    # compute bounding box by projection
    def compute_box(self, pose, points):
        x3d = np.transpose(points)
        RT = np.zeros((3, 4), dtype=np.float32)
        RT[:3, :3] = quat2mat(pose[:4])
        RT[:, 3] = pose[4:]
        x2d = np.matmul(self.intrinsics, np.matmul(RT, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

        x1 = np.min(x2d[0, :])
        y1 = np.min(x2d[1, :])
        x2 = np.max(x2d[0, :])
        y2 = np.max(x2d[1, :])
        box = np.array([x1, y1, x2, y2])
        return box


    # evaluate particles according to the RGBD images
    def evaluate_particles_rgbd(self, image,
                           uv, z,
                           render_dist, gaussian_std,
                           depth, mask=None,
                           initialization=False):

        image = torch.cat((image.float(), depth.clone().float()), dim=2)
        z_tensor = torch.from_numpy(z).float().cuda().unsqueeze(2).unsqueeze(3)

        if initialization:
            # crop the input image according to uv z
            images_roi_cuda, scale_roi = get_rois_cuda(image.detach(), uv, z,
                                                             self.target_obj_cfg.PF.FU,
                                                             self.target_obj_cfg.PF.FV,
                                                             render_dist, out_size=256)

            # just crop in the center
            roi_info = torch.zeros(images_roi_cuda.size(0), 5).float().cuda()
            roi_info[:, 0] = torch.arange(images_roi_cuda.size(0))
            roi_info[:, 1] = 128.0 - 128.0 / 2
            roi_info[:, 2] = 128.0 - 128.0 / 2
            roi_info[:, 3] = 128.0 + 128.0 / 2
            roi_info[:, 4] = 128.0 + 128.0 / 2

            # compute the codes
            codes, codes_depth = self.target_obj_encoder.forward_pf(images_roi_cuda.detach(),
                                                                            roi_info.clone(),
                                                                            z_tensor,
                                                                            self.target_box_sz)
            codes = codes.detach().view(uv.shape[0], -1)
            codes_depth = codes_depth.detach().view(uv.shape[0], -1)

        else:
            # compute mean uv and mean z and crop the image
            uv_bar = np.mean(uv, axis=0, keepdims=True)
            z_bar = np.mean(z, keepdims=True)
            images_roi_cuda, scale_roi = get_rois_cuda(image.detach(), uv_bar, z_bar,
                                                             self.target_obj_cfg.PF.FU,
                                                             self.target_obj_cfg.PF.FV,
                                                             render_dist, out_size=256)

            # adjust rois according to the particles
            delta_uv = (uv - uv_bar) / scale_roi[0, 0]
            roi_centers = torch.from_numpy(
                np.array([255.0 / 2, 255.0 / 2, 1], dtype=np.float32) + delta_uv).float().cuda()
            delta_z = z / z_bar
            roi_sizes = torch.from_numpy(128.0 / delta_z[:, 0]).float().cuda()
            roi_info = torch.zeros(uv.shape[0], 5).float().cuda()
            roi_info[:, 0] = 0
            roi_info[:, 1] = roi_centers[:, 0] - roi_sizes / 2
            roi_info[:, 2] = roi_centers[:, 1] - roi_sizes / 2
            roi_info[:, 3] = roi_centers[:, 0] + roi_sizes / 2
            roi_info[:, 4] = roi_centers[:, 1] + roi_sizes / 2

            # computer the codes
            codes, codes_depth = self.target_obj_encoder.forward_pf(images_roi_cuda.detach(),
                                                                                roi_info.clone(),
                                                                                z_tensor,
                                                                                self.target_box_sz)
            codes = codes.detach().view(uv.shape[0], -1)
            codes_depth = codes_depth.detach().view(uv.shape[0], -1)

        # compute the similarity between particles' codes and the codebook
        cosine_distance_matrix_depth = self.aae_full.compute_distance_matrix(codes_depth,
                                                                             self.target_obj_codebook_depth)
        cosine_distance_matrix_rgb = self.aae_full.compute_distance_matrix(codes, self.target_obj_codebook)

        cosine_distance_matrix = cosine_distance_matrix_rgb * self.target_obj_cfg.PF.FUSION_WT_RGB\
                                 + cosine_distance_matrix_depth * (1 - self.target_obj_cfg.PF.FUSION_WT_RGB)

        # get the maximum similarity for each particle
        v_sims, i_sims = torch.max(cosine_distance_matrix, dim=1)
        self.cos_dist_mat = v_sims

        _, j_sim = torch.max(v_sims, dim=0)
        self.max_sim_rgb = cosine_distance_matrix_rgb[j_sim, i_sims[j_sim]].cpu().numpy()
        self.max_sim_depth = cosine_distance_matrix_depth[j_sim, i_sims[j_sim]].cpu().numpy()

        # evaluate particles with depth images
        depth_scores = torch.from_numpy(np.ones_like(z)).cuda().float()
        if initialization:
            depth_scores = self.renderer.evaluate_depths_init(cls_id=self.target_obj_idx,
                                                              depth=depth, uv=uv, z=z,
                                                              q_idx=i_sims.cpu().numpy(),
                                                              intrinsics=self.intrinsics,
                                                              render_dist=render_dist, codepose=self.rbpf_codepose,
                                                              delta=self.target_obj_cfg.PF.DEPTH_DELTA,
                                                              tau=self.target_obj_cfg.PF.DEPTH_TAU,
                                                              mask=mask)
            depth_scores = torch.from_numpy(depth_scores).float().cuda()
        else:
            depth_scores, vis_ratio = self.renderer.evaluate_depths_tracking(rbpf=self.rbpf,
                                                                             cls_id=self.target_obj_idx,
                                                                             depth=depth, uv=uv, z=z,
                                                                             q_idx=i_sims.cpu().numpy(),
                                                                             intrinsics=self.intrinsics,
                                                                             render_dist=render_dist,
                                                                             codepose=self.rbpf_codepose,
                                                                             delta=self.target_obj_cfg.PF.DEPTH_DELTA,
                                                                             tau=self.target_obj_cfg.PF.DEPTH_TAU,
                                                                             rbpf_ready=self.rbpf_ok,
                                                                             mask=mask
                                                                             )
            max_vis_ratio = torch.max(vis_ratio).cpu().numpy()
            self.max_vis_ratio = max_vis_ratio

            # reshape the depth score
            if torch.max(depth_scores) > 0:
                depth_scores = depth_scores / torch.max(depth_scores)
                depth_scores = mat2pdf(depth_scores, 1.0, self.target_obj_cfg.PF.DEPTH_STD)
            else:
                depth_scores = torch.ones_like(depth_scores)
                depth_scores /= torch.sum(depth_scores)

        max_sim_all = torch.max(v_sims)
        pdf_matrix = mat2pdf(cosine_distance_matrix / max_sim_all, 1, gaussian_std)

        # combine RGB and D
        pdf_matrix = torch.mul(pdf_matrix, depth_scores)

        # determine system failure
        self.rbpf.max_sim_all = max_sim_all
        if max_sim_all < 0.6:
            self.rbpf_ok = False

        self.log_max_sim.append(max_sim_all)

        return pdf_matrix

    def evaluate_particles_rgb(self, image, uv, z,
                                render_dist, gaussian_std, depth=None, initialization=False):

        images_roi_cuda, scale_roi = get_rois_cuda(image.detach(), uv, z,
                                                   self.target_obj_cfg.PF.FU,
                                                   self.target_obj_cfg.PF.FV,
                                                   render_dist, out_size=128)

        # forward passing
        n_particles = z.shape[0]
        class_info = torch.ones((1, 1, 128, 128), dtype=torch.float32)
        class_info_cuda = class_info.cuda().repeat(n_particles, 1, 1, 1)
        images_input_cuda = torch.cat((images_roi_cuda.detach(), class_info_cuda.detach()), dim=1)
        codes = self.target_obj_encoder.forward(images_input_cuda).view(images_input_cuda.size(0), -1).detach()

        # compute the similarity between particles' codes and the codebook
        cosine_distance_matrix_rgb = self.aae_full.compute_distance_matrix(codes, self.target_obj_codebook)
        max_rgb = torch.max(cosine_distance_matrix_rgb)
        self.max_sim_rgb = max_rgb.cpu().numpy()
        cosine_distance_matrix = cosine_distance_matrix_rgb

        # get the maximum similarity for each particle
        v_sims, i_sims = torch.max(cosine_distance_matrix, dim=1)
        self.cos_dist_mat = v_sims

        max_sim_all = torch.max(v_sims)
        pdf_matrix = mat2pdf(cosine_distance_matrix / max_sim_all, 1, gaussian_std)

        # render and compare for PoseRBPF only using RGB embeddings
        if depth is not None:
            depth_scores = torch.from_numpy(np.ones_like(z)).cuda().float()
            if initialization:
                depth_scores = self.renderer.evaluate_depths_init(cls_id=self.target_obj_idx,
                                                                  depth=depth, uv=uv, z=z,
                                                                  q_idx=i_sims.cpu().numpy(),
                                                                  intrinsics=self.intrinsics,
                                                                  render_dist=render_dist, codepose=self.rbpf_codepose,
                                                                  delta=self.target_obj_cfg.PF.DEPTH_DELTA,
                                                                  tau=self.target_obj_cfg.PF.DEPTH_TAU)
                depth_scores = torch.from_numpy(depth_scores).float().cuda()
            else:
                depth_scores, vis_ratio = self.renderer.evaluate_depths_tracking(rbpf=self.rbpf,
                                                                                 cls_id=self.target_obj_idx,
                                                                                 depth=depth, uv=uv, z=z,
                                                                                 q_idx=i_sims.cpu().numpy(),
                                                                                 intrinsics=self.intrinsics,
                                                                                 render_dist=render_dist,
                                                                                 codepose=self.rbpf_codepose,
                                                                                 delta=self.target_obj_cfg.PF.DEPTH_DELTA,
                                                                                 tau=self.target_obj_cfg.PF.DEPTH_TAU,
                                                                                 rbpf_ready=self.rbpf_ok
                                                                                 )
                max_vis_ratio = torch.max(vis_ratio).cpu().numpy()
                self.max_vis_ratio = max_vis_ratio

                # reshape the depth score
                if torch.max(depth_scores) > 0:
                    depth_scores = depth_scores / torch.max(depth_scores)
                    depth_scores = mat2pdf(depth_scores, 1.0, self.target_obj_cfg.PF.DEPTH_STD)
                else:
                    depth_scores = torch.ones_like(depth_scores)
                    depth_scores /= torch.sum(depth_scores)

            # combine RGB and D
            pdf_matrix = torch.mul(pdf_matrix, depth_scores)

        # determine system failure
        self.rbpf.max_sim_all = max_sim_all
        if max_sim_all < 0.6:
            self.rbpf_ok = False

        self.log_max_sim.append(max_sim_all)

        return pdf_matrix

    # filtering
    def process_poserbpf(self, image, intrinsics, depth=None, mask=None, apply_motion_prior=False, use_detection_prior=False):
        # propagation
        if apply_motion_prior:
            self.rbpf.propagate_particles(self.T_c1c0, self.T_o0o1, 0, 0, intrinsics)
            uv_noise = self.target_obj_cfg.PF.UV_NOISE
            z_noise = self.target_obj_cfg.PF.Z_NOISE
            self.rbpf.add_noise_r3(uv_noise, z_noise)
            self.rbpf.add_noise_rot()
        else:
            uv_noise = self.target_obj_cfg.PF.UV_NOISE
            z_noise = self.target_obj_cfg.PF.Z_NOISE
            self.rbpf.add_noise_r3(uv_noise, z_noise)
            self.rbpf.add_noise_rot()

        if use_detection_prior:
            self.use_detection_priors(int(self.rbpf.n_particles/2))

        # compute pdf matrix for each particle
        if self.modality == 'rgbd':
            est_pdf_matrix = self.evaluate_particles_rgbd(image, self.rbpf.uv, self.rbpf.z,
                                                       self.target_obj_cfg.TRAIN.RENDER_DIST[0],
                                                       self.target_obj_cfg.PF.WT_RESHAPE_VAR,
                                                       depth=depth,
                                                       mask=mask, initialization=False)
        else:
            est_pdf_matrix = self.evaluate_particles_rgb(image, self.rbpf.uv, self.rbpf.z,
                                                         self.target_obj_cfg.TRAIN.RENDER_DIST[0],
                                                         self.target_obj_cfg.PF.WT_RESHAPE_VAR,
                                                         depth=depth,
                                                         initialization=False)


        # most likely particle
        index_star = my_arg_max(est_pdf_matrix)
        self.rbpf.update_trans_star(self.rbpf.uv[index_star[0], :], self.rbpf.z[index_star[0], :], intrinsics)
        self.rbpf.update_rot_star_R(quat2mat(self.rbpf_codepose[index_star[1]][3:]))

        # match rotation distribution
        self.rbpf.rot = torch.clamp(self.rbpf.rot, 1e-5, 1)
        rot_dist = torch.exp(torch.add(torch.log(est_pdf_matrix), torch.log(self.rbpf.rot.view(self.rbpf.n_particles, -1))))
        normalizers = torch.sum(rot_dist, dim=1)

        normalizers_cpu = normalizers.cpu().numpy()
        if np.sum(normalizers_cpu) == 0:
            return 0
        self.rbpf.weights = normalizers_cpu / np.sum(normalizers_cpu)

        rot_dist /= normalizers.unsqueeze(1).repeat(1, self.target_obj_codebook.size(0))

        # matched distributions
        self.rbpf.rot = rot_dist.view(self.rbpf.n_particles, 1, 37, 72, 72)

        # resample
        self.rbpf.resample_ddpf(self.rbpf_codepose, intrinsics, self.target_obj_cfg.PF)

        # update roi
        pose = np.zeros((7,), dtype=np.float32)
        pose[4:] = self.rbpf.trans_bar
        pose[:4] = mat2quat(self.rbpf.rot_bar)
        points = self.points_list[self.target_obj_idx]
        box = self.compute_box(pose, points)
        self.rbpf.roi[0, 2:6] = box

        # visualization
        self.est_bbox_weights = self.rbpf.weights
        self.est_bbox_center = self.rbpf.uv
        self.est_bbox_size = 128 * self.target_obj_cfg.TRAIN.RENDER_DIST[0] * np.ones_like(self.rbpf.z) / self.rbpf.z

        return 0

    def propagate_with_forward_kinematics(self, target_instance_idx):
        self.switch_target_obj(target_instance_idx)
        self.rbpf.propagate_particles(self.T_c1c0, self.T_o0o1, 0, 0, torch.from_numpy(self.intrinsics).unsqueeze(0))

    # function used in ros node
    def pose_estimation_single(self, target_instance_idx, roi, image, depth, visualize=False, dry_run=False):

        # set target object
        self.switch_target_obj(target_instance_idx)

        if not roi is None:
            center = np.array([0.5 * (roi[2] + roi[4]), 0.5 * (roi[3] + roi[5]), 1], dtype=np.float32)

            # todo: use segmentation masks
            # idx_obj = self.obj_list_all.index(self.target_obj) + 1
            # # there is no large clamp
            # if self.target_obj == '061_foam_brick':
            #     idx_obj -= 1
            # mask_obj = (mask==idx_obj).float().repeat(1,1,3)
            # depth_input = depth * mask_obj[:, :, [0]]

            self.prior_uv = center

            if self.rbpf_ok_list[target_instance_idx] == False:
                # sample around the center of bounding box
                self.initialize_poserbpf(image, self.intrinsics,
                                       self.prior_uv[:2], 100, depth=depth)

                if self.max_sim_rgb > self.target_obj_cfg.PF.SIM_RGB_THRES and not dry_run:
                    print('===================is initialized!======================')
                    self.rbpf_ok_list[target_instance_idx] = True
                    self.process_poserbpf(image,
                                      torch.from_numpy(self.intrinsics).unsqueeze(0),
                                      depth=depth, use_detection_prior=True)

            else:
                self.process_poserbpf(image,
                                      torch.from_numpy(self.intrinsics).unsqueeze(0),
                                      depth=depth, use_detection_prior=True)
                if self.log_max_sim[-1] < 0.75:
                    print('{} low similarity, mark untracked'.format(self.instance_list[target_instance_idx]))
                    self.rbpf_ok_list[target_instance_idx] = False

            if not dry_run:
                print('Estimating {}, rgb sim = {}, depth sim = {}'.format(self.instance_list[target_instance_idx], self.max_sim_rgb, self.max_sim_depth))

        if visualize:
            render_rgb, render_depth = self.renderer.render_pose(self.data_intrinsics,
                                                                 self.rbpf_list[target_instance_idx].trans_bar,
                                                                 self.rbpf_list[target_instance_idx].rot_bar,
                                                                 self.target_obj_idx)

            render_rgb = render_rgb[0].permute(1, 2, 0).cpu().numpy()
            image_disp = 0.4 * image.cpu().numpy() + 0.6 * render_rgb
            image_disp = np.clip(image_disp, 0, 1.0)
            plt.figure()
            plt.imshow(image_disp)
            plt.show()

        Tco = np.eye(4, dtype=np.float32)
        Tco[:3, :3] = self.rbpf.rot_bar
        Tco[:3, 3] = self.rbpf.trans_bar
        max_sim = self.log_max_sim[-1]
        return Tco, max_sim


    # save result to mat file
    def save_results_mat(self, filename):

        # collect rois from rbpfs
        rois = np.zeros((0, 7), dtype=np.float32)
        poses = np.zeros((0, 7), dtype=np.float32)
        for i in range(len(self.instance_list)):
            roi = np.zeros((1, 7), dtype=np.float32)
            roi[0, :6] = self.rbpf_list[i].roi
            roi[0, 6] = self.rbpf_list[i].max_sim_all
            rois = np.concatenate((rois, roi), axis=0)
            pose = np.zeros((1, 7), dtype=np.float32)
            pose[0, :4] = mat2quat(self.rbpf_list[i].rot_bar)
            pose[0, 4:] = self.rbpf_list[i].trans_bar
            poses = np.concatenate((poses, pose), axis=0)

        result = {'poses': poses, 'rois': rois, 'intrinsic_matrix': self.intrinsics}
        scipy.io.savemat(filename, result, do_compression=True)
        print('%s: %d objects' % (filename, rois.shape[0]))


    # run SDF pose refine for multiple objects at once
    def pose_refine_multiple(self, sdf_optimizer, posecnn_classes, index_sdf, im_depth, im_pcloud, im_label=None, steps=10, visualize=False):

        width = im_depth.shape[1]
        height = im_depth.shape[0]
        # compare the depth
        depth_meas_roi = im_pcloud[:, :, 2]
        mask_depth_meas = depth_meas_roi > 0
        mask_depth_valid = torch.isfinite(depth_meas_roi)

        # prepare data
        num = len(index_sdf)
        pose = np.zeros((7,), dtype=np.float32)
        T_oc_init = np.zeros((num, 4, 4), dtype=np.float32)
        cls_index = torch.cuda.FloatTensor(0, 1)
        obj_index = torch.cuda.FloatTensor(0, 1)
        pix_index = torch.cuda.LongTensor(0, 2)
        for i in range(num):

            # pose
            ind = index_sdf[i]
            pose[4:] = self.rbpf_list[ind].trans_bar
            pose[:4] = mat2quat(self.rbpf_list[ind].rot_bar)

            T_co = np.eye(4, dtype=np.float32)
            T_co[:3, :3] = quat2mat(pose[:4])
            T_co[:3, 3] = pose[4:]
            T_oc_init[i] = np.linalg.inv(T_co)

            # filter out points far away
            z = float(pose[6])
            roi = self.rbpf_list[ind].roi.flatten()
            extent = 1.2 * np.mean(self.extents[int(roi[1]), :]) / 2
            mask_distance = torch.abs(depth_meas_roi - z) < extent

            # mask label
            cls = int(roi[1])
            w = roi[4] - roi[2]
            h = roi[5] - roi[3]
            x1 = max(int(roi[2] - w / 2), 0)
            y1 = max(int(roi[3] - h / 2), 0)
            x2 = min(int(roi[4] + w / 2), width - 1)
            y2 = min(int(roi[5] + h / 2), height - 1)
            if im_label is not None:
                labels = torch.zeros_like(im_label)
                labels[y1:y2, x1:x2] = im_label[y1:y2, x1:x2]
                cls_train = posecnn_classes.index(self.obj_list[cls])
                mask_label = labels == cls_train
            else:
                mask_label = torch.zeros_like(mask_depth_meas)
                mask_label[y1:y2, x1:x2] = 1

            mask = mask_label * mask_depth_meas * mask_depth_valid * mask_distance
            index_p = torch.nonzero(mask)
            n = index_p.shape[0]

            if n > 100:
                pix_index = torch.cat((pix_index, index_p), dim=0)
                index = cls * torch.ones((n, 1), dtype=torch.float32, device=0)
                cls_index = torch.cat((cls_index, index), dim=0)
                index = i * torch.ones((n, 1), dtype=torch.float32, device=0)
                obj_index = torch.cat((obj_index, index), dim=0)
                if visualize:
                    print('sdf {} points for object {}, class {} {}'.format(n, i, cls, self.obj_list[cls]))
            else:
                if visualize:
                    print('sdf {} points for object {}, class {} {}, no refinement'.format(n, i, cls, self.obj_list[cls]))

            if visualize and n <= 100:
                fig = plt.figure()
                ax = fig.add_subplot(2, 3, 1)
                plt.imshow(mask_label.cpu().numpy())
                ax.set_title('mask label')
                ax = fig.add_subplot(2, 3, 2)
                plt.imshow(mask_depth_meas.cpu().numpy())
                ax.set_title('mask_depth_meas')
                ax = fig.add_subplot(2, 3, 3)
                plt.imshow(mask_depth_valid.cpu().numpy())
                ax.set_title('mask_depth_valid')
                ax = fig.add_subplot(2, 3, 4)
                plt.imshow(mask_distance.cpu().numpy())
                ax.set_title('mask_distance')
                print(extent, z)
                ax = fig.add_subplot(2, 3, 5)
                plt.imshow(depth_meas_roi.cpu().numpy())
                ax.set_title('depth')
                plt.show()

        # data
        n = pix_index.shape[0]
        if visualize:
            print('sdf with {} points'.format(n))
        if n == 0:
            return
        points = im_pcloud[pix_index[:, 0], pix_index[:, 1], :]
        points = torch.cat((points, cls_index, obj_index), dim=1)
        T_oc_opt = sdf_optimizer.refine_pose_layer(T_oc_init, points, steps=steps)

        # update poses and bounding boxes
        for i in range(num):
            RT_opt = T_oc_opt[i]
            if RT_opt[3, 3] > 0:
                RT_opt = np.linalg.inv(RT_opt)
                ind = index_sdf[i]
                self.rbpf_list[ind].rot_bar = RT_opt[:3, :3]
                self.rbpf_list[ind].trans_bar = RT_opt[:3, 3]
                pose[:4] = mat2quat(RT_opt[:3, :3])
                pose[4:] = RT_opt[:3, 3]
                pc = self.points_list[int(self.rbpf_list[ind].roi[0, 1])]
                self.rbpf_list[ind].roi[0, 2:6] = self.compute_box(pose, pc)

        if visualize:

            points = points.cpu().numpy()
            for i in range(num):

                ind = index_sdf[i]
                roi = self.rbpf_list[ind].roi.flatten()
                cls = int(roi[1])
                T_co_init = np.linalg.inv(T_oc_init[i])

                T_co_opt = np.eye(4, dtype=np.float32)
                T_co_opt[:3, :3] = self.rbpf_list[ind].rot_bar
                T_co_opt[:3, 3] = self.rbpf_list[ind].trans_bar

                index = np.where(points[:, 4] == i)[0]
                if len(index) == 0:
                    continue
                pts = points[index, :4].copy()
                pts[:, 3] = 1.0

                # show points
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                points_obj = self.points_list[cls]
                points_init = np.matmul(np.linalg.inv(T_co_init), pts.transpose()).transpose()
                points_opt = np.matmul(np.linalg.inv(T_co_opt), pts.transpose()).transpose()

                ax.scatter(points_obj[::5, 0], points_obj[::5, 1], points_obj[::5, 2], color='yellow')
                ax.scatter(points_init[::5, 0], points_init[::5, 1], points_init[::5, 2], color='red')
                ax.scatter(points_opt[::5, 0], points_opt[::5, 1], points_opt[::5, 2], color='blue')

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.set_xlim(sdf_optimizer.xmins[cls], sdf_optimizer.xmaxs[cls])
                ax.set_ylim(sdf_optimizer.ymins[cls], sdf_optimizer.ymaxs[cls])
                ax.set_zlim(sdf_optimizer.zmins[cls], sdf_optimizer.zmaxs[cls])
                ax.set_title(self.obj_list[cls])
                plt.show()


    def pose_refine_single(self, depth, steps, mask_input=None):
        T_co_init = np.eye(4, dtype=np.float32)
        T_co_init[:3, :3] = self.rbpf.rot_bar
        T_co_init[:3, 3] = self.rbpf.trans_bar
        ps_c, visible_ratio, mask_viz = self.renderer.estimate_visibile_points(T_co_init, self.target_obj_idx, depth,
                                                                             self.target_obj_cfg.TRAIN.RENDER_DIST,
                                                                             intrinsics=self.data_intrinsics,
                                                                             delta=0.03,
                                                                             mask_input=mask_input)

        if ps_c is None:
            self.rbpf_ok = False
            return mask_viz, 0.0

        T_co_opt, _ = self.sdf_refiner.refine_pose(T_co_init,
                                                   ps_c,
                                                   sdf_input=self.target_sdf,
                                                   sdf_limits_input=self.target_sdf_limits,
                                                   steps=steps)

        self.Tco_list.append(T_co_opt.copy())
        self.pc_list.append(ps_c.clone())

        self.rbpf.rot_bar = T_co_opt[:3, :3]
        self.rbpf.trans_bar = T_co_opt[:3, 3]

        return mask_viz

    def run_dataset(self, val_dataset, sequence, only_track_kf=False, kf_skip=1, demo=False):
        self.log_err_r = []
        self.log_err_r_star = []
        self.log_err_t = []
        self.log_err_t_star = []
        self.log_dir = self.target_obj_cfg.PF.SAVE_DIR+'seq_{}'.format(sequence)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                    shuffle=False, num_workers=0)
        steps = len(val_dataset)
        step = 0
        kf_step = 0

        for inputs in val_generator:
            if val_dataset.dataset_type == 'ycb':
                if step == 0:
                    print('RUNNING YCB DATASET ! ')
                images, depths, poses_gt, intrinsics, class_mask, file_name, is_kf, \
                center_posecnn, z_posecnn, t_posecnn, q_posecnn, masks = inputs

                self.prior_uv = np.array([center_posecnn[0, 0], center_posecnn[0, 1], 1], dtype=np.float32)
                self.prior_z = z_posecnn[0].numpy().astype(np.float32)
                self.prior_t = t_posecnn[0].cpu().numpy()
                self.prior_R = quat2mat(q_posecnn[0].cpu().numpy())

                self.data_intrinsics = intrinsics[0].numpy()
                self.intrinsics = intrinsics[0].numpy()
                self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
                self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
                self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
                self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]

                self.data_with_est_center = True
                self.data_with_gt = True

                # ground truth for visualization
                pose_gt = poses_gt.numpy()[0, :, :]
                self.gt_t = pose_gt[:3, 3]
                self.gt_rotm = pose_gt[:3, :3]
                gt_center = np.matmul(intrinsics, self.gt_t)
                if gt_center.shape[0] == 1:
                    gt_center = gt_center[0]
                gt_center = gt_center / gt_center[2]
                self.gt_uv[:2] = gt_center[:2]
                self.gt_z = self.gt_t[2]

            elif val_dataset.dataset_type == 'tless':
                images, depths, poses_gt, intrinsics, class_mask, \
                file_name, is_kf, bbox = inputs

                self.prior_uv = np.array([bbox[0, 0] + 0.5 * bbox[0, 2], bbox[0, 1] + 0.5 * bbox[0, 3], 1],
                                         dtype=np.float32)

                self.data_intrinsics = intrinsics[0].numpy()
                self.intrinsics = intrinsics[0].numpy()
                self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
                self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
                self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
                self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]
                self.renderer.set_intrinsics(self.intrinsics, im_w=images.size(2), im_h=images.size(1))

                self.data_with_est_center = True
                self.data_with_gt = True

                # ground truth for visualization
                pose_gt = poses_gt.numpy()[0, :, :]
                self.gt_t = pose_gt[:3, 3]
                self.gt_rotm = pose_gt[:3, :3]
                gt_center = np.matmul(intrinsics, self.gt_t)
                if gt_center.shape[0] == 1:
                    gt_center = gt_center[0]
                gt_center = gt_center / gt_center[2]
                self.gt_uv[:2] = gt_center[:2]
                self.gt_z = self.gt_t[2]

                is_kf = (step % 20 == 0)
            else:
                print('*** INCORRECT DATASET SETTING! ***')
                break

            self.step = step
            self.iskf = is_kf

            # skip kfs for larger baseline, motion model test
            if only_track_kf:
                if is_kf == 1:
                    kf_step += 1
                if is_kf == 0 or (kf_step+1) % kf_skip != 0:
                    step += 1
                    continue

            # motion prior
            self.T_c1o[:3, :3] = self.gt_rotm
            self.T_c1o[:3, 3] = self.gt_t
            if np.linalg.norm(self.T_c0o[:3, 3]) == 0:
                self.T_c1c0 = np.eye(4, dtype=np.float32)
            else:
                self.T_c1c0 = np.matmul(self.T_c1o, np.linalg.inv(self.T_c0o))
            self.T_c0o = self.T_c1o.copy()

            if self.modality == 'rgbd':
                depth_data = depths[0]
            else:
                depth_data = None

            # initialization
            if step == 0 or self.rbpf_ok is False:

                print('[Initialization] Initialize PoseRBPF with detected center ... ')
                if np.linalg.norm(self.prior_uv[:2] - self.gt_uv[:2]) > 40:
                    self.prior_uv[:2] = self.gt_uv[:2]

                self.initialize_poserbpf(images[0].detach(), self.data_intrinsics,
                                         self.prior_uv[:2], self.target_obj_cfg.PF.N_INIT,
                                         depth=depth_data)

                if self.data_with_gt:
                    init_error = np.linalg.norm(self.rbpf.trans_star - self.gt_t)
                    print('     Initial translation error = {:.4} cm'.format(init_error * 100))
                    init_rot_error = abs(single_orientation_error(mat2quat(self.gt_rotm), mat2quat(self.rbpf.rot_star)))
                    print('     Initial rotation error    = {:.4} deg'.format(init_rot_error * 57.3))

                self.rbpf_ok = True

                # to avoid initialization to symmetric view and cause abnormal ADD results
                if self.obj_ctg == 'ycb' and init_rot_error * 57.3 > 60:
                    self.rbpf_ok = False

            # filtering
            if self.rbpf_ok:
                torch.cuda.synchronize()
                time_start = time.time()
                self.process_poserbpf(images[0], intrinsics, depth=depth_data)

                if self.refine:
                    self.pose_refine(depth_data.float(), 50)

                torch.cuda.synchronize()
                time_elapse = time.time() - time_start
                print('[Filtering] fps = ', 1 / time_elapse)

                # logging
                if self.data_with_gt:
                    self.display_result(step, steps)
                    self.save_log(sequence, file_name, tless=(self.obj_ctg == 'tless'))

                    # visualization
                    if demo:
                        image_disp = images[0].float().numpy()
                        image_est_render, _ = self.renderer.render_pose(self.intrinsics,
                                                                        self.rbpf.trans_bar,
                                                                        self.rbpf.rot_bar,
                                                                        self.target_obj_idx)
                        image_est_disp = image_est_render[0].permute(1, 2, 0).cpu().numpy()
                        image_disp = 0.4 * image_disp + 0.6 * image_est_disp
                        cv2.imshow('show tracking', cv2.cvtColor(image_disp, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(10)
                    elif is_kf:
                        image_disp = images[0].float().numpy()

                        image_est_render, _ = self.renderer.render_pose(self.intrinsics,
                                                                         self.rbpf.trans_bar,
                                                                         self.rbpf.rot_bar,
                                                                         self.target_obj_idx)

                        image_est_disp = image_est_render[0].permute(1, 2, 0).cpu().numpy()

                        image_disp = 0.4 * image_disp + 0.6 * image_est_disp
                        self.visualize_roi(image_disp, self.rbpf.uv, self.rbpf.z, step, error=False, uncertainty=self.show_prior)
                        plt.close()

            if step == steps-1:
                break
            step += 1

        self.display_overall_result()

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from networks.aae_models import *
import numpy.ma as ma
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pdb
import glob
import copy
from config.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
import pprint
from transforms3d.axangles import *
from pose_rbpf.particle_filter import *
from pose_rbpf.render_wrapper import *
from datasets.render_ycb_dataset import *
from datasets.render_tless_dataset import *
import matplotlib.patches as patches
import os
from functools import partial
import pickle

class PoseRBPF:
    def __init__(self, obj_list, cfg_list, ckpt_list, codebook_list, obj_ctg, modality, cad_model_dir, visualize=True):

        self.visualize = visualize
        self.obj_list = obj_list
        self.obj_ctg = obj_ctg

        # ycb class names
        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            self.obj_list_all = class_name_file.read().split('\n')

        # load the object information
        self.cfg_list = cfg_list

        # load encoders and poses
        self.aae_list = []
        self.codebook_list = []
        self.codebook_list_depth = []
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
                    dataset_code = ycb_codebook_online_generator(cad_model_dir, [obj],
                                                                 self.cfg_list[self.obj_list.index(obj)].TRAIN.RENDER_DIST[0],
                                                                 output_size=(256, 256),
                                                                 gpu_id=self.cfg_list[self.obj_list.index(obj)].GPU_ID)
                elif self.obj_ctg == 'tless':
                    dataset_code = tless_codebook_online_generator(cad_model_dir, [obj],
                                                                 self.cfg_list[self.obj_list.index(obj)].TRAIN.RENDER_DIST[0],
                                                                 output_size=(256, 256),
                                                                 gpu_id=self.cfg_list[self.obj_list.index(obj)].GPU_ID)
                if modality == 'rgbd':
                    self.aae_full.compute_codebook_rgbd(dataset_code, codebook_file, save=True)
                else:
                    self.aae_full.compute_codebook(dataset_code, codebook_file, save=True)

                self.codebook_list.append(torch.load(codebook_file)[0])
                if self.cfg_list[0].TRAIN.DEPTH_EMBEDDING:
                    self.codebook_list_depth.append(torch.load(codebook_file)[2])

            self.rbpf_codepose = torch.load(codebook_file)[1].cpu().numpy()  # all are identical
            idx_obj = self.obj_list.index(obj)
            self.rbpf_list.append(particle_filter(self.cfg_list[idx_obj].PF, n_particles=self.cfg_list[idx_obj].PF.N_PROCESS))
            self.rbpf_ok_list.append(False)

        # renderer
        self.intrinsics = np.array([[self.cfg_list[0].PF.FU, 0, self.cfg_list[0].PF.U0],
                               [0, self.cfg_list[0].PF.FV, self.cfg_list[0].PF.V0],
                               [0, 0, 1.]], dtype=np.float32)

        self.renderer = render_wrapper(self.obj_list, self.intrinsics, gpu_id=self.cfg_list[0].GPU_ID,
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
        for i in range(len(self.rbpf_list)):
            self.mope_Tbo_list.append(np.eye(4, dtype=np.float32))
            self.mope_pc_b_list.append(None)

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

    # specify the target object for tracking
    def set_target_obj(self, target_object):
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
        self.rbpf = self.rbpf_list[self.target_obj_idx]
        self.rbpf_ok = self.rbpf_ok_list[self.target_obj_idx]
        self.rbpf_init_max_sim = 0

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

        # posecnn prior
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

    def save_log(self, sequence, filename, with_gt=True):
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
                                                       initialzation=True)
        else:
            distribution = self.evaluate_particles_rgb(image, uv_h, z,
                                                        self.target_obj_cfg.TRAIN.RENDER_DIST[0], 0.1)

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

    # evaluate particles according to the RGBD images
    def evaluate_particles_rgbd(self, image,
                           uv, z,
                           render_dist, gaussian_std,
                           depth, mask=None,
                           initialzation=False):

        image = torch.cat((image.float(), depth.clone().float()), dim=2)
        z_tensor = torch.from_numpy(z).float().cuda().unsqueeze(2).unsqueeze(3)

        if initialzation:
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
        max_rgb = torch.max(cosine_distance_matrix_rgb)
        max_depth = torch.max(cosine_distance_matrix_depth)
        self.max_sim_rgb = max_rgb.cpu().numpy()
        self.max_sim_depth = max_depth.cpu().numpy()
        cosine_distance_matrix = cosine_distance_matrix_rgb * 0.5 + cosine_distance_matrix_depth * 0.5

        # get the maximum similarity for each particle
        v_sims, i_sims = torch.max(cosine_distance_matrix, dim=1)
        self.cos_dist_mat = v_sims

        # evaluate particles with depth images
        depth_scores = torch.from_numpy(np.ones_like(z)).cuda().float()
        if initialzation:
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
        if max_sim_all < 0.6:
            self.rbpf_ok = False

        self.log_max_sim.append(max_sim_all)

        return pdf_matrix

    def evaluate_particles_rgb(self, image, uv, z,
                                render_dist, gaussian_std):

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

        # determine system failure
        if max_sim_all < 0.6:
            self.rbpf_ok = False

        self.log_max_sim.append(max_sim_all)

        return pdf_matrix

    # filtering
    def process_poserbpf(self, image, intrinsics, depth=None, mask=None, apply_motion_prior=False):

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

        # compute pdf matrix for each particle
        if self.modality == 'rgbd':
            est_pdf_matrix = self.evaluate_particles_rgbd(image, self.rbpf.uv, self.rbpf.z,
                                                       self.target_obj_cfg.TRAIN.RENDER_DIST[0],
                                                       self.target_obj_cfg.PF.WT_RESHAPE_VAR,
                                                       depth=depth,
                                                       mask=mask, initialzation=False)
        else:
            est_pdf_matrix = self.evaluate_particles_rgb(image, self.rbpf.uv, self.rbpf.z,
                                                          self.target_obj_cfg.TRAIN.RENDER_DIST[0],
                                                          self.target_obj_cfg.PF.WT_RESHAPE_VAR)


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

        # visualization
        self.est_bbox_weights = self.rbpf.weights
        self.est_bbox_center = self.rbpf.uv
        self.est_bbox_size = 128 * self.target_obj_cfg.TRAIN.RENDER_DIST[0] * np.ones_like(self.rbpf.z) / self.rbpf.z

        return 0

    def run_dataset(self, val_dataset, sequence, only_track_kf=False, kf_skip=1):

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
                if self.obj_ctg == 'ycb' and init_rot_error * 57.3 > 100:
                    self.rbpf_ok = False

            # filtering
            if self.rbpf_ok:
                torch.cuda.synchronize()
                time_start = time.time()
                self.process_poserbpf(images[0], intrinsics, depth=depth_data)
                torch.cuda.synchronize()
                time_elapse = time.time() - time_start
                print('[Filtering] fps = ', 1 / time_elapse)

                # logging
                if self.data_with_gt:
                    self.display_result(step, steps)
                    self.save_log(sequence, file_name)

                    # visualization
                    if is_kf:
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

from ycb_render.ycb_renderer import *
from ycb_render.tless_renderer_tensor import *
from utils.poserbpf_utils import *
import numpy.ma as ma
import matplotlib.pyplot as plt
import time

class render_wrapper:
    def __init__(self, models, intrinsics, gpu_id, model_dir, model_ctg='ycb', im_w=640, im_h=480, initialize_render=True):

        obj_paths = []
        texture_paths = []
        assert model_ctg in ['ycb', 'tless'], "model {} is not supported".format(model_ctg)
        for item in models:
            if model_ctg == 'ycb':
                if os.path.exists('{}/ycb_models/{}/texture_map.png'.format(model_dir, item)):
                    obj_paths.append('{}/ycb_models/{}/textured_simple.obj'.format(model_dir, item))
                    texture_paths.append('{}/ycb_models/{}/texture_map.png'.format(model_dir, item))
                else:
                    obj_paths.append('{}/ycb_models/{}/textured_simple.ply'.format(model_dir, item))
                    texture_paths.append(''.format(model_dir, item))
            elif model_ctg == 'tless':
                obj_paths.append('{}/tless_models/{}.ply'.format(model_dir, item))
                texture_paths.append(''.format(model_dir, item))

        self.obj_paths = obj_paths
        self.texture_paths = texture_paths

        if initialize_render:
            if model_ctg == 'ycb':
                self.renderer = YCBRenderer(im_w, im_h, gpu_id=gpu_id)
            else:
                self.renderer = TLessTensorRenderer(im_w, im_h, gpu_id=gpu_id)
            self.renderer.load_objects(obj_paths, texture_paths)
            self.renderer.set_light_pos(cfg.TRAIN.TARGET_LIGHT1_POS)
            self.renderer.set_light_color([1.0, 1.0, 1.0])
            self.renderer_cam_pos = [0, 0, 0]
            self.renderer.set_camera_default()
            self.renderer.set_projection_matrix(im_w, im_h,
                                                intrinsics[0, 0], intrinsics[1, 1],
                                                intrinsics[0, 2], intrinsics[1, 2], 0.01, 10)
        else:
            self.renderer = None
        self.intrinsics = intrinsics
        self.im_w = im_w
        self.im_h = im_h

        # predefined parameters
        self.ymap_full = np.array([[j for i in range(int(self.im_w))] for j in range(self.im_h)])
        self.xmap_full = np.array([[i for i in range(int(self.im_w))] for j in range(self.im_h)])

        self.ymap_full_torch = torch.from_numpy(self.ymap_full).float().unsqueeze(2)
        self.xmap_full_torch = torch.from_numpy(self.xmap_full).float().unsqueeze(2)

        self.xmap_full_torch = self.xmap_full_torch.cuda()
        self.ymap_full_torch = self.ymap_full_torch.cuda()

        class_file = './datasets/ycb_video_classes.txt'
        if model_ctg == 'tless':
            class_file = './datasets/tless_classes.txt'

        with open(class_file, 'r') as class_name_file:
            self.class_names_all = class_name_file.read().split('\n')
        self.model_idx = []
        for model in models:
            self.model_idx.append(self.class_names_all.index(model) + 1)

    def set_intrinsics(self, intrinsics, im_w=640, im_h=480):
        self.renderer.set_camera_default()
        self.renderer.set_projection_matrix(im_w, im_h,
                                            intrinsics[0, 0], intrinsics[1, 1],
                                            intrinsics[0, 2], intrinsics[1, 2], 0.01, 10)
        self.intrinsics = intrinsics
        self.im_w = im_w
        self.im_h = im_h

    def render_pose(self, intrinsics, t, R, cls_id): # cls_id: target object index in the loaded classes
        self.set_intrinsics(intrinsics, im_w=self.im_w, im_h=self.im_h)
        t_v = t
        q_v = mat2quat(R)
        pose_v = np.zeros((7,))
        pose_v[:3] = t_v
        pose_v[3:] = q_v
        frame_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        seg_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        pc_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        self.renderer.set_poses([pose_v])
        self.renderer.render([cls_id], frame_cuda, seg_cuda, pc2_tensor=pc_cuda)
        frame_cuda = frame_cuda.flip(0)
        frame_cuda = frame_cuda[:, :, :3].float()
        frame_cuda = frame_cuda.permute(2, 0, 1).unsqueeze(0)
        pc_cuda = pc_cuda.flip(0)
        depth_render = pc_cuda[:, :, 2]
        return frame_cuda, depth_render

    def render_pose_multiple(self, intrinsics, Tco_list, cls_id_list):
        self.set_intrinsics(intrinsics)

        # list of poses
        poses = []
        pose_v = np.zeros((7,))
        for Tco in Tco_list:
            t_v = Tco[:3, 3]
            q_v = mat2quat(Tco[:3, :3])
            pose_v[:3] = t_v
            pose_v[3:] = q_v
            poses.append(pose_v.copy())

        # render
        frame_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        seg_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        pc_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        self.renderer.set_poses(poses)
        self.renderer.render(cls_id_list, frame_cuda, seg_cuda, pc2_tensor=pc_cuda)

        frame_cuda = frame_cuda.flip(0)
        frame_cuda = frame_cuda[:, :, :3].float()
        frame_cuda = frame_cuda.permute(2, 0, 1).unsqueeze(0)
        pc_cuda = pc_cuda.flip(0)
        depth_render = pc_cuda[:, :, 2]

        seg_cuda = seg_cuda.flip(0)
        seg_render_np = seg_cuda[:, :, 0].cpu().numpy()
        seg_est_np = np.zeros_like(seg_render_np)
        for i in cls_id_list:
            select = np.isclose(seg_render_np, float(self.renderer.colors[i][0]))
            seg_est_np[select] = self.model_idx[i]

        return frame_cuda, depth_render, seg_est_np

    def estimate_visibile_points(self, T_co_est, cls_id, depth, render_dist, intrinsics,
                                 delta=0.02, mask_input=None, output_cuda=True):

        t_v = T_co_est[:3, 3]
        R = T_co_est[:3, :3]

        # pose
        pose_v = np.zeros((7,))
        pose_v[:3] = t_v
        pose_v[3:] = mat2quat(R)

        # render
        frame_cuda = torch.cuda.FloatTensor(self.im_h, self.im_w, 4)
        seg_cuda = torch.cuda.FloatTensor(self.im_h, self.im_w, 4)
        pc_cuda = torch.cuda.FloatTensor(self.im_h, self.im_w, 4)

        self.renderer.set_poses([pose_v])
        self.renderer.render([cls_id], frame_cuda, seg_cuda, pc2_tensor=pc_cuda)

        uv = project(t_v, intrinsics)
        uv = np.expand_dims(uv, axis=0)
        z = np.array([[t_v[2]]])

        render_roi_cuda, _ = get_rois_cuda(pc_cuda.flip(0), uv, z,
                                                 intrinsics[0, 0], self.intrinsics[1, 1],
                                                 render_dist)
        if output_cuda:
            pc_cuda = render_roi_cuda[0, :3, :, :].permute(1, 2, 0)
        else:
            pc_cuda = render_roi_cuda[0, :3, :, :].cpu().permute(1, 2, 0)

        # get point cloud from depth
        pt2 = depth
        if output_cuda:
            pt2 = pt2.cuda()
        pt0 = (self.xmap_full_torch - intrinsics[0, 2]) * pt2 / intrinsics[0, 0]
        pt1 = (self.ymap_full_torch - intrinsics[1, 2]) * pt2 / intrinsics[1, 1]
        pc_c_torch = torch.cat((pt0, pt1, pt2), dim=2)

        if mask_input is not None:
            if np.sum(mask_input.numpy() == self.model_idx[cls_id]) == 0:
                mask_input = None

        if mask_input is None:
            pc_roi_cuda, _ = get_rois_cuda(pc_c_torch, uv, z, intrinsics[0, 0], intrinsics[1, 1],
                                                    render_dist)
            # compare the depth
            depth_meas_roi = pc_roi_cuda[0, 2, :, :].cpu().numpy()
            depth_render_roi = pc_cuda[:, :, 2].cpu().numpy()
            mask_depth_meas = ma.getmaskarray(ma.masked_not_equal(depth_meas_roi, 0))
            mask_depth_render = ma.getmaskarray(ma.masked_greater(depth_render_roi, 0))
            mask_depth_vis = ma.getmaskarray(ma.masked_less(np.abs(depth_render_roi - depth_meas_roi), delta))
            mask = mask_depth_meas * mask_depth_render * mask_depth_vis
            mask_disp = mask.copy()
            choose = mask.flatten().nonzero()[0]

            if np.sum(mask_depth_render) == 0:
                visible_ratio = 0
                ps_c = None
                mask_disp = None
                return ps_c, visible_ratio, mask_disp

            visible_ratio = np.sum(mask) * 1.0 / (np.sum(mask_depth_render) * 1.0)
            pt2_valid = depth_meas_roi.flatten()[choose][:, np.newaxis].astype(np.float32)
            pt0_valid = pc_roi_cuda[0, 0, :, :].cpu().numpy().flatten()[choose][:, np.newaxis].astype(np.float32)
            pt1_valid = pc_roi_cuda[0, 1, :, :].cpu().numpy().flatten()[choose][:, np.newaxis].astype(np.float32)
        else:
            pc_roi_cuda, _ = get_rois_cuda(pc_c_torch, uv, z, intrinsics[0, 0], intrinsics[1, 1],
                                                 render_dist)
            # compare the depth
            depth_meas_roi = pc_roi_cuda[0, 2, :, :].cpu().numpy()
            depth_render_roi = pc_cuda[:, :, 2].cpu().numpy()
            mask_depth_meas = ma.getmaskarray(ma.masked_not_equal(depth_meas_roi, 0))
            mask_depth_render = ma.getmaskarray(ma.masked_greater(depth_render_roi, 0))
            mask_depth_vis = ma.getmaskarray(ma.masked_less(np.abs(depth_render_roi - depth_meas_roi), delta))
            mask_prbpf = mask_depth_meas * mask_depth_render * mask_depth_vis
            visible_ratio_prbpf = np.sum(mask_prbpf) * 1.0 / (np.sum(mask_depth_render) * 1.0)

            mask_roi_cuda, _ = get_rois_cuda(mask_input, uv, z, intrinsics[0, 0], intrinsics[1, 1],
                                                   render_dist)
            mask_roi = mask_roi_cuda[0, 0, :, :].cpu().numpy()
            mask_input = ma.getmaskarray(ma.masked_equal(mask_roi, self.model_idx[cls_id]))
            mask_depth_vis2 = ma.getmaskarray(ma.masked_less(np.abs(depth_render_roi - depth_meas_roi), delta * 2.5))
            mask_fuse = mask_depth_meas * mask_input * mask_depth_vis2 * mask_depth_render
            visible_ratio_fuse = np.sum(mask_fuse) * 1.0 / (np.sum(mask_depth_render) * 1.0)

            if visible_ratio_fuse > 1.2 * visible_ratio_prbpf:
                choose = mask_fuse.flatten().nonzero()[0]
                visible_ratio = visible_ratio_fuse
                mask_disp = mask_fuse.copy()
            else:
                choose = mask_prbpf.flatten().nonzero()[0]
                visible_ratio = visible_ratio_prbpf
                mask_disp = mask_prbpf.copy()

            pt2_valid = depth_meas_roi.flatten()[choose][:, np.newaxis].astype(np.float32)
            pt0_valid = pc_roi_cuda[0, 0, :, :].cpu().numpy().flatten()[choose][:, np.newaxis].astype(np.float32)
            pt1_valid = pc_roi_cuda[0, 1, :, :].cpu().numpy().flatten()[choose][:, np.newaxis].astype(np.float32)

        ps_c = np.concatenate((pt0_valid, pt1_valid, pt2_valid, np.ones_like(pt0_valid)), axis=1)
        ps_c = torch.from_numpy(ps_c)
        if output_cuda:
            ps_c = ps_c.cuda()

        return ps_c, visible_ratio, mask_disp

    def estimate_visibile_points_multi(self, Tco_list, cls_id_list, depth, intrinsics, mask_input,
                                        delta=0.02, output_cuda=True):

        # render with current pose
        self.set_intrinsics(intrinsics)

        # list of poses
        poses = []
        pose_v = np.zeros((7,))
        for Tco in Tco_list:
            t_v = Tco[:3, 3]
            q_v = mat2quat(Tco[:3, :3])
            pose_v[:3] = t_v
            pose_v[3:] = q_v
            poses.append(pose_v.copy())

        # render
        frame_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        seg_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        pc_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        self.renderer.set_poses(poses)
        self.renderer.render(cls_id_list, frame_cuda, seg_cuda, pc2_tensor=pc_cuda)

        frame_cuda = frame_cuda.flip(0)
        frame_cuda = frame_cuda[:, :, :3].float()
        frame_cuda = frame_cuda.permute(2, 0, 1).unsqueeze(0)
        pc_cuda = pc_cuda.flip(0)
        depth_render_np = pc_cuda[:, :, 2].cpu().numpy()
        seg_cuda = seg_cuda.flip(0)
        seg_render_np = seg_cuda[:, :, :3].cpu().numpy()

        seg_input_np = mask_input[:, :, 0].cpu().numpy()
        seg_est_np = np.zeros_like(seg_input_np)
        for i in cls_id_list:
            select = np.isclose(seg_render_np[:, :, 0], float(self.renderer.colors[i][0]))
            seg_est_np[select] = self.model_idx[i]

        # convert depth to point cloud
        pt2 = depth
        if output_cuda:
            pt2 = pt2.cuda()
        pt0 = (self.xmap_full_torch - intrinsics[0, 2]) * pt2 / intrinsics[0, 0]
        pt1 = (self.ymap_full_torch - intrinsics[1, 2]) * pt2 / intrinsics[1, 1]

        depth_meas_np = depth[:, :, 0].cpu().numpy()
        mask_depth_meas = ma.getmaskarray(ma.masked_not_equal(depth_meas_np, 0))

        if not mask_input is None:
            delta = delta * 2
        mask_depth_vis = ma.getmaskarray(ma.masked_less(np.abs(depth_render_np - depth_meas_np), delta))

        pt2_np = pt2.cpu().numpy()
        pt0_np = pt0.cpu().numpy()
        pt1_np = pt1.cpu().numpy()

        pc_list = []

        for i in cls_id_list:
            mask_seg_est = ma.getmaskarray(ma.masked_equal(seg_est_np, self.model_idx[i]))
            mask_seg_meas = ma.getmaskarray(ma.masked_equal(seg_input_np, self.model_idx[i]))

            mask_out_merge = mask_depth_meas * mask_seg_est * mask_seg_meas * mask_depth_meas * mask_depth_vis

            choose = mask_out_merge.flatten().nonzero()[0]

            pt2_valid = pt2_np.flatten()[choose][:, np.newaxis].astype(np.float32)
            pt0_valid = pt0_np.flatten()[choose][:, np.newaxis].astype(np.float32)
            pt1_valid = pt1_np.flatten()[choose][:, np.newaxis].astype(np.float32)

            ps_c = np.concatenate((pt0_valid, pt1_valid, pt2_valid, np.ones_like(pt0_valid)), axis=1)
            ps_c = torch.from_numpy(ps_c).clone()
            if output_cuda:
                ps_c = ps_c.cuda()

            pc_list.append(ps_c)

        return pc_list

    # evaluate particles according to depth measurements
    def evaluate_depths_init(self, cls_id, depth, uv, z, q_idx, intrinsics, render_dist, codepose, delta=0.03, tau=0.05, mask=None):

        score = np.zeros_like(z)

        # crop rois
        depth_roi_cuda, _ = get_rois_cuda(depth.detach(), uv, z, intrinsics[0, 0], intrinsics[1, 1],
                                                 render_dist)
        depth_roi_np = depth_roi_cuda.cpu().numpy()

        if mask is not None:
            mask_roi_cuda, _ = get_rois_cuda(mask.detach(), uv, z, intrinsics[0, 0], intrinsics[1, 1],
                                                    render_dist)
            mask_roi_np = mask_roi_cuda.cpu().numpy()

        # render
        pose_v = np.zeros((7,))
        frame_cuda = torch.cuda.FloatTensor(self.im_h, self.im_w, 4)
        seg_cuda = torch.cuda.FloatTensor(self.im_h, self.im_w, 4)
        pc_cuda = torch.cuda.FloatTensor(self.im_h, self.im_w, 4)

        q_idx_unique, idx_inv = np.unique(q_idx, return_inverse=True)
        pc_render_all = np.zeros((q_idx_unique.shape[0], 128, 128, 3), dtype=np.float32)
        q_render = codepose[q_idx_unique][:, 3:]
        for i in range(q_render.shape[0]):
            pose_v[:3] = [0, 0, render_dist]
            pose_v[3:] = q_render[i]
            self.renderer.set_poses([pose_v])
            self.renderer.render([cls_id], frame_cuda, seg_cuda, pc2_tensor=pc_cuda)
            render_roi_cuda, _ = get_rois_cuda(pc_cuda.flip(0),
                                                     np.array([[intrinsics[0, 2], intrinsics[1, 2], 1]]),
                                                     np.array([[render_dist]]),
                                                     intrinsics[0, 0], intrinsics[1, 1],
                                                     render_dist)
            pc_render_all[i] = render_roi_cuda[0, :3, :, :].permute(1, 2, 0).cpu().numpy()

        # evaluate every particle
        for i in range(uv.shape[0]):

            pc_render = pc_render_all[idx_inv[i]].copy()
            depth_mask = pc_render[:, :, 2] > 0
            pc_render[:, :, 2][depth_mask] = pc_render[:, :, 2][depth_mask] - render_dist + z[i]

            depth_render = pc_render[:, :, [2]]

            depth_meas_np = depth_roi_np[i, 0, :, :]
            depth_render_np = depth_render[:, :, 0]

            # compute visibility mask
            if mask is None:
                visibility_mask = estimate_visib_mask_numba(depth_meas_np, depth_render_np, delta=delta)
            else:
                visibility_mask = np.logical_and((mask_roi_np[i, 0, :, :] > 0), np.logical_and(depth_meas_np>0, depth_render_np>0))

            if np.sum(visibility_mask) == 0:
                continue

            # compute depth error
            depth_error = np.abs(depth_meas_np[visibility_mask] - depth_render_np[visibility_mask])
            depth_error /= tau
            depth_error[depth_error > 1] = 1

            # score computation
            total_pixels = np.sum((depth_render_np > 0).astype(np.float32))
            if total_pixels is not 0:
                vis_ratio = np.sum(visibility_mask.astype(np.float32)) / total_pixels
                score[i] = (1 - np.mean(depth_error)) * vis_ratio
            else:
                score[i] = 0

        return score

    # evaluate particles according to depth measurements
    def evaluate_depths_tracking(self, rbpf, cls_id, depth, uv, z, q_idx, intrinsics, render_dist, codepose, rbpf_ready,
                        delta=0.03, tau=0.05, mask=None):

        if mask is not None:
            depth = depth.float() * mask

        # crop rois
        depth_roi_cuda, _ = get_rois_cuda(depth.detach(), uv, z, intrinsics[0, 0], intrinsics[1, 1],
                                                 render_dist)
        depth_meas_all = depth_roi_cuda[:, 0, :, :]

        # render
        pose_v = np.zeros((7,))
        frame_cuda = torch.cuda.FloatTensor(self.im_h, self.im_w, 4)
        seg_cuda = torch.cuda.FloatTensor(self.im_h, self.im_w, 4)
        pc_cuda = torch.cuda.FloatTensor(self.im_h, self.im_w, 4)

        fast_rendering = False
        if rbpf_ready and np.linalg.norm(rbpf.trans_bar) > 0: # fast rendering for tracking
            pose_v[:3] = rbpf.trans_bar
            pose_v[3:] = mat2quat(rbpf.rot_bar)
            self.renderer.set_poses([pose_v])
            self.renderer.render([cls_id], frame_cuda, seg_cuda, pc2_tensor=pc_cuda)
            uv_crop = project(rbpf.trans_bar, intrinsics)
            uv_crop = np.repeat(np.expand_dims(uv_crop, axis=0), uv.shape[0], axis=0)
            z_crop = np.ones_like(z) * rbpf.trans_bar[2]
            render_roi_cuda, _ = get_rois_cuda(pc_cuda.flip(0), uv_crop, z_crop,
                                                      intrinsics[0, 0],
                                                      intrinsics[1, 1],
                                                      render_dist)
            depth_render_all = render_roi_cuda[:, 2, :, :]
            fast_rendering = True
        else:
            q_idx_unique, idx_inv = np.unique(q_idx, return_inverse=True)
            depth_render_uni = torch.zeros((q_idx_unique.shape[0], 128, 128), dtype=torch.float32).cuda()
            q_render = codepose[q_idx_unique][:, 3:]
            for i in range(q_render.shape[0]):
                pose_v[:3] = [0, 0, render_dist]
                pose_v[3:] = q_render[i]
                self.renderer.set_poses([pose_v])
                self.renderer.render([cls_id], frame_cuda, seg_cuda, pc_cuda)
                render_roi_cuda, _ = get_rois_cuda(pc_cuda.flip(0),
                                                         np.array([[intrinsics[0, 2], intrinsics[1, 2], 1]]),
                                                         np.array([[render_dist]]),
                                                         intrinsics[0, 0], intrinsics[1, 1],
                                                         render_dist)
                depth_render_uni[i] = render_roi_cuda[0, 2, :, :].clone()
            depth_render_all = torch.zeros((q_idx.shape[0], 128, 128), dtype=torch.float32).cuda()
            for i in range(q_idx.shape[0]):
                depth_render_all[i] = depth_render_uni[idx_inv[i]].clone()

        # eval particles
        # shift the rendered image
        depth_invalid_mask = depth_render_all == 0
        if fast_rendering:
            depth_shift_np = z[:, 0] - rbpf.trans_bar[2]
        else:
            depth_shift_np = z[:, 0] - render_dist
        depth_shift = torch.from_numpy(depth_shift_np).cuda().float().repeat(depth_render_all.size(1),
                                                                             depth_render_all.size(2),
                                                                             1).permute(2, 0, 1)
        depth_render_all += depth_shift
        depth_render_all[depth_invalid_mask] = 0

        # compute visibility mask
        visibility_mask_cuda = estimate_visib_mask_cuda(depth_meas_all, depth_render_all, delta=delta)

        # compute scores
        # depth errors
        depth_error = torch.ones_like(depth_render_all).cuda()
        depth_error[visibility_mask_cuda] = torch.abs(depth_meas_all[visibility_mask_cuda] -
                                                      depth_render_all[visibility_mask_cuda]) / tau
        depth_error = torch.clamp(depth_error, 0, 1.0)
        depth_error_mean = torch.mean(depth_error, (2, 1))

        # visible ratio
        total_pixels = torch.sum(depth_render_all > 0, (2, 1)).float()
        total_pixels[total_pixels == 0] = 10000
        vis_ratio = torch.sum(visibility_mask_cuda, (2, 1)).float() / total_pixels

        # scores
        score = (torch.ones_like(depth_error_mean) - depth_error_mean) * vis_ratio

        return score.unsqueeze(1), vis_ratio

    def transform_points(self, Transformation, points):
        return torch.matmul(Transformation, points.permute(1, 0)).permute(1, 0)
from utils.poserbpf_utils import *
from utils.averageQuaternions import *
import torch
import torch.nn as nn
import scipy.stats as sci_stats
import time
import matplotlib.pyplot as plt

class particle_filter():
    def __init__(self, cfg_pf, n_particles=100, resample_method='systematic'):
        # particles
        self.uv = np.zeros((n_particles, 3), dtype=float)
        self.uv[:, 2] = 1
        self.z = np.zeros((n_particles, 1), dtype=float)

        # rotation distribution
        # discretize the so3 into space with 5 degrees interval, elevation - azimuth - in plane rotation
        self.rot = torch.ones((n_particles, 1, 37, 72, 72)).cuda().float()

        # for shifting the distribution with grid sample
        self.elevation_space = np.linspace(-1.0, 1.0, 37)
        self.azimuth_space = np.linspace(-1.0, 1.0, 72)
        self.bank_space = np.linspace(-1.0, 1.0, 72)

        # processing noise
        self.rot_gk3 = GaussianKernel3D(cfg_pf.ROT_GAUSSIAN_KERNEL_SZ, cfg_pf.ROT_GAUSSIAN_KERNEL_STD)
        self.rot_filter = nn.Conv3d(1, 1, cfg_pf.ROT_GAUSSIAN_KERNEL_SZ, bias=False, padding=(cfg_pf.ROT_GAUSSIAN_KERNEL_SZ//2,
                                                                                              cfg_pf.ROT_GAUSSIAN_KERNEL_SZ//2,
                                                                                              cfg_pf.ROT_GAUSSIAN_KERNEL_SZ//2))
        self.rot_filter.weight.data = torch.from_numpy(self.rot_gk3).float().cuda().expand(1,
                                                                                           1,
                                                                                           cfg_pf.ROT_GAUSSIAN_KERNEL_SZ,
                                                                                           cfg_pf.ROT_GAUSSIAN_KERNEL_SZ,
                                                                                           cfg_pf.ROT_GAUSSIAN_KERNEL_SZ)
        self.rot_filter.weight.requires_grad = False
        self.ele_pad = nn.ReplicationPad3d((0, 0, 0, 0, cfg_pf.ROT_GAUSSIAN_KERNEL_SZ//2, cfg_pf.ROT_GAUSSIAN_KERNEL_SZ//2))

        # star
        self.uv_star = [0, 0, 1]
        self.z_star = 0
        self.rot_star = np.identity(3)
        self.trans_star = [0, 0, 0]

        # expectation
        self.uv_bar = [0, 0, 1]
        self.z_bar = 0
        self.trans_bar = [0, 0, 0]
        self.rot_bar = np.identity(3)
        self.rot_bar_prev = np.identity(3)
        self.rot_var = cfg_pf.ROT_VAR

        # motion model
        self.delta_rot = np.identity(3)
        self.delta_uv = [0, 0, 0]
        self.delta_z = 0

        # weights
        self.weights = np.ones((n_particles, ), dtype=float)/n_particles

        # sampling method
        assert resample_method in ['systematic', 'stratified'], 'Resampling method is not supported!'
        self.resample_method = resample_method
        self.resample_idx = np.ones((n_particles, ), dtype=int)

        # particle number
        self.n_particles = n_particles

        # re-initialization
        self.d_angle = 0
        self.uv_star_prev = [0, 0, 1]
        self.trans_star_prev = [0, 0, 0]
        self.prev_q_stars = []

        self.cfg_pf = cfg_pf

        # visualization
        self.views_vis = np.zeros((cfg_pf.N_E_ROT, 3), dtype=np.float)
        self.wt_vis = np.ones((cfg_pf.N_E_ROT, 1), dtype=np.float)

        # rotation distribution expectation
        self.E_Rot = torch.ones((37 * 72 * 37,)).cuda().float()

    def check_rot_expectation(self):
        n_q_stars = 30

        q_star = mat2quat(self.rot_star)
        q_bar = mat2quat(self.rot_bar)

        d_bar_star = rot_diff(q_star, q_bar)

        if len(self.prev_q_stars) == n_q_stars and d_bar_star > 1.0:
                d_angles_bar = sum(rots_diff(q_bar, self.prev_q_stars))
                d_angles_star = sum(rots_diff(q_star, self.prev_q_stars))
                if d_angles_bar > d_angles_star:
                    self.rot_bar = self.rot_star.copy()
                    self.prev_q_stars = []

        if d_bar_star < 1.5:
            self.prev_q_stars.insert(0, q_star)

        if len(self.prev_q_stars) > n_q_stars:
            self.prev_q_stars.pop()


    def add_noise_rot(self):
        uni_v, uni_i = np.unique(self.resample_idx, return_inverse=True)
        # rot_pad = self.rot[uni_v].detach().repeat(1, 1, 1, 3, 3)[:, :, :,
        #           (72 - self.cfg_pf.ROT_GAUSSIAN_KERNEL_SZ//2):(72 * 2 + self.cfg_pf.ROT_GAUSSIAN_KERNEL_SZ//2),
        #           (72 - self.cfg_pf.ROT_GAUSSIAN_KERNEL_SZ//2):(72 * 2 + self.cfg_pf.ROT_GAUSSIAN_KERNEL_SZ//2)]
        # rot_pad = self.ele_pad(rot_pad).detach()
        rot_filtered = self.rot_filter(self.rot[uni_v].detach())
        self.rot = rot_filtered[uni_i]

    def search_neighbourhood(self, q_index, threshold = 0.2):
        view_q = self.view_code[q_index].repeat(self.view_code.size(0), 1)
        view_diff = self.view_code.detach() - view_q
        # view_diff[view_diff > np.pi] -= np.pi * 2
        # view_diff[view_diff < -np.pi] += np.pi * 2
        # view_diff_norm = torch.norm(view_diff, 2, 1)
        view_diff = torch.abs(view_diff)
        view_diff_max, _ = torch.max(view_diff, dim=1)
        idx = (view_diff_max < threshold).nonzero()
        return idx

    def update_trans_star_uvz(self, uv_star, z_star, intrinsics):
        self.uv_star_prev = self.uv_star
        self.uv_star = uv_star
        self.z_star = z_star

        self.trans_star_prev = self.trans_star

        self.trans_star = back_project(uv_star,intrinsics, z_star)

        if len(self.trans_star.shape) > 1:
            self.trans_star = self.trans_star.squeeze(1)

        self.uv = np.repeat(np.expand_dims(uv_star, axis=0), self.n_particles, axis=0)
        self.z = np.repeat([z_star], self.n_particles, axis=0)

    def update_trans_star(self, uv_star, z_star, intrinsics):
        self.uv_star_prev = self.uv_star
        self.uv_star = uv_star
        self.z_star = z_star

        self.trans_star_prev = self.trans_star

        self.trans_star = back_project(uv_star,intrinsics, z_star).squeeze(1)

    def update_rot_star_q(self, q_star):
        self.rot_star = quat2mat(add_trans_q(self.trans_star, q_star))
        self.rot = np.repeat([self.rot_star], self.n_particles, axis=0)
        self.rot_bar = self.mean_orientation()

    def update_rot_star_R(self, R_star):
        # self.d_angle = single_orientation_error(mat2quat(R_star), mat2quat(self.rot_star))
        self.rot_star = R_star
        # self.rot_bar
        # self.rot = np.repeat([self.rot_star], self.n_particles, axis=0)

    def update_weights(self, weights):
        self.weights = weights
        self.weights /= np.sum(self.weights)

    def resample_from_index_trans(self, indexes):
        self.uv = self.uv[indexes]
        self.z = self.z[indexes]

        self.uv_bar = np.mean(self.uv, axis=0)
        self.z_bar = np.mean(self.z, axis=0)

        self.weights = np.ones((self.n_particles,), dtype=float) / self.n_particles

    def resample_from_index_rot(self, indexes):
        self.rot = self.rot[indexes]

        self.weights = np.ones((self.n_particles,), dtype=float) / self.n_particles

    def resample_from_index(self, indexes):
        self.rot = self.rot[indexes]
        self.uv = self.uv[indexes]
        self.z = self.z[indexes]
        self.weights = np.ones((self.n_particles,), dtype=float) / self.n_particles

    def resample_trans(self, intrinsics):
        self.weights[-1] *= 0
        if self.resample_method == 'systematic':
            indexes = systematic_resample(self.weights)
        else:
            indexes = stratified_resample(self.weights)

        self.resample_from_index_trans(indexes)
        self.trans_bar = self.mean_translation(intrinsics)

    def resample_rot(self):
        if self.resample_method == 'systematic':
            indexes = systematic_resample(self.weights)
        else:
            indexes = stratified_resample(self.weights)

        self.resample_from_index_rot(indexes)

        rot_bar_prev = self.rot_bar
        self.rot_bar = self.mean_orientation()
        self.d_angle = single_orientation_error(mat2quat(rot_bar_prev), mat2quat(self.rot_bar))

    def resample_all(self):
        if self.resample_method == 'systematic':
            indexes = systematic_resample(self.weights)
        else:
            indexes = stratified_resample(self.weights)

        self.resample_from_index(indexes)

        rot_bar_prev = self.rot_bar
        self.rot_bar = self.mean_orientation()
        self.d_angle = single_orientation_error(mat2quat(rot_bar_prev), mat2quat(self.rot_bar))

    def pairwise_distances(self, x, y=None):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

        return torch.clamp(dist, 0.0, np.inf)

    def retrieve_idx_q(self, q, codepose):
        q_query = codepose[:, 3:]
        distance_matrix = self.pairwise_distances(q, q_query)
        distance_max, idx = torch.min(distance_matrix, dim=1)
        return idx

    def resample_ddpf(self, codepose, intrinsics, cfg_pf):
        if self.resample_method == 'systematic':
            indexes = systematic_resample(self.weights)
        else:
            indexes = stratified_resample(self.weights)

        self.resample_idx = indexes

        self.resample_from_index(indexes)
        self.delta_uv = np.mean(self.uv, axis=0) - self.uv_bar
        self.delta_z = np.mean(self.z, axis=0) - self.z_bar
        self.uv_bar = np.mean(self.uv, axis=0)
        self.z_bar = np.mean(self.z, axis=0)
        self.trans_bar = back_project(self.uv_bar, intrinsics, self.z_bar).squeeze(1)

        # get the top 100 and compute the weighted mean
        # rot_sum, _ = torch.max(self.rot.view(self.n_particles, -1), dim=0)
        rot_sum = torch.sum(self.rot.view(self.n_particles, -1), dim=0)
        rot_wt, rot_idx = torch.topk(rot_sum, cfg_pf.N_E_ROT)
        rot_wt /= torch.sum(rot_wt)
        rot_wt = rot_wt.cpu().numpy()

        # rot_meam = torch.sum(self.rot.view(self.n_particles, -1), dim=0)
        # self.E_Rot = rot_meam / torch.sum(rot_meam)
        # self.E_Rot = rot_sum / torch.sum(rot_sum)

        # visualization
        # for i in range(rot_wt.shape[0]):
        #     Rco = quat2mat(codepose[rot_idx[i], 3:].cpu().numpy())
        #     Roc = Rco.transpose(1, 0)
        #     self.views_vis[i] = Roc[:, 2]
        #
        # self.wt_vis = np.expand_dims(rot_wt, 1)

        q_rot = codepose[rot_idx.cpu().numpy(), 3:]
        add_trans_qs_numba(self.trans_bar, q_rot)
        d_axis, d_angle = mat2axangle(self.delta_rot)
        self.delta_rot = axangle2mat(d_axis, cfg_pf.MOTION_R_FACTOR * d_angle)
        q_mean = weightedAverageQuaternions_star_numba(q_rot,
                                                       mat2quat(np.matmul(self.delta_rot, self.rot_bar)),
                                                         rot_wt,
                                                         cfg_pf.ROT_RANGE,
                                                         self.rot_var)
        self.delta_rot = np.matmul(quat2mat(q_mean), np.transpose(self.rot_bar))
        self.rot_bar = quat2mat(q_mean)

    def propagate_particles(self, Tc1c0, To0o1, n_t, n_R, intrinsics):
        Tco = np.eye(4, dtype=np.float32)
        Tco[:3, :3] = self.rot_bar

        rotation_prev = rotm2viewpoint(self.rot_bar) * 57.3
        rotation_curr = rotm2viewpoint(np.matmul(Tc1c0[:3, :3], np.matmul(self.rot_bar, To0o1[:3, :3]))) * 57.3

        elevation_shift = (rotation_curr[0] - rotation_prev[0]) / 5 / 37
        azimuth_shift = (rotation_curr[1] - rotation_prev[1]) / 5 / 72
        bank_shift = (rotation_curr[2] - rotation_prev[2]) / 5 / 72

        elevations = self.elevation_space + elevation_shift
        azimuths = self.azimuth_space + azimuth_shift
        azimuths[azimuth_shift > 1] -= 2
        azimuths[azimuth_shift < -1] += 2
        banks = self.bank_space + bank_shift
        banks[banks > 1] -= 2
        banks[banks < -1] += 2

        ele_grid, azi_grid, ban_grid = np.meshgrid(elevations, azimuths, banks, indexing='ij')
        grid = np.stack((ban_grid, azi_grid, ele_grid), axis=-1)
        grid_torch = torch.from_numpy(grid).unsqueeze(0).float().cuda()

        for i in range(int(self.n_particles)):
            t_v = back_project(self.uv[i], intrinsics, self.z[i]).squeeze(1)
            Tco[:3, 3] = t_v
            Tc1o = np.matmul(Tc1c0, np.matmul(Tco, To0o1))
            t_v = Tc1o[:3, 3]
            self.z[i] = t_v[2]
            self.uv[i] = project(t_v, intrinsics)
            self.rot[[i]] = F.grid_sample(self.rot[[i]], grid_torch)

        # Tbar
        Tco[:3, :3] = self.rot_bar
        Tco[:3, 3] = self.trans_bar
        Tco = np.matmul(Tc1c0, np.matmul(Tco, To0o1))
        self.rot_bar = Tco[:3, :3]
        self.trans_bar = Tco[:3, 3]


    def add_noise_se3(self, uv_noise, z_noise, rot_noise):
        self.uv[:, :2] += np.random.uniform(-uv_noise, uv_noise, (self.n_particles, 2))
        self.z += np.random.uniform(-z_noise * self.z, z_noise * self.z, (self.n_particles, 1))
        for i in range(self.n_particles):
            # rot_noise_euler = np.multiply(np.random.randn(3), rot_noise)
            rot_noise_euler = np.random.uniform(-rot_noise, rot_noise, (3,))
            rot_m = euler2mat(rot_noise_euler[0], rot_noise_euler[1], rot_noise_euler[2])
            self.rot[i, :, :] = np.matmul(self.rot[i, :, :], rot_m)

    def add_noise_r3(self, uv_noise, z_noise):
        self.uv[:, :2] += np.repeat([self.delta_uv[:2]], self.n_particles, axis=0) * self.cfg_pf.MOTION_T_FACTOR
        # self.uv[:, :2] += np.random.uniform(-uv_noise, uv_noise, (self.n_particles, 2))
        self.uv[:, :2] += np.random.randn(self.n_particles, 2) * uv_noise
        self.z += self.delta_z * self.cfg_pf.MOTION_T_FACTOR
        self.z += np.random.randn(self.n_particles, 1) * z_noise

    def add_noise_so3(self, rot_noise):
        for i in range(self.n_particles):
            rot_noise_euler = np.random.uniform(-rot_noise, rot_noise, (3,))
            rot_m = euler2mat(rot_noise_euler[0], rot_noise_euler[1], rot_noise_euler[2])
            self.rot[i, :, :] = np.matmul(self.rot[i, :, :], rot_m)

    def set_rot_with_qt(self, qs):
        assert qs.shape[0] == self.n_particles, ' size mismatch ! '
        for i in range(self.n_particles):
            self.rot[i] = quat2mat(add_trans_q(self.trans_star, qs[i]))
        self.rot_bar = self.mean_orientation()

    def compute_rot_errors(self, q_gt):
        rot_errors = np.zeros((self.n_particles,))
        for i in range(self.n_particles):
            rot_errors[i] = abs(single_orientation_error(q_gt, mat2quat(self.rot[i])))

        return rot_errors

    def mean_orientation(self):
        # convert rotm to quaternion representation
        Q = np.zeros((self.n_particles, 4))
        for i in range(self.n_particles):
            Q[i, :] = mat2quat(self.rot[i])
        q_mean = averageQuaternions(Q)
        self.rot_bar = quat2mat(q_mean)
        return self.rot_bar

    def mean_translation(self, intrinsics):
        self.trans_bar = [0, 0, 0]
        for i in range(self.n_particles):
            self.trans_bar += back_project(self.uv[i], intrinsics, self.z[i]).squeeze(1)

        self.trans_bar = self.trans_bar / self.n_particles

        return self.trans_bar


    def compute_rot_prob(self, q_gt, trans_gt, codepose):

        # convert to allocentric
        q_gt = compensate_trans_q(trans_gt, q_gt)

        q_dist = q_dist_numba(q_gt, codepose[:, 3:])

        n = 1
        idx_q = q_dist.argsort()[:n]

        rot_dist = self.E_Rot.cpu().numpy()

        prob_q = np.mean(rot_dist[idx_q])

        prob_cumulated = np.sum(rot_dist[np.where(rot_dist > prob_q)])

        # fig, ax = plt.subplots()
        # ax.plot(rot_dist)
        # ax.plot((idx_q[0]), (prob_q), 'o', color='r')
        # plt.show()

        return prob_cumulated

    def compute_rot_prob_2(self, q_gt, trans_gt, codepose):

        # convert to allocentric
        q_gt = compensate_trans_q(trans_gt, q_gt)

        q_dist = q_dist_numba(q_gt, codepose[:, 3:])
        # min_q_difference = np.min(q_dist)  # find the closest orientation

        min_q_difference = [np.min(q_dist), 5/57.3, 10/57.3, 20/57.3]

        rot_dist = self.E_Rot.cpu().numpy()
        rot_dist /= np.sum(rot_dist)  # convert to probability

        order = np.argsort(rot_dist)

        prob_cumulated = np.zeros((len(min_q_difference,)))

        for k in range(prob_cumulated.shape[0]):
            for ii in range(len(order)-1, -1, -1): # iterate the probabilities in descending order
                i = order[ii]
                # if we have not got to the min angle distance we should keep accumulating
                if q_dist[i] > min_q_difference[k]:
                    prob_cumulated[k] += rot_dist[i]
                else:
                    break

        rot_dist_reorder = np.flip(rot_dist[order])
        rot_dist_cum_sum = np.cumsum(rot_dist_reorder)

        percentiles = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        n_percentiles = np.ones_like(percentiles) * rot_dist_cum_sum.shape[0]

        for ip in range(percentiles.shape[0]):
            percentile = percentiles[ip]
            n_percentiles[ip] = np.argmax(rot_dist_cum_sum > percentile)+1

        return prob_cumulated, n_percentiles

    def compute_rot_dist(self, q, trans, codepose):

        q_gt = compensate_trans_q(trans, q)
        q_dist = q_dist_numba(q_gt, codepose[:, 3:])

        return q_dist

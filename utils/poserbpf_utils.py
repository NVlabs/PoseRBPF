# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import numpy as np
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *
import scipy.stats as sci_stats
from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import stratified_resample
import torch
import torch.nn.functional as F
from config.config import cfg
import cv2
import scipy.ndimage.filters as sci_filter
import numba
from numba import jit

from utils.RoIAlign.layer_utils.roi_layers import ROIAlign


def CropAndResizeFunction(image, rois):
    return ROIAlign((128, 128), 1.0, 0)(image, rois)


def CropAndResizeFunction2x(image, rois):
    return ROIAlign((256, 256), 1.0, 0)(image, rois)


def GaussianKernel3D(kernlen=3, nsig=1):
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return sci_filter.gaussian_filter(inp, nsig)


# pytorch argmax is slow
def my_arg_max(input):
    v_sims, i_sims = torch.max(input, dim=1)
    _, j_sim = torch.max(v_sims, dim=0)
    return [j_sim, i_sims[j_sim]]

def weight_computation(distance, var=0.05):
    distance_max = np.max(distance)
    weights = sci_stats.norm(distance_max, var).pdf(distance)
    return weights

def pose_error(particles, pose_gt):
        # particles in nxSE3: nx3x4 tensor
        # pose_gt in SE3: 3x4 matrix

        n_particles = particles.shape[0]
        trans_error = np.zeros(n_particles)
        rot_error = np.zeros(n_particles)

        for i in range(n_particles):
            R_co = particles[i, :3, :3]
            t_co = particles[i, :3, 3]

            R_gt = pose_gt[:3, :3]
            t_gt = pose_gt[:3, 3]

            trans_error[i] = np.linalg.norm(t_co - t_gt)
            _, rot_error[i] = mat2axangle(np.matmul(R_co,np.transpose(R_gt)))

        return trans_error, np.abs(rot_error)


def neff(weights):
        return 1. / np.sum(np.square(weights))


def resample_from_index(particles, weights, indexes):
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        weights.fill(1.0 / len(weights))


def add_noise_se3(particles, trans_noise, rot_noise):
    for i in range(particles.shape[0]):
        particles[i, :, 3] += np.multiply(np.random.randn(3), trans_noise)
        rot_noise_euler = np.multiply(np.random.randn(3), rot_noise)
        rot_m = euler2mat(rot_noise_euler[0], rot_noise_euler[1], rot_noise_euler[2])
        particles[i, :3, :3] = np.matmul(particles[i, :3, :3], rot_m)

def add_noise_so3(particles_rot, rot_noise):
    for i in range(particles_rot.shape[0]):
        # rot_noise_euler = np.multiply(np.random.randn(3), rot_noise)
        rot_noise_euler = np.random.uniform(-rot_noise, rot_noise, (3,))
        rot_m = euler2mat(rot_noise_euler[0], rot_noise_euler[1], rot_noise_euler[2])
        particles_rot[i, :, :] = np.matmul(particles_rot[i, :, :], rot_m)


def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])

def rotm2viewpoint(Rco):
    Roc = Rco.transpose()
    z_axis = Roc[:3, 2]
    elevation = np.arcsin(np.clip(-z_axis[2], -1, 1))
    azimuth = np.arctan2(-z_axis[1], -z_axis[0])
    y_c = np.cross(z_axis, np.array([0, 0, 1], dtype=np.float32))
    x_c = np.cross(y_c, z_axis)
    bank = np.arctan2(np.dot(Roc[:3, 0], y_c), np.dot(Roc[:3, 0], x_c))
    return np.array([elevation, azimuth, bank], dtype=np.float32)

def compute_orientation_error(pose_gt, poses_est):
    error = np.array([])
    for q in range(poses_est.size(0)):
        q_est = poses_est[q, 3:].cpu().numpy()
        q_gt = pose_gt[3:]
        q_diff = qmult(qinverse(q_est), q_gt)
        _, d_angle = quat2axangle(q_diff)
        if d_angle > np.pi:
            d_angle = d_angle - 2 * np.pi
            d_angle = - d_angle
        error = np.append(error, d_angle / np.pi * 180)
    return error


def project(t_co, intrinsics):
    # return homogeneous coordinate todo: maybe make this generic if time permits
    uv = np.matmul(intrinsics, t_co)
    if uv.shape[0] == 1:
        uv = uv[0]
    uv_h = uv / uv[2]
    return uv_h


def back_project(uv, intrinsics, z):
    # here uv is the homogeneous coordinates todo: maybe make this generic if time permits
    xyz = np.matmul(np.linalg.inv(intrinsics), np.transpose(uv))
    xyz = np.multiply(np.transpose(xyz), z)
    return xyz


def single_orientation_error(q_gt, q_est):
    q_diff = qmult(qinverse(q_est), q_gt)
    _, d_angle = quat2axangle(q_diff)
    if d_angle > np.pi:
        d_angle = d_angle - 2 * np.pi
        d_angle = - d_angle
    return d_angle

def single_orientation_error_axes(q_gt, q_est):
    q_diff = qmult(qinverse(q_est), q_gt)
    axis, d_angle = quat2axangle(q_diff)
    if d_angle > np.pi:
        d_angle = d_angle - 2 * np.pi
        d_angle = - d_angle
    return d_angle * axis


def allo2ego_q(translation, q):
    Rc1o1 = quat2mat(q)
    tco = translation
    p = [0, 0, 1]
    q = tco / np.linalg.norm(tco)
    r = np.cross(p, q)
    r_x = skew(r)
    Rc = np.identity(3) + r_x + np.matmul(r_x, r_x) * 1 / (1 + np.inner(p, q))
    Rc1o1 = np.matmul(Rc, Rc1o1)
    qco = mat2quat(Rc1o1)
    return qco


def allo2ego_R(translation, R):
    Rc1o1 = R
    tco = translation
    p = [0, 0, 1]
    q = tco / np.linalg.norm(tco)
    r = np.cross(p, q)
    r_x = skew(r)
    Rc = np.identity(3) + r_x + np.matmul(r_x, r_x) * 1 / (1 + np.inner(p, q))
    Rc1o1 = np.matmul(Rc, Rc1o1)
    return Rc1o1


def allo2ego_Rs(translation, Rs):
    for i in range(Rs.shape[0]):
        Rs[i, :, :] = allo2ego_R(translation, Rs[i, :, :])

def allo2ego_qs(translation, qs):
    for i in range(qs.shape[0]):
        qs[i, :] = allo2ego_q(translation, qs[i, :])

def ego2allo_q(translation, q):
    Rc1o1 = quat2mat(q)
    tco = translation
    p = [0, 0, 1]
    q = tco / np.linalg.norm(tco)
    r = np.cross(p, q)
    r_x = skew(r)
    Rc = np.identity(3) + r_x + np.matmul(r_x, r_x) * 1 / (1 + np.inner(p, q))
    Rc1o1 = np.matmul(np.transpose(Rc), Rc1o1)
    qco = mat2quat(Rc1o1)
    return qco


def ego2allo_R(translation, R):
    Rc1o1 = R
    tco = translation
    p = [0, 0, 1]
    q = tco / np.linalg.norm(tco)
    r = np.cross(p, q)
    r_x = skew(r)
    Rc = np.identity(3) + r_x + np.matmul(r_x, r_x) * 1 / (1 + np.inner(p, q))
    Rc1o1 = np.matmul(np.transpose(Rc), Rc1o1)
    return Rc1o1

def get_rois_cuda(image, uvs, zs, pf_fu, pf_fv, target_distance=2.5, out_size=128):
    image = image.permute(2, 0, 1).float().unsqueeze(0).cuda()

    bbox_u = target_distance * (1 / zs) / cfg.TRAIN.FU * pf_fu * out_size / image.size(3)
    bbox_u = torch.from_numpy(bbox_u).cuda().float().squeeze(1)
    bbox_v = target_distance * (1 / zs) / cfg.TRAIN.FV * pf_fv * out_size / image.size(2)
    bbox_v = torch.from_numpy(bbox_v).cuda().float().squeeze(1)

    center_uvs = torch.from_numpy(uvs).cuda().float()

    center_uvs[:, 0] /= image.size(3)
    center_uvs[:, 1] /= image.size(2)

    boxes = torch.zeros(center_uvs.size(0), 5).cuda()
    boxes[:, 1] = (center_uvs[:, 0] - bbox_u/2) * float(image.size(3))
    boxes[:, 2] = (center_uvs[:, 1] - bbox_v/2) * float(image.size(2))
    boxes[:, 3] = (center_uvs[:, 0] + bbox_u/2) * float(image.size(3))
    boxes[:, 4] = (center_uvs[:, 1] + bbox_v/2) * float(image.size(2))

    out = ROIAlign((out_size, out_size), 1.0, 0)(image, boxes)

    uv_scale = target_distance * (1 / zs) / cfg.TRAIN.FU * pf_fu

    return out, uv_scale

def mat2pdf(distance_matrix, mean, std):
    coeff = torch.ones_like(distance_matrix) * (1/(np.sqrt(2*np.pi) * std))
    mean = torch.ones_like(distance_matrix) * mean
    std = torch.ones_like(distance_matrix) * std
    pdf = coeff * torch.exp(- (distance_matrix - mean)**2 / (2 * std**2))
    return pdf

def mat2pdf_np(matrix, mean, std):
    coeff = np.ones_like(matrix) * (1/(np.sqrt(2*np.pi) * std))
    mean = np.ones_like(matrix) * mean
    std = np.ones_like(matrix) * std
    pdf = coeff * np.exp(- (matrix - mean)**2 / (2 * std**2))
    return pdf

# numba functions
@jit(nopython=True)
def cross_numba(p, q):
    return np.array([
        p[1]*q[2] - p[2]*q[1],
        p[2]*q[0] - p[0]*q[2],
        p[0]*q[1] - p[1]*q[0]
    ])

@jit(nopython=True)
def inner_numba(p, q):
    return p[0]*q[0] + p[1]*q[1] + p[2]*q[2]

@jit(nopython=True)
def inner_q_numba(p, q):
    return p[0]*q[0] + p[1]*q[1] + p[2]*q[2] + p[3]*q[3]

@jit(nopython=True)
def q_dist_numba(p, qs):
    distance = np.zeros((qs.shape[0],),dtype=np.float32)
    for i in range(qs.shape[0]):
        distance[i] = np.abs(rot_diff(p, qs[i]))
    return distance

@jit(nopython=True)
def quat2mat_numba(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    # if Nq < np.finfo(np.float).eps:
    #     return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

@jit(nopython=True)
def mat2quat_numba(M):
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q_v = vecs[:, np.argmax(vals)]
    q = np.zeros_like(q_v)
    q[0] = q_v[3]
    q[1] = q_v[0]
    q[2] = q_v[1]
    q[3] = q_v[2]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q

@jit(nopython=True)
def allo2ego_Rs_numba(translation, Rs):
    for i in range(Rs.shape[0]):
        Rc1o1 = Rs[i, :, :]
        tco = translation
        p = [0, 0, 1]
        q = tco / np.linalg.norm(tco)
        r = cross_numba(p, q)
        r_x = np.array([[0, -r[2], r[1]],
                     [r[2], 0, -r[0]],
                     [-r[1], r[0], 0]])
        Rc = np.identity(3) + r_x + r_x.dot(r_x) * 1 / (1 + inner_numba(p, q))
        Rc1o1 = Rc.dot(Rc1o1)
        Rs[i, :, :] = Rc1o1

@jit(nopython=True)
def allo2ego_dR_numba(translation):
    tco = translation
    p = [0, 0, 1]
    q = tco / np.linalg.norm(tco)
    r = cross_numba(p, q)
    r_x = np.array([[0, -r[2], r[1]],
                 [r[2], 0, -r[0]],
                 [-r[1], r[0], 0]])
    Rc = np.identity(3) + r_x + r_x.dot(r_x) * 1 / (1 + inner_numba(p, q))
    return Rc

@jit(nopython=True)
def allo2ego_qs_numba(translation, qs):
    for i in range(qs.shape[0]):
        Rc1o1 = quat2mat_numba(qs[i])
        tco = translation
        p = [0, 0, 1]
        q = tco / np.linalg.norm(tco)
        r = cross_numba(p, q)
        r_x = np.array([[0, -r[2], r[1]],
                     [r[2], 0, -r[0]],
                     [-r[1], r[0], 0]])
        Rc = np.identity(3) + r_x + r_x.dot(r_x) * 1 / (1 + inner_numba(p, q))
        Rc1o1 = Rc.dot(Rc1o1)
        qs[i] = mat2quat_numba(Rc1o1)

@jit(nopython=True)
def qmult_numba(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])

@jit(nopython=True)
def quat2axangle_numba(quat):
    w, x, y, z = quat
    Nq = w * w + x * x + y * y + z * z
    if Nq != 1:  # Normalize if not normalized
        s = math.sqrt(Nq)
        w, x, y, z = w / s, x / s, y / s, z / s
    len2 = x * x + y * y + z * z
    theta = 2 * math.acos(max(min(w, 1), -1))
    return  np.array([x, y, z]) / math.sqrt(len2), theta

@jit(nopython=True)
def quat2angle_numba(quat):
    w, x, y, z = quat
    Nq = w * w + x * x + y * y + z * z
    if Nq != 1:  # Normalize if not normalized
        s = math.sqrt(Nq)
        w, x, y, z = w / s, x / s, y / s, z / s
    theta = 2 * math.acos(max(min(w, 1), -1))
    return theta

@jit(nopython=True)
def qconjugate_numba(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

@jit(nopython=True)
def qnorm_numba(q):
    return np.sqrt(np.dot(q, q))

@jit(nopython=True)
def qinverse_numba(q):
    return qconjugate_numba(q) / qnorm_numba(q)

@jit(nopython=True)
def weightedAverageQuaternions_star_numba(Q, q_star, w, rot_range, rot_var):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4, 4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i,:]
        q_diff = qmult_numba(qinverse_numba(q_star), q)
        _, d_angle = quat2axangle_numba(q_diff)
        if d_angle > np.pi:
            d_angle = d_angle - 2 * np.pi
            d_angle = - d_angle

        if np.abs(d_angle) > rot_range:
            continue

        wt_motion = 1/np.sqrt(2*np.pi*rot_var**2)*np.exp(-d_angle**2/(2*rot_var**2))

        A = w[i] * wt_motion * np.outer(q,q) + A
        weightSum += w[i] * wt_motion

    # scale
    if not weightSum == 0:
        A = (1.0/weightSum) * A
    else:
        A += np.outer(q_star,q_star)

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eigh(A)

    # return the real part of the largest eigenvector (has only real part)
    return eigenVectors[:, np.argmax(eigenValues)]

@jit(nopython=True)
def weightedAverageQuaternions_numba(Q, w):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4, 4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A
        weightSum += w[i]

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eigh(A)

    # return the real part of the largest eigenvector (has only real part)
    return eigenVectors[:, np.argmax(eigenValues)]

@jit(nopython=True)
def rot_diff(p, q):
    q_diff = qmult_numba(qinverse_numba(p), q)
    d_angle = quat2angle_numba(q_diff)
    if d_angle > np.pi:
        d_angle = d_angle - 2 * np.pi
        d_angle = - d_angle
    return d_angle

@jit(nopython=True)
def rots_diff(p, quats):
    d_angles = []
    for q in quats:
        d_angles.append(rot_diff(p, q))
    return d_angles

@jit(nopython=True)
def matmul_numba(A, B):
    return A.dot(B)

@jit(nopython=True)
def project_numba(t_co, intrinsics):
    uv = matmul_numba(intrinsics, t_co)
    uv_h = uv / uv[2]
    return uv_h

@jit(nopython=True)
def back_project_numba(uv, intrinsics, z):
    xyz = matmul_numba(np.linalg.inv(intrinsics), np.transpose(uv))
    xyz = np.multiply(np.transpose(xyz), z)
    return xyz

def transform_points_numba(uv, z, Tc1c0, To0o1, Tco, intrinsics):
    for i in range(0, uv.shape[0]):
        t_v = back_project_numba(uv[[i]], intrinsics, z[i])[0]
        Tco[:3, 3] = t_v
        Tc1o = matmul_numba(Tc1c0, matmul_numba(Tco, To0o1))
        t_v = Tc1o[:3, 3]
        z[i] = t_v[2]
        uv[i] = project_numba(t_v.astype(np.float32), intrinsics.astype(np.float32))
    return uv, z

@jit(nopython=True)
def estimate_visib_mask_numba(d_test, d_model, delta = 0.1):

    assert(d_test.shape == d_model.shape)
    mask_valid = np.logical_and(d_test > 0, d_model > 0)

    d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
    visib_mask = np.logical_and(np.abs(d_diff) <= delta, mask_valid)

    return visib_mask

def estimate_visib_mask_cuda(d_test, d_model, delta = 0.1):

    assert(d_test.size() == d_model.size())
    mask_valid = torch.mul(d_test > 0, d_model > 0)

    d_diff = d_model - d_test
    visib_mask = torch.mul(torch.abs(d_diff) <= delta, mask_valid)

    return visib_mask

@jit(nopython=True)
def compute_depth_score_numba(t_v, render_dist, pc_render_np, depth_zoom_np, tau, delta):
    # convert pc from allocentric to egocentric
    dR = allo2ego_dR_numba(t_v)
    pc_flatten = pc_render_np.reshape((-1, 3))
    pc_flatten[:, 2] -= render_dist
    pc_flatten = matmul_numba(dR.astype(np.float32), pc_flatten.transpose().astype(np.float32))
    pc_render_np = pc_flatten.transpose().reshape((128, 128, 3))
    depth_render_np = pc_render_np[:, :, 2] + t_v[2]

    # compute visibility mask
    visibility_mask = estimate_visib_mask_numba(depth_zoom_np, depth_render_np, delta=delta)
    visibility_mask_float = visibility_mask.astype(np.float32)

    total_visible_pixels = np.sum(visibility_mask_float)

    if total_visible_pixels == 0:
        return 0

    # compute depth error
    depth_error = np.abs(np.multiply(visibility_mask_float, depth_zoom_np) - np.multiply(visibility_mask_float, depth_render_np))
    depth_error /= tau
    depth_error = np.where(depth_error < 1.0, depth_error, np.ones_like(depth_error))

    # score computation
    total_pixels = np.sum(depth_render_np > 0)
    if total_pixels is not 0:
        vis_ratio = np.sum(visibility_mask) / total_pixels
        score = (1 - np.sum(depth_error) / total_visible_pixels) * vis_ratio
    else:
        score = 0

    return score

@jit(nopython=True)
def compute_depth_scores_numba(t_vs, render_dist, pc_render_all, depth_zoom_all, tau, delta):
    scores = np.zeros_like((t_vs.shape[0],))
    for i in range(t_vs.shape[0]):
        scores[i] = compute_depth_score_numba(t_vs[0, :],
                                              render_dist,
                                              np.ascontiguousarray(pc_render_all[0, :, :, :]),
                                              depth_zoom_all[0, :, :, :],
                                              tau,
                                              delta)
    return scores


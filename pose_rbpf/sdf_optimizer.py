# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from transforms3d.quaternions import *
from transforms3d.axangles import *
import torch.nn.functional as F
import time
from utils.sdf_layer.sdf_matching_loss import *


def load_sdf(sdf_file, into_gpu=True):

    assert sdf_file[-3:] == 'pth', "cannot load this type of data"

    print(' start loading sdf from {} ... '.format(sdf_file))

    sdf_info = torch.load(sdf_file)
    min_coords = sdf_info['min_coords']
    max_coords = sdf_info['max_coords']
    xmin, ymin, zmin = min_coords
    xmax, ymax, zmax = max_coords
    sdf_torch = sdf_info['sdf_torch'][0, 0].permute(1, 0, 2)

    sdf_limits = torch.tensor([xmin, ymin, zmin, xmax, ymax, zmax], dtype=torch.float32, requires_grad=False)

    if into_gpu:
        sdf_torch = sdf_torch.cuda()
        sdf_limits = sdf_limits.cuda()

    print('     sdf size = {}x{}x{}'.format(sdf_torch.size(0), sdf_torch.size(1), sdf_torch.size(2)))
    print('     minimal coordinate = ({:.4f}, {:.4f}, {:.4f}) cm'.format(xmin * 100, ymin * 100, zmin * 100))
    print('     maximal coordinate = ({:.4f}, {:.4f}, {:.4f}) cm'.format(xmax * 100, ymax * 100, zmax * 100))
    print(' finished loading sdf ! ')

    return sdf_torch, sdf_limits

class sdf_optimizer():
    def __init__(self, lr=0.01, use_gpu=True):

        self.use_gpu = use_gpu

        if use_gpu:
            self.dpose = torch.tensor([0, 0, 0, 1e-12, 1e-12, 1e-12], dtype=torch.float32, requires_grad=True, device=0)
        else:
            self.dpose = torch.tensor([0, 0, 0, 1e-12, 1e-12, 1e-12], dtype=torch.float32, requires_grad=True)

        self.sdf_loss = SDFLoss()
        self.sdf_optimizer = optim.Adam([self.dpose], lr=lr)

        if use_gpu:
            self.sdf_loss = self.sdf_loss.cuda()

    def compute_sdf(self, T_co_0, points, sdf_input=None, sdf_limits_input=None):
        if points.size(1) > 3:
            points = points[:, :3].contiguous()

        if sdf_input is None or sdf_limits_input is None:
            return

        # construct initial pose
        T_oc_0 = np.linalg.inv(T_co_0)

        if self.use_gpu:
            pose_init = torch.from_numpy(T_oc_0).cuda()
        else:
            pose_init = torch.from_numpy(T_oc_0)

        self.dpose.data[:3] *= 0
        self.dpose.data[3:] = self.dpose.data[3:] * 0 + 1e-12

        loss, sdf_values, T_oc_opt = self.sdf_loss(self.dpose.detach(), pose_init, sdf_input, sdf_limits_input, points)

        return sdf_values


    def refine_pose(self, T_co_0, points, sdf_input=None, sdf_limits_input=None, steps=100):
        # input T_co_0: 4x4
        #       points: nx3 in camera

        if points.size(1) > 3:
            points = points[:, :3].contiguous()

        if sdf_input is None or sdf_limits_input is None:
            return

        # construct initial pose
        T_oc_0 = np.linalg.inv(T_co_0)

        if self.use_gpu:
            pose_init = torch.from_numpy(T_oc_0).cuda()
        else:
            pose_init = torch.from_numpy(T_oc_0)

        self.dpose.data[:3] *= 0
        self.dpose.data[3:] = self.dpose.data[3:] * 0 + 1e-12

        for i in range(steps):
            self.sdf_optimizer.zero_grad()
            loss, sdf_values, T_oc_opt = self.sdf_loss(self.dpose, pose_init, sdf_input, sdf_limits_input, points)
            loss.backward()
            self.sdf_optimizer.step()

        T_co_opt = np.linalg.inv(T_oc_opt.cpu().detach().numpy())
        return T_co_opt, sdf_values
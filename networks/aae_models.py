# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os
import datetime
import re
import matplotlib.pyplot as plt
import time
from transforms3d.quaternions import *
from decimal import *
import cv2
from config.config import cfg
from collections import OrderedDict
from utils.RoIAlign.layer_utils.roi_layers import ROIAlign

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s \n' % (prefix, bar, percent, suffix))
    # # Print New Line on Complete
    # if iteration == total:
    #     print()

def isSubstring(s1, s2):
    M = len(s1)
    N = len(s2)

    for i in range(N - M + 1):
        for j in range(M):
            if (s2[i + j] != s1[j]):
                break
        if j + 1 == M:
            return i

    return -1

class encoder_rgbd(nn.Module):
    def __init__(self, capacity=1, code_dim=128, encode_depth=False):
        super(encoder_rgbd, self).__init__()
        self.encode_depth = encode_depth
        self.capacity = capacity

        # for RGB
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128 * capacity, 256 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256 * capacity, 256 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256 * capacity, 512 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.roi_layer = ROIAlign((16, 16), 1.0, 0)
        self.fc_rgb = nn.Linear(512 * 8 * 8 * capacity, code_dim)

        # for depth
        self.layer1_d = nn.Sequential(
            nn.Conv2d(1, 64 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer2_d = nn.Sequential(
            nn.Conv2d(64 * capacity, 128 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer3_d = nn.Sequential(
            nn.Conv2d(128 * capacity, 128 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer4_d = nn.Sequential(
            nn.Conv2d(128 * capacity, 256 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.fc_depth = nn.Linear(256 * 8 * 8 * capacity, code_dim)
        self.roi_layer_d = ROIAlign((128, 128), 1.0, 0)

    def forward_rgb(self, x, roi):
        # note here roi is represented in 256x256 image [upper_left_u, upper_left_v, lower_right_u, lower_right_v]
        out = self.layer1(x[:, :3])
        out = self.layer2(out)
        out = self.layer3(out)
        roi_rgb = roi.clone()
        roi_rgb[:, 1:] /= 8.0
        out = self.roi_layer(out, roi_rgb)
        out = self.layer4(out)
        out = out.view(-1, 512 * 8 * 8 * self.capacity)
        code_rgb = self.fc_rgb(out)
        output = code_rgb

        return output

    def forward_pf(self, x, roi, z, box_size):
        # note here roi is represented in 256x256 image [upper_left_u, upper_left_v, lower_right_u, lower_right_v]
        out = self.layer1(x[:, :3])
        out = self.layer2(out)
        out = self.layer3(out)
        roi_rgb = roi.clone()
        roi_rgb[:, 1:] /= 8.0
        out = self.roi_layer(out, roi_rgb)
        out = self.layer4(out)
        out = out.view(-1, 512 * 8 * 8 * self.capacity)
        code_rgb = self.fc_rgb(out)

        out_d = self.roi_layer_d(x[:, [3]], roi)
        out_d = (out_d - z) / box_size + 0.5
        out_d = torch.clamp(out_d, 0, 1)
        out_d = self.layer1_d(out_d)
        out_d = self.layer2_d(out_d)
        out_d = self.layer3_d(out_d)
        out_d = self.layer4_d(out_d)
        out_d = out_d.view(-1, 256 * 8 * 8 * self.capacity)
        out = out_d
        code_d = self.fc_depth(out)
        output = (code_rgb, code_d)

        return output

    def forward(self, x, roi):
        out = self.layer1(x[:, :3])
        out = self.layer2(out)
        out = self.layer3(out)
        roi_rgb = roi.clone()
        roi_rgb[:, 1:] /= 8.0
        out = self.roi_layer(out, roi_rgb)
        out = self.layer4(out)
        out = out.view(-1, 512 * 8 * 8 * self.capacity)
        code_rgb = self.fc_rgb(out)

        out_d = self.roi_layer_d(x[:, [3]], roi)
        out_d = self.layer1_d(out_d)
        out_d = self.layer2_d(out_d)
        out_d = self.layer3_d(out_d)
        out_d = self.layer4_d(out_d)
        out_d = out_d.view(-1, 256 * 8 * 8 * self.capacity)
        out = out_d
        code_d = self.fc_depth(out)
        output = (code_rgb, code_d)
        return output


class encoder_rgb(nn.Module):
    def __init__(self, object_total_n, capacity=1, code_dim=128):
        super(encoder_rgb, self).__init__()
        self.object_total_n = object_total_n
        self.layer1 = nn.Sequential(
            nn.Conv2d(3 + self.object_total_n, 128 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128 * capacity, 256 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256 * capacity, 256 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256 * capacity, 512 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(512 * 8 * 8 * capacity, code_dim)

        self.capacity = capacity

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 512 * 8 * 8 * self.capacity)
        out = self.fc(out)
        return out

class decoder_depth(nn.Module):
    def __init__(self, capacity=1, code_dim=128):
        super(decoder_depth, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256 * capacity, 128 * capacity, 5, 2, padding=2, output_padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128 * capacity, 128 * capacity, 5, 2, padding=2, output_padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128 * capacity, 64 * capacity, 5, 2, padding=2, output_padding=1),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64 * capacity, 1, 5, 2, padding=2, output_padding=1)
        )

        self.fc = nn.Linear(code_dim, 256 * 8 * 8 * capacity)

        self.dropout = nn.Dropout(0.5)

        self.capacity = capacity

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, 256 * self.capacity, 8, 8)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class decoder_rgb(nn.Module):
    def __init__(self, capacity=1, code_dim=128):
        super(decoder_rgb, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(512 * capacity, 256 * capacity, 5, 2, padding=2, output_padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256 * capacity, 256 * capacity, 5, 2, padding=2, output_padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256 * capacity, 128 * capacity, 5, 2, padding=2, output_padding=1),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128 * capacity, 3, 5, 2, padding=2, output_padding=1)
        )

        self.fc = nn.Linear(code_dim, 512 * 8 * 8 * capacity)

        self.dropout = nn.Dropout(0.5)

        self.capacity = capacity

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, 512 * self.capacity, 8, 8)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0.0)


def pairwise_cosine_distances(x, y, eps=1e-8):
    """
    :param x: batch of code from the encoder (batch size x code size)
    :param y: code book (codebook size x code size)
    :return: cosine similarity matrix (batch size x code book size)
    """
    dot_product = torch.mm(x, torch.t(y))
    x_norm = torch.norm(x, 2, 1).unsqueeze(1)
    y_norm = torch.norm(y, 2, 1).unsqueeze(1)
    normalizer = torch.mm(x_norm, torch.t(y_norm))

    return dot_product / normalizer.clamp(min=eps)


class BootstrapedMSEloss(nn.Module):
    def __init__(self, b=200):
        super(BootstrapedMSEloss, self).__init__()
        self.b = b

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        batch_size = pred.size(0)
        diff = torch.sum((target - pred)**2, 1)
        diff = diff.view(batch_size, -1)
        diff = torch.topk(diff, self.b, dim=1)
        self.loss = diff[0].mean()
        return self.loss

class AAE(nn.Module):
    def __init__(self, object_names, modality='rgbd', capacity=1, code_dim=128, model_path=None):
        super(AAE, self).__init__()
        # dir
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')

        self.model_dir = './checkpoints'

        self.object_names = object_names
        self.code_dim = code_dim

        self.modality = modality

        if self.modality == 'rgbd':
            self.encoder = encoder_rgbd(capacity=capacity, code_dim=code_dim)
            self.decoder = decoder_rgb(capacity=capacity, code_dim=code_dim)
            self.depth_decoder = decoder_depth(capacity=capacity, code_dim=code_dim)
            self.encoder.apply(weights_init)
            self.decoder.apply(weights_init)
            self.depth_decoder.apply(weights_init)
            self.optimizer = optim.Adam(list(self.encoder.parameters()) + \
                                        list(self.decoder.parameters()) + \
                                        list(self.depth_decoder.parameters()),
                                        lr=0.0002)
        else:
            self.encoder = encoder_rgb(object_total_n=1, capacity=capacity, code_dim=code_dim)
            self.decoder = decoder_rgb(capacity=capacity, code_dim=code_dim)
            self.encoder.apply(weights_init)
            self.decoder.apply(weights_init)

            self.optimizer = optim.Adam(list(self.encoder.parameters()) + \
                                        list(self.decoder.parameters()),
                                        lr=0.0002)

        self.model_path = model_path

        self.B_loss = BootstrapedMSEloss(cfg.TRAIN.BOOTSTRAP_CONST)
        self.L1_loss = torch.nn.L1Loss()
        self.Cos_loss = nn.CosineEmbeddingLoss()

        # GPU
        self.use_GPU = (torch.cuda.device_count() > 0)

        if self.use_GPU:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            if self.modality == 'rgbd':
                self.depth_decoder = self.depth_decoder.cuda()
            self.B_loss = self.B_loss.cuda()
            self.Cos_loss = self.Cos_loss.cuda()

    def load_ckpt_weights(self, state_dict):
        print('Start assigning weights to AAE ...')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            m_idx = isSubstring('module', str(k))
            if m_idx == -1:
                name = k
            else:
                name = k[:m_idx] + k[m_idx+7:]

            new_state_dict[name] = v

        self.load_state_dict(new_state_dict)
        self.weights_loaded = True
        print('Finished assigning weights to AAE !')

    # codebook generation and saving
    def compute_codebook(self, code_dataset, save_dir, batch_sz=1000, code_dim=128, save=True):
        assert self.weights_loaded, "need to load pretrained weights!"
        codebook_batch_size = batch_sz
        code_generator = torch.utils.data.DataLoader(code_dataset, batch_size=codebook_batch_size, shuffle=False, num_workers=0)
        print('code book size {}'.format(len(code_dataset)))
        step = 0
        self.encoder.eval()

        codebook_cpt = torch.zeros(len(code_dataset), code_dim).cuda()
        codepose_cpt = torch.zeros(len(code_dataset), 7).cuda()

        for inputs in code_generator:
            poses, rgb, depth = inputs

            if self.use_GPU:
                poses = poses.cuda()
                rgb = rgb.cuda()

            code = self.encoder.forward(rgb).detach().view(rgb.size(0), -1)

            print(code.size())

            codebook_cpt[step * codebook_batch_size:step * codebook_batch_size + code.size(0), :] = code
            codepose_cpt[step * codebook_batch_size:step * codebook_batch_size + code.size(0), :] = poses.squeeze(1)

            step += 1
            print('finished {}/{}'.format(step, len(code_generator)))

        if save:
            torch.save((codebook_cpt, codepose_cpt), save_dir)
            print('code book is saved to {}'.format(save_dir))

    def compute_codebook_rgbd(self, code_dataset, save_dir, batch_sz=250, code_dim=128, save=True):
        codebook_batch_size = batch_sz
        code_generator = torch.utils.data.DataLoader(code_dataset, batch_size=codebook_batch_size, shuffle=False, num_workers=0)
        print('code book size {}'.format(len(code_dataset)))
        step = 0
        self.encoder = self.encoder.cuda()
        self.encoder.eval()
        n_single_object = int(len(code_dataset))
        codebook_cpt = torch.zeros(n_single_object, code_dim).cuda()
        codepose_cpt = torch.zeros(n_single_object, 7).cuda()
        codebook_depth_cpt = torch.zeros(n_single_object, code_dim).cuda()

        for inputs in code_generator:
            images, poses, depths = inputs
            images = images.cuda()
            poses = poses.cuda()
            depths = depths.cuda()

            roi_info = torch.zeros(images.size(0), 5).float().cuda()
            roi_info[:, 0] = torch.arange(images.size(0))
            roi_info[:, 1] = 128.0 - 128.0 / 2
            roi_info[:, 2] = 128.0 - 128.0 / 2
            roi_info[:, 3] = 128.0 + 128.0 / 2
            roi_info[:, 4] = 128.0 + 128.0 / 2

            code, code_depth = self.encoder.forward(torch.cat((images.detach(), depths.detach()), dim=1),
                                                    roi_info.clone().detach())
            code = code.detach().view(images.size(0), -1)
            code_depth = code_depth.detach().view(images.size(0), -1)
            codebook_depth_cpt[step * codebook_batch_size:step * codebook_batch_size + code.size(0), :] = code_depth
            codebook_cpt[step * codebook_batch_size:step * codebook_batch_size + code.size(0), :] = code
            codepose_cpt[step * codebook_batch_size:step * codebook_batch_size + code.size(0), :] = poses.squeeze(1)

            step += 1
            print('finished {}/{}'.format(step, len(code_generator)))

        if save:
            torch.save((codebook_cpt, codepose_cpt, codebook_depth_cpt), save_dir)
            print('code book is saved to {}'.format(save_dir))

    def forward_rgbd(self, x, roi):
        code_rgb, code_depth = self.encoder.forward(x, roi)
        rgb_recon = self.decoder.forward(code_rgb)
        depth_recon = self.depth_decoder.forward(code_depth)
        out = torch.cat((rgb_recon, depth_recon), dim=1)
        return [out, code_rgb, code_depth]

    def forward(self, x):
        code = self.encoder(x)
        out = self.decoder(code)
        return [out, code]

    def compute_distance_matrix(self, code, codebook):
        assert codebook.size(0) > 0, "codebook is empty"
        return pairwise_cosine_distances(code, codebook)

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





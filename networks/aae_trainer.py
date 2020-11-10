# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from __future__ import division
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
from transforms3d.euler import *
from transforms3d.axangles import *
import scipy
from ycb_render.ycb_renderer import *
from decimal import *
import cv2
from shutil import copyfile
from networks.aae_models import *
from config.config import cfg
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import axes3d, Axes3D
import gc

class aae_trainer(nn.Module):
    def __init__(self, cfg_path, object_names, modality, config_new=None,
                 aae_capacity=1, aae_code_dim=128, ckpt_path=None, obj_ctg='ycb'):
        super(aae_trainer, self).__init__()

        self.cfg_path = cfg_path

        if config_new != None:
            self.cfg_all = config_new
        else:
            self.cfg_all = cfg

        self.obj_ctg = obj_ctg
        self.modality = modality

        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        self.ckpt_dir = './checkpoints'

        self.AAE = AAE(object_names=object_names,
                       modality=modality,
                       capacity=aae_capacity,
                       code_dim=aae_code_dim,
                       model_path=ckpt_path)

        self.object_names = object_names

        self.use_GPU = (torch.cuda.device_count() > 0)

        self.code_dim = aae_code_dim

        if self.modality == 'rgbd':
            self.optimizer = optim.Adam(list(self.AAE.encoder.parameters()) + \
                                        list(self.AAE.decoder.parameters()) + \
                                        list(self.AAE.depth_decoder.parameters()),
                                        lr=0.0002)
        else:
            self.optimizer = optim.Adam(list(self.AAE.encoder.parameters()) + \
                                        list(self.AAE.decoder.parameters()),
                                        lr=0.0002)

        self.mseloss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.l1_recon_loss = nn.L1Loss(reduction='mean')

        if self.use_GPU:
            self.mseloss = self.mseloss.cuda()
            self.l1_loss = self.l1_loss.cuda()
            self.l1_recon_loss = self.l1_recon_loss.cuda()

        self.loss_history_recon = []
        self.val_loss_history_recon = []

        self.batch_size_train = self.cfg_all.TRAIN.BATCH_SIZE
        self.batch_size_val = self.cfg_all.TRAIN.VAL_BATCH_SIZE
        self.start_epoch = 1

        self.codebook_dir = None
        self.log_dir = None
        self.checkpoint_path = None

        self.lb_shift = self.cfg_all.TRAIN.SHIFT_MIN
        self.ub_shift = self.cfg_all.TRAIN.SHIFT_MAX
        self.lb_scale = self.cfg_all.TRAIN.SCALE_MIN
        self.ub_scale = self.cfg_all.TRAIN.SCALE_MAX

        if ckpt_path is not None:
            self.load_ckpt(ckpt_path=ckpt_path)

    def set_log_dir(self, dataset_name='', model_path=None, now=None, ):
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        if now == None:
            now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/trans\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)), int(m.group(6)))

        # Directory for training logs
        self.log_dir = os.path.join(self.ckpt_dir, "{}_{:%Y%m%dT%H%M%S}_{}_{}".format(
            dataset_name, now, self.object_names[0], self.cfg_all.EXP_NAME))
        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "ckpt_{}_*epoch*.pth".format(
            self.obj_ctg))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def save_ckpt(self, epoch):
        print('=> Saving checkpoint to {} ...'.format(self.checkpoint_path.format(epoch)))
        torch.save({
            'epoch': epoch,
            'log_dir': self.log_dir,
            'checkpoint_path': self.checkpoint_path,
            'aae_state_dict': self.AAE.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_history_recon': self.loss_history_recon,
            'val_loss_history_recon': self.val_loss_history_recon,
        }, self.checkpoint_path.format(epoch))
        print('=> Finished saving checkpoint to {} ! '.format(self.checkpoint_path.format(epoch)))

    def load_ckpt(self, ckpt_path):
        if os.path.isfile(ckpt_path):
            print("=> Loading checkpoint from {} ...".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.log_dir = checkpoint['log_dir']
            self.checkpoint_path = checkpoint['checkpoint_path']
            self.AAE.load_ckpt_weights(checkpoint['aae_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.loss_history_recon = checkpoint['loss_history_recon']
            self.val_loss_history_recon = checkpoint['val_loss_history_recon']
            print("=> Finished loading checkpoint from {} (epoch {})"
                  .format(ckpt_path, checkpoint['epoch']))
        else:
            print('=> Cannot find checkpoint file in {} !'.format(ckpt_path))

    def plot_loss(self, loss, title, save=True, log_dir=None):
        loss = np.array(loss)
        plt.figure(title)
        plt.gcf().clear()
        plt.plot(loss, label='train')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        if save:
            save_path = os.path.join(log_dir, "{}.png".format(title))
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.1)

    def train_model(self, train_dataset, epochs, dstr_dataset=None, save_frequency=5):
        self.AAE.encoder.train()
        self.AAE.decoder.train()
        if self.modality == 'rgbd':
            self.AAE.depth_decoder.train()

        train_set = train_dataset
        print('train set size {}'.format(len(train_set)))

        if self.log_dir == None:
            self.set_log_dir(dataset_name=train_dataset._name)
            if not os.path.exists(self.log_dir) and save_frequency > 0:
                print('Create folder at {}'.format(self.log_dir))
                os.makedirs(self.log_dir)
                copyfile(self.cfg_path, self.log_dir + '/config.yml')

        print('dataset workers %d' % (self.cfg_all.TRAIN.WORKERS))
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size_train,
                                                      shuffle=True, num_workers=self.cfg_all.TRAIN.WORKERS)

        if dstr_dataset != None:
            print('background workers %d' % (self.cfg_all.TRAIN.DISTRACTOR_WORKERS))
            train_dstr_generator = torch.utils.data.DataLoader(dstr_dataset, batch_size=self.batch_size_train,
                                                              shuffle=True, num_workers=self.cfg_all.TRAIN.DISTRACTOR_WORKERS)
        else:
            train_dstr_generator = None

        train_steps = np.floor(len(train_set)/self.batch_size_train)
        train_steps = np.floor(train_steps/4)

        for epoch in np.arange(start=self.start_epoch, stop=(self.start_epoch+epochs)):
            print("Epoch {}/{}.".format(epoch, (self.start_epoch+epochs)-1))

            if self.modality == 'rgbd':
                recon_loss_rgb, recon_loss_train = self.train_epoch_rgbd(train_generator, self.optimizer,
                                                                          train_steps, epoch, self.start_epoch + epochs - 1,
                                                                          dstrgenerator=train_dstr_generator)
                self.loss_history_recon.append(recon_loss_rgb + recon_loss_train)

            else:
                recon_loss_rgb, recon_loss_train = self.train_epoch_rgb(train_generator, self.optimizer,
                                        train_steps, epoch, self.start_epoch+epochs-1,
                                        dstrgenerator=train_dstr_generator)
                self.loss_history_recon.append(recon_loss_rgb + recon_loss_train)


            self.plot_loss(self.loss_history_recon, 'recon loss', save=True, log_dir=self.log_dir)
            if save_frequency > 0 and epoch % save_frequency == 0:
                self.save_ckpt(epoch)

    def train_epoch_rgbd(self, datagenerator, optimizer, steps, epoch, total_epoch, dstrgenerator=None):
        loss_sum_rgb = 0
        loss_sum_depth = 0
        step = 0
        optimizer.zero_grad()

        if dstrgenerator != None:
            enum_dstrgenerator = enumerate(dstrgenerator)

        for inputs in datagenerator:

            # receiving data from the renderer
            images, images_target, pose_cam, mask,\
            translation_target, scale_target, \
            affine_target, roi_center, roi_size, roi_affine, depths, depths_target = inputs

            if self.use_GPU:
                images = images.cuda()
                images_target = images_target.cuda()
                mask = mask.cuda()
                roi_affine = roi_affine.cuda()
                roi_center = roi_center.cuda().float()
                roi_size = roi_size.cuda().float()

            # warp the images according to the center and size of rois
            grids = F.affine_grid(roi_affine, images.size())
            images = F.grid_sample(images, grids)
            depths = F.grid_sample(depths, grids)
            mask = F.grid_sample(mask, grids)
            mask = 1 - mask

            # add random background and gaussian noise
            if dstrgenerator != None:
                _, images_dstr = next(enum_dstrgenerator)
                if images_dstr.size(0) != images.size(0):
                    enum_dstrgenerator = enumerate(dstrgenerator)
                    _, images_dstr = next(enum_dstrgenerator)
                if self.use_GPU:
                    images_dstr = images_dstr.cuda()
                images = images + mask * images_dstr
            noise_level = np.random.uniform(0, 0.05)
            images += torch.randn_like(images) * noise_level

            # add random background to depth image
            _, depth_dstr = next(enum_dstrgenerator)
            if depth_dstr.size(0) != depths.size(0):
                enum_dstrgenerator = enumerate(dstrgenerator)
                _, depth_dstr = next(enum_dstrgenerator)
            if self.use_GPU:
                depth_dstr = depth_dstr.cuda()
            depths_background = torch.sum(mask * depth_dstr, dim=1, keepdim=True) / np.random.uniform(0.5, 2.0)
            depths = depths + depths_background
            depths += torch.rand_like(depths) * np.random.uniform(0, 0.05)
            depths = torch.clamp(depths, 0, 1)

            # construct tensor for roi information
            roi_info = torch.zeros(images.size(0), 5).float().cuda()
            roi_center += torch.from_numpy(np.random.uniform(self.cfg_all.TRAIN.SHIFT_MIN,
                                                                   self.cfg_all.TRAIN.SHIFT_MAX,
                                                                   size=(roi_info.size(0), 2))).float().cuda()
            roi_size = roi_size * torch.from_numpy(np.random.uniform(self.cfg_all.TRAIN.SCALE_MIN,
                                                                     self.cfg_all.TRAIN.SCALE_MAX,
                                                                     size=(roi_info.size(0), ))).float().cuda()
            roi_info[:, 0] = torch.arange(images.size(0))
            roi_info[:, 1] = roi_center[:, 0] - roi_size / 2
            roi_info[:, 2] = roi_center[:, 1] - roi_size / 2
            roi_info[:, 3] = roi_center[:, 0] + roi_size / 2
            roi_info[:, 4] = roi_center[:, 1] + roi_size / 2

            roi_info_copy = roi_info.clone()
            roi_info_copy_depth = roi_info.clone()

            # # visualization for debugging
            # roi_info_copy2 = roi_info.clone()
            # images_roi = ROIAlign((128, 128), 1.0, 0)(images, roi_info_copy)
            # images_roi_disp = images_roi[0].permute(1, 2, 0).cpu().numpy()
            # depth_roi = ROIAlign((128, 128), 1.0, 0)(depths, roi_info_copy2)
            # depth_roi_disp = depth_roi[0, 0].cpu().numpy()
            # image_disp = images[0].permute(1, 2, 0).cpu().numpy()
            # depth_disp = depths[0, 0].cpu().numpy()
            # depth_target_disp = depths_target[0, 0].cpu().numpy()
            # mask_disp = mask[0].permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy()
            # plt.figure()
            # plt.subplot(2, 3, 1)
            # plt.imshow(np.concatenate((image_disp, mask_disp), axis=1))
            # plt.subplot(2, 3, 2)
            # plt.imshow(images_roi_disp)
            # plt.subplot(2, 3, 3)
            # plt.imshow(images_target[0].permute(1, 2, 0).cpu().numpy())
            # plt.subplot(2, 3, 4)
            # plt.imshow(depth_disp)
            # plt.subplot(2, 3, 5)
            # plt.imshow(depth_roi_disp)
            # plt.subplot(2, 3, 6)
            # plt.imshow(depth_target_disp)
            # plt.show()

            # AAE forward pass
            images_input = torch.cat((images, depths), dim=1)
            outputs = self.AAE.forward_rgbd(images_input, roi_info)
            images_recnst = outputs[0]
            loss_reconstr = self.AAE.B_loss(images_recnst[:, :3, :, :], images_target.detach())
            loss_depth = self.AAE.B_loss(images_recnst[:, [3], :, :], depths_target)
            loss = loss_reconstr + loss_depth

            loss_aae_rgb_data = loss_reconstr.data.cpu().item()
            loss_aae_depth_data = loss_depth.data.cpu().item()

            # AAE backward pass
            optimizer.zero_grad()
            try:
                loss.backward()
            except:
                pass

            optimizer.step()

            printProgressBar(step + 1, steps, prefix="\t{}/{}: {}/{}".format(epoch, total_epoch, step + 1, steps),
                             suffix="Complete [Training] - loss_rgb: {:.4f}, loss depth: {:.4f}". \
                             format(loss_aae_rgb_data, loss_aae_depth_data), length=10)

            loss_sum_rgb += loss_aae_rgb_data / steps
            loss_sum_depth += loss_aae_depth_data / steps

            # display
            plot_n_comparison = 20
            if step < plot_n_comparison:
                images_roi = ROIAlign((128, 128), 1.0, 0)(images, roi_info_copy)
                image = images_roi[0].detach().permute(1, 2, 0).cpu().numpy()
                image_target = images_target[0].permute(1, 2, 0).cpu().numpy()
                depths_roi = ROIAlign((128, 128), 1.0, 0)(depths, roi_info_copy_depth)
                depth = depths_roi[0, 0].detach().cpu().numpy()
                depth_target = depths_target[0, 0].cpu().numpy()
                depth_recon = images_recnst[0, 3].detach().cpu().numpy()
                image_recon = images_recnst[0, :3].detach().permute(1, 2, 0).cpu().numpy()
                disp = (image, image_target, image_recon, depth, depth_target, depth_recon)
                self.plot_comparison(disp, str(step))

            if step==steps-1:
                break
            step += 1

        return loss_sum_rgb, loss_sum_depth

    def train_epoch_rgb(self, datagenerator, optimizer, steps, epoch, total_epoch, dstrgenerator=None, visualize=False):
        loss_sum_rgb = 0
        loss_sum_depth = 0
        step = 0
        optimizer.zero_grad()

        if dstrgenerator != None:
            enum_dstrgenerator = enumerate(dstrgenerator)

        for inputs in datagenerator:

            # receiving data from the renderer
            images, images_target, pose_cam, mask,\
            translation_target, scale_target, \
            affine_target, roi_center, roi_size, roi_affine, depths, depths_target = inputs

            if self.use_GPU:
                images = images.cuda()
                images_target = images_target.cuda()
                mask = mask.cuda()
                roi_affine = roi_affine.cuda()
                roi_center = roi_center.cuda().float()
                roi_size = roi_size.cuda().float()

            # warp the images according to the center and size of rois
            grids = F.affine_grid(roi_affine, images.size())
            images = F.grid_sample(images, grids)
            depths = F.grid_sample(depths, grids)
            mask = F.grid_sample(mask, grids)
            mask = 1 - mask

            # add random background and gaussian noise
            if dstrgenerator != None:
                _, images_dstr = next(enum_dstrgenerator)
                if images_dstr.size(0) != images.size(0):
                    enum_dstrgenerator = enumerate(dstrgenerator)
                    _, images_dstr = next(enum_dstrgenerator)
                if self.use_GPU:
                    images_dstr = images_dstr.cuda()
                images = images + mask * images_dstr
            noise_level = np.random.uniform(0, 0.05)
            images += torch.randn_like(images) * noise_level

            class_info = torch.ones((images.size(0), 1, 128, 128), dtype=torch.float32).cuda()

            # visualization
            if visualize:
                image_disp = images[0].permute(1, 2, 0).cpu().numpy()
                image_target_disp = images_target[0].permute(1, 2, 0).cpu().numpy()
                depth_disp = depths[0, 0].cpu().numpy()
                depth_target_disp = depths_target[0, 0].cpu().numpy()
                mask_disp = mask[0].permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy()
                plt.figure()
                plt.subplot(2, 2, 1)
                im = np.concatenate((image_disp, mask_disp), axis=1)
                im = np.clip(im * 255, 0, 255).astype(np.uint8)
                plt.imshow(im)
                plt.subplot(2, 2, 2)
                plt.imshow(np.clip(image_target_disp * 255, 0, 255).astype(np.uint8))
                plt.subplot(2, 2, 3)
                plt.imshow(depth_disp)
                plt.subplot(2, 2, 4)
                plt.imshow(depth_target_disp)
                plt.show()

            # AAE forward pass
            images_input = torch.cat((images, class_info), dim=1)
            outputs = self.AAE.forward(images_input)
            images_recnst = outputs[0]
            loss_reconstr = self.AAE.B_loss(images_recnst[:, :3, :, :], images_target.detach())
            loss = loss_reconstr

            loss_aae_rgb_data = loss_reconstr.data.cpu().item()
            loss_aae_depth_data = 0

            # AAE backward pass
            optimizer.zero_grad()
            try:
                loss.backward()
            except:
                pass

            optimizer.step()

            printProgressBar(step + 1, steps, prefix="\t{}/{}: {}/{}".format(epoch, total_epoch, step + 1, steps),
                             suffix="Complete [Training] - loss_rgb: {:.4f}, loss depth: {:.4f}". \
                             format(loss_aae_rgb_data, loss_aae_depth_data), length=10)

            loss_sum_rgb += loss_aae_rgb_data / steps
            loss_sum_depth += loss_aae_depth_data / steps

            # display
            plot_n_comparison = 20
            if step < plot_n_comparison:
                image = images[0].detach().permute(1, 2, 0).cpu().numpy()
                image_target = images_target[0].permute(1, 2, 0).cpu().numpy()
                image_recon = images_recnst[0, :3].detach().permute(1, 2, 0).cpu().numpy()
                disp = (image, image_target, image_recon)
                self.plot_comparison(disp, str(step))

            if step==steps-1:
                break
            step += 1

        return loss_sum_rgb, loss_sum_depth

    # visualization
    def plot_comparison(self, images, name, save=True):
        if len(images) == 3:
            comparison_row = np.concatenate(images, 1)
            plt.figure("compare"+name)
            plt.gcf().clear()
            plt.imshow(comparison_row)
        else:
            comparison_row = np.concatenate(images[:3], 1)
            comparison_row2 = np.concatenate(images[3:], 1)
            plt.figure("compare" + name)
            plt.gcf().clear()
            plt.subplot(2, 1, 1)
            plt.imshow(comparison_row)
            plt.subplot(2, 1, 2)
            plt.imshow(comparison_row2)

        if save:
            save_path = os.path.join(self.log_dir, "compare_{}.png".format(name))
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.1)

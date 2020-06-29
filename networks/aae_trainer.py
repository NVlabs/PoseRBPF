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

class AAE_Trainer(nn.Module):
    def __init__(self, cfg_path, model_category, config_new=None,
                 aae_capacity=1, aae_code_dim=128, ckpt_path=None):
        super(AAE_Trainer, self).__init__()

        self.cfg_path = cfg_path

        if config_new != None:
            self.cfg_all = config_new
        else:
            self.cfg_all = cfg

        self.category = model_category

        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        self.ckpt_dir = './checkpoints'

        self.embed_depth = self.cfg_all.TRAIN.DEPTH_EMBEDDING

        self.AAE = AAE(object_names=model_category,
                       capacity=aae_capacity,
                       code_dim=aae_code_dim)

        self.object_names = model_category

        self.use_GPU = (torch.cuda.device_count() > 0)

        self.code_dim = aae_code_dim

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

        if ckpt_path is not None:
            self.load_ckpt(ckpt_path=ckpt_path)

    def set_log_dir(self, model_path=None, now=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

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
        self.log_dir = os.path.join(self.ckpt_dir, "{}{:%Y%m%dT%H%M%S}_{}".format(
            self.category, now, self.cfg_all.EXP_NAME))
        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "ckpt_{}_*epoch*.pth".format(
            self.category))
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

    def plot_loss(self, loss, val_loss, title, save=True, log_dir=None):
        loss = np.array(loss)
        plt.figure(title)
        plt.gcf().clear()
        plt.plot(loss, label='train')
        plt.plot(val_loss, label='val')
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

    def train_model(self, train_dataset, val_dataset, epochs, dstr_dataset=None, save_frequency=20):
        self.AAE.encoder.train()
        self.AAE.decoder.train()

        train_set = train_dataset
        print('train set size {}'.format(len(train_set)))

        if self.log_dir == None:
            self.set_log_dir()
            if not os.path.exists(self.log_dir) and save_frequency > 0:
                print('Create folder at {}'.format(self.log_dir))
                os.makedirs(self.log_dir)
                copyfile(self.cfg_path, self.log_dir + '/config.yml')

        train_generator = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size_train,
                                                      shuffle=True, num_workers=0)
        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size_train,
                                                      shuffle=True, num_workers=0)

        if dstr_dataset != None:
            train_dstr_generator = torch.utils.data.DataLoader(dstr_dataset, batch_size=self.batch_size_train,
                                                              shuffle=True, num_workers=self.cfg_all.TRAIN.DISTRACTOR_WORKERS)
        else:
            train_dstr_generator = None

        train_steps = np.floor(len(train_set)/self.batch_size_train)
        val_steps = 30

        if self.cfg_all.TRAIN.ONLINE_RENDERING:
            train_steps = np.floor(train_steps/4)

        for epoch in np.arange(start=self.start_epoch, stop=(self.start_epoch+epochs)):
            print("Epoch {}/{}.".format(epoch, (self.start_epoch+epochs)-1))

            recon_loss_train, trans_loss_train = self.train_epoch(train_generator, self.optimizer,
                                                                  train_steps, epoch, self.start_epoch+epochs-1,
                                                                  train_dstr_generator)

            recon_loss_val, trans_loss_val = self.val_epoch(val_generator, self.optimizer,
                                                              val_steps, epoch, self.start_epoch + epochs - 1)

            self.loss_history_recon.append(recon_loss_train)
            self.val_loss_history_recon.append(recon_loss_val)

            self.plot_loss(self.loss_history_recon, self.val_loss_history_recon, 'recon loss', save=True, log_dir=self.log_dir)
            if save_frequency > 0 and epoch % save_frequency == 0:
                self.save_ckpt(epoch)

    def train_epoch(self, datagenerator, optimizer, steps, epoch, total_epoch, dstrgenerator=None):
        self.AAE.encoder.train()
        self.AAE.decoder.train()
        loss_sum_recon = 0
        loss_sum_trans = 0
        step = 0
        optimizer.zero_grad()

        if dstrgenerator != None:
            enum_dstrgenerator = enumerate(dstrgenerator)

        for inputs in datagenerator:

            # receiving data from the renderer
            depths, depths_target, affine, shift, scale, mask = inputs
            if self.use_GPU:
                depths = depths.cuda()
                depths_target = depths_target.cuda()
                affine = affine.cuda()
                shift = shift.cuda().float()
                scale = scale.cuda().float()

            # shift and scale the object in the input image
            grids = F.affine_grid(affine, depths.size())
            depths = F.grid_sample(depths, grids)
            mask = F.grid_sample(mask, grids)
            mask = 1 - mask

            if dstrgenerator != None:
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

            # AAE forward pass
            outputs = self.AAE.forward(depths)
            depth_recnst = outputs[0]
            loss_depth = self.AAE.B_loss(depth_recnst, depths_target)
            loss = loss_depth
            loss_aae_data = loss_depth.data.cpu().item()

            # AAE backward pass
            optimizer.zero_grad()
            try:
                loss.backward()
            except:
                pass

            optimizer.step()

            printProgressBar(step + 1, steps, prefix="\t{}/{}: {}/{}".format(epoch, total_epoch, step + 1, steps),
                             suffix="Complete [Training] - loss_recon: {:.4f}". \
                             format(loss_aae_data), length=10)

            # display
            plot_n_comparison = 20
            if step < plot_n_comparison:
                depth = depths[0, 0].detach().cpu().numpy()
                depth_target = depths_target[0, 0].cpu().numpy()
                depth_recon = depth_recnst[0, 0].detach().cpu().numpy()
                disp = (depth, depth_target, depth_recon)
                self.plot_comparison(disp, str(step)+'_train')

            loss_sum_recon += loss_aae_data / steps

            if step==steps-1:
                break
            step += 1

        return loss_sum_recon, loss_sum_trans

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
from ycb_renderer import *
from ycb_renderer_tensor import *

import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
import os
import time
import sys
import os.path
import numpy as np
from transforms3d.euler import quat2euler, mat2euler
from transforms3d.quaternions import *
import torch
from scipy.io import loadmat


class YCBPoseDataset_CUDA(data.Dataset):
    def __init__(self, model_dir):
        print('Processing the data:')

        self.renderer = YCBTensorRenderer(512, 512)

        models = [
            "003_cracker_box",
            "002_master_chef_can",
            "011_banana",
        ]

        obj_paths = [
            '{}/models/{}/textured_simple.obj'.format(model_dir, item) for item in models]
        texture_paths = [
            '{}/models/{}/texture_map.png'.format(model_dir, item) for item in models]
        
        self.renderer.load_objects(obj_paths, texture_paths)
        self.renderer.set_fov(48.6)
        self.renderer.set_light_pos([0, 1, 1])
        self.renderer.V = np.eye(4)
        self.renderer.colors = [[0.3, 0, 0], [0.6, 0, 0], [0.9, 0, 0]] * 100
        self.tensor1 = torch.cuda.ByteTensor(1, 512, 512, 4)
        self.tensor_seg = torch.cuda.ByteTensor(1, 512, 512, 4)
        self.tensor2 = torch.cuda.ByteTensor(1, 512, 512, 4)

    def __len__(self):
        return 10

    def load(self):
        # render tensor1 with all objects
        self.renderer.instances = [0,1,2]
        self.renderer.set_poses(np.array(
            [[0,0,-1,1,0,0,0],[0,0.5,-2,1,0,0,0],[0,1,-3,1,0,0,0]]))

        pose = self.renderer.get_poses()
        print(pose)

        self.renderer.render(
            self.tensor1,
            self.tensor_seg)

        # render tensor2 with only two objects
        self.renderer.instances = [0]
        self.renderer.set_poses(np.array(
            [[0,0.5,-2,1,0,0,0]]))

        pose = self.renderer.get_poses()
        print(pose)

        self.renderer.render(
            self.tensor2,
            self.tensor_seg)

        #show tensor1 and tensor2
        plt.subplot(1,2,1)
        plt.imshow(self.tensor1.cpu().data.numpy()[0])
        plt.subplot(1,2,2)
        plt.imshow(self.tensor2.cpu().data.numpy()[0])
        plt.show()


        img1 = self.tensor1.clone()
        img1 = img1[:, :, :, :3]
        img1 = img1.permute(0, 3, 1, 2).float() / 255.0


        img2 = self.tensor2.clone()
        img2 = img2[:, :, :, :3]
        img2 = img2.permute(0, 3, 1, 2).float() / 255.0
            
        return img1, img2


    def __getitem__(self, idx):
        return self.load()


if __name__ == "__main__":
    dataset = YCBPoseDataset_CUDA(sys.argv[1])
    print(len(dataset))
    print(dataset[0])
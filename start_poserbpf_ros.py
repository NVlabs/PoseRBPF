#!/usr/bin/env python

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np

import rospy
from ros.poserbpf_listener import ImageListener
from config.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
import glob
import copy

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Start PoseRBPF ROS Node with Multiple Object Models')
    parser.add_argument('--test_config', dest='test_cfg_file',
                        help='configuration for testing',
                        required=True, type=str)
    parser.add_argument('--pf_config_dir', dest='pf_cfg_dir',
                        help='directory for poserbpf configuration files',
                        default='./config/test/YCB/', type=str)
    parser.add_argument('--train_config_dir', dest='train_cfg_dir',
                        help='directory for AAE training configuration files',
                        default='./checkpoints/ycb_configs_roi_rgbd/', type=str)
    parser.add_argument('--ckpt_dir', dest='ckpt_dir',
                        help='directory for AAE ckpts',
                        default='./checkpoints/ycb_ckpts_roi_rgbd/', type=str)
    parser.add_argument('--codebook_dir', dest='codebook_dir',
                        help='directory for codebooks',
                        default='./checkpoints/ycb_codebooks_roi_rgbd/', type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to test on',
                        default='ycb_video', type=str)
    parser.add_argument('--dataset_dir', dest='dataset_dir',
                        help='relative dir of the dataset',
                        default='../YCB_Video_Dataset/data/',
                        type=str)
    parser.add_argument('--cad_dir', dest='cad_dir',
                        help='directory of objects CAD models',
                        default='./cad_models',
                        type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(args)

    # load the configurations
    test_cfg_file = args.test_cfg_file
    cfg_from_file(test_cfg_file)

    # load the test objects
    print('Testing with objects: ')
    print(cfg.TEST.OBJECTS)
    obj_list = cfg.TEST.OBJECTS
    with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
        obj_list_all = class_name_file.read().split('\n')

    # pf config files
    pf_config_files = sorted(glob.glob(args.pf_cfg_dir + '*yml'))
    cfg_list = []
    for obj in obj_list:
        obj_idx = obj_list_all.index(obj)
        train_config_file = args.train_cfg_dir + '{}.yml'.format(obj)
        pf_config_file = pf_config_files[obj_idx]
        cfg_from_file(train_config_file)
        cfg_from_file(pf_config_file)
        cfg_list.append(copy.deepcopy(cfg))
    # pprint.pprint(cfg_list)

    # checkpoints and codebooks
    checkpoint_list = []
    codebook_list = []
    for obj in obj_list:
        checkpoint_list.append(args.ckpt_dir+'{}_py3.pth'.format(obj))
        if not os.path.exists(args.codebook_dir):
            os.makedirs(args.codebook_dir)
        codebook_list.append(args.codebook_dir+'{}.pth'.format(obj))

    # image listener
    listener = ImageListener(obj_list, cfg_list, checkpoint_list, codebook_list,
                             modality='rgbd', cad_model_dir=args.cad_dir)

    while not rospy.is_shutdown():
        if listener.input_rgb is not None:
            listener.process_data()



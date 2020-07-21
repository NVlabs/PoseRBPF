# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
import matplotlib
# matplotlib.use('Agg')
from pose_rbpf.pose_rbpf import *
from datasets.ycb_video_dataset import *
from datasets.tless_dataset import *
from config.config import cfg, cfg_from_file
import pprint
import glob
import copy

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test PoseRBPF on YCB Video or T-LESS Datasets (RGBD)')
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
    parser.add_argument('--n_seq', dest='n_seq',
                        help='index of sequence',
                        default=1,
                        type=int)
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

    if args.dataset_name == 'ycb_video':
        print('Test on YCB Video Dataset ... ')
        object_category = 'ycb'
        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            obj_list_all = class_name_file.read().split('\n')
    elif args.dataset_name == 'tless':
        print('Test on TLESS Dataset ... ')
        object_category = 'tless'
        with open('./datasets/tless_classes.txt', 'r') as class_name_file:
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
    pprint.pprint(cfg_list)

    # checkpoints and codebooks
    checkpoint_list = []
    codebook_list = []
    for obj in obj_list:
        checkpoint_list.append(args.ckpt_dir+'{}_py3.pth'.format(obj))
        if not os.path.exists(args.codebook_dir):
            os.makedirs(args.codebook_dir)
        codebook_list.append(args.codebook_dir+'{}.pth'.format(obj))

    # setup the poserbpf
    pose_rbpf = PoseRBPF(obj_list, cfg_list, checkpoint_list, codebook_list, object_category, modality='rgbd', cad_model_dir=args.cad_dir)

    target_obj = cfg.TEST.OBJECTS[0]
    target_cfg = pose_rbpf.set_target_obj(target_obj)

    # test the system on ycb or tless datasets
    if args.dataset_name == 'ycb_video':
        test_list_file = './datasets/YCB/{}/seq{}.txt'.format(target_obj, args.n_seq)
        dataset_test = ycb_video_dataset(class_ids=[0],
                                         object_names=[target_obj],
                                         class_model_num=1,
                                         path=args.dataset_dir,
                                         list_file=test_list_file)
        pose_rbpf.run_dataset(dataset_test, args.n_seq, only_track_kf=False, kf_skip=1)
    elif args.dataset_name == 'tless':
        test_list_file = './datasets/TLess/{}/{}.txt'.format(target_obj, args.n_seq)
        dataset_test = tless_dataset(class_ids=[0],
                                     object_names=[target_obj],
                                     class_model_num=1,
                                     path=args.dataset_dir,
                                     list_file=test_list_file)
        pose_rbpf.run_dataset(dataset_test, args.n_seq, only_track_kf=False, kf_skip=1)


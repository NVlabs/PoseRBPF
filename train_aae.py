# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import argparse
from datasets.render_ycb_dataset import *
from datasets.render_tless_dataset import *
from datasets.ycb_video_dataset import *
from networks.aae_trainer import *
from ycb_render.ycb_renderer import *
from ycb_render.tless_renderer_tensor import *
from config.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
import pprint

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Pose Estimation Network on ycb Dataset')
    parser.add_argument('--cfg_dir', dest='cfg_dir',
                        help='directory for configuration files',
                        required=True, type=str)
    parser.add_argument('--obj', dest='obj',
                        help='object instance',
                        required=True, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train',
                        default=100, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--save', dest='save_frequency',
                        help='checkpoint saving frequency',
                        default=50, type=int)
    parser.add_argument('--obj_ctg', dest='obj_ctg',
                        help='object category: ycb or tless',
                        default='ycb', type=str)
    parser.add_argument('--dis_dir', dest='dis_dir',
                        help='relative dir of the distration set',
                        default='../coco/val2017',
                        type=str)
    parser.add_argument('--modality', dest='modality',
                        help='modality: rgb or rgbd',
                        default='rgbd', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(args)

    cfg_file = '{}{}.yml'.format(args.cfg_dir, args.obj)
    cfg_from_file(cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    # device
    print('GPU device {:d}'.format(args.gpu_id))

    cfg.MODE = 'TRAIN'
    print(cfg.TRAIN.OBJECTS)
    print(cfg.TRAIN.RENDER_SZ)
    print(cfg.TRAIN.INPUT_IM_SIZE)

    # set up render
    model_path = './cad_models'
    models = cfg.TRAIN.OBJECTS[:]
    if args.obj_ctg == 'ycb':
        renderer = YCBRenderer(cfg.TRAIN.RENDER_SZ, cfg.TRAIN.RENDER_SZ, cfg.GPU_ID)
        if cfg.TRAIN.USE_OCCLUSION:
            with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
                class_names_all = class_name_file.read().split('\n')
                for class_name in class_names_all:
                    if class_name not in models:
                        models.append(class_name)

        obj_paths = ['{}/ycb_models/{}/textured_simple.obj'.format(model_path, item) for item in models]
        texture_paths = ['{}/ycb_models/{}/texture_map.png'.format(model_path, item) for item in models]
        renderer.load_objects(obj_paths, texture_paths)
        renderer.set_projection_matrix(cfg.TRAIN.RENDER_SZ, cfg.TRAIN.RENDER_SZ, fu=cfg.TRAIN.FU, 
            fv=cfg.TRAIN.FU, u0=cfg.TRAIN.U0, v0=cfg.TRAIN.V0, znear=0.01, zfar=10)
    elif args.obj_ctg == 'tless':
        renderer = TLessTensorRenderer(cfg.TRAIN.RENDER_SZ, cfg.TRAIN.RENDER_SZ)
        if cfg.TRAIN.USE_OCCLUSION:
            with open('./datasets/tless_classes.txt', 'r') as class_name_file:
                class_names_all = class_name_file.read().split('\n')
                for class_name in class_names_all:
                    if class_name not in models:
                        models.append(class_name)

        class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                            (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64),
                            (155, 0, 0), (0, 155, 0), (0, 0, 155), (155, 155, 0), (155, 0, 155), (0, 155, 155),
                            (200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0),
                            (200, 0, 200), (0, 200, 200)
                            ]
        obj_paths = ['{}/tless_models/{}.ply'.format(model_path, item) for item in models]
        texture_paths = ['' for cls in models]
        renderer.load_objects(obj_paths, texture_paths, class_colors_all)
        renderer.set_projection_matrix(cfg.TRAIN.RENDER_SZ, cfg.TRAIN.RENDER_SZ, cfg.TRAIN.FU, cfg.TRAIN.FU,
                                       cfg.TRAIN.RENDER_SZ/2.0, cfg.TRAIN.RENDER_SZ/2.0, 0.01, 10)
    renderer.set_camera_default()
    renderer.set_light_pos([0, 0, 0])

    # dataset
    if args.obj_ctg == 'ycb':
        dataset_train = ycb_multi_render_dataset(model_path, cfg.TRAIN.OBJECTS, renderer,
                                                 render_size=cfg.TRAIN.RENDER_SZ,
                                                 output_size=cfg.TRAIN.INPUT_IM_SIZE)
    elif args.obj_ctg == 'tless':
        dataset_train = tless_multi_render_dataset(model_path, cfg.TRAIN.OBJECTS, renderer,
                                                 render_size=cfg.TRAIN.RENDER_SZ,
                                                 output_size=cfg.TRAIN.INPUT_IM_SIZE)

    dataset_dis = DistractorDataset(args.dis_dir, cfg.TRAIN.CHM_RAND_LEVEL,
                                    size_crop=(cfg.TRAIN.INPUT_IM_SIZE[1],
                                               cfg.TRAIN.INPUT_IM_SIZE[0]))

    trainer = aae_trainer(cfg_path=cfg_file,
                          object_names=cfg.TRAIN.OBJECTS,
                          modality=args.modality,
                          aae_capacity=cfg.CAPACITY,
                          aae_code_dim=cfg.CODE_DIM,
                          ckpt_path=args.pretrained,
                          obj_ctg=args.obj_ctg)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        trainer.AAE.encoder = nn.DataParallel(trainer.AAE.encoder)
        trainer.AAE.decoder = nn.DataParallel(trainer.AAE.decoder)
        if args.modality == 'rgbd':
            trainer.AAE.depth_decoder = nn.DataParallel(trainer.AAE.depth_decoder)

    trainer.train_model(dataset_train,
                          epochs=args.epochs,
                          dstr_dataset=dataset_dis,
                          save_frequency=args.save_frequency,
                          )

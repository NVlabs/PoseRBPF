from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import argparse
from datasets.render_ycb_dataset import *
from datasets.render_tless_dataset import *
from datasets.ycb_video_dataset import *
from networks.aae_trainer import *
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

    if args.obj_ctg == 'ycb':
        model_path = './cad_models'
        dataset_train = ycb_multi_render_dataset(model_path, cfg.TRAIN.OBJECTS,
                                                 render_size=cfg.TRAIN.RENDER_SZ,
                                                 output_size=cfg.TRAIN.INPUT_IM_SIZE)
    elif args.obj_ctg == 'tless':
        model_path = './cad_models'
        dataset_train = tless_multi_render_dataset(model_path, cfg.TRAIN.OBJECTS,
                                                 render_size=cfg.TRAIN.RENDER_SZ,
                                                 output_size=cfg.TRAIN.INPUT_IM_SIZE)

    dataset_dis = DistractorDataset(args.dis_dir, cfg.TRAIN.CHM_RAND_LEVEL,
                                    size_crop=(cfg.TRAIN.INPUT_IM_SIZE[1],
                                               cfg.TRAIN.INPUT_IM_SIZE[0]))

    trainer = aae_trainer(cfg_path=cfg_file,
                          object_names=cfg.TRAIN.OBJECTS,
                          modality='rgbd',
                          aae_capacity=cfg.CAPACITY,
                          aae_code_dim=cfg.CODE_DIM,
                          ckpt_path=args.pretrained,
                          obj_ctg=args.obj_ctg)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        trainer.AAE.encoder = nn.DataParallel(trainer.AAE.encoder)
        trainer.AAE.decoder = nn.DataParallel(trainer.AAE.decoder)
        trainer.AAE.depth_decoder = nn.DataParallel(trainer.AAE.depth_decoder)

    trainer.train_model(dataset_train,
                          epochs=args.epochs,
                          dstr_dataset=dataset_dis,
                          save_frequency=args.save_frequency,
                          )

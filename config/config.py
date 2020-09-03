# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""PoseRBPF Configuration File
"""

import os
import os.path as osp
import numpy as np
import math
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# for pose estimation network
__C.CODE_DIM = 128
__C.CAPACITY = 1
__C.FULLY_CONV = False

__C.MODE = 'TRAIN'

__C.MODE = 'TEST'

#
# Training options
#

__C.TRAIN = edict()

__C.TRAIN.DISPLAY = 20

__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.VAL_BATCH_SIZE = 64

# Added for pose estimation networks:
__C.TRAIN.TRAIN_AAE = True
__C.TRAIN.TRAIN_TRANSNET = False
__C.TRAIN.OBJECTS = []
__C.TRAIN.BOOTSTRAP_CONST = 2000
__C.TRAIN.USE_TRIPLET_LOSS = False
__C.TRAIN.TRANS_LOSS_WT = 1.0
__C.TRAIN.TRIPLET_LOSS_WT = 10.0
__C.TRAIN.TRIPLET_MARGIN = 0.01
__C.TRAIN.WORKERS = 8
__C.TRAIN.DISTRACTOR_WORKERS = 8
__C.TRAIN.DYNAMIC_TRIPLET = False
__C.TRAIN.CHM_RAND_LEVEL = [0.1, 0.3, 0.3]
__C.TRAIN.TRANS_SEP = False
__C.TRAIN.ONLINE_RENDERING = True
__C.TRAIN.DEEPER_TRANSNET = False
__C.TRAIN.TRANSNET_CAPACITY = [128, 256, 256]
__C.TRAIN.DEPTH_LOSS_WT = 5.0

# Use
__C.TRAIN.TRANS_ENCODER = False
__C.TRAIN.RoI_ACC = False

# Noise
__C.TRAIN.SHIFT_MIN = -20.0
__C.TRAIN.SHIFT_MAX = 20.0
__C.TRAIN.SCALE_MIN = 0.7
__C.TRAIN.SCALE_MAX = 1.3
__C.TRAIN.TRANS_SHIFT_MIN = -50.0
__C.TRAIN.TRANS_SHIFT_MAX = 50.0
__C.TRAIN.TRANS_SCALE_MIN = 0.5
__C.TRAIN.TRANS_SCALE_MAX = 1.5
__C.TRAIN.LIGHT_INT_MIN = 1.0
__C.TRAIN.LIGHT_INT_MAX = 2.5
__C.TRAIN.LIGHT_R_MIN = 0.3
__C.TRAIN.LIGHT_R_MAX = 0.5
__C.TRAIN.LIGHT_COLOR_VAR = 0.4
__C.TRAIN.TRANS_ONLY = False
__C.TRAIN.ANGLE_PERTURB_DEG = 20.0
__C.TRAIN.INVALID_SCALE_PROB = 0.15
__C.TRAIN.INVALID_SCALE_LOW = 0.5
__C.TRAIN.INVALID_SCALE_UP = 1.5

# Depth Embedding
__C.TRAIN.DEPTH_EMBEDDING = False
__C.TRAIN.DEPTH_UB = 0.2
__C.TRAIN.DEPTH_LB = -0.2
__C.TRAIN.DEPTH_MARGIN = 0.1  # margin for depth
__C.TRAIN.NORMALIZE_DEPTH = True
__C.TRAIN.DEPTH_FUSION = False

# Occlusion
__C.TRAIN.USE_OCCLUSION = True
__C.TRAIN.N_OCCLUDERS = 3
__C.TRAIN.OC_XY_RANGE = [-0.075, 0.075]
__C.TRAIN.OC_Z_RANGE = [0.10, 1.00]
# __C.TRAIN.OC_XY_RANGE = [-0.2, 0.2]
# __C.TRAIN.OC_Z_RANGE = [0.30, 0.60]
__C.TRAIN.OC_PROB = 0.3

# Truncation
__C.TRAIN.TRUNC_PROB = 0.2
__C.TRAIN.TRUNC_X_MAX = 0.6
__C.TRAIN.TRUNC_Y_MAX = 0.6

# Render distance
__C.TRAIN.RENDER_DIST = [2.5]
__C.TRAIN.FU = 1066.778
__C.TRAIN.FV = 1056.487
__C.TRAIN.U0 = 312.987 / 640 * 128
__C.TRAIN.V0 = 241.311 / 480 * 128
__C.TRAIN.W = 128.0
__C.TRAIN.H = 128.0
__C.TRAIN.TARGET_INTENSITY = 1.2
__C.TRAIN.TARGET_LIGHT1_POS = [0, 0, 0]
__C.TRAIN.TARGET_LIGHT2_POS = [0, 0, 0]
__C.TRAIN.TARGET_LIGHT3_POS = [0, 0, 0]

# Self-supervision
__C.TRAIN.SV_RENDER_DIST = [1.0]
__C.TRAIN.SV_FU = 500.0
__C.TRAIN.SV_FV = 500.0
__C.TRAIN.SV_U0 = 64.0
__C.TRAIN.SV_V0 = 64.0
__C.TRAIN.SV_TARGET_INTENSITY = 1.0

# RoI Acceleration
__C.TRAIN.ROI_CENTER_RANGE = [100.0, 100.0, 156.0, 156.0]
__C.TRAIN.ROI_SIZE_RANGE = [116.4, 140.8]
# __C.TRAIN.ROI_CENTER_RANGE = [128, 128, 128, 128]
# __C.TRAIN.ROI_SIZE_RANGE = [128, 128]
__C.TRAIN.INPUT_IM_SIZE = [128, 128]
__C.TRAIN.RENDER_SZ = 128

# Photo realistic data
__C.TRAIN.USE_PHOTO_REALISTIC_DATA = False
__C.TRAIN.PHOTO_REALISTIC_DATA_PATH = '../YCB_Video_DPF/'

#
# Testing options
#

__C.TEST = edict()
# Default GPU device id
__C.GPU_ID = 0
__C.TEST.OBJECTS = []

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'
__C.EXP_NAME = 'default'

# Deep particle filter setting
__C.PF = edict()
__C.PF.USE_TRANSNET = False
__C.PF.USE_DEPTH = False
__C.PF.RENDER_FULL = False
__C.PF.INIT_GLOBALLY = False
__C.PF.DEPTH_DELTA = 0.03
__C.PF.DEPTH_TAU = 0.03
__C.PF.DEPTH_STD = 0.1
__C.PF.UV_NOISE = 5.0
__C.PF.Z_NOISE = 0.05
__C.PF.UV_NOISE_PRIOR = 5.0
__C.PF.Z_NOISE_PRIOR = 0.05
__C.PF.ROT_NOISE = 0.05
__C.PF.INIT_UV_NOISE = 30
__C.PF.INIT_Z_RANGE = [0.2, 3.0]
__C.PF.INIT_ROT_WT_VAR = 0.05
__C.PF.TRANS_WT_VAR = 0.05
__C.PF.ROT_WT_VAR = 0.05
__C.PF.N_INIT = 500
__C.PF.N_PROCESS = 50
__C.PF.FU = 1066.778
__C.PF.FV = 1056.487
__C.PF.U0 = 312.987
__C.PF.V0 = 241.311
__C.PF.W = 640.0
__C.PF.H = 480.0
__C.PF.VISUALIZE = True
__C.PF.SAVE_DIR = './results/tmp/'
__C.PF.TRACK_OBJ = ' '
__C.PF.WT_RESHAPE_VAR = 0.035
__C.PF.N_E_ROT = 100
__C.PF.MOTION_T_FACTOR = 1.0
__C.PF.MOTION_R_FACTOR = 0.5
__C.PF.ROT_RANGE = 0.2
__C.PF.ROT_GAUSSIAN_KERNEL_SZ = 5
__C.PF.ROT_GAUSSIAN_KERNEL_STD = 1
__C.PF.ROT_VAR = 0.05
__C.PF.SIM_RGB_THRES = 0.75
__C.PF.SIM_DEPTH_THRES = 0.7
__C.PF.FUSION_WT_RGB = 0.65

def get_output_dir(imdb, net):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return osp.join(path, net)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def write_selected_class_file(filename, index):
    # read file
    with open(filename) as f:
        lines = [x for x in f.readlines()]
    lines_selected = [lines[i] for i in index]

    # write new file
    filename_new = filename + '.selected'
    f = open(filename_new, 'w')
    for i in range(len(lines_selected)):
        f.write(lines_selected[i])
    f.close()
    return filename_new

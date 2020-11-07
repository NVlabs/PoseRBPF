# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
import matplotlib
# matplotlib.use('Agg')
from pose_rbpf.pose_rbpf import *
from datasets.ycb_video_dataset import *
from datasets.dex_ycb_dataset import *
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
    parser.add_argument('--demo', dest='demo',
                        help='run as demo mode',
                        default=False,
                        type=bool)
    args = parser.parse_args()
    return args


def _get_bb3D(extent):
    bb = np.zeros((3, 8), dtype=np.float32)
    xHalf = extent[0] * 0.5
    yHalf = extent[1] * 0.5
    zHalf = extent[2] * 0.5
    bb[:, 0] = [xHalf, yHalf, zHalf]
    bb[:, 1] = [-xHalf, yHalf, zHalf]
    bb[:, 2] = [xHalf, -yHalf, zHalf]
    bb[:, 3] = [-xHalf, -yHalf, zHalf]
    bb[:, 4] = [xHalf, yHalf, -zHalf]
    bb[:, 5] = [-xHalf, yHalf, -zHalf]
    bb[:, 6] = [xHalf, -yHalf, -zHalf]
    bb[:, 7] = [-xHalf, -yHalf, -zHalf]    
    return bb


def _vis_minibatch(sample, classes, class_colors):

    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    image = sample['image_color'].numpy()
    depth = sample['image_depth'].numpy()
    label = sample['label'].numpy()
    gt_poses = sample['gt_poses'].numpy()
    intrinsics = sample['intrinsic_matrix'].numpy()
    gt_boxes = sample['gt_boxes'].numpy()
    extents = sample['extents'][0, :, :].numpy()
    points = sample['points'][0, :, :].numpy()
    poses_result = sample['poses_result'].numpy()
    rois_result = sample['rois_result'].numpy()
    video_ids = sample['video_id']
    image_ids = sample['image_id']

    m = 2
    n = 3
    for i in range(image.shape[0]):
        fig = plt.figure()
        start = 1

        video_id = video_ids[i]
        image_id = image_ids[i]
        print(video_id, image_id)

        # show image
        im = image[i, :, :, :].copy() * 255.0
        im = im[:, :, (2, 1, 0)]
        im = np.clip(im, 0, 255)
        im = im.astype(np.uint8)
        ax = fig.add_subplot(m, n, 1)
        plt.imshow(im)
        ax.set_title('color: %s_%s' % (video_id, image_id))
        start += 1

        # show depth
        im_depth = depth[i].copy()
        ax = fig.add_subplot(m, n, start)
        plt.imshow(im_depth)
        ax.set_title('depth')
        start += 1

        # project the 3D box to image
        intrinsic_matrix = intrinsics[i]
        pose_blob = gt_poses[i]
        boxes = gt_boxes[i]
        for j in range(pose_blob.shape[0]):
            class_id = int(boxes[j, -1])
            bb3d = _get_bb3D(extents[class_id, :])
            x3d = np.ones((4, 8), dtype=np.float32)
            x3d[0:3, :] = bb3d
            
            # projection
            RT = pose_blob[:, :, j]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            x1 = np.min(x2d[0, :])
            x2 = np.max(x2d[0, :])
            y1 = np.min(x2d[1, :])
            y2 = np.max(x2d[1, :])
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

        # show gt boxes
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im)
        ax.set_title('gt boxes')
        for j in range(boxes.shape[0]):
            x1 = boxes[j, 0]
            y1 = boxes[j, 1]
            x2 = boxes[j, 2]
            y2 = boxes[j, 3]
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=3, clip_on=False))

        # show label
        im_label = label[i]
        ax = fig.add_subplot(m, n, start)
        start += 1
        plt.imshow(im_label)
        ax.set_title('label')

        # show posecnn pose
        ax = fig.add_subplot(m, n, start)
        start += 1
        ax.set_title('posecnn poses')
        plt.imshow(im)
        rois = rois_result[i]
        poses = poses_result[i]
        for j in range(rois.shape[0]):
            cls = int(rois[j, 1])
            print('%s: detection score %s' % (classes[cls], rois[j, -1]))

            # extract 3D points
            x3d = np.ones((4, points.shape[1]), dtype=np.float32)
            x3d[0, :] = points[cls,:,0]
            x3d[1, :] = points[cls,:,1]
            x3d[2, :] = points[cls,:,2]

            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(poses[j, 2:6])
            RT[:, 3] = poses[j, 6:]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
            plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(class_colors[cls], 255.0), alpha=0.1)

        # show posecnn detection
        ax = fig.add_subplot(m, n, start)
        start += 1
        ax.set_title('posecnn detections')
        plt.imshow(im)
        for j in range(rois.shape[0]):
            cls = int(rois[j, 1])
            x1 = rois[j, 2]
            y1 = rois[j, 3]
            x2 = rois[j, 4]
            y2 = rois[j, 5]
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=np.divide(class_colors[cls], 255.0), linewidth=3, clip_on=False))

        plt.show()


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
    elif 'dex_ycb' in args.dataset_name:
        print('Test on DEX-YCB Dataset ... ')
        object_category = 'ycb'
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
    print('%d cfg files' % (len(cfg_list)))

    # checkpoints and codebooks
    checkpoint_list = []
    codebook_list = []
    for obj in obj_list:
        checkpoint_list.append(args.ckpt_dir+'{}_py3.pth'.format(obj))
        if not os.path.exists(args.codebook_dir):
            os.makedirs(args.codebook_dir)
        codebook_list.append(args.codebook_dir+'{}.pth'.format(obj))
    print('checkpoint files:', checkpoint_list)
    print('codebook files:', codebook_list)

    # dataset
    if 'dex_ycb' in args.dataset_name:
        names = args.dataset_name.split('_')
        setup = names[-2]
        split = names[-1]
        print(setup, split)
        dataset_test = dex_ycb_dataset(setup, split, obj_list)
    dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

    # loop the dataset
    for i, sample in enumerate(dataloader):

        _vis_minibatch(sample, obj_list, dataset_test._class_colors)

    # setup the poserbpf
    pose_rbpf = PoseRBPF(obj_list, cfg_list, checkpoint_list, codebook_list, object_category, modality='rgbd', cad_model_dir=args.cad_dir)

    target_obj = cfg.TEST.OBJECTS[0]
    pose_rbpf.add_object_instance(target_obj)
    target_cfg = pose_rbpf.set_target_obj(0)


    pose_rbpf.run_dataset(dataset_test, args.n_seq, only_track_kf=False, kf_skip=1, demo=args.demo)

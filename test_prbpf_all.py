# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
import matplotlib
import pprint
import glob
import copy
import posecnn_cuda

from pose_rbpf.pose_rbpf import *
from pose_rbpf.sdf_multiple_optimizer import sdf_multiple_optimizer
from datasets.ycb_video_dataset import *
from datasets.dex_ycb_dataset import *
from datasets.tless_dataset import *
from config.config import cfg, cfg_from_file
from utils.cython_bbox import bbox_overlaps

posecnn_classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', \
                   '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', \
                   '011_banana', '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', \
                   '036_wood_block', '037_scissors', '040_large_marker', '052_extra_large_clamp', '061_foam_brick')

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
    parser.add_argument('--modality', dest='modality',
                        help='modality',
                        default='rgbd', type=str)
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
    parser.add_argument('--depth_refinement', dest='refine',
                        help='sdf refinement',
                        action='store_true')
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
    labels_result = sample['labels_result'].numpy()
    video_ids = sample['video_id']
    image_ids = sample['image_id']

    m = 3
    n = 3
    for i in range(image.shape[0]):
        fig = plt.figure()
        start = 1

        video_id = video_ids[i]
        image_id = image_ids[i]
        print(video_id, image_id)

        # show image
        im = image[i, :, :, :].copy() * 255.0
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
            print('%d %s: detection score %s' % (cls, classes[cls], rois[j, -1]))

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

        # show predicted label
        im_label = labels_result[i, :, :]
        if im_label.shape[0] > 0:
            ax = fig.add_subplot(m, n, start)
            start += 1
            plt.imshow(im_label)
            ax.set_title('posecnn labels')

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
        if args.modality == 'rgbd':
            checkpoint_list.append(args.ckpt_dir+'{}_py3.pth'.format(obj))
        else:
            checkpoint_list.append(args.ckpt_dir+'{}.pth'.format(obj))
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

    # setup the poserbpf
    pose_rbpf = PoseRBPF(obj_list, cfg_list, checkpoint_list, codebook_list, 
        object_category, modality=args.modality, cad_model_dir=args.cad_dir, gpu_id=args.gpu_id)

    if args.refine:
        print('loading SDFs')
        sdf_files = []
        for cls in obj_list:
            sdf_file = '{}/ycb_models/{}/textured_simple_low_res.pth'.format(args.cad_dir, cls)
            sdf_files.append(sdf_file)
        reg_trans = 1000.0
        reg_rot = 10.0
        sdf_optimizer = sdf_multiple_optimizer(obj_list, sdf_files, reg_trans, reg_rot)

    # output directory
    output_dir = os.path.join('output', args.dataset_name, args.modality)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #'''
    # loop the dataset
    visualize = False
    video_id = ''
    epoch_size = len(dataloader)
    for k, sample in enumerate(dataloader):

        # _vis_minibatch(sample, obj_list, dataset_test._class_colors)

        # if 'is_testing' in sample and sample['is_testing'] == 0:
        #    continue

        # prepare data
        image_input = sample['image_color'][0]
        if args.modality == 'rgb':
            image_depth = None
        else:
            image_depth = sample['image_depth'][0]
            im_depth = image_depth.cuda().float()
            image_depth = image_depth.unsqueeze(2)
        image_label = sample['labels_result'][0].cuda()
        if image_label.shape[0] == 0:
            image_label = None
        width = image_input.shape[1]
        height = image_input.shape[0]
        intrinsics = sample['intrinsic_matrix'][0].numpy()
        image_id = sample['image_id'][0]

        # start a new video
        if video_id != sample['video_id'][0]:
            pose_rbpf.reset_poserbpf()
            pose_rbpf.set_intrinsics(intrinsics, width, height)
            video_id = sample['video_id'][0]
            print('start video %s' % (video_id))
            print(intrinsics)

        print('video %s, frame %s' % (video_id, image_id))

        # detection from posecnn
        rois = sample['rois_result'][0].numpy()

        # collect rois from rbpfs
        rois_rbpf = np.zeros((0, 6), dtype=np.float32)
        index_rbpf = []
        for i in range(len(pose_rbpf.instance_list)):
            if pose_rbpf.rbpf_ok_list[i]:
                roi = pose_rbpf.rbpf_list[i].roi
                rois_rbpf = np.concatenate((rois_rbpf, roi), axis=0)
                index_rbpf.append(i)
                pose_rbpf.rbpf_list[i].roi_assign = None

        # data association based on bounding box overlap
        num_rois = rois.shape[0]
        num_rbpfs = rois_rbpf.shape[0]
        assigned_rois = np.zeros((num_rois, ), dtype=np.int32)
        if num_rbpfs > 0 and num_rois > 0:
            # overlaps: (rois x gt_boxes) (batch_id, x1, y1, x2, y2)
            overlaps = bbox_overlaps(np.ascontiguousarray(rois_rbpf[:, (1, 2, 3, 4, 5)], dtype=np.float),
                np.ascontiguousarray(rois[:, (1, 2, 3, 4, 5)], dtype=np.float))

            # assign rois to rbpfs
            assignment = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)
            unassigned = []
            for i in range(num_rbpfs):
                if max_overlaps[i] > 0.2:
                    pose_rbpf.rbpf_list[index_rbpf[i]].roi_assign = rois[assignment[i]]
                    assigned_rois[assignment[i]] = 1
                else:
                    unassigned.append(i)

            # check if there are un-assigned rois
            index = np.where(assigned_rois == 0)[0]

            # if there is un-assigned rbpfs
            if len(unassigned) > 0 and len(index) > 0:
                for i in range(len(unassigned)):
                    for j in range(len(index)):
                        if assigned_rois[index[j]] == 0 and pose_rbpf.rbpf_list[index_rbpf[unassigned[i]]].roi[0, 1] == rois[index[j], 1]:
                            pose_rbpf.rbpf_list[index_rbpf[unassigned[i]]].roi_assign = rois[index[j]]
                            assigned_rois[index[j]] = 1
        elif num_rbpfs == 0 and num_rois == 0:
            continue

        # filter tracked objects
        for i in range(len(pose_rbpf.instance_list)):
            if pose_rbpf.rbpf_ok_list[i]:
                roi = pose_rbpf.rbpf_list[i].roi_assign
                Tco, max_sim = pose_rbpf.pose_estimation_single(i, roi, image_input, image_depth, visualize=visualize)

        # initialize new object
        for i in range(num_rois):
            if assigned_rois[i]:
                continue
            roi = rois[i]
            obj_idx = int(roi[1])
            target_obj = pose_rbpf.obj_list[obj_idx]
            add_new_instance = True

            # associate the same object, assume one instance per object
            for j in range(len(pose_rbpf.instance_list)):
                if pose_rbpf.instance_list[j] == target_obj and pose_rbpf.rbpf_ok_list[j] == False:
                    print('initialize previous object: %s' % (target_obj))
                    add_new_instance = False
                    Tco, max_sim = pose_rbpf.pose_estimation_single(j, roi, image_input,
                                                                    image_depth, visualize=visualize)
            if add_new_instance:
                print('initialize new object: %s' % (target_obj))
                pose_rbpf.add_object_instance(target_obj)
                Tco, max_sim = pose_rbpf.pose_estimation_single(len(pose_rbpf.instance_list)-1, roi, image_input,
                                                                image_depth, visualize=visualize)

        # save result
        if not visualize:
            filename = os.path.join(output_dir, video_id + '_' + image_id + '.mat')
            pose_rbpf.save_results_mat(filename)

        # SDF refinement for multiple objects
        if args.refine and image_label is not None:
            # backproject depth
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            px = intrinsics[0, 2]
            py = intrinsics[1, 2]
            im_pcloud = posecnn_cuda.backproject_forward(fx, fy, px, py, im_depth)[0]

            index_sdf = []
            for i in range(len(pose_rbpf.instance_list)):
                if pose_rbpf.rbpf_ok_list[i]:
                    index_sdf.append(i)
            if len(index_sdf) > 0:
                pose_rbpf.pose_refine_multiple(sdf_optimizer, posecnn_classes, index_sdf, im_depth, 
                    im_pcloud, image_label, steps=50)

        print('=========[%d/%d]==========' % (k, epoch_size))
    #'''

    filename = os.path.join(output_dir, 'results_poserbpf.mat')
    if os.path.exists(filename):
        os.remove(filename)

    # evaluation
    dataset_test.evaluation(output_dir, args.modality)

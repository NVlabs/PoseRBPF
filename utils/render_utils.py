# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import torch
import time
import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from config.config_posecnn import cfg
from transforms3d.quaternions import quat2mat

def render_one(dataset, cls, pose):

    cls_id = cfg.TEST.CLASSES.index(cls)
    intrinsic_matrix = dataset._intrinsic_matrix
    height = dataset._height
    width = dataset._width

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 10.0
    znear = 0.25

    image_tensor = torch.cuda.FloatTensor(height, width, 4)
    seg_tensor = torch.cuda.FloatTensor(height, width, 4)

    # set renderer
    cfg.renderer.set_light_pos([0, 0, 0])
    cfg.renderer.set_light_color([1, 1, 1])
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    # render images
    cls_indexes = []
    poses_all = []
    cls_indexes.append(cls_id)
    poses_all.append(pose)

    # rendering
    cfg.renderer.set_poses(poses_all)
    cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
    image_tensor = image_tensor.flip(0)
    seg_tensor = seg_tensor.flip(0)
    seg = seg_tensor[:,:,2] + 256*seg_tensor[:,:,1] + 256*256*seg_tensor[:,:,0]
    image_tensor[seg == 0] = 0.5

    im_render = image_tensor.cpu().numpy()
    im_render = np.clip(im_render, 0, 1)
    im_render = im_render[:, :, :3] * 255
    im_render = im_render.astype(np.uint8)

    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(im_render)
    plt.show()
    '''

    mask = seg.cpu().numpy()
    y, x = np.where(mask > 0)
    x1 = np.min(x)
    x2 = np.max(x)
    y1 = np.min(y)
    y2 = np.max(y)
    s = max((y2 - y1), (x2 - x1))

    return im_render, s


def render_images(dataset, poses):

    intrinsic_matrix = dataset._intrinsic_matrix
    height = dataset._height
    width = dataset._width

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 10.0
    znear = 0.25
    num = poses.shape[0]

    im_output = np.zeros((num, height, width, 3), dtype=np.uint8)
    image_tensor = torch.cuda.FloatTensor(height, width, 4)
    seg_tensor = torch.cuda.FloatTensor(height, width, 4)

    # set renderer
    cfg.renderer.set_light_pos([0, 0, 0])
    cfg.renderer.set_light_color([1, 1, 1])
    cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    # render images
    for i in range(num):

        cls_indexes = []
        poses_all = []

        cls_index = cfg.TRAIN.CLASSES[0] - 1
        cls_indexes.append(cls_index)
        poses_all.append(poses[i,:])

        # rendering
        cfg.renderer.set_poses(poses_all)
        cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)

        im_render = image_tensor.cpu().numpy()
        im_render = np.clip(im_render, 0, 1)
        im_render = im_render[:, :, :3] * 255
        im_render = im_render.astype(np.uint8)
        im_output[i] = im_render

    return im_output


# only render rois and segmentation masks
def render_image_detection(dataset, im, rois, labels):
    # label image
    label_image = dataset.labels_to_image(labels)
    im_label = im[:, :, (2, 1, 0)].copy()
    I = np.where(labels != 0)
    im_label[I[0], I[1], :] = 0.5 * label_image[I[0], I[1], :] + 0.5 * im_label[I[0], I[1], :]

    num = rois.shape[0]
    classes = dataset._classes
    class_colors = dataset._class_colors

    for i in range(num):
        cls = int(rois[i, 1])
        if rois[i, -1] > cfg.TEST.DET_THRESHOLD:
            print(dataset._classes[cls])
            # draw roi
            x1 = rois[i, 2]
            y1 = rois[i, 3]
            x2 = rois[i, 4]
            y2 = rois[i, 5]
            cv2.rectangle(im_label, (x1, y1), (x2, y2), class_colors[cls], 2)

    return im_label


def render_image(dataset, im, rois, poses, poses_refine, labels, cls_render_ids=None):

    # label image
    label_image = dataset.labels_to_image(labels)
    im_label = im[:, :, (2, 1, 0)].copy()
    I = np.where(labels != 0)
    im_label[I[0], I[1], :] = 0.5 * label_image[I[0], I[1], :] + 0.5 * im_label[I[0], I[1], :]

    num = poses.shape[0]
    classes = dataset._classes_test
    class_colors = dataset._class_colors_test

    cls_indexes = []
    poses_all = []
    poses_refine_all = []
    for i in range(num):
        if cls_render_ids is not None and len(cls_render_ids) == num:
            cls_index = cls_render_ids[i]
        else:
            if cfg.MODE == 'TEST':
                cls_index = int(rois[i, 1]) - 1
            else:
                cls_index = cfg.TRAIN.CLASSES[int(rois[i, 1])] - 1

        if cls_index < 0:
            continue

        cls_indexes.append(cls_index)
        qt = np.zeros((7, ), dtype=np.float32)
        qt[:3] = poses[i, 4:7]
        qt[3:] = poses[i, :4]
        poses_all.append(qt.copy())

        if cfg.TEST.POSE_REFINE and poses_refine is not None:
            qt[:3] = poses_refine[i, 4:7]
            qt[3:] = poses_refine[i, :4]
            poses_refine_all.append(qt.copy())

        cls = int(rois[i, 1])
        print(classes[cls], rois[i, -1], cls_index)
        if cls > 0 and rois[i, -1] > cfg.TEST.DET_THRESHOLD:
            # draw roi
            x1 = rois[i, 2]
            y1 = rois[i, 3]
            x2 = rois[i, 4]
            y2 = rois[i, 5]
            cv2.rectangle(im_label, (x1, y1), (x2, y2), class_colors[cls], 2)

    # rendering
    if len(cls_indexes) > 0:

        height = im.shape[0]
        width = im.shape[1]
        intrinsic_matrix = dataset._intrinsic_matrix
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 10.0
        znear = 0.01
        image_tensor = torch.cuda.FloatTensor(height, width, 4)
        seg_tensor = torch.cuda.FloatTensor(height, width, 4)

        # set renderer
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        cfg.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        # pose
        cfg.renderer.set_poses(poses_all)
        frame = cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        im_render = image_tensor.cpu().numpy()
        im_render = np.clip(im_render, 0, 1)
        im_render = im_render[:, :, :3] * 255
        im_render = im_render.astype(np.uint8)
        im_output = 0.8 * im[:,:,(2, 1, 0)].astype(np.float32) + 2.0 * im_render.astype(np.float32)
        im_output = np.clip(im_output, 0, 255)

        # pose refine
        if cfg.TEST.POSE_REFINE and poses_refine is not None:
             cfg.renderer.set_poses(poses_refine_all)
             frame = cfg.renderer.render(cls_indexes, image_tensor, seg_tensor)
             image_tensor = image_tensor.flip(0)
             im_render = image_tensor.cpu().numpy()
             im_render = np.clip(im_render, 0, 1)
             im_render = im_render[:, :, :3] * 255
             im_render = im_render.astype(np.uint8)
             im_output_refine = 0.8 * im[:,:,(2, 1, 0)].astype(np.float32) + 2.0 * im_render.astype(np.float32)
             im_output_refine = np.clip(im_output_refine, 0, 255)
             im_output_refine = im_output_refine.astype(np.uint8)
        else:
             im_output_refine = im_output.copy()
    else:
        im_output = 0.4 * im[:,:,(2, 1, 0)]
        im_output_refine = im_output.copy()

    return im_output.astype(np.uint8), im_output_refine.astype(np.uint8), im_label


def overlay_image(dataset, im, rois, poses, labels):

    im = im[:, :, (2, 1, 0)]
    classes = dataset._classes
    class_colors = dataset._class_colors
    points = dataset._points_all
    intrinsic_matrix = dataset._intrinsic_matrix
    height = im.shape[0]
    width = im.shape[1]

    label_image = dataset.labels_to_image(labels)
    im_label = im.copy()
    I = np.where(labels != 0)
    im_label[I[0], I[1], :] = 0.5 * label_image[I[0], I[1], :] + 0.5 * im_label[I[0], I[1], :]

    for j in xrange(rois.shape[0]):
        cls = int(rois[j, 1])
        print(classes[cls], rois[j, -1])
        if cls > 0 and rois[j, -1] > cfg.TEST.DET_THRESHOLD:

            # draw roi
            x1 = rois[j, 2]
            y1 = rois[j, 3]
            x2 = rois[j, 4]
            y2 = rois[j, 5]
            cv2.rectangle(im_label, (x1, y1), (x2, y2), class_colors[cls], 2)

            # extract 3D points
            x3d = np.ones((4, points.shape[1]), dtype=np.float32)
            x3d[0, :] = points[cls,:,0]
            x3d[1, :] = points[cls,:,1]
            x3d[2, :] = points[cls,:,2]

            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(poses[j, :4])
            RT[:, 3] = poses[j, 4:7]
            x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
            x = np.round(np.divide(x2d[0, :], x2d[2, :]))
            y = np.round(np.divide(x2d[1, :], x2d[2, :]))
            index = np.where((x >= 0) & (x < width) & (y >= 0) & (y < height))[0]
            x = x[index].astype(np.int32)
            y = y[index].astype(np.int32)
            im[y, x, 0] = class_colors[cls][0]
            im[y, x, 1] = class_colors[cls][1]
            im[y, x, 2] = class_colors[cls][2]

    return im, im_label


def convert_to_image(im_blob):
    return np.clip(255 * im_blob, 0, 255).astype(np.uint8)

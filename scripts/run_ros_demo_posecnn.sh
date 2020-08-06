#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

./ros/start_posecnn_ros.py --gpu 0 \
  --instance 0 \
  --network posecnn \
  --pretrained checkpoints/posecnn/vgg16_ycb_object_self_supervision_all_epoch_8.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg config/posecnn/ycb_object_subset_realsense.yml

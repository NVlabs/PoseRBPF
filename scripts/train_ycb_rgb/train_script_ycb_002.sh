#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0 

python3 train_aae.py \
  --cfg_dir ./config/train/YCB/ \
  --obj 002_master_chef_can \
  --epochs 200 \
  --save 50 \
  --modality rgb \
  --dis_dir data/coco/val2014/val2014

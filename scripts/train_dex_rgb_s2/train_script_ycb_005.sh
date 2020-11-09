#!/usr/bin/env bash

python3 train_aae.py \
  --cfg_dir ./config/train/YCB/ \
  --obj 005_tomato_soup_can \
  --epochs 200 \
  --save 50 \
  --modality rgb \
  --dis_dir data/coco/train2014/train2014 \
  --dataset dex_ycb_s2_train \

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python ./ros/start_poserbpf_ros.py \
  --modality 'rgb' \
  --test_config './config/test/test_single_obj_list_ycb/003.yml' \
  --train_config_dir './checkpoints/ycb_configs_rgb/' \
  --ckpt_dir './checkpoints/ycb_ckpts_rgb/' \
  --codebook_dir './checkpoints/ycb_codebooks_rgb/' \
  --use_depth 'False'

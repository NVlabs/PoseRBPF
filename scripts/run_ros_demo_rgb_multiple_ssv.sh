#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python ./ros/start_poserbpf_ros.py \
  --modality 'rgb' \
  --test_config './config/test/test_multi_obj_list_ycb/example_ssv.yml' \
  --train_config_dir './checkpoints/ycb_configs_rgb_ssv/' \
  --ckpt_dir './checkpoints/ycb_ckpts_rgb_ssv/' \
  --codebook_dir './checkpoints/ycb_codebooks_rgb_ssv/' \
  --use_depth 'True' \
  --use_ssv_ckpts 'True'

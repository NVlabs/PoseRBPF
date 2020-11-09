#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1 

python3.6 test_prbpf_all.py \
  --gpu $1 \
  --modality 'rgbd' \
  --test_config './config/test/test_multi_obj_list_ycb/dex_ycb.yml' \
  --dataset 'dex_ycb_s2_test' \
  --train_config_dir './checkpoints/ycb_configs_roi_rgbd/' \
  --ckpt_dir './checkpoints/ycb_ckpts_roi_rgbd/' \
  --codebook_dir './checkpoints/ycb_codebooks_roi_rgbd/' \


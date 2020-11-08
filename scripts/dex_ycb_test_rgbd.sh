#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1 

python3.6 test_prbpf_rgbd_all.py \
  --test_config './config/test/test_multi_obj_list_ycb/dex_ycb_debug.yml' \
  --dataset 'dex_ycb_s0_test' \
  --depth_refinement

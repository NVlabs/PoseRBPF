#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python ./ros/start_poserbpf_ros.py \
  --test_config './config/test/test_multi_obj_list_ycb/example.yml'

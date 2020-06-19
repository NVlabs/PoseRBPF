#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python test_prbpf_rgbd.py --test_config './config/test/test_single_obj_list_ycb/006.yml' --n_seq $2;
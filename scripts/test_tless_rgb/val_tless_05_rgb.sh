#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python test_prbpf_rgb.py --test_config './config/test/test_single_obj_list_tless/05.yml' \
                                                  --pf_config_dir './config/test/TLess/' \
                                                  --train_config_dir './checkpoints/tless_configs_rgb/' \
                                                  --ckpt_dir './checkpoints/tless_ckpts_rgb/' \
                                                  --codebook_dir './checkpoints/tless_codebooks_rgb/' \
                                                  --dataset 'tless' \
                                                  --dataset_dir '../TLess/t-less_v2/test_primesense/' \
                                                  --n_seq $2;
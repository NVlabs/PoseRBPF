#!/usr/bin/env bash
python train_aae.py --cfg_dir ./config/train/YCB/ --obj 002_master_chef_can --epochs 200 --save 50 --modality rgb

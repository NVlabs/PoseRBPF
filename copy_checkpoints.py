#!/usr/bin/env python3

import os
import os.path as osp
import glob
from shutil import copyfile

classes = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
           '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
           '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
           '052_extra_large_clamp', '061_foam_brick')

modality = 'rgb'

src_dir = '/mnt/poserbpf/checkpoints'
dst_dir = 'checkpoints/dex_s0_ckpts_' + modality

# list src dir
subdirs = os.listdir(src_dir)
count = 0
for d in subdirs:
    for cls in classes:
        if cls in d:
            cls_name = cls
            break

    # list checkpoints
    filename = os.path.join(src_dir, d, '*.pth')
    files = sorted(glob.glob(filename))

    # copy file
    src_filename = files[-1]
    dst_filename = osp.join(dst_dir, cls_name + '.pth')
    print('===============================')
    print('copy %s to \n %s' % (src_filename, dst_filename))
    copyfile(src_filename, dst_filename)
    count += 1
print('%d files copied' % (count))

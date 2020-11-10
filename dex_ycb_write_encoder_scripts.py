import os
import os.path as osp

classes = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
           '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
           '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
           '052_extra_large_clamp', '061_foam_brick')

setup = ['s0', 's1', 's2', 's3']

modality = 'rgb'

for s in setup:
    for i in range(len(classes)):
        cls = classes[i]

        dirname = osp.join('scripts', 'train_dex_' + modality + '_' + s)
        if not osp.exists(dirname):
            os.makedirs(dirname)

        # write training script
        filename = osp.join(dirname, 'train_script_ycb_' + cls[:3] + '.sh')
        print(filename)
        with open(filename, 'w') as f:
            f.write('#!/usr/bin/env bash\n\n')
            f.write('python3 train_aae.py \\\n')
            f.write('  --cfg_dir ./config/train/YCB/ \\\n')
            f.write('  --obj %s \\\n' %(cls))
            f.write('  --epochs 200 \\\n')
            f.write('  --save 25 \\\n')
            f.write('  --modality %s \\\n' % (modality))
            f.write('  --dis_dir data/coco/train2014/train2014 \\\n')
            f.write('  --dataset dex_ycb_%s_train \\\n' % (s))
        f.close()
        cmd = 'chmod +x ' + filename
        os.system(cmd)

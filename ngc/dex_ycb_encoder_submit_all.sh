#!/bin/bash

declare -a arr=("002_master_chef_can" "003_cracker_box" "004_sugar_box" "005_tomato_soup_can" "006_mustard_bottle" \
                "007_tuna_fish_can" "008_pudding_box" "009_gelatin_box" "010_potted_meat_can" "011_banana" "019_pitcher_base" \
                "021_bleach_cleanser" "024_bowl" "025_mug" "035_power_drill" "036_wood_block" "037_scissors" "040_large_marker" \
                "052_extra_large_clamp" "061_foam_brick")

for i in "${arr[@]}"
do
    echo $i
    ./ngc/train_ngc.sh dex_$i ./scripts/train_dex_rgb_s0/train_script_ycb_${i:0:3}.sh 1
done

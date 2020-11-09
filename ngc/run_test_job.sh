#!/bin/bash

# Usage: ./train_ngc.sh [run-name] [script-path]

set -e
set -v

ngc batch run \
    --instance "dgx1v.16g.2.norm" \
    --name "test_job" \
    --image "nvcr.io/nvidian/robotics/posecnn-pytorch:latest" \
    --result /result \
    --datasetid "58777:/deepim/data/models" \
    --datasetid "58774:/deepim/data/backgrounds" \
    --datasetid "8187:/deepim/data/coco" \
    --datasetid "11888:/deepim/data/YCB_Video/YCB_Video_Dataset" \
    --datasetid "68150:/deepim/data/DEX_YCB/data" \
    --datasetid "68317:/deepim/data/SUNRGBD" \
    --workspace deepim:/deepim \
    --commandline "sleep 168h"

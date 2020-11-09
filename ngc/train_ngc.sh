#!/bin/bash

# Usage: ./train_ngc.sh [run-name] [script-path] [num-gpus]

set -e
set -v

NAME=$1
SCRIPT=$2
NGPU=$3

ngc batch run \
    --instance "dgx1v.16g.$NGPU.norm" \
    --name "$NAME" \
    --image "nvcr.io/nvidian/robotics/posecnn-pytorch:latest" \
    --result /result \
    --datasetid "58777:/deepim/data/models" \
    --datasetid "58774:/deepim/data/backgrounds" \
    --datasetid "8187:/deepim/data/coco" \
    --datasetid "11888:/deepim/data/YCB_Video/YCB_Video_Dataset" \
    --datasetid "68150:/deepim/data/DEX_YCB/data" \
    --datasetid "68317:/deepim/data/SUNRGBD" \
    --workspace deepim:/deepim \
    --commandline "cd /deepim; bash $SCRIPT" \
    --total-runtime 7D \
    --port 6006

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
    --datasetid "58777:/poserbpf/cad_models/ycb_models" \
    --datasetid "8187:/poserbpf/data/coco" \
    --datasetid "68150:/poserbpf/data/DEX_YCB/data" \
    --workspace poserbpf:/poserbpf \
    --commandline "cd /poserbpf; bash $SCRIPT" \
    --total-runtime 7D \
    --port 6006

#!/bin/bash

# Usage: ./train_ngc.sh [run-name] [script-path]

set -e
set -v

ngc batch run \
    --instance "dgx1v.16g.1.norm" \
    --name "test_job" \
    --image "nvcr.io/nvidian/robotics/posecnn-pytorch:latest" \
    --result /result \
    --datasetid "58777:/poserbpf/cad_models/ycb_models" \
    --datasetid "8187:/poserbpf/data/coco" \
    --datasetid "68150:/poserbpf/data/DEX_YCB/data" \
    --workspace poserbpf:/poserbpf \
    --commandline "sleep 168h"

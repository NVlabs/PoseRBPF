#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $DIR

if [ -z "$1" ]
  then
    echo "No target path supplied"
fi

set -ex
mkdir -p $1

rsync -rlvcP \
  --exclude 'data' \
  --exclude 'data_files' \
  --exclude 'cad_models' \
  --exclude 'checkpoints' \
  --exclude 'detections' \
  --exclude 'results' \
  --exclude 'docker' \
  --exclude 'output' \
  --exclude 'ngc' \
  --exclude 'pics' \
  --exclude 'utils/bbox.c' \
  --exclude '.ipynb_checkpoints' \
  --exclude '__pycache__' \
  --exclude '.cache' \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '*.o' \
  --exclude '*.so' \
  --exclude '*.pyc' \
  --exclude '*.egg-info' \
  --exclude '.idea' \
  --exclude 'build' \
  --exclude 'dist' \
  --exclude 'notebooks' \
  --exclude '.vscode' \
  $DIR/.. $1 --delete

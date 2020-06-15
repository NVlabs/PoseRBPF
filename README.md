# PoseRBPF: A Rao-Blackwellized Particle Filter for 6D Object Pose Tracking
* [[paper](https://arxiv.org/abs/1905.09304)]

## Citing PoseRBPF
If you find the PoseRBPF code useful, please consider citing:

```bibtex
@inproceedings{deng2019pose,
author    = {Xinke Deng and Arsalan Mousavian and Yu Xiang and Fei Xia and Timothy Bretl and Dieter Fox},
title     = {PoseRBPF: A Rao-Blackwellized Particle Filter for 6D Object Pose Tracking},
booktitle = {Robotics: Science and Systems (RSS)},
year      = {2019}
}
```

## Installation 
```bash
git clone https://github.com/XinkeAE/PoseRBPF.git --recursive
```

Install dependencies:
- install anaconda according to [the official website](https://docs.anaconda.com/anaconda/install/).
- create the virtual env with ```pose_rbpf_env.yml```:
```angular2html
conda env create -f pose_rbpf_env.yml
conda activate pose_rbpf_env
``` 
- compile the YCB Renderer according to the [instruction](./ycb_render/README.md).
- compile the utility functions with:
```angular2html
sh build.sh
```

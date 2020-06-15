# PoseRBPF: A Rao-Blackwellized Particle Filter for 6D Object Pose Tracking

## Installation 
```bash
git clone https://github.com/XinkeAE/ActivePoseEstimation.git --recursive
```

Install dependencies:
- compile the YCB Render with tensor rendering in this repo
- python 3.6.6
- pytorch 0.4.1
- easyDict ```pip install easydict```
- filterpy 1.4.5 ```pip install filterpy```
- mayavi for uncertainty visualization (optional) ```pip install vtk mayavi```
- numba ```pip install numba```
- ruamel.yaml ```pip install ruamel.yaml```

For dependencies, I recommend to use anaconda for setting up the virtual env. Download the ```pytorch_env.yaml``` 
file from my google drive link and set the env up with
```angular2html
conda env create -f pytorch_env.yaml
``` 

## Testing example
- Download the weights from my google drive and put in checkpoints folder
- My structure is like:
```angular2html
Documents
├── ...
├── ActivePoseEstimation
|   |── checkpoints
|   |   |── ycbxxxxTxxxx(xx)
|   |   |── tlessxxxxTxxxxxx
|   |   └── ...  
|   |── config                      # configuration files for training and DPF
|   |── models                      # networks and particle filter
|   |── scripts                     # scripts for training and testing
|   └── ...
|── modelAndPose                    # to store ycb models, download from my google drive
|   |── models  
|   |   |── 003_cracker_box
|   |   └── ...                   
|   └── ...
|── TLessModels
|   └── model_cad_color_hd
|       |── obj_01.ply
|       └── ...
|── coco                            # distractors use coco val images
|                 
└── ...
```
- Compile YCB Render
- Update the ```--dataset_dir``` in ```scripts/val_script_tracking_dope_003.sh``` to specify the path to ycb or tless dataset 
(be aware of the sub directories, see the example in the scripts.)
- run
```sh scripts/val_script_tracking_dope_003.sh 1``` to evaluate the cracker box
- result will be saved to ```./results```

## Training
- run
```sh scripts/train_script_dope_003.sh``` to train locally.
- on the SaturnV: set up structure is like:
```angular2html
SaturnV_NFS
├── ...
├── ActivePoseEstimation
|   |── checkpoints
|   |   |── ycbxxxxTxxxx(xx)
|   |   |── tlessxxxxTxxxxxx
|   |   └── ...  
|   |── config                      # configuration files for training and DPF
|   |── models                      # networks and particle filter
|   |── scripts                     # scripts for training and testing
|   └── ...
|── modelAndPose                    # to store ycb models, download from my google drive
|   |── models  
|   |   |── 003_cracker_box
|   |   └── ...                   
|   └── ...
|── TLessModels
|   └── model_cad_color_hd
|       |── obj_01.ply
|       └── ...
|── coco                            # distractors use coco val images
|                 
└── ...
```
- you will find the scripts ```sync_ckpt.sh``` and ```sync_saturn.sh```. ```sync_ckpt.sh``` downloads the checkpoints
from the server and ```sync_saturn.sh``` upload your files to NFS.
- I always do:
```sh sync_saturn.sh; dgx job submit -f scripts/jobs/xxxx.json``` to submit the jobs.

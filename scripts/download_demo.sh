#!/usr/bin/env bash
# download checkpoints
mkdir checkpoints;
sh ./scripts/gdrive_download.sh 1gUvcB_MJ2xLveF5qQgwFIorW8JY1aCeT checkpoints.tar.gz;
tar -xvf checkpoints.tar.gz -C ./checkpoints/;
rm ./checkpoints.tar.gz

# download object models
mkdir cad_models;
sh ./scripts/gdrive_download.sh 1RBY88gfQ9UDxlPpnYF6XzDIKHN-w1ra5 ycb_models.tar.gz;
tar -xvf ycb_models.tar.gz -C ./cad_models/;
rm ./ycb_models.tar.gz

# download detections
mkdir detections;
sh ./scripts/gdrive_download.sh 1DQC7-ur-3czr5URD7LQF2o3p5j8Nu6AQ detections.tar.gz;
tar -xvf detections.tar.gz -C ./detections/;
rm ./detections.tar.gz

# download data
mkdir ../YCB_Video_Dataset
sh ./scripts/gdrive_download.sh 13b_GHxpnKjPbpBvijx3QO6e79TKzjBNh demo_data.tar.gz;
tar -xvf demo_data.tar.gz -C ../YCB_Video_Dataset/;
rm ./demo_data.tar.gz
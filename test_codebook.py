import torch
from datasets.render_ycb_dataset import *
from networks.aae_models import *

if __name__ == "__main__":

    aae_full = AAE(['002_master_chef_can'], 'rgbd')
    aae_full.encoder.eval()
    aae_full.decoder.eval()
    for param in aae_full.encoder.parameters():
        param.requires_grad = False
    for param in aae_full.decoder.parameters():
        param.requires_grad = False
    checkpoint = torch.load('./checkpoints/ycb_ckpts_roi_rgbd/002_master_chef_can_py3.pth')

    aae_full.load_ckpt_weights(checkpoint['aae_state_dict'])
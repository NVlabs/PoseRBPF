import math
from torch import nn
from torch.autograd import Function
import torch
import sdf_layer_cuda

class SDFLossFunction(Function):
    @staticmethod
    def forward(ctx, pose_delta, pose_init, sdf_grids, sdf_limits, points):
        outputs = sdf_layer_cuda.sdf_loss_forward(pose_delta, pose_init, sdf_grids, sdf_limits, points)

        loss = outputs[0]
        sdf_values = outputs[1]
        se3 = outputs[2]
        variables = outputs[3:]
        ctx.save_for_backward(*variables)

        return loss, sdf_values, se3

    @staticmethod
    def backward(ctx, grad_loss, grad_sdf_values, grad_se3):
        outputs = sdf_layer_cuda.sdf_loss_backward(grad_loss, *ctx.saved_variables)
        d_delta = outputs[0]

        return d_delta, None, None, None, None


class SDFLoss(nn.Module):
    def __init__(self):
        super(SDFLoss, self).__init__()

    def forward(self, pose_delta, pose_init, sdf_grids, sdf_limits, points):
        return SDFLossFunction.apply(pose_delta, pose_init, sdf_grids, sdf_limits, points)

#include <torch/torch.h>

#include <vector>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/************************************************************
 sdf matching loss layer
*************************************************************/

std::vector<at::Tensor> sdf_loss_cuda_forward(
    at::Tensor pose_delta,
    at::Tensor pose_init,
    at::Tensor sdf_grids,
    at::Tensor sdf_limits,
    at::Tensor points);

std::vector<at::Tensor> sdf_loss_cuda_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff);

std::vector<at::Tensor> sdf_loss_forward(
    at::Tensor pose_delta,
    at::Tensor pose_init,
    at::Tensor sdf_grids,
    at::Tensor sdf_limits,
    at::Tensor points)
{
  CHECK_INPUT(pose_delta);
  CHECK_INPUT(pose_init);
  CHECK_INPUT(sdf_grids);
  CHECK_INPUT(sdf_limits);
  CHECK_INPUT(points);

  return sdf_loss_cuda_forward(pose_delta, pose_init, sdf_grids, sdf_limits, points);
}

std::vector<at::Tensor> sdf_loss_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff) 
{
  CHECK_INPUT(grad_loss);
  CHECK_INPUT(bottom_diff);

  return sdf_loss_cuda_backward(grad_loss, bottom_diff);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sdf_loss_forward", &sdf_loss_forward, "SDF loss forward (CUDA)");
  m.def("sdf_loss_backward", &sdf_loss_backward, "SDF loss backward (CUDA)");
}

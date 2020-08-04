#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

__device__ __host__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ __host__ float3 operator-(const float3 &a, const float3 &b)
{
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

template <typename Dtype>
inline __device__ __host__ const Dtype & getValue(const int3 & v, const int3 & dim, const Dtype* sdf_grids)
{
  return sdf_grids[v.x * dim.y * dim.z + v.y * dim.z + v.z];
}

template <typename Dtype>
inline __device__ __host__ Dtype getValueInterpolated(const float3 & pGrid, const int3 & dim, const Dtype* sdf_grids)
{
  const int x0 = (int)(pGrid.x - 0.5); const float fx = (pGrid.x - 0.5) - x0;
  const int y0 = (int)(pGrid.y - 0.5); const float fy = (pGrid.y - 0.5) - y0;
  const int z0 = (int)(pGrid.z - 0.5); const float fz = (pGrid.z - 0.5) - z0;

  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const int z1 = z0 + 1;

  if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y && z0 >=0 && z1 < dim.z) )
    return 0.1;

  const float dx00 = lerp( getValue(make_int3(x0,y0,z0), dim, sdf_grids), getValue(make_int3(x1,y0,z0), dim, sdf_grids), fx);
  const float dx01 = lerp( getValue(make_int3(x0,y0,z1), dim, sdf_grids), getValue(make_int3(x1,y0,z1), dim, sdf_grids), fx);
  const float dx10 = lerp( getValue(make_int3(x0,y1,z0), dim, sdf_grids), getValue(make_int3(x1,y1,z0), dim, sdf_grids), fx);
  const float dx11 = lerp( getValue(make_int3(x0,y1,z1), dim, sdf_grids), getValue(make_int3(x1,y1,z1), dim, sdf_grids), fx);

  const float dxy0 = lerp( dx00, dx10, fy );
  const float dxy1 = lerp( dx01, dx11, fy );
  float dxyz = lerp( dxy0, dxy1, fz );

  // penalize inside objects
  //if (dxyz < 0)
  //  dxyz *= 10;

  return dxyz;
}

template <typename Dtype>
inline __device__ __host__ float3 getGradientInterpolated(const float3 & pGrid, const int3 & dim, const Dtype* sdf_grids)
{
  const float3 delta_x = make_float3(1,0,0);
  const float3 delta_y = make_float3(0,1,0);
  const float3 delta_z = make_float3(0,0,1);

  Dtype f_px = getValueInterpolated(pGrid + delta_x, dim, sdf_grids);
  Dtype f_py = getValueInterpolated(pGrid + delta_y, dim, sdf_grids);
  Dtype f_pz = getValueInterpolated(pGrid + delta_z, dim, sdf_grids);

  Dtype f_mx = getValueInterpolated(pGrid - delta_x, dim, sdf_grids);
  Dtype f_my = getValueInterpolated(pGrid - delta_y, dim, sdf_grids);
  Dtype f_mz = getValueInterpolated(pGrid - delta_z, dim, sdf_grids);

  float3 grad;
  grad.x = 0.5*(f_px - f_mx);
  grad.y = 0.5*(f_py - f_my);
  grad.z = 0.5*(f_pz - f_mz);
  return grad;
}


template <typename Dtype>
__global__ void SDFdistanceForward(const int nthreads, const Dtype* pose_delta, const Dtype* pose_init,
    const Dtype* sdf_grids, const Dtype* sdf_limits, const Dtype* points, 
    const int num_points, const int d0, const int d1, const int d2, Dtype* losses, Dtype* top_values, Dtype* diffs, Dtype* top_se3) 
{
  typedef Sophus::SE3<Dtype> SE3;
  typedef Eigen::Matrix<Dtype,3,1,Eigen::DontAlign> Vec3;

  // index is the index of point
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // convert delta pose
    Eigen::Matrix<Dtype,6,1> deltaPose;
    deltaPose << pose_delta[0], pose_delta[1], pose_delta[2], pose_delta[3], pose_delta[4], pose_delta[5];
    SE3 deltaPoseMatrix = SE3::exp(deltaPose);

    // convert initial pose
    Eigen::Matrix<Dtype,4,4> initialPose;
    initialPose << pose_init[0], pose_init[1], pose_init[2], pose_init[3],
                   pose_init[4], pose_init[5], pose_init[6], pose_init[7],
                   pose_init[8], pose_init[9], pose_init[10], pose_init[11],
                   pose_init[12], pose_init[13], pose_init[14], pose_init[15];
    SE3 initialPoseMatrix = SE3(initialPose);

    if (index == 0)
    {
      SE3 pose = deltaPoseMatrix * initialPoseMatrix;
      Eigen::Matrix<Dtype,3,4> matrix = pose.matrix3x4();
      int count = 0;
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 4; j++)
          top_se3[count++] = matrix(i, j);
      }
      top_se3[15] = 1.0;
    }

    // convert point
    Vec3 point;
    point << points[3 * index], points[3 * index + 1], points[3 * index + 2];

    // transform the point
    const Vec3 updatedPoint = deltaPoseMatrix * initialPoseMatrix * point;

    // obtain sdf value
    float px = (updatedPoint(0) - sdf_limits[0]) / (sdf_limits[3] - sdf_limits[0]) * d0;
    float py = (updatedPoint(1) - sdf_limits[1]) / (sdf_limits[4] - sdf_limits[1]) * d1;
    float pz = (updatedPoint(2) - sdf_limits[2]) / (sdf_limits[5] - sdf_limits[2]) * d2;

    float3 pGrid = make_float3(px, py, pz);
    int3 dim = make_int3(d0, d1, d2);
    Dtype value = getValueInterpolated(pGrid, dim, sdf_grids);

    int flag = 1;
    if (value < 0)
      flag = -1;
    losses[index] = flag * value;
    top_values[index] = losses[index];

    // L2 penalty on translation
    float lambda = 0.001;
    losses[index] += 0.5 * lambda * (pose_delta[0] * pose_delta[0] + pose_delta[1] * pose_delta[1] + pose_delta[2] * pose_delta[2]);

    // compute gradient
    float3 grad = getGradientInterpolated(pGrid, dim, sdf_grids);
    Vec3 sdfUpdate;
    sdfUpdate << grad.x, grad.y, grad.z;

    Eigen::Matrix<Dtype,3,6> dUpdate;
    dUpdate << 1, 0, 0,                     0,  updatedPoint(2), -updatedPoint(1),
               0, 1, 0, -updatedPoint(2),                     0,  updatedPoint(0),
               0, 0, 1,  updatedPoint(1), -updatedPoint(0),                     0;

    Eigen::Matrix<Dtype,1,6> grad_pose = sdfUpdate.transpose() * dUpdate;

    // assign gradient
    for (int i = 0; i < 6; i++)
      diffs[6 * index + i] = flag * grad_pose(i);

    // L2 penalty on translation
    diffs[6 * index + 0] += lambda * pose_delta[0];
    diffs[6 * index + 1] += lambda * pose_delta[1];
    diffs[6 * index + 2] += lambda * pose_delta[2];
  }
}

/* diffs: num_points x num_channels */
/* bottom_diff: num_channels */
template <typename Dtype>
__global__ void sum_gradients(const int nthreads, const Dtype* diffs, const int num_points, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    bottom_diff[index] = 0;
    int num_channels = 6;
    for (int p = 0; p < num_points; p++)
    {
      int index_diff = p * num_channels + index;
      bottom_diff[index] += diffs[index_diff];
    }
  }
}


/***************************/
/* pose_delta: 1 x 6       */
/* pose_init: 4 x 4        */
/* sdf_grid: c x h x w     */
/* points: n x 3           */
/***************************/
std::vector<at::Tensor> sdf_loss_cuda_forward(
    at::Tensor pose_delta,
    at::Tensor pose_init,
    at::Tensor sdf_grids,
    at::Tensor sdf_limits,
    at::Tensor points) 
{
  // run kernels
  cudaError_t err;
  const int kThreadsPerBlock = 512;
  const int num_channels = 6;
  int output_size;

  // temp losses
  const int num_points = points.size(0); 
  const int3 dim = make_int3(sdf_grids.size(0), sdf_grids.size(1), sdf_grids.size(2));
  auto losses = at::zeros({num_points}, points.options());
  auto top_values = at::zeros({num_points}, points.options());
  auto top_data = at::zeros({1}, points.options());
  auto top_se3 = at::zeros({4, 4}, points.options());

  // temp diffs
  auto diffs = at::zeros({num_points, num_channels}, points.options());
  auto bottom_diff = at::zeros({num_channels}, points.options());

  // compute the losses and gradients
  output_size = num_points;
  SDFdistanceForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, pose_delta.data<float>(), pose_init.data<float>(), sdf_grids.data<float>(), sdf_limits.data<float>(),
      points.data<float>(), num_points, dim.x, dim.y, dim.z, losses.data<float>(), top_values.data<float>(), diffs.data<float>(), top_se3.data<float>());
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // sum the diffs
  output_size = num_channels;
  sum_gradients<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, diffs.data<float>(), num_points, bottom_diff.data<float>());
  cudaDeviceSynchronize();

  // sum the loss
  thrust::device_ptr<float> losses_ptr(losses.data<float>());
  float loss = thrust::reduce(losses_ptr, losses_ptr + num_points) / num_points;
  cudaMemcpy(top_data.data<float>(), &loss, sizeof(float), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return {top_data, top_values, top_se3, bottom_diff};
}


template <typename Dtype>
__global__ void SDFdistanceBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_diff, Dtype* output) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    output[index] = top_diff[0] * bottom_diff[index];
  }
}


std::vector<at::Tensor> sdf_loss_cuda_backward(
    at::Tensor grad_loss,
    at::Tensor bottom_diff)
{
  cudaError_t err;
  const int kThreadsPerBlock = 512;
  int output_size;
  const int batch_size = bottom_diff.size(0);
  auto grad_pose = at::zeros({batch_size}, bottom_diff.options());

  output_size = batch_size;
  SDFdistanceBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, grad_loss.data<float>(), bottom_diff.data<float>(), grad_pose.data<float>());

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return {grad_pose};
}

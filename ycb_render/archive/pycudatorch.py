import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import torch

x = torch.cuda.FloatTensor(8)

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer():
        return self.t.data_ptr()

a = np.random.randn(400).astype(np.float32)
b = np.random.randn(400).astype(np.float32)

a = torch.from_numpy(a).cuda()
b = torch.from_numpy(b).cuda()
dest = torch.Tensor(a.size()).cuda()

multiply_them(
        Holder(dest),
        Holder(a),
        Holder(b),
        block=(400,1,1), grid=(1,1))

torch.cuda.synchronize()

print(dest-a*b)

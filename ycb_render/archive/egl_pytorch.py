from contextlib import contextmanager
import numpy as np
import torch
from torch import Tensor, ByteTensor
import torch.nn.functional as F
from torch.autograd import Variable
import pycuda.driver
from pycuda.gl import graphics_map_flags
try:
    from .get_available_devices import *
except:
    from get_available_devices import *

import CppYCBRenderer
import OpenGL.GL as GL
from PIL import Image
import matplotlib.pyplot as plt

from progressbar import progressbar as bar

@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0,0)
    mapping.unmap()


def loadTexture(path):
    img = Image.open(path)
    img_data = np.fromstring(img.tobytes(), np.uint8)
    width, height = img.size

    texture = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, width, height, 0,
                    GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

    cuda_buffer = pycuda.gl.RegisteredImage(
        int(texture), GL.GL_TEXTURE_2D, pycuda.gl.graphics_map_flags.NONE)

    return texture, cuda_buffer, (width, height, 3)


def main():
    # setup pycuda and torch
    import pycuda.gl.autoinit
    import pycuda.gl
    assert torch.cuda.is_available()
    print('pytorch using GPU {}'.format(torch.cuda.current_device()))
    img = Image.open('misc/husky.jpg').transpose(Image.FLIP_TOP_BOTTOM)
    width, height = img.size

    state = torch.cuda.FloatTensor(width, height, 4)
    state = state.byte().contiguous()
    tex, cuda_buffer, sz = loadTexture('misc/husky.jpg')

    nbytes = state.numel() * state.element_size()

    for _ in bar(range(20000)):
        with cuda_activate(cuda_buffer) as ary:
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_dst_device(state.data_ptr())
            cpy.set_src_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = nbytes//height
            cpy.height = height
            cpy(aligned=False)
            torch.cuda.synchronize()

    img_np = state.data.cpu().numpy().reshape(height, width, 4)
    print(img_np.shape)
    img_np[:, :, 3] = 255
    plt.imshow(img_np)
    plt.show()

gl_dev = get_available_devices()[0]
print('opengl using GPU {}'.format(gl_dev))
r = CppYCBRenderer.CppYCBRenderer(512, 512, gl_dev)
r.init()

main()

r.release()
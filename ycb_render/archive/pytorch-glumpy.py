from contextlib import contextmanager
import numpy as np
import torch
from torch import Tensor, ByteTensor
import torch.nn.functional as F
from torch.autograd import Variable
import pycuda.driver
from pycuda.gl import graphics_map_flags
from glumpy import app, gloo, gl

@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0,0)
    mapping.unmap()

def create_shared_texture(w, h, c=4,
        map_flags=graphics_map_flags.WRITE_DISCARD,
        dtype=np.uint8):
    """Create and return a Texture2D with gloo and pycuda views."""
    tex = np.zeros((h,w,c), dtype).view(gloo.Texture2D)
    tex.activate() # force gloo to create on GPU
    tex.deactivate()
    cuda_buffer = pycuda.gl.RegisteredImage(
        int(tex.handle), tex.target, map_flags)
    return tex, cuda_buffer

def setup():
    global screen, cuda_buffer, state
    w, h = window.get_size()
    # setup pycuda and torch
    import pycuda.gl.autoinit
    import pycuda.gl
    assert torch.cuda.is_available()
    print('using GPU {}'.format(torch.cuda.current_device()))
    # torch.nn layers expect batch_size, channels, height, width
    state = torch.cuda.FloatTensor(1,3,h,w)
    state.uniform_()
    state = Variable(state, volatile=True)
    # create a buffer with pycuda and gloo views
    tex, cuda_buffer = create_shared_texture(w,h,4)
    print(tex, tex.handle)
    # create a shader to program to draw to the screen
    vertex = """
    uniform float scale;
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        v_texcoord = texcoord;
        gl_Position = vec4(scale*position, 0.0, 1.0);
    } """
    fragment = """
    uniform sampler2D tex;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(tex, v_texcoord);
    } """
    # Build the program and corresponding buffers (with 4 vertices)
    screen = gloo.Program(vertex, fragment, count=4)
    # Upload data into GPU
    screen['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
    screen['texcoord'] = [(0,0), (0,1), (1,0), (1,1)]
    screen['scale'] = 1.0
    screen['tex'] = tex

def torch_process(state):
    """Random convolutions."""
    fs = 11
    filters, sgns = (
        Variable(init(torch.cuda.FloatTensor(3,3,fs,fs)), volatile=True)
        for init in (
            lambda x: x.normal_(),
            lambda x: x.bernoulli_(0.52)
        ))
    filters = F.softmax(filters)*(sgns*2-1)
    state = F.conv2d(state, filters, padding=fs//2)
    state = state-state.mean().expand(state.size())
    state = state/state.std().expand(state.size())
    return state

# create window with OpenGL context
app.use('glfw')
window = app.Window(512, 512, fullscreen=False)

@window.event
def on_draw(dt):
    global state
    window.set_title(str(
    window.fps).encode("ascii"))
    tex = screen['tex']
    h,w = tex.shape[:2]
    # mutate state in torch
    state = torch_process(state).detach() # prevent autograd from filling memory
    img = F.tanh(state).abs()
    # convert into proper format
    print(img.size())
    tensor = img.squeeze().transpose(0,2).transpose(1,0).data # put in texture order
    tensor = torch.cat((tensor, tensor[:,:,0:1]),2) # add the alpha channel
    tensor[:,:,3] = 1 # set alpha
    # check that tensor order matches texture:
    # img[:,:,2] = 1 # set blue
    # img[100,:,:] = 1 # horizontal white line
    # img[:,200,0] = 1 # vertical magenta line
    tensor = (255*tensor).byte().contiguous() # convert to ByteTensor
    # copy from torch into buffer
    assert tex.nbytes == tensor.numel()*tensor.element_size()
    with cuda_activate(cuda_buffer) as ary:
        cpy = pycuda.driver.Memcpy2D()
        cpy.set_src_device(tensor.data_ptr())
        cpy.set_dst_array(ary)
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tex.nbytes//h
        cpy.height = h
        cpy(aligned=False)
        torch.cuda.synchronize()
    # draw to screen
    window.clear()
    screen.draw(gl.GL_TRIANGLE_STRIP)

# not sure why this doesn't work right
@window.event
def on_close():
    pycuda.gl.autoinit.context.pop()

if __name__=='__main__':
    setup()
    app.run()

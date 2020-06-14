import cv2
import numpy as np
#from multistep import SimplePoseReg
#from multistep_recurrent import SimplePoseRNN
import sys
import torch
from PIL import Image
import glutils.glcontext as glcontext

import OpenGL.GL as gl
import glutils.glrenderer as glrenderer
import glutils.meshutil as mu

def main():

    context = glcontext.Context()
    context.create_opengl_context((800, 800))
    print(gl.glGetString(gl.GL_VERSION))

    renderer = glrenderer.MeshRenderer((800, 800))
    renderer.fovy = 90

    position = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
    color = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 1]]
    face = [[0, 6, 4], [0, 2, 6], [0, 3, 2], [0, 1, 3], [2, 7, 6], [2, 3, 7], [4, 6, 7], [4, 7, 6], [0, 4, 5], [0, 5, 1]
        , [1, 5, 7], [1, 7, 3]]

    img = renderer.render_mesh(position, color, face, modelview=mu.lookat([2,2,2], [0,0,0]))
    img, alpha = img[..., :3], img[..., 3]

    q = None
    while q!= ord('q'):
        cv2.imshow('test', img)
        q = cv2.waitKey(10)

if __name__ == '__main__':
    main()
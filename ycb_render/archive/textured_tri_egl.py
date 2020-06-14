"""Quick hack of 'modern' OpenGL example using pysdl2 and pyopengl
that shows a textured triangle; assumes there is a 'hazard.png' image
file in working directory.

Based on:

pysdl2 OpenGL example
http://www.tomdalling.com/blog/modern-opengl/02-textures/
http://www.arcsynthesis.org/gltut/Basics/Tut02%20Vertex%20Attributes.html
http://schi.iteye.com/blog/1969710
https://www.opengl.org/wiki/Vertex_Specification_Best_Practices#Vertex_Layout_Specification
http://docs.gl/gl3/glVertexAttribPointer
https://gist.github.com/jawa0/4003034
https://github.com/tomdalling/opengl-series/blob/master/linux/02_textures/source/main.cpp
"""
import sys
import ctypes
import numpy

from PIL import Image
import glutils.glcontext as glcontext

import OpenGL.GL as GL
from numpy import array
import cv2
import numpy as np
from pyassimp import *
from glutils.meshutil import perspective, lookat

shaderProgram = None
VAO = None
VBO = None
sampleTexture = None
texUnitUniform = None
global shaders
shaders = None

WIDTH = 512
HEIGHT = 512

def main():
    global shaders

    context = glcontext.Context()
    context.create_opengl_context((WIDTH, HEIGHT))
    print(GL.glGetString(GL.GL_VERSION))
    from OpenGL.GL import shaders
    #print(shaders)
    initialize()

    q = None
    frame = 0
    while q != ord('q'):
        img = render(frame)
        #img, alpha = img[..., :3], img[..., 3]
        cv2.imshow('test', img)
        q = cv2.waitKey(1)
        if frame % 100 == 0:
            print(frame)
        frame += 1

def loadTexture(path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    print(img.size)
    img_data = numpy.fromstring(img.tobytes(), numpy.uint8)
    #print(img_data.shape)
    width, height = img.size

    # glTexImage2D expects the first element of the image data to be the
    # bottom-left corner of the image.  Subsequent elements go left to right,
    # with subsequent lines going from bottom to top.

    # However, the image data was created with PIL Image tostring and numpy's
    # fromstring, which means we have to do a bit of reorganization. The first
    # element in the data output by tostring() will be the top-left corner of
    # the image, with following values going left-to-right and lines going
    # top-to-bottom.  So, we need to flip the vertical coordinate (y). 
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
    return texture

def initialize():
    global shaderProgram
    global VAO
    global VBO
    global texUnitUniform
    global sampleTexture
    global shaders
    global faces
    #print(shaders)

    vertexShader = shaders.compileShader("""
#version 460
uniform mat4 MVP;
layout (location=0) in vec3 position;
layout (location=1) in vec2 texCoords;

out vec2 theCoords;

void main()
{
    gl_Position = MVP * vec4(position, 1);
    theCoords = texCoords;
}
""", GL.GL_VERTEX_SHADER)

    fragmentShader = shaders.compileShader("""
#version 460

uniform sampler2D texUnit;

in vec2 theCoords;

out vec4 outputColour;

void main()
{
    outputColour = texture(texUnit, theCoords);
}
""", GL.GL_FRAGMENT_SHADER)

    shaderProgram = shaders.compileProgram(vertexShader, fragmentShader)

    '''vertexData = numpy.array([
         # X,    Y,   Z     U,   V
         0.0,  0.8, 0.0,  0.0, 1.0,
        -0.8, -0.8, 0.0,  0.0, 0.0,
         0.8, -0.8, 0.0,  1.0, 0.0,
         0.0,  -0.8, 0.0, 0.0, 1.0,
         0.8, 0.8, 0.0, 0.0, 0.0,
         -0.8, 0.8, 0.0, 1.0, 0.0,
    ], dtype=numpy.float32)

    faces = np.array([[0,1,2], [3,4,5]])
    faces = np.ascontiguousarray(faces, np.uint32)
    '''

    scene = load('/home/fei/Downloads/models/002_master_chef_can/textured_simple.obj')
    print(len(scene.meshes))

    mesh = scene.meshes[0]
    faces = mesh.faces
    vertices = np.concatenate([mesh.vertices, mesh.texturecoords[0, :, :2]], axis=-1)
    vertexData = vertices.astype(np.float32)
    release(scene)

    # Core OpenGL requires that at least one OpenGL vertex array be bound
    VAO = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(VAO)

    # Need VBO for triangle vertices and texture UV coordinates
    VBO = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData,
        GL.GL_STATIC_DRAW)

    # enable array and set up data
    positionAttrib = GL.glGetAttribLocation(shaderProgram, 'position')
    coordsAttrib = GL.glGetAttribLocation(shaderProgram, 'texCoords')

    GL.glEnableVertexAttribArray(0)
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 20,
        None)
    # the last parameter is a pointer
    GL.glVertexAttribPointer(coordsAttrib, 2, GL.GL_FLOAT, GL.GL_TRUE, 20,
        ctypes.c_void_p(12))

    # load texture and assign texture unit for shaders
    sampleTexture = loadTexture('/home/fei/Downloads/models/002_master_chef_can/texture_map.png')
    texUnitUniform = GL.glGetUniformLocation(shaderProgram, 'texUnit')

    # Finished
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    GL.glBindVertexArray(0)


def render(frame):
    global sampleTexture
    global shaderProgram
    global texUnitUniform
    global VAO
    global faces

    GL.glClearColor(0, 0, 0, 1)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    # active shader program
    GL.glUseProgram(shaderProgram)
    proj = perspective(10, 1, 0.01, 100)
    modelview = lookat([np.sin(frame/100.0) + np.cos(frame/100.0), np.cos(frame/100.0) - np.sin(frame/100.0),1], [0,0,0])
    MVP = modelview.T.dot(proj)
    MVP = np.ascontiguousarray(MVP, np.float32)

    GL.glUniformMatrix4fv(GL.glGetUniformLocation(shaderProgram, 'MVP'), 1, GL.GL_FALSE, MVP)
    GL.glEnable(GL.GL_DEPTH_TEST)

    try:
        # Activate texture
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, sampleTexture)
        GL.glUniform1i(texUnitUniform, 0)

        # Activate array
        GL.glBindVertexArray(VAO)

        # draw triangles
        GL.glDrawElements(GL.GL_TRIANGLES, faces.size, GL.GL_UNSIGNED_INT, faces)

    finally:
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    GL.glDisable(GL.GL_DEPTH_TEST)

    frame = GL.glReadPixels(0, 0, WIDTH, HEIGHT, GL.GL_BGRA, GL.GL_FLOAT)
    frame = frame[::-1, ::-1]

    return frame

if __name__ == "__main__":
    main()
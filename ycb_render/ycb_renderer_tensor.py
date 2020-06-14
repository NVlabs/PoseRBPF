import ctypes
import torch
from pprint import pprint
from PIL import Image
import glutils.glcontext as glcontext
import OpenGL.GL as GL
import cv2
import numpy as np
from pyassimp import *
from glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat
import CppYCBRenderer
from progressbar import progressbar as bar
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import sys

try:
    from .get_available_devices import *
except:
    from get_available_devices import *

MAX_NUM_OBJECTS = 3
from glutils.utils import colormap


def loadTexture(path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.fromstring(img.tobytes(), np.uint8)
    # print(img_data.shape)
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


class YCBTensorRenderer:
    def __init__(self, width=512, height=512, gpu_id=0, render_rotation=False):
        self.shaderProgram = None
        self.VAOs = []
        self.VBOs = []
        self.textures = []
        self.objects = []
        self.texUnitUniform = None
        self.width = width
        self.height = height
        self.faces = []
        self.poses_trans = []
        self.poses_rot = []
        self.instances = []
        # self.context = glcontext.Context()
        # self.context.create_opengl_context((self.width, self.height))

        self.render_rotation = render_rotation

        self.r = CppYCBRenderer.CppYCBRenderer(width, height, get_available_devices()[gpu_id])
        self.r.init()

        self.glstring = GL.glGetString(GL.GL_VERSION)
        from OpenGL.GL import shaders

        self.shaders = shaders
        self.colors = [[0.9, 0, 0], [0.6, 0, 0], [0.3, 0, 0],
                       [0.0, 0.3, 0], [0, 0.6, 0], [0, 0, 0.9]]
        self.lightcolor1 = [1, 1, 1]
        self.lightcolor2 = [1, 0.5, 1]
        self.lightcolor3 = [1, 1, 0.5]

        self.cuda_buffer = None
        self.cuda_buffer2 = None
        vertexShader = self.shaders.compileShader("""
        #version 450
        uniform mat4 V;
        uniform mat4 P;
        uniform mat4 pose_rot;
        uniform mat4 pose_trans;
        uniform vec3 instance_color; 

        layout (location=0) in vec3 position;
        layout (location=1) in vec3 normal;
        layout (location=2) in vec2 texCoords;
        out vec2 theCoords;
        out vec3 Normal;
        out vec3 FragPos;
        out vec3 Normal_cam;
        out vec3 Instance_color;
        out vec3 Pos_cam;
        out vec3 Pos_obj;
        void main() {
            gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
            vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
            FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
            Normal = normalize(mat3(pose_rot) * normal); // in world coordinate
            Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate

            vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
            Pos_cam = pos_cam4.xyz / pos_cam4.w;
            Pos_obj = position;

            theCoords = texCoords;
            Instance_color = instance_color;
        }
        """, GL.GL_VERTEX_SHADER)

        fragmentShader = self.shaders.compileShader("""
        #version 450
        uniform sampler2D texUnit;
        in vec2 theCoords;
        in vec3 Normal;
        in vec3 Normal_cam;
        in vec3 FragPos;
        in vec3 Instance_color;
        in vec3 Pos_cam;
        in vec3 Pos_obj;

        layout (location = 0) out vec4 outputColour;
        layout (location = 1) out vec4 NormalColour;
        layout (location = 2) out vec4 InstanceColour;
        layout (location = 3) out vec4 PCColour;
        layout (location = 4) out vec4 PCObject;

        uniform vec3 light_position1;  // in world coordinate
        uniform vec3 light_position2;  // in world coordinate
        uniform vec3 light_position3;  // in world coordinate
        uniform vec3 light_color1; // light color
        uniform vec3 light_color2; // light color
        uniform vec3 light_color3; // light color

        void main() {
            float ambientStrength = 0.4;
            float diffuseStrength = 0.6;
            vec3 ambient_color = vec3(1,1,1);
            vec3 ambient = ambientStrength * ambient_color;
            vec3 lightDir1 = normalize(light_position1 - FragPos);
            vec3 lightDir2 = normalize(light_position2 - FragPos);
            vec3 lightDir3 = normalize(light_position3 - FragPos);
            float diff1 = max(dot(Normal, lightDir1), 0.0);
            float diff2 = max(dot(Normal, lightDir2), 0.0);
            float diff3 = max(dot(Normal, lightDir3), 0.0);
            vec3 diffuse = (diff1 * light_color1 + diff2 * light_color2 + diff3 * light_color3) * diffuseStrength;

            outputColour =  texture(texUnit, theCoords) * vec4(diffuse + ambient, 1);
            NormalColour =  vec4((Normal_cam + 1) / 2,1);
            InstanceColour = vec4(Instance_color,1);
            PCColour = vec4(Pos_cam,1);
            PCObject = vec4(Pos_obj,1);
        }
        """, GL.GL_FRAGMENT_SHADER)

        vertexShader_simple = self.shaders.compileShader("""
            #version 450
            uniform mat4 V;
            uniform mat4 P;

            layout (location=0) in vec3 position;
            layout (location=1) in vec3 normal;
            layout (location=2) in vec2 texCoords;
            out vec3 Normal_cam;
            void main() {
                gl_Position = P * V * vec4(position,1);
                Normal_cam = normal;
            }
            """, GL.GL_VERTEX_SHADER)

        fragmentShader_simple = self.shaders.compileShader("""
            #version 450
            in vec3 Normal_cam;
            layout (location = 0) out vec4 outputColour;
            layout (location = 1) out vec4 NormalColour;
            layout (location = 2) out vec4 InstanceColour;
            layout (location = 3) out vec4 PCColour;
            void main() {
                outputColour = vec4(Normal_cam, 1.0);
                NormalColour = vec4(0,0,0,0);
                InstanceColour = vec4(0,0,0,0);
                PCColour = vec4(0,0,0,0);
            }
            """, GL.GL_FRAGMENT_SHADER)

        self.shaderProgram = self.shaders.compileProgram(vertexShader, fragmentShader)
        self.texUnitUniform = GL.glGetUniformLocation(self.shaderProgram, 'texUnit')

        self.shaderProgram_simple = self.shaders.compileProgram(
            vertexShader_simple, fragmentShader_simple)

        self.lightpos1 = [0, 1, 0]
        self.lightpos2 = [1, 0, 0]
        self.lightpos3 = [0, 0, 1]

        self.fbo = GL.glGenFramebuffers(1)
        self.color_tex = GL.glGenTextures(1)
        self.color_tex_2 = GL.glGenTextures(1)
        self.color_tex_3 = GL.glGenTextures(1)
        self.color_tex_4 = GL.glGenTextures(1)
        self.color_tex_5 = GL.glGenTextures(1)

        self.depth_tex = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_2)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_3)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_4)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_5)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)

        GL.glTexImage2D.wrappedOperation(
            GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH24_STENCIL8, self.width, self.height, 0,
            GL.GL_DEPTH_STENCIL, GL.GL_UNSIGNED_INT_24_8, None)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, self.color_tex, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1,
                                  GL.GL_TEXTURE_2D, self.color_tex_2, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2,
                                  GL.GL_TEXTURE_2D, self.color_tex_3, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3,
                                  GL.GL_TEXTURE_2D, self.color_tex_4, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4,
                                  GL.GL_TEXTURE_2D, self.color_tex_5, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL.GL_TEXTURE_2D, self.depth_tex, 0)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffers(5, [GL.GL_COLOR_ATTACHMENT0,
                             GL.GL_COLOR_ATTACHMENT1,
                             GL.GL_COLOR_ATTACHMENT2,
                             GL.GL_COLOR_ATTACHMENT3,
                             GL.GL_COLOR_ATTACHMENT4])

        assert GL.glCheckFramebufferStatus(
            GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

        self.fov = 20
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        P = perspective(self.fov, float(self.width) /
                        float(self.height), 0.01, 100)
        V = lookat(
            self.camera,
            self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)

        if render_rotation:
            initial_distribution = np.ones((37 * 72, ), dtype=np.float32)
            # initial_distribution /= np.sum(initial_distribution)

            self.views_vis = np.load('./code/views_vis.npy') * 0.5

            print(self.views_vis)

            self.rotation_visualization = self.generate_rotation_visualization(initial_distribution)

    def generate_rotation_visualization(self, distribution):
        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)

        vertexData = []

        for i in range(int(self.views_vis.shape[0])):
            if distribution[i] >  0.001:
                x, y, z = self.views_vis[i, 0], self.views_vis[i, 1], self.views_vis[i, 2]
                vertexData.append([-x * (1 - distribution[i]), -y * (1 - distribution[i]), -z * (1 - distribution[i]), 0.1, 1.8, 0.1, 0, 0])
                vertexData.append([-x, -y, -z, 1.8, 0.1, 0.1, 0, 0])

        if len(vertexData) > 0:
            vertexData = np.array(vertexData).astype(np.float32) * 3
            # Need VBO for triangle vertices and texture UV coordinates
            VBO = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData,
                            GL.GL_STATIC_DRAW)

            # enable array and set up data
            positionAttrib = GL.glGetAttribLocation(
                self.shaderProgram_simple, 'position')
            normalAttrib = GL.glGetAttribLocation(
                self.shaderProgram_simple, 'normal')

            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(1)

            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32,
                                     None)
            GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32,
                                     ctypes.c_void_p(12))

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)

        return VAO

    def load_object(self, obj_path, texture_path):
        texture = loadTexture(texture_path)
        self.textures.append(texture)

        scene = load(obj_path)
        mesh = scene.meshes[0]
        faces = mesh.faces
        vertices = np.concatenate(
            [mesh.vertices, mesh.normals, mesh.texturecoords[0, :, :2]], axis=-1)
        vertexData = vertices.astype(np.float32)
        release(scene)

        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)

        # Need VBO for triangle vertices and texture UV coordinates
        VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData,
                        GL.GL_STATIC_DRAW)

        # enable array and set up data
        positionAttrib = GL.glGetAttribLocation(self.shaderProgram, 'position')
        normalAttrib = GL.glGetAttribLocation(self.shaderProgram, 'normal')
        coordsAttrib = GL.glGetAttribLocation(self.shaderProgram, 'texCoords')

        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(1)
        GL.glEnableVertexAttribArray(2)

        GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32,
                                 None)
        GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32,
                                 ctypes.c_void_p(12))
        # the last parameter is a pointer
        GL.glVertexAttribPointer(coordsAttrib, 2, GL.GL_FLOAT, GL.GL_TRUE, 32,
                                 ctypes.c_void_p(24))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

        self.VAOs.append(VAO)
        self.VBOs.append(VBO)
        self.faces.append(faces)
        self.objects.append(obj_path)
        self.poses_rot.append(np.eye(4))
        self.poses_trans.append(np.eye(4))

    def load_objects(self, obj_paths, texture_paths):
        for i in range(len(obj_paths)):
            self.load_object(obj_paths[i], texture_paths[i])
            self.instances.append(len(self.instances))


    def set_camera(self, camera, target, up):
        self.camera = camera
        self.target = target
        self.up = up
        V = lookat(
            self.camera,
            self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)

    def set_fov(self, fov):
        self.fov = fov
        P = perspective(self.fov, float(self.width)/float(self.height), 0.01, 100)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_light_color(self, color1, color2, color3):
        self.lightcolor1 = color1
        self.lightcolor2 = color2
        self.lightcolor3 = color3

    def set_perspective(self, P):
        self.P = np.ascontiguousarray(P, np.float32)
        self.V = np.identity(4, np.float32)

    def render(self, image, seg, pc=None, normal=None):
        frame = 0
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        if self.render_rotation:
            # render some grid and directions
            GL.glUseProgram(self.shaderProgram_simple)
            GL.glBindVertexArray(self.rotation_visualization)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(
                self.shaderProgram, 'V'), 1, GL.GL_TRUE, self.V)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(
                self.shaderProgram, 'P'), 1, GL.GL_FALSE, self.P)
            GL.glDrawElements(GL.GL_LINES, 6000,
                              GL.GL_UNSIGNED_INT, np.arange(6000, dtype=np.int))
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)

        for i in range(len(self.instances)):
            # active shader program
            GL.glUseProgram(self.shaderProgram)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(
                self.shaderProgram, 'V'), 1, GL.GL_TRUE, self.V)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(
                self.shaderProgram, 'P'), 1, GL.GL_FALSE, self.P)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram, 'pose_trans'), 1, GL.GL_FALSE,
                                  self.poses_trans[i])
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram, 'pose_rot'), 1, GL.GL_TRUE,
                                  self.poses_rot[i])
            GL.glUniform3f(GL.glGetUniformLocation(
                self.shaderProgram, 'light_position1'), *self.lightpos1)
            GL.glUniform3f(GL.glGetUniformLocation(
                self.shaderProgram, 'light_position2'), *self.lightpos2)
            GL.glUniform3f(GL.glGetUniformLocation(
                self.shaderProgram, 'light_position3'), *self.lightpos3)
            GL.glUniform3f(GL.glGetUniformLocation(
                self.shaderProgram, 'instance_color'), *self.colors[i])
            GL.glUniform3f(GL.glGetUniformLocation(
                self.shaderProgram, 'light_color1'), *self.lightcolor1)
            GL.glUniform3f(GL.glGetUniformLocation(
                self.shaderProgram, 'light_color2'), *self.lightcolor2)
            GL.glUniform3f(GL.glGetUniformLocation(
                self.shaderProgram, 'light_color3'), *self.lightcolor3)

            try:
                # Activate texture
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.textures[self.instances[i]])
                GL.glUniform1i(self.texUnitUniform, 0)
                # Activate array
                GL.glBindVertexArray(self.VAOs[self.instances[i]])
                # draw triangles
                GL.glDrawElements(
                    GL.GL_TRIANGLES, self.faces[self.instances[i]].size, GL.GL_UNSIGNED_INT, self.faces[self.instances[i]])

            finally:
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)

        GL.glDisable(GL.GL_DEPTH_TEST)

        self.r.map_tensor(int(self.color_tex), int(self.width), int(self.height), image.data_ptr())
        self.r.map_tensor(int(self.color_tex_3), int(self.width), int(self.height), seg.data_ptr())
        if pc is not None:
            self.r.map_tensor(int(self.color_tex_4), int(self.width), int(self.height), pc.data_ptr())
        if normal is not None:
            self.r.map_tensor(int(self.color_tex_2), int(self.width), int(self.height), normal.data_ptr())

    def set_light_pos(self, light1, light2, light3):
        self.lightpos1 = light1
        self.lightpos2 = light2
        self.lightpos3 = light3

    def get_num_objects(self):
        return len(self.objects)
    
    def get_num_instances(self):
        return len(self.instances)
    
    def set_poses(self, poses):
        self.poses_rot = [np.ascontiguousarray(quat2rotmat(item[3:])) for item in poses]
        self.poses_trans = [np.ascontiguousarray(xyz2mat(item[:3])) for item in poses]

    def release(self):
        print(self.glstring)
        self.clean()
        self.r.release()

    def set_projection_matrix(self, w, h, fu, fv, u0, v0, znear, zfar):
        L = -(u0) * znear / fu;
        R = +(w-u0) * znear / fu;
        T = -(v0) * znear / fv;
        B = +(h-v0) * znear / fv;

        P = np.zeros((4, 4), dtype=np.float32);
        P[0, 0] = 2 * znear / (R-L);
        P[1, 1] = 2 * znear / (T-B);
        P[2, 0] = (R+L)/(L-R);
        P[2, 1] = (T+B)/(B-T);
        P[2, 2] = (zfar +znear) / (zfar - znear);
        P[2, 3] = 1.0;
        P[3, 2] = (2*zfar*znear)/(znear - zfar);
        self.P = P

    def clean(self):
        GL.glDeleteTextures([self.color_tex, self.color_tex_2, self.color_tex_3, self.color_tex_4, self.depth_tex])
        self.color_tex = None
        self.color_tex_2 = None
        self.color_tex_3 = None
        self.color_tex_4 = None
        self.depth_tex = None
        GL.glDeleteFramebuffers(1, [self.fbo])
        self.fbo = None
        GL.glDeleteBuffers(len(self.VAOs), self.VAOs)
        self.VAOs = []
        GL.glDeleteBuffers(len(self.VBOs), self.VBOs)
        self.VBOs = []
        GL.glDeleteTextures(self.textures)
        self.textures = []
        self.objects = []  # GC should free things here
        self.faces = []  # GC should free things here
        self.poses_trans = []  # GC should free things here
        self.poses_rot = []  # GC should free things here

    def transform_vector(self, vec):
        vec = np.array(vec)
        zeros = np.zeros_like(vec)
        vec_t = self.transform_point(vec)
        zero_t = self.transform_point(zeros)
        v = vec_t - zero_t
        return v

    def transform_point(self, vec):
        vec = np.array(vec)
        if vec.shape[0] == 3:
            v = self.V.dot(np.concatenate([vec, np.array([1])]))
            return v[:3] / v[-1]
        elif vec.shape[0] == 4:
            v = self.V.dot(vec)
            return v / v[-1]
        else:
            return None

    def transform_pose(self, pose):
        pose_rot = quat2rotmat(pose[3:])
        pose_trans = xyz2mat(pose[:3])
        pose_cam = self.V.dot(pose_trans.T).dot(pose_rot).T
        return np.concatenate([mat2xyz(pose_cam), safemat2quat(pose_cam[:3, :3].T)])

    def get_poses(self):
        mat = [self.V.dot(self.poses_trans[i].T).dot(self.poses_rot[i]).T for i in range(self.get_num_instances())]
        poses = [np.concatenate([mat2xyz(item), safemat2quat(item[:3, :3].T)]) for item in mat]
        return poses


if __name__ == '__main__':
    model_path = sys.argv[1]
    w = 640
    h = 480

    renderer = YCBTensorRenderer(w, h, render_rotation=False)
    models = [
        "061_foam_brick",
        # "002_master_chef_can",
        # "011_banana",
    ]
    obj_paths = ['{}/models/{}/textured_simple.obj'.format(model_path, item) for item in models]
    texture_paths = ['{}/models/{}/texture_map.png'.format(model_path, item) for item in models]
    renderer.load_objects(obj_paths, texture_paths)

    # mat = pose2mat(pose)
    pose = np.array([0, 0, 0, 1, 0, 0, 0])
    # pose2 = np.array([-0.56162935, 0.05060109, -0.028915625, 0.6582951, 0.03479896, -0.036391996, -0.75107396])
    # pose3 = np.array([0.22380374, 0.019853603, 0.12159989, -0.40458265, -0.036644224, -0.6464779, 0.64578354])

    theta = 0
    phi = 0
    psi = 0
    r = 1
    cam_pos = [np.sin(theta) * np.cos(phi) * r, np.sin(phi) * r, np.cos(theta) * np.cos(phi) * r]
    renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])
    renderer.set_fov(40)
    renderer.set_poses([pose])
    renderer.set_light_pos(cam_pos, [0, 1, 0], [1, 0, 0])
    renderer.set_light_color([1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5])
    tensor = torch.cuda.ByteTensor(h, w, 4)
    tensor2 = torch.cuda.ByteTensor(h, w, 4)

    tensor = torch.cuda.FloatTensor(h, w, 4)
    tensor2 = torch.cuda.FloatTensor(h, w, 4)
    pc_tensor = torch.cuda.FloatTensor(h, w, 4)
    normal_tensor = torch.cuda.FloatTensor(h, w, 4)

    while True:
        # renderer.set_light_pos([0,-1 + 0.01 * i, 0])
        renderer.render(tensor, tensor2, pc=None, normal=normal_tensor)

        img_np = tensor.flip(0).data.cpu().numpy().reshape(h, w, 4)
        img_np2 = tensor2.flip(0).data.cpu().numpy().reshape(h, w, 4)
        img_normal = normal_tensor.flip(0).data.cpu().numpy().reshape(h, w, 4)

        img_disp = np.concatenate((img_np[:, :, :3], img_normal[:, :, :3]), axis=1)

        if len(sys.argv) > 2 and sys.argv[2] == 'headless':
            # print(np.mean(frame[0]))
            theta += 0.001
            if theta > 1: break
        else:
            cv2.imshow('test', cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR))
            q = cv2.waitKey(16)
            if q == ord('w'):
                phi += 0.1
            elif q == ord('s'):
                phi -= 0.1
            elif q == ord('a'):
                theta -= 0.1
            elif q == ord('d'):
                theta += 0.1
            elif q == ord('n'):
                r -= 0.1
            elif q == ord('m'):
                r += 0.1
            elif q == ord('p'):
                Image.fromarray((img_np[:, :, :3] * 255).astype(np.uint8)).save('test.png')
            elif q == ord('q'):
                break

            # print(renderer.get_poses())
        cam_pos = [np.sin(theta) * np.cos(phi) * r, np.sin(phi) * r, np.cos(theta) * np.cos(phi) * r]
        renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])
        renderer.set_light_pos(cam_pos, cam_pos, cam_pos)

    renderer.release()
from pyassimp import *
import numpy as np

scene = load('/home/fei/Downloads/models/002_master_chef_can/textured_simple.obj')
print(len(scene.meshes))

mesh = scene.meshes[0]

print(len(mesh.vertices))
print(len(mesh.faces))

faces = mesh.faces
vertices = np.concatenate([mesh.vertices, mesh.texturecoords[0,:,:2]], axis=-1)
#from IPython import embed; embed()
print(faces.shape, vertices.shape)

release(scene)

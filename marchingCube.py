import numpy as np
from engine.config import config
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import constants as const

from skimage.draw import ellipsoid
def marchingCube(voxels):
    width = config['world_width']
    height = config['world_height']
    depth =config['world_depth']

    verts, faces, normals, values = measure.marching_cubes(voxels, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlim((config['world_width']-const.FRAME_SIZE)/2, (config['world_width']+const.FRAME_SIZE)/2)
    ax.set_ylim((config['world_height']-const.FRAME_SIZE)/2, (config['world_height']+const.FRAME_SIZE)/2)
    ax.set_zlim(0, const.FRAME_SIZE)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ellip_base = ellipsoid(6, 10, 16)
    verts, faces, normals, values = measure.marching_cubes(ellip_base, 0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")

    ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 20)  # b = 10
    ax.set_zlim(0, 32)  # c = 16

    plt.tight_layout()
    plt.show()
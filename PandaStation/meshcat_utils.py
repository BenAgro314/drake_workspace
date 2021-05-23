import numpy as np
import pydrake.all

import meshcat.geometry as g
import meshcat.transformations as tf

def draw_open3d_point_cloud(meshcat, pcd, normals_scale=0.0, size=0.001):
    pts = np.asarray(pcd.points)
    meshcat.set_object(g.PointCloud(pts.T, np.asarray(pcd.colors).T, size=size))
    if pcd.has_normals() and normals_scale > 0.0:
        normals = np.asarray(pcd.normals)
        vertices = np.hstack(
            (pts, pts + normals_scale * normals)).reshape(-1, 3).T
        meshcat["normals"].set_object(
            g.LineSegments(g.PointsGeometry(vertices),
                           g.MeshBasicMaterial(color=0x000000)))


def draw_points(meshcat, points, color, **kwargs):
    """Helper for sending a 3xN points of a single color to MeshCat"""
    points = np.asarray(points)
    assert points.shape[0] == 3
    if points.size == 3:
        points.shape = (3, 1)
    colors = np.tile(np.asarray(color).reshape(3, 1), (1, points.shape[1]))
    meshcat.set_object(g.PointCloud(points, colors, **kwargs))

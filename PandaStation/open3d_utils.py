import numpy as np
import open3d as o3d


def create_open3d_point_cloud(point_cloud):
    indices = np.all(np.isfinite(point_cloud.xyzs()), axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.xyzs()[:, indices].T)

    if point_cloud.has_rgbs():
        pcd.colors = o3d.utility.Vector3dVector(point_cloud.rgbs()[:, indices].T
                                                / 255.)

    return pcd


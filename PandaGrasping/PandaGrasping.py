import numpy as np
from PandaStation import (
    PandaStation, FindResource, AddPackagePaths, AddRgbdSensors, 
    draw_points, draw_open3d_point_cloud, create_open3d_point_cloud) 
from pydrake.all import (
        RotationMatrix, RigidTransform, Solve
        )
from PandaInverseKinematics import *
import open3d as o3d


def grasp_candidate_cost(plant_context, cloud, plant, scene_graph, scene_graph_context):

    def is_minimally_valid(X_G):
        plant.SetFreeBodyPose(plant_context, body, X_G)
        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
        
        if query_object.HasCollisions():
            return False

        # Check collisions between the gripper and the point cloud
        margin = 0.0  # must be smaller than the margin used in the point cloud preprocessing.
        for pt in cloud.points:
            distances = query_object.ComputeSignedDistanceToPoint(pt, threshold=margin)
            if distances:
                return False

        return True


    body = plant.GetBodyByName("panda_hand")
    X_G = plant.GetFreeBodyPose(plant_context, body)

    X_GW = X_G.inverse()
    pts = np.asarray(cloud.points).T
    p_GC = X_GW.multiply(pts)

    # Crop to a region inside of the finger box.
    crop_min = [-0.009, -0.04, 0.066]
    crop_max = [0.009, 0.04, 0.12]
    indices = np.all((crop_min[0] <= p_GC[0,:], p_GC[0,:] <= crop_max[0],
                      crop_min[1] <= p_GC[1,:], p_GC[1,:] <= crop_max[1],
                      crop_min[2] <= p_GC[2,:], p_GC[2,:] <= crop_max[2]), 
                     axis=0)

    X_G_lower = RigidTransform(X_G)
    X_G_upper = RigidTransform(X_G)
    if np.sum(indices)>0:
        p_GC_y = p_GC[1, indices]
        p_Gcenter_y = (p_GC_y.min() + p_GC_y.max())/2.0
        X_G.set_translation(X_G.translation() + X_G.rotation().multiply([0, p_Gcenter_y, 0]))
        X_G_upper.set_translation(X_G_upper.translation() + X_G_upper.rotation().multiply([0, p_GC_y.min() + 0.04, 0]))
        X_GW = X_G.inverse()

    if not is_minimally_valid(X_G):
        return np.inf, None, None, None

    # we know the grasp is valid, lets find a range of valid z values 
    # (for flexibility in the IK solution)

    p_GC_z = p_GC[2, indices]
    zs = np.linspace(p_GC_z.min() - 0.0995, p_GC_z.min() - 0.066, 10)

    valids = []
    for p_Gzshift in zs:
        X_Gtest = RigidTransform(X_G)
        X_Gtest.set_translation(X_Gtest.translation() + X_Gtest.rotation().multiply([0,0,p_Gzshift]))
        if is_minimally_valid(X_Gtest):
            valids.append(p_Gzshift)

    if len(valids) != 0:
        X_G_lower.set_translation(X_G_lower.translation() + X_G_lower.rotation().multiply([0, 0, min(valids)]))
        X_G_upper.set_translation(X_G_upper.translation() + X_G_upper.rotation().multiply([0, 0, max(valids)]))

    n_GC = X_GW.rotation().multiply(np.asarray(cloud.normals)[indices,:].T)

    # Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    cost = 20.0*X_G.rotation().matrix()[2, 1]

    # Reward sum |dot product of normals with gripper y|^2
    cost -= np.sum(n_GC[1,:]**2)

    return cost, X_G_lower, X_G, X_G_upper
    
def generate_grasp_candidate_antipodal(env, env_context, cloud, rng): 
    """
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel. 
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].
    """
    
    plant = env.get_multibody_plant()
    plant_context = env.GetSubsystemContext(plant, env_context)
    scene_graph = env.get_scene_graph()
    scene_graph_context = env.GetSubsystemContext(scene_graph, env_context)

    body = plant.GetBodyByName("panda_hand")

    index = rng.integers(0,len(cloud.points)-1)

    # Use S for sample point/frame.
    p_WS = np.asarray(cloud.points[index])
    n_WS = np.asarray(cloud.normals[index])

    assert np.isclose(np.linalg.norm(n_WS), 1.0), "This error is likely because you don't have version 0.10.0.0 of open3d"

    Gy = n_WS # gripper y axis aligns with normal
    # make orthonormal z axis, aligned with world down
    z = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(z,Gy)) < 1e-6:
        return np.inf, None, None, None

    Gz = z - np.dot(z,Gy)*Gy
    Gx = np.cross(Gy, Gz)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0, 0.04, 0.1]

    # Try orientations from the center out
    min_pitch=-np.pi/3.0
    max_pitch=np.pi/3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    thetas = (min_pitch + (max_pitch - min_pitch)*alpha)

    for theta in thetas: 
        # Rotate the object in the hand by a rotation (around the normal).
        R_WG2 = R_WG.multiply(RotationMatrix.MakeYRotation(theta))
        # Use G for gripper frame.
        p_SG_W = - R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W 

        X_G = RigidTransform(R_WG2, p_WG)
        
        plant.SetFreeBodyPose(plant_context, body, X_G)

        cost, X_G_lower, X_G, X_G_upper = grasp_candidate_cost(plant_context, cloud, plant, scene_graph, scene_graph_context)
        
        if np.isfinite(cost):
            return cost, X_G_lower, X_G, X_G_upper

    return np.inf, None, None, None




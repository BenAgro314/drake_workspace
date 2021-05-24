import numpy as np
from PandaStation import (
    PandaStation, FindResource, AddPackagePaths, AddRgbdSensors, 
    draw_points, draw_open3d_point_cloud, create_open3d_point_cloud) 
from pydrake.all import (
        RotationMatrix, RigidTransform, Solve
        )
from PandaInverseKinematics import *
import open3d as o3d

#TODO(ben): add inverse kinematics to cost

def grasp_candidate_cost(X_G, plant_context, cloud, plant, scene_graph, scene_graph_context):
    
    body = plant.GetBodyByName("panda_hand")

    X_GW = X_G.inverse()
    pts = np.asarray(cloud.points).T
    p_GC = X_GW.multiply(pts)

    # Crop to a region inside of the finger box.
    crop_min = [-0.009, -0.035, 0.06]
    crop_max = [0.009, 0.035, 0.115]
    indices = np.all((crop_min[0] <= p_GC[0,:], p_GC[0,:] <= crop_max[0],
                      crop_min[1] <= p_GC[1,:], p_GC[1,:] <= crop_max[1],
                      crop_min[2] <= p_GC[2,:], p_GC[2,:] <= crop_max[2]), 
                     axis=0)



    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    collision_pairs = query_object.ComputePointPairPenetration()
    
    # check collisions with objects in the world 
    if query_object.HasCollisions():
        print('collision with world')
        return np.inf

    # Check collisions between the gripper and the point cloud
    margin = 0.0  # must be smaller than the margin used in the point cloud preprocessing.
    for pt in cloud.points:
        distances = query_object.ComputeSignedDistanceToPoint(pt, threshold=margin)
        if distances:
            print('collision with point cloud')
            return np.inf

    n_GC = X_GW.rotation().multiply(np.asarray(cloud.normals)[indices,:].T)

    # Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    cost = 20.0*X_G.rotation().matrix()[2, 1]

    # Reward sum |dot product of normals with gripper x|^2
    cost -= np.sum(n_GC[0,:]**2)

    return cost

def process_bin_point_cloud(diagram, context, cameras, bin_name):

    plant = diagram.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    # Compute crop box.
    bin_instance = plant.GetModelInstanceByName(bin_name)
    bin_body = plant.GetBodyByName("bin_base", bin_instance)
    X_B = plant.EvalBodyPoseInWorld(plant_context, bin_body)
    margin = 0.001  # only because simulation is perfect!
    a = X_B.multiply([-.22+0.025+margin, -.29+0.025+margin, 0.015+margin])
    b = X_B.multiply([.22-0.1-margin, .29-0.025-margin, 2.0])
    crop_min = np.minimum(a,b)
    crop_max = np.maximum(a,b)

    # Evaluate the camera output ports to get the images.
    merged_pcd = o3d.geometry.PointCloud()
    for c in cameras:
        point_cloud = diagram.GetOutputPort(f"{c}_point_cloud").Eval(context)
        pcd = create_open3d_point_cloud(point_cloud)

        # Crop to region of interest.
        pcd = pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min,
                                                max_bound=crop_max))    

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))

        camera = plant.GetModelInstanceByName(c)
        body = plant.GetBodyByName("base", camera)
        X_C = plant.EvalBodyPoseInWorld(plant_context, body)
        pcd.orient_normals_towards_camera_location(X_C.translation())
        
        # Merge point clouds.
        merged_pcd += pcd

    # Voxelize down-sample.  (Note that the normals still look reasonable)
    return merged_pcd.voxel_down_sample(voxel_size=0.005)



def generate_grasp_candidate_antipodal(plant_context, cloud, plant, scene_graph, scene_graph_context, panda, rng, meshcat=None):
    """
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel. 
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].
    """
    body = plant.GetBodyByName("panda_hand")

    index = rng.integers(0,len(cloud.points)-1)

    # Use S for sample point/frame.
    p_WS = np.asarray(cloud.points[index])
    n_WS = np.asarray(cloud.normals[index])

    if meshcat:
        vertices = np.empty((3,2))
        vertices[:, 0] = p_WS
        vertices[:, 1] = p_WS + 0.05*n_WS
        meshcat.set_object(g.LineSegments(g.PointsGeometry(vertices),
                            g.MeshBasicMaterial(color=0xff0000)))
  

    if not np.isclose(np.linalg.norm(n_WS), 1.0):
        return np.inf, None

    Gy = n_WS # gripper y axis aligns with normal
    # make orthonormal z axis, aligned with world down
    z = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(z,Gy)) < 1e-6:
        # normal was pointing straight down.  reject this sample.
        print('here')
        return np.inf, None

    Gz = z - np.dot(z,Gy)*Gy
    Gx = np.cross(Gy, Gz)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0, 0.035, 0.11]

    # Try orientations from the center out
    min_pitch=-np.pi/3.0
    max_pitch=np.pi/3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    thetas = [0]
    for theta in (min_pitch + (max_pitch - min_pitch)*alpha):
        # Rotate the object in the hand by a random rotation (around the normal).
        R_WG2 = R_WG.multiply(RotationMatrix.MakeYRotation(theta))

        # Use G for gripper frame.
        p_SG_W = - R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W 

        X_G = RigidTransform(R_WG2, p_WG)
        ik = PandaInverseKinematics(plant, plant_context, panda)
        p_WQ = X_G.translation()
        tol = np.ones(3)*0.01
        q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.]) 
        ik.AddPositionConstraint(p_WQ+tol, p_WQ+tol)
        ik.AddOrientationConstraint(X_G.rotation(), 0.01)
        prog = ik.get_prog()
        q = ik.get_q()
        prog.AddQuadraticErrorCost(np.identity(len(q)), q_nominal, q)
        prog.SetInitialGuess(q, q_nominal)
        result = Solve(prog)

        if not result.is_success():
            continue
        
        plant.SetPositions(plant_context, panda, result.GetSolution(q))
        print('evaluating cost')
        cost = grasp_candidate_cost(X_G, plant_context, cloud, plant, scene_graph, scene_graph_context)
        #X_G = plant.GetFreeBodyPose(plant_context, body)
        if np.isfinite(cost):
            return cost, X_G

        #draw_grasp_candidate(X_G)

    return np.inf, None

import numpy as np
from PandaStation import *
from pydrake.all import (
        RotationMatrix, RigidTransform, Solve, 
        Box, Sphere, Cylinder, InverseKinematics,
        LoadModelDirectives, ProcessModelDirectives,
        Parser, FixedOffsetFrame
        )
from .GroundTruthGrasping import *

class PlacementSurface:

    def __init__(self, shape_info, z, bb_min, bb_max):
        # bb: bounding_box coords in frame of shape (np array)
        # z: upwards normal in world frame (np array)
        self.shape_info = shape_info
        self.z = z
        self.bb_min = bb_min
        self.bb_max = bb_max


def is_safe_to_place(placement_shape_info, station, station_context):

    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)

    shape = placement_shape_info.shape
    G = placement_shape_info.frame
    X_WG = G.CalcPoseInWorld(plant_context)

    drop_height = 0.05
    theta_tol = np.pi*0.1

    if type(shape) == Sphere:
        return False, None # don't try to place things on a sphere
    if type(shape) == Cylinder:
        # the z axis of the cylinder needs to be aligned with the world z axis
        z_W = np.array([0, 0, 1])
        z_G = X_WG.rotation().col(2)
        # acute angle between vectors
        dot = z_W.dot(z_G)
        theta = np.arccos(np.clip(dot, -1, 1))
        if (theta < np.pi - theta_tol) and (theta > theta_tol):
            return False, None
        else:
            z_G = z_G * np.sign(dot) # get upwards pointing vector
            bb_min = np.array([-shape.radius(), -shape.radius(), shape.length()/2])
            bb_min[2] = bb_min[2]*np.sign(dot)
            bb_max = np.array([shape.radius(), shape.radius(), shape.length()/2])
            bb_max[2] = bb_min[2]*np.sign(dot)
            if np.sign(dot) > 0:
                bb_max[2] = bb_max[2] + drop_height
            else:
                bb_min[2] = bb_min[2] - drop_height
            return True, PlacementSurface(placement_shape_info, z_G, bb_min, bb_max)
    if type(shape) == Box:
        # check if any of the axes are aligned with the world z axis
        z_W = np.array([0, 0, 1])
        for a in range(3):
            z_cand = X_WG.rotation().col(a)
            dot = z_W.dot(z_cand)
            theta = np.arccos(np.clip(dot, -1, 1))
            if (theta < np.pi - theta_tol) and (theta > theta_tol):
                continue 
            z_G = z_cand * np.sign(dot) # get upwards pointing vector
            bb_min = np.array([-shape.width()/2, -shape.depth()/2, -shape.height()/2])
            bb_max = np.array([shape.width()/2, shape.depth()/2, shape.height()/2])
            bb_min[a] = bb_min[a]*(-1)*np.sign(dot)
            bb_max[a] = bb_max[a]*np.sign(dot)
            if np.sign(dot) > 0:
                bb_max[a] = bb_max[a] + drop_height
            else:
                bb_min[a] = bb_min[a] - drop_height
            return True, PlacementSurface(placement_shape_info, z_G, bb_min, bb_max)
        return False, None

def is_placeable(holding_shape_info):
    shape = holding_shape_info.shape
    if type(shape) == Sphere:
        if shape.radius() < 0.001:
            return False
    return True

def place_pose(holding_body_info, place_body_info, station, station_context,
        q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.]), initial_guess = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])):
    """
    Returns the best generalized coorindates, q (np.array), for the panda in the 
    PandaStation station which as holding_body_info welded to it's end effector 
    to place that object stabley onto place_model_name
    """

    placement_shape_infos = place_body_info.shape_infos

    qs = []
    costs = []
    for placement_shape_info in placement_shape_infos:
        stat, surface = is_safe_to_place(placement_shape_info, 
                station, 
                station_context)
        if not stat:
            continue
        for holding_shape_info in holding_body_info.shape_infos:
            # filter out tiny shapes 
            if not is_placeable(holding_shape_info): 
                q, cost = sphere_placement_pose(holding_shape_info, 
                        surface,
                        station, 
                        station_context, 
                        q_nominal = q_nominal, 
                        initial_guess = initial_guess)
            if holding_shape_info.type == Cylinder:
                q, cost = cylinder_placement_pose(holding_shape_info, 
                        surface,
                        station, 
                        station_context, 
                        q_nominal = q_nominal, 
                        initial_guess = initial_guess)
            if holding_shape_info.type == Box:
                q, cost = box_placement_pose(holding_shape_info, 
                        surface,
                        station, 
                        station_context, 
                        q_nominal = q_nominal, 
                        initial_guess = initial_guess)
            if holding_shape_info.type == Sphere:
                q, cost = sphere_placement_pose(holding_shape_info,
                        surface,
                        station, 
                        station_context, 
                        q_nominal = q_nominal, 
                        initial_guess = initial_guess)
            qs.append(q)
            costs.append(cost)
    indices = np.argsort(costs)
    return qs[indices[0]], costs[indices[0]]

def extract_box_corners(box, axis, sign):
    x = np.array([box.width()/2, 0, 0])
    y = np.array([0, box.depth()/2, 0])
    z = np.array([0, 0, box.height()/2])
    vecs = [x, y, z]
    a = vecs.pop(axis)
    a = a*sign
    corners = []
    for i in range(4):
        signs = [ (i & 0b1)*2 - 1, ((i>>1) & 0b1)*2 - 1]
        corner = a
        for s, v in zip(signs, vecs):
            corner = corner + s*v
        corners.append(corner)
    return corners

def extract_cylinder_corners(cylinder, sign):
    dim = cylinder.radius()*np.sqrt(2)
    x = np.array([dim, 0, 0])
    y = np.array([0, dim, 0])
    z = np.array([0, 0, cylinder.length()/2])
    vecs = [x, y]
    a = z*sign
    corners = []
    for i in range(4):
        signs = [ (i & 0b1)*2 - 1, ((i>>1) & 0b1)*2 - 1]
        corner = a
        for s, v in zip(signs, vecs):
            corner = corner + s*v
        corners.append(corner)
    return corners

def sphere_placement_pose(holding_shape_info,
                        surface,
                        station, 
                        station_context, 
                        q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.]), 
                        initial_guess = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])):

    # only try to place cylinders on their flat sides

    weights = np.array([0,1])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid weights"
    weights = weights/norm

    plant = station.get_multibody_plant()
    assert (q_nominal is None) or plant.num_positions() == len(q_nominal), "incorret length of q_nominal"
    plant_context = station.GetSubsystemContext(plant, station_context)
    H = holding_shape_info.frame
    sphere = holding_shape_info.shape
    P = surface.shape_info.frame

    a = None
    for i in range(len(surface.bb_min)):
        if np.isclose(surface.bb_min[i], surface.bb_max[i]*-1):
            surface.bb_min[i] = surface.bb_min[i] + sphere.radius() 
            surface.bb_max[i] = surface.bb_max[i] - sphere.radius() 
        else:
            sign = np.sign(surface.bb_max[i])
            surface.bb_max[i] = surface.bb_max[i] + sign * sphere.radius()
            surface.bb_min[i] = surface.bb_min[i] + sign * sphere.radius()
    
    ik = InverseKinematics(plant, plant_context)
    ik.AddMinimumDistanceConstraint(0.001, 0.1)
    ik.AddPositionConstraint(
            H,
            np.zeros(3),
            P,
            surface.bb_min,
            surface.bb_max)

    prog = ik.prog()
    q = ik.q()
    AddDeviationFromVerticalCost(prog, q, 
            plant, plant_context, weight = weights[1])
    if q_nominal is not None:
        prog.AddQuadraticErrorCost(weights[0]*np.identity(len(q)), 
                q_nominal, q)
    if initial_guess is not None:
        prog.SetInitialGuess(q, initial_guess)

    result = Solve(prog)
    cost = result.get_optimal_cost()

    if not result.is_success():
        cost = np.inf

    return result.GetSolution(q), cost

def cylinder_placement_pose(holding_shape_info,
                        surface,
                        station, 
                        station_context, 
                        q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.]), 
                        initial_guess = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])):

    # only try to place cylinders on their flat sides

    weights = np.array([0,1])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid weights"
    weights = weights/norm

    plant = station.get_multibody_plant()
    assert (q_nominal is None) or plant.num_positions() == len(q_nominal), "incorret length of q_nominal"
    plant_context = station.GetSubsystemContext(plant, station_context)
    H = holding_shape_info.frame
    cylinder = holding_shape_info.shape
    P = surface.shape_info.frame

    signs = [-1,1]
    costs = []
    qs = []

    theta_tol = np.pi*0.01

    # create conservative bounding box around cylinder

    for sign in signs:

        ik = InverseKinematics(plant, plant_context)
        ik.AddMinimumDistanceConstraint(0.001, 0.1)
        # conservative cylinder corners must lie in above placement surface face
        corners = extract_cylinder_corners(cylinder, sign)
        for corner in corners:
            ik.AddPositionConstraint(
                    H,
                    corner,
                    P,
                    surface.bb_min,
                    surface.bb_max)

        n = np.array([0, 0, -1])*sign
        ik.AddAngleBetweenVectorsConstraint(
                H,
                n,
                plant.world_frame(),
                surface.z,
                0,
                theta_tol)
        prog = ik.prog()
        q = ik.q()
        AddDeviationFromVerticalCost(prog, q, 
                plant, plant_context, weight = weights[1])
        if q_nominal is not None:
            prog.AddQuadraticErrorCost(weights[0]*np.identity(len(q)), 
                    q_nominal, q)

        if initial_guess is not None:
            prog.SetInitialGuess(q, initial_guess)

        result = Solve(prog)
        cost = result.get_optimal_cost()

        if not result.is_success():
            cost = np.inf

        costs.append(cost)
        qs.append(result.GetSolution(q))
            
    indices = np.argsort(costs)
    return qs[indices[0]], min(costs)# return lowest cost


def box_placement_pose(holding_shape_info,
                        surface,
                        station, 
                        station_context, 
                        q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.]), 
                        initial_guess = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])):

    # weighting parameters in order:
    # deviation_from_nominal_weight
    # deviation_from_vertical_weight 
    weights = np.array([0,1])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid weights"
    weights = weights/norm

    plant = station.get_multibody_plant()
    assert (q_nominal is None) or plant.num_positions() == len(q_nominal), "incorret length of q_nominal"
    plant_context = station.GetSubsystemContext(plant, station_context)
    H = holding_shape_info.frame
    box = holding_shape_info.shape
    P = surface.shape_info.frame

    axes = [0, 1, 2] # x y z
    signs = [-1,1]
    costs = []
    qs = []

    theta_tol = np.pi*0.01

    
    for sign in signs:
        for a in axes:
            ik = InverseKinematics(plant, plant_context)
            ik.AddMinimumDistanceConstraint(0.001, 0.1)
            dim = box_dim_from_index(a, box)
            #corners of face must lie in bounding box
            corners = extract_box_corners(box, a, sign)
            for corner in corners:
                ik.AddPositionConstraint(
                        H,
                        corner,
                        P,
                        surface.bb_min,
                        surface.bb_max)

            n = np.zeros(3)
            n[a] = -sign
            ik.AddAngleBetweenVectorsConstraint(
                    H,
                    n,
                    plant.world_frame(),
                    surface.z,
                    0,
                    theta_tol)
            prog = ik.prog()
            q = ik.q()
            AddDeviationFromVerticalCost(prog, q, 
                    plant, plant_context, weight = weights[1])
            if q_nominal is not None:
                prog.AddQuadraticErrorCost(weights[0]*np.identity(len(q)), 
                        q_nominal, q)

            if initial_guess is not None:
                prog.SetInitialGuess(q, initial_guess)

            result = Solve(prog)
            cost = result.get_optimal_cost()

            if not result.is_success():
                cost = np.inf

            costs.append(cost)
            qs.append(result.GetSolution(q))
            #TODO(deviation from placement surface center)
            
    indices = np.argsort(costs)
    return qs[indices[0]], min(costs)# return lowest cost



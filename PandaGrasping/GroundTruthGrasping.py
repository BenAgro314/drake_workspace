import numpy as np
from PandaStation import *
from pydrake.all import (
        RotationMatrix, RigidTransform, Solve, 
        Box, Sphere, Cylinder, InverseKinematics,
        LoadModelDirectives, ProcessModelDirectives,
        Parser, FixedOffsetFrame
        )

class BodyInfo:

    def __init__(self, body_index):
        self.body_index = body_index
        self.shape_infos= []

    def add_shape_info(self, shape_info):
        self.shape_infos.append(shape_info)

class ShapeInfo:

    def __init__(self, shape, frame):
        self.shape = shape
        self.frame = frame
        self.type = type(shape)

    def __str__(self):
        s = None
        if self.type == Box:
            s = "box"
        if self.type == Cylinder:
            s = "cylinder"
        if self.type == Sphere:
            s = "sphere"
        return s + " " + str(self.frame)

def create_welded_station(station, station_context, omni = False,
        body_index_to_weld_to_hand = None):
    """
    Given a PandaStation, return a version with everything welded in place 
    except for the panda arm (fingers are welded as well)
    
    For each body, it iteraties through all collision geometries
    that are relevant to manipulation (the correct size),
    and adds a frame for each of them

    Return the welded station and a list of the relevant ShapeInfo objects
    for the collision geometries that are relevant to manipulation 
    
    Args:
        station: PandaStation instance
        station_context: the context for the panda station
    """
    directive = station.directive 
    if not omni:
        welded_station = PandaStation()
    else:
        welded_station = OmniStation()

    # setup same environment with directive
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    welded_plant = welded_station.get_multibody_plant()
    parser = Parser(welded_plant)
    AddPackagePaths(parser)
    ProcessModelDirectives(LoadModelDirectives(directive), welded_plant, parser)

    #setup hand and arm
    welded_station.SetupDefaultStation(welded_hand = True)
    welded_hand = welded_plant.GetModelInstanceByName("hand")

    scene_graph = station.get_scene_graph()
    scene_graph_context = station.GetSubsystemContext(scene_graph, station_context)
    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    inspector = query_object.inspector()

    # add and weld all the models
    # TODO(ben): currently this only supports models with one body

    welded_body_infos = []

    for path, info in list(station.body_info.items()):
        model_name, body_index = info
        body = plant.get_body(body_index)
        X_WB = body.EvalPoseInWorld(plant_context)
        welded_model = parser.AddModelFromFile(path, model_name)
        if (body_index_to_weld_to_hand is not None) and (body_index == body_index_to_weld_to_hand):
            body_frame = body.body_frame()
            X_HB = body_frame.CalcPose(plant_context, plant.GetFrameByName("panda_hand"))
            welded_plant.WeldFrames(welded_plant.GetFrameByName("panda_hand"), 
                    welded_plant.GetFrameByName(body.name(), welded_model),
                    X_HB)
            continue
        else:
            welded_plant.WeldFrames(welded_plant.world_frame(), 
                    welded_plant.GetFrameByName(body.name(), welded_model),
                    X_WB)
        indices = welded_plant.GetBodyIndices(welded_model)
        assert len(indices) == 1
        welded_body_info = BodyInfo(indices[0])
        welded_body = welded_plant.get_body(indices[0])
        for i, geom_id in enumerate(plant.GetCollisionGeometriesForBody(body)):
            shape = inspector.GetShape(geom_id)
            X_BG = inspector.GetPoseInFrame(geom_id)
            frame_name = "frame_" + model_name+ "_" + welded_body.name() + "_" + str(i)
            frame = welded_plant.AddFrame(FixedOffsetFrame(frame_name, welded_body.body_frame(),
                                        X_BG))
            welded_body_info.add_shape_info(ShapeInfo(shape, frame))
        welded_body_infos.append(welded_body_info)
    
    welded_station.Finalize()

    return welded_station, welded_body_infos

def is_graspable(shape_info):
    shape = shape_info.shape
    if type(shape) == Sphere:
        if shape.radius() < 0.001 or shape.radius() > 0.055: 
            return False
    if type(shape) == Cylinder:
        if (shape.radius() > 0.04) and (shape.length() > 0.08 - 0.006*2): 
            return False
    if type(shape) == Box:
        min_dim = min([shape.depth(), shape.width(), shape.height()])
        min_margin = 0.006
        if  min_dim  >= 0.08 - min_margin*2: 
            return False
    return True

def grasp_pose(body_info, station, station_context, 
        q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])):
    """
    Returns the best generalized coordinates, q (np.array), for the panda
    in the PandaStation station to grasp the shape in shape_info

    Args:
        shape_info: a shape info instance
        station: a PandaStation system
    """
    qs = []
    costs = []
    for shape_info in body_info.shape_infos:
        if not is_graspable(shape_info):
            continue 
        if shape_info.type == Cylinder:
            q, cost = cylinder_grasp_pose(shape_info, station, station_context, q_nominal = q_nominal)
        if shape_info.type == Box:
            q, cost = box_grasp_pose(shape_info, station, station_context, q_nominal = q_nominal)
        if shape_info.type == Sphere:
            q, cost = sphere_grasp_pose(shape_info, station, station_context, q_nominal = q_nominal)
        qs.append(q)
        costs.append(cost)
    indices = np.argsort(costs)
    return qs[indices[0]], costs[indices[0]]

def sphere_grasp_pose(shape_info, station, station_context,
        q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])):
    """
    Returns the best generalized coordinates, q (np.array), for the panda
    in the PandaStation station to grasp the sphere in shape_info

    Args:
        shape_info: a shape info instance with type Sphere
        station: a PandaStation system
    """

    assert shape_info.type == Sphere, "This shape is not a Sphere"

    weights = np.array([1, 100])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid weights"
    weights = weights/norm

    plant = station.get_multibody_plant()
    assert (q_nominal is None) or plant.num_positions() == len(q_nominal), "Incorrect length of q_nominal"
    plant_context = station.GetSubsystemContext(plant, station_context)
    hand = station.GetHand()
    hand_frame = plant.GetFrameByName("panda_hand", hand)

    cylinder = shape_info.shape
    G = shape_info.frame
    X_WG = G.CalcPoseInWorld(plant_context)

    p_tol = 10e-3
    theta_tol = 0.01
    finger_width = 0.020

    ik = InverseKinematics(plant, plant_context)
    ik.AddMinimumDistanceConstraint(0, 0.1)
    ik.AddPositionConstraint(
            hand_frame,
            [0, 0, 0.1],
            G,
            [-p_tol, -p_tol, -p_tol],
            [p_tol, p_tol, p_tol])
    prog = ik.prog()
    q = ik.q()
    AddDeviationFromVerticalCost(prog, q, 
            plant, plant_context, weight = weights[1])
    if q_nominal is not None:
        prog.AddQuadraticErrorCost(weights[0]*np.identity(len(q)), 
                q_nominal, q)
        prog.SetInitialGuess(q, q_nominal)
    result = Solve(prog)
    cost = result.get_optimal_cost()

    if not result.is_success():
        cost = np.inf

    return result.GetSolution(q), cost


def cylinder_grasp_pose(shape_info, station, station_context, 
        q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])):
    """
    Returns the best generalized coordinates, q (np.array), for the panda
    in the PandaStation station to grasp the cylinder in shape_info

    Args:
        shape_info: a shape info instance with type Cylinder
        station: a PandaStation system
    """

    assert shape_info.type == Cylinder, "This shape is not a Cylinder"

    weights = np.array([0, 1, 1])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid weights"
    weights = weights/norm

    plant = station.get_multibody_plant()
    assert (q_nominal is None) or plant.num_positions() == len(q_nominal), "Incorrect length of q_nominal"
    plant_context = station.GetSubsystemContext(plant, station_context)
    hand = station.GetHand()
    hand_frame = plant.GetFrameByName("panda_hand", hand)

    cylinder = shape_info.shape
    G = shape_info.frame
    X_WG = G.CalcPoseInWorld(plant_context)

    costs = []
    qs = []

    p_tol = 10e-3
    theta_tol = 0.01
    finger_width = 0.020

    if cylinder.radius() < 0.04:
        lower_z_bound = min(-p_tol, -cylinder.length()/2 + finger_width/2) 
        upper_z_bound = max(p_tol, cylinder.length()/2 - finger_width/2) 
        ik = InverseKinematics(plant, plant_context)
        ik.AddMinimumDistanceConstraint(0, 0.1)
        ik.AddPositionConstraint(
                hand_frame,
                [0, 0, 0.1], 
                G, 
                [-p_tol, -p_tol, lower_z_bound], 
                [p_tol,p_tol, upper_z_bound])
        ik.AddAngleBetweenVectorsConstraint(
                hand_frame,
                [0, 1, 0],
                G, 
                [0, 0, 1],
                np.pi/2-theta_tol,
                np.pi/2+theta_tol)
        prog = ik.prog()
        q = ik.q()
        AddDeviationFromVerticalCost(prog, q, 
                plant, plant_context, weight = weights[1])
        AddDeviationFromCylinderMiddleCost(prog, q,
                plant, plant_context, G, weight = weights[2])
        if q_nominal is not None:
            prog.AddQuadraticErrorCost(weights[0]*np.identity(len(q)), 
                    q_nominal, q)
            prog.SetInitialGuess(q, q_nominal)
        result = Solve(prog)
        cost = result.get_optimal_cost()

        if not result.is_success():
            cost = np.inf

        costs.append(cost)
        qs.append(result.GetSolution(q))

    if cylinder.length() < 0.08:
        signs = [-1, 1]
        for flip in signs:
            margin = 0.08 - cylinder.length()
            radius = cylinder.radius()
            lower_xy_bound = min(-radius + finger_width/2, -p_tol)
            upper_xy_bound = max(radius - finger_width/2, p_tol)
            ik = InverseKinematics(plant, plant_context)
            ik.AddMinimumDistanceConstraint(0, 0.1)
            ik.AddPositionConstraint(
                    hand_frame,
                    [0, 0.04*flip, 0.1], 
                    G, 
                    [lower_xy_bound, lower_xy_bound, cylinder.length()/2], 
                    [upper_xy_bound, upper_xy_bound, cylinder.length()/2 + margin])
            ik.AddPositionConstraint(
                    hand_frame,
                    [0, -0.04*flip, 0.1], 
                    G, 
                    [lower_xy_bound, lower_xy_bound, -cylinder.length()/2 - margin], 
                    [upper_xy_bound, upper_xy_bound, -cylinder.length()/2])
            ik.AddAngleBetweenVectorsConstraint(
                    hand_frame, 
                    [0, flip, 0],
                    plant.world_frame(),
                    X_WG.rotation().col(2),
                    0.0,
                    0.01)
            prog = ik.prog()
            q = ik.q()
            AddDeviationFromVerticalCost(prog, q, 
                    plant, plant_context, weight = weights[1])
            AddDeviationFromCylinderMiddleCost(prog, q,
                    plant, plant_context, G, weight = weights[2])
            if q_nominal is not None:
                prog.AddQuadraticErrorCost(weights[0]*np.identity(len(q)), 
                        q_nominal, q)
                prog.SetInitialGuess(q, q_nominal)
            result = Solve(prog)
            cost = result.get_optimal_cost()

            if not result.is_success():
                cost = np.inf

            costs.append(cost)
            qs.append(result.GetSolution(q))

    indices = np.argsort(costs)
    return qs[indices[0]], min(costs)# return lowest cost

def AddDeviationFromCylinderMiddleCost(prog, q, plant, plant_context, G, weight = 1):
    plant_ad = plant.ToAutoDiffXd()
    plant_context_ad = plant_ad.CreateDefaultContext()
    G_ad = plant_ad.GetFrameByName(G.name())

    def deviation_from_cylinder_middle_cost(q):
        # H: hand frame
        # G: cylinder frame
        p_HC_H = [0, 0, 0.1] 
        plant_ad.SetPositions(plant_context_ad, q)
        hand_frame = plant_ad.GetFrameByName("panda_hand")
        X_GH_G = hand_frame.CalcPose(plant_context_ad, G_ad)
        R_GH = X_GH_G.rotation()
        p_GH_G = X_GH_G.translation()
        # distance from cylinder center to hand middle
        p_GC_G = p_GH_G + R_GH.multiply(p_HC_H)
        return p_GC_G.dot(p_GC_G)

    cost = lambda q: weight*deviation_from_cylinder_middle_cost(q)
    prog.AddCost(cost, q) 

def box_grasp_pose(shape_info, station, station_context,
        q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])):
    """
    Returns the best generalized coordinates, q (np.array), for the panda
    in the PandaStation station to grasp the box in shape_info

    Args:
        shape_info: a shape info instance with type box
        station: a PandaStation system
    """

    assert shape_info.type == Box, "This shape is not a Box"

    """
    Here we try and grab each of the 3 pairs of faces. 
    Then try the same pairs with the gripper flipped
    The constraints and costs are:
        - no collisions with other objects
        - quadratic error cost from nominal joint position
        - PositionConstraint the point on the gripper finger 
          must be within the bounding box of the side
        - AngleBetweenVectorsConstraint that the 
          angle between the normal of the box side and the 
          y axis of the gripper frame must be small
        - TODO(ben): quadratic cost for cartesian distance from
          center of mass from body
    """

    # weighting parameters in order:
    # deviation_from_nominal_weight
    # deviation_from_vertical_weight 
    # deviation_from_box_center_weight 
    # TODO(ben): make something clever that depends on object size
    weights = np.array([0,1,100])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid weights"
    weights = weights/norm



    plant = station.get_multibody_plant()
    assert (q_nominal is None) or plant.num_positions() == len(q_nominal), "incorret length of q_nominal"
    plant_context = station.GetSubsystemContext(plant, station_context)
    hand = station.GetHand()
    hand_frame = plant.GetFrameByName("panda_hand", hand)

    box = shape_info.shape
    G = shape_info.frame
    X_WG = G.CalcPoseInWorld(plant_context)

    axes = [0, 1, 2] # x y z
    signs = [-1,1]
    costs = []
    qs = []
    for flip in signs:
        for a in axes:
            #for axis in axes:
            ik = InverseKinematics(plant, plant_context)
            ik.AddMinimumDistanceConstraint(0, 0.1)
            dim = None
            if a == 0: #x
                dim = box.width()
            if a == 1: #y
                dim = box.depth()
            if a == 2: #z
                dim = box.height()
            margin = 0.08 - dim
            eps = 1e-3
            if (margin < 0.006 + eps):
                continue 
            unit_vec = np.zeros(3)
            unit_vec[a] += 1
            p_GQu_G = [box.width()/2 + margin/2, box.depth()/2, box.height()/2]
            p_GQu_G[a] += margin
            p_GQl_G = [-box.width()/2, - box.depth()/2, - box.height()/2]
            p_GQl_G[a]*= -1.0
            ik.AddPositionConstraint(hand_frame, 
                    [0, 0.04*flip, 0.1],
                    G,
                    p_GQl_G,
                    p_GQu_G)

            p_GQu_G = [box.width()/2, box.depth()/2, box.height()/2]
            p_GQu_G[a]*= -1.0
            p_GQl_G = [-box.width()/2, - box.depth()/2, - box.height()/2]
            p_GQl_G[a] -= margin
            ik.AddPositionConstraint(hand_frame, 
                    [0, -0.04*flip, 0.1],
                    G,
                    p_GQl_G,
                    p_GQu_G)

            ik.AddAngleBetweenVectorsConstraint(hand_frame, 
                    [0, flip*1, 0],
                    plant.world_frame(),
                    X_WG.rotation().col(a),
                    0.0,
                    0.01)

            prog = ik.prog()
            q = ik.q()
            AddDeviationFromVerticalCost(prog, q, 
                    plant, plant_context, weight = weights[1])
            AddDeviationFromBoxCenterCost(prog, q,
                    plant, plant_context, X_WG.translation(),
                    weight = weights[2])
            if q_nominal is not None:
                prog.AddQuadraticErrorCost(weights[0]*np.identity(len(q)), 
                        q_nominal, q)
                prog.SetInitialGuess(q, q_nominal)
            result = Solve(prog)
            cost = result.get_optimal_cost()

            if not result.is_success():
                cost = np.inf

            costs.append(cost)
            qs.append(result.GetSolution(q))


    indices = np.argsort(costs)
    return qs[indices[0]], min(costs)# return lowest cost


def AddDeviationFromVerticalCost(prog, q, plant, plant_context, weight = 1):
    plant_ad = plant.ToAutoDiffXd()
    plant_context_ad = plant_ad.CreateDefaultContext()

    def deviation_from_vertical_cost(q):
        plant_ad.SetPositions(plant_context_ad, q)
        hand_frame = plant_ad.GetFrameByName("panda_hand")
        X_WH = hand_frame.CalcPoseInWorld(plant_context_ad)
        R_WH = X_WH.rotation()
        z_H = R_WH.matrix().dot(np.array([0,0,1])) # extract the z direction 
        return z_H.dot(np.array([0,0,1]))

    cost = lambda q: weight*deviation_from_vertical_cost(q)
    prog.AddCost(cost, q) 

#TODO(ben): geometric center -> mass center
def AddDeviationFromBoxCenterCost(prog, q, plant, plant_context, p_WC, weight = 1):
    plant_ad = plant.ToAutoDiffXd()
    plant_context_ad = plant_ad.CreateDefaultContext()

    def deviation_from_box_center_cost(q):
        plant_ad.SetPositions(plant_context_ad, q)
        hand_frame = plant_ad.GetFrameByName("panda_hand")
        # C: center of box
        # H: hand frame
        # W: world frame
        # M: point in between fingers
        X_WH = hand_frame.CalcPoseInWorld(plant_context_ad)
        R_WH = X_WH.rotation()
        p_WH = X_WH.translation()
        p_HM_H = np.array([0, 0, 0.1])
        p_WM = p_WH + R_WH.multiply(p_HM_H)
        # we do not care about z (as much?) TODO(ben): look into this
        return ((p_WM - p_WC)[:-1]).dot((p_WM - p_WC)[:-1]) 

    cost = lambda q: weight*deviation_from_box_center_cost(q)
    prog.AddCost(cost, q) 


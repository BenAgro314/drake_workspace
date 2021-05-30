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

def create_welded_station(station, station_context):
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
    welded_station = PandaStation()

    # setup same environment with directive
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    welded_plant = welded_station.get_multibody_plant()
    parser = Parser(welded_plant)
    AddPackagePaths(parser)
    ProcessModelDirectives(LoadModelDirectives(directive), welded_plant, parser)

    #setup hand and arm
    welded_station.SetupDefaultStation(welded_hand = True)

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








# DEPRECIATED
def parse_manipuland_shapes(station, station_context):
    """
    Return a list of ShapeInfo instances for all of the 
    manipulands that are possible to grasp in the 
    station

    Args:
        station: PandaStation system
        station_context: context for the PandaStation system
    """
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    scene_graph = station.get_scene_graph()
    scene_graph_context = station.GetSubsystemContext(scene_graph, station_context)
    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    inspector = query_object.inspector()
    shape_infos = []
    for body_id in station.object_ids:
        body = plant.get_body(body_id)
        X_WB = body.EvalPoseInWorld(plant_context)
        for geom_id in plant.GetCollisionGeometriesForBody(body):
            shape = inspector.GetShape(geom_id)
            X_BG = inspector.GetPoseInFrame(geom_id)
            if type(shape) == Sphere:
                if shape.radius() < 0.001 or shape.radius() > 0.055: 
                    # we won't consider picking up a sphere with these dimensions
                    continue 
                else:
                    shape_infos.append(ShapeInfo(body, X_WB, X_BG, shape))
            if type(shape) == Cylinder:
                if shape.radius() > 0.04: 
                    # we won't consider picking up a cylinder with these dimensions
                    continue 
                else:
                    shape_infos.append(ShapeInfo(body, X_WB, X_BG, shape))
            if type(shape) == Box:
                max_dim = max([shape.depth(), shape.width(), shape.height()])
                if  max_dim  > 0.08: 
                    # we won't consider picking up a cylinder with these dimensions
                    continue 
                else:
                    shape_infos.append(ShapeInfo(body, X_WB, X_BG, shape))

    return shape_infos

def is_graspable(shape_info):
    shape = shape_info.shape
    if type(shape) == Sphere:
        if shape.radius() < 0.001 or shape.radius() > 0.055: 
            return False
    if type(shape) == Cylinder:
        if shape.radius() > 0.04: 
            return False
    if type(shape) == Box:
        max_dim = max([shape.depth(), shape.width(), shape.height()])
        if  max_dim  > 0.08: 
            return False
    return True

def grasp_pose(body_info, station, station_context):
    """
    Returns the best generalized coordinates, q (np.array), for the panda
    in the PandaStation station to grasp the shape in shape_info

    Args:
        shape_info: a shape info instance
        station: a PandaStation system
    """
    grasp_qs = []
    for shape_info in body_info.shape_infos:
        if not is_graspable(shape_info):
            continue 

        if shape_info.type == Cylinder:
            grasp_qs.append(cylinder_grasp_pose(shape_info, station, station_context))
        if shape_info.type == Box:
            grasp_qs.append(box_grasp_pose(shape_info, station, station_context))
        if shape_info.type == Sphere:
            grasp_qs.append(sphere_grasp_pose(shape_info, station, station_context))

    return grasp_qs

def box_grasp_pose(shape_info, station, station_context):
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


    plant = station.get_multibody_plant()
    assert plant.num_positions() == 7, "This plant is not suitable for inverse kinematics"
    plant_context = station.GetSubsystemContext(plant, station_context)
    hand = station.GetHand()
    hand_frame = plant.GetFrameByName("panda_hand", hand)

    box = shape_info.shape
    G = shape_info.frame
    X_WG = G.CalcPoseInWorld(plant_context)

    axes = [0, 1, 2] # x y z
    #for axis in axes:
    ik = InverseKinematics(plant, plant_context)
    ik.AddMinimumDistanceConstraint(0, 0.1)
    # grabbing it widthwise
    margin = 0.08 - box.width()
    print('width', box.width())
    col_margin = 10e-3
    print(margin)
    if (margin < 0):
        return None
    p_GQu_G = [box.width()/2 + margin/2, box.depth()/2, box.height()/2]
    p_GQl_G = [box.width()/2, - box.depth()/2, - box.height()/2]
    print('lower', p_GQl_G, 'upper', p_GQu_G)
    ik.AddPositionConstraint(hand_frame, 
            [0, 0.04, 0.1],
            G,
            p_GQl_G,
            p_GQu_G)

    p_GQu_G = [-box.width()/2, box.depth()/2, box.height()/2]
    p_GQl_G = [-box.width()/2 - margin/2, - box.depth()/2, - box.height()/2]
    print('lower', p_GQl_G, 'upper', p_GQu_G)
    ik.AddPositionConstraint(hand_frame, 
            [0, -0.04, 0.1],
            G,
            p_GQl_G,
            p_GQu_G)

    ik.AddAngleBetweenVectorsConstraint(hand_frame, 
            [0, 1, 0],
            plant.world_frame(),
            X_WG.rotation().col(0),
            0.0,
            0.01)

    prog = ik.prog()
    q = ik.q()
    q_nominal = np.array([ 0., 0.55, 0., -1.45, 0., 1.58, 0.])
    prog.AddQuadraticErrorCost(np.identity(len(q)), q_nominal, q)
    prog.SetInitialGuess(q, q_nominal)
    result = Solve(prog)
    cost = result.get_optimal_cost()
    print(result.is_success())
    print(result.get_optimal_cost())
    return result.GetSolution(q)



    

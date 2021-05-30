from PandaStation import (
    PandaStation, FindResource, AddPackagePaths, AddRgbdSensors, draw_points, draw_open3d_point_cloud, 
    create_open3d_point_cloud)

from PandaGrasping import *

# Start a single meshcat server instance to use for the remainder of this notebook.
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])

# Let's do all of our imports here, too.
import numpy as np
import ipywidgets
import pydot
import pydrake.all
import os
from IPython.display import display, SVG
import open3d as o3d
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


import pydrake.all
from pydrake.geometry import Cylinder, Box
from pydrake.all import (
    RigidTransform, RotationMatrix, AngleAxis, RollPitchYaw, InverseKinematics, MultibodyPlant, Parser,
    FindResourceOrThrow, Solve, PiecewisePolynomial, TrajectorySource, SceneGraph, DiagramBuilder,
    AddMultibodyPlantSceneGraph, LinearBushingRollPitchYaw, MathematicalProgram, AutoDiffXd, GenerateHtml, Role,
    MakeRenderEngineVtk, DepthRenderCamera, RenderCameraCore, CameraInfo, ClippingRange,  DepthImageToPointCloud,
    BaseField, RenderEngineVtkParams, ConnectMeshcatVisualizer, DepthRange, RgbdSensor, MeshcatPointCloudVisualizer,
    LoadModelDirectives, ProcessModelDirectives, Box, Sphere, Cylinder
    )
from PandaInverseKinematics import PandaInverseKinematics, PandaIKTraj, Waypoint, Trajectory
from RRT import PandaRRTPlanner, PandaRRTompl
from collections import OrderedDict

import matplotlib.pyplot as plt

ycb = {"cracker": "drake/manipulation/models/ycb/sdf/003_cracker_box.sdf", 
    "sugar": "drake/manipulation/models/ycb/sdf/004_sugar_box.sdf", 
    "soup": "drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf", 
    "mustard": "drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf", 
    "gelatin": "drake/manipulation/models/ycb/sdf/009_gelatin_box.sdf", 
    "meat": "drake/manipulation/models/ycb/sdf/010_potted_meat_can.sdf",
    "brick": "drake/examples/manipulation_station/models/061_foam_brick.sdf"}

def ycb_resource(name):
    global ycb
    return FindResourceOrThrow(ycb[name])

def random_ycb_resource():
    global ycb
    ycb_items = list(ycb.items())
    index = np.random.randint(0, len(ycb_items))
    return ycb_resource(ycb_items[index][0])

builder = DiagramBuilder()

station = builder.AddSystem(PandaStation())
station.SetupTableStation(welded_hand = True)
#station.AddModelFromFile(ycb_resource("soup"), RigidTransform(RotationMatrix.MakeXRotation(0), [1, 1, 0]))
station.AddModelFromFile(ycb_resource("brick"), RigidTransform(RotationMatrix(), [0.6, 0, 0.05]))
station.Finalize()

station_context = station.CreateDefaultContext()
scene_graph = station.get_scene_graph()
scene_graph_context = station.GetSubsystemContext(scene_graph, station_context)
plant = station.get_multibody_plant()
plant_context = station.GetSubsystemContext(plant, station_context)


meshcat = pydrake.systems.meshcat_visualizer.ConnectMeshcatVisualizer(builder,
          scene_graph,
          output_port=station.GetOutputPort("query_object"),
          delete_prefix_on_load=True,                                      
          zmq_url=zmq_url, role = Role.kProximity)



shapes = parse_manipuland_shapes(station, station_context)
print(shapes)
box = shapes[0]
q = grasp_pose(box, station, station_context)
print(q)
plant.SetPositions(plant_context, q)

diagram = builder.Build()
context = diagram.CreateDefaultContext()

meshcat.load()
diagram.Publish(context)

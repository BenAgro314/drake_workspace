from PandaStation import *
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
import subprocess


import pydrake.all
from pydrake.geometry import Cylinder, Box
from pydrake.all import (
    RigidTransform, RotationMatrix, AngleAxis, RollPitchYaw, InverseKinematics, MultibodyPlant, Parser,
    FindResourceOrThrow, Solve, PiecewisePolynomial, TrajectorySource, SceneGraph, DiagramBuilder,
    AddMultibodyPlantSceneGraph, LinearBushingRollPitchYaw, MathematicalProgram, AutoDiffXd, GenerateHtml, Role,
    LeafSystem, AbstractValue, PublishEvent, TriggerType, BasicVector, PiecewiseQuaternionSlerp,
    RandomGenerator, UniformlyRandomRotationMatrix, ConnectMeshcatVisualizer
    )
import pydrake.perception as mut
import open3d as o3d
from ompl import base as ob
from ompl import geometric as og
import time
from enum import Enum

ycb = {"cracker": "drake/manipulation/models/ycb/sdf/003_cracker_box.sdf", 
    "sugar": "drake/manipulation/models/ycb/sdf/004_sugar_box.sdf", 
    "soup": "drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf", 
    "mustard": "drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf", 
    "gelatin": "drake/manipulation/models/ycb/sdf/009_gelatin_box.sdf", 
    "meat": "drake/manipulation/models/ycb/sdf/010_potted_meat_can.sdf",
    "brick": "drake/examples/manipulation_station/models/061_foam_brick.sdf"}

for key in ycb.keys():
    ycb[key] = FindResourceOrThrow(ycb[key])
    
ycb["puck"] = FindResource("models/puck.urdf")
ycb["marble"] = FindResource("models/marble.urdf")

def ycb_resource(name):
    global ycb
    return ycb[name]

def random_ycb_resource():
    global ycb
    ycb_items = list(ycb.items())
    index = np.random.randint(0, len(ycb_items))
    return ycb_resource(ycb_items[index][0])

#--------------------------------------------------------------------------

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.001)
parser = Parser(plant)

brickA = parser.AddModelFromFile(
        FindResource("models/puck.sdf"), "brickA")
brickB = parser.AddModelFromFile(
        FindResource("models/puck.sdf"), "brickB")
table = parser.AddModelFromFile(
        FindResource("models/table.urdf"), "table")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link", table), RigidTransform(RotationMatrix(), [0, 0, 0]))

A = plant.GetFrameByName("top", brickA)
B = plant.GetFrameByName("bottom", brickB)

#k0 = np.array([100.,100.,100.])
#d0 = np.array([100.,100.,100.])
#kx = np.array([1000,1000,1000.])
#dx = np.array([100,100,100])

k0 = np.array([0,0,0])
d0 = np.array([0,0,0])
kx = np.array([10,10,10])
dx = np.array([1,1,1])

joint = LinearBushingRollPitchYaw(A, B, k0, d0, kx, dx)
plant.AddForceElement(joint)

plant.Finalize()

body = plant.get_body(plant.GetBodyIndices(brickA)[0])
plant.SetDefaultFreeBodyPose(body, RigidTransform(RotationMatrix(), [0, 0, 0.0627]))
body = plant.get_body(plant.GetBodyIndices(brickB)[0])
plant.SetDefaultFreeBodyPose(body, RigidTransform(RotationMatrix(), [0, 0, 0.0881]))

meshcat = pydrake.systems.meshcat_visualizer.ConnectMeshcatVisualizer(builder,
          scene_graph,
          output_port=scene_graph.get_query_output_port(),
          delete_prefix_on_load=True,                                      
          zmq_url=zmq_url, role = Role.kProximity)# <- this commented part allows visualization of the collisions
meshcat.load()
diagram = builder.Build()
simulator = pydrake.systems.analysis.Simulator(diagram)
simulator_context = simulator.get_mutable_context()
meshcat.start_recording()
t_max = 5
t = 0
dt = 0.01
t_crit = 3
while t <= t_max - dt:
    t += dt
    simulator.AdvanceTo(t)
    """
    if np.isclose(t, t_crit):
        print("here")
        plant_context = plant.GetMyContextFromRoot(simulator_context)
        joint.SetForceStiffnessConstants(plant_context, np.array([0,0,0])) # testing changing sim context based on certain events
        joint.SetTorqueStiffnessConstants(plant_context, np.array([0,0,0]))
        joint.SetTorqueDampingConstants(plant_context, np.array([0,0,0]))
        joint.SetForceDampingConstants(plant_context, np.array([0,0,0]))
    """
#simulator.AdvanceTo(5)
meshcat.stop_recording()
meshcat.publish_recording()
print("done simulation")
input("press Enter to exit simulation")
